"""Bounded HTTP benchmark accounting, independent of model/training dependencies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

STATES = {"NSW", "VIC", "QLD", "SA", "TAS"}


def synthetic_request():
    """Use one retained source-scale synthetic history; no dataset/test access."""
    return {
        "state": "TAS",
        "history": [15 + 20 * math.sin(2 * math.pi * i / 48) for i in range(512)],
    }


def digest(value):
    """Hash canonical JSON, including the complete declared input and settings."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Create an evidence file without replacing an earlier run."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_plan(plan):
    """Reject unsafe budgets, nonlocal endpoints and incomplete declarations."""
    if plan["evidence_kind"] not in {"fixture", "selected-target"}:
        raise ValueError("Declare fixture or selected-target evidence")
    if plan["purpose"] not in {"discovery", "confirmation"}:
        raise ValueError("Declare discovery or confirmation")
    url = urlsplit(plan["base_url"])
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or not url.port
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
        or url.username
        or url.password
    ):
        raise ValueError("Run on the target against http://127.0.0.1:PORT only")
    request = plan["request"]
    if set(request) != {"state", "history"} or request["state"] not in STATES:
        raise ValueError("Use the PR #86 state/history request contract")
    history = request["history"]
    if (
        not isinstance(history, list)
        or len(history) != 512
        or any(
            type(v) not in {int, float} or not math.isfinite(v) or abs(v) > 3.4028234663852886e38
            for v in history
        )
    ):
        raise ValueError("Declare exactly 512 finite float32-representable values")
    measurement = plan["measurement"]
    for key in (
        "budget_seconds",
        "request_timeout_seconds",
        "startup_seconds",
        "warmup_requests",
        "requests_per_level",
        "overload_requests",
        "consecutive_timeout_limit",
        "memory_limit_bytes",
    ):
        if type(measurement[key]) not in {int, float} or not math.isfinite(measurement[key]):
            raise ValueError(f"Invalid {key}")
        if measurement[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in (
        "warmup_requests",
        "requests_per_level",
        "overload_requests",
        "consecutive_timeout_limit",
    ):
        if type(measurement[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if not 10 <= measurement["budget_seconds"] <= 900:
        raise ValueError("Each lifecycle must fit 10..900 seconds; two runs total at most 1800")
    if not 0 < measurement["request_timeout_seconds"] <= measurement["budget_seconds"] - 5:
        raise ValueError("Request timeout must leave five seconds for cleanup")
    if not 0 < measurement["startup_seconds"] <= measurement["budget_seconds"] - 5:
        raise ValueError("Startup deadline must fit within the total budget")
    if measurement["concurrency"] != [1, 2]:
        raise ValueError("This protocol declares concurrency [1, 2], at most two HTTP workers")
    if measurement["requests_per_level"] < 20:
        raise ValueError("Declare at least 20 observations per warm level")
    margin = measurement["proposal_margin_fraction"]
    if type(margin) not in {int, float} or not math.isfinite(margin) or not 0 < margin <= 1:
        raise ValueError("Predeclare an NFR proposal margin in (0, 1]")
    expected = plan["expected"]
    if (
        expected["model_id"] != "chronos-t5-small-fine-tuned"
        or not re.fullmatch(r"[0-9a-f]{64}", expected["checkpoint"]["sha256"])
        or not re.fullmatch(r"[0-9a-f]{32}", expected["checkpoint"]["training_run_id"])
        or expected["model_revision"] != "local-sha256:" + expected["checkpoint"]["sha256"]
        or expected["context_length"] != 512
        or expected["horizon"] != 48
        or expected["value_scale"] != "source"
    ):
        raise ValueError("Declare the selected checkpoint and source-scale contract")
    if digest(expected["settings"]) != digest(
        {
            "num_samples": 20,
            "seed": 7,
            "device": "cpu",
            "cpu_threads": 2,
            "dtype": "float32",
            "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
            "quantile_method": "linear",
            "point_statistic": "median",
        }
    ):
        raise ValueError("Preserve selected inference settings, including 20 samples and seed 7")
    for key in ("service_commit", "benchmark_commit", "params_sha256"):
        size = 64 if key == "params_sha256" else 40
        if not re.fullmatch(r"[0-9a-f]{" + str(size) + r"}", plan["source"][key]):
            raise ValueError(f"Declare a full {key}")
    hardware = plan["hardware"]
    for key in ("host_identity", "os", "cpu"):
        if not isinstance(hardware[key], str) or not hardware[key].strip():
            raise ValueError(f"Declare actual hardware {key}")
    for key in ("cpu_count", "memory_bytes"):
        if type(hardware[key]) is not int or hardware[key] <= 0:
            raise ValueError(f"Declare measured {key}")
    startup = plan["startup"]
    if not Path(startup["cwd"]).is_absolute():
        raise ValueError("Lifecycle cwd must be absolute")
    for key in ("argv", "stop_argv"):
        if not isinstance(startup[key], list) or any(
            not isinstance(arg, str) or not arg for arg in startup[key]
        ):
            raise ValueError("Lifecycle commands must be argv arrays, without a shell")
    if not startup["argv"]:
        raise ValueError("Cold start needs an owned foreground lifecycle")
    resource = plan["resource"]
    if resource["scope"] not in {"process", "container"}:
        raise ValueError("Declare process or container memory scope")
    if resource["scope"] == "container" and (
        not startup["stop_argv"] or not Path(resource["cgroup_path"]).is_absolute()
    ):
        raise ValueError("Container lifecycle needs explicit cleanup and an isolated cgroup")
    if resource["scope"] == "container" and (
        type(resource.get("cgroup_version")) is not int or resource["cgroup_version"] not in {1, 2}
    ):
        raise ValueError("Declare the actual memory cgroup version, 1 or 2")
    if resource["scope"] == "container" and not re.fullmatch(
        r"sha256:[0-9a-f]{64}", plan["source"].get("image_id") or ""
    ):
        raise ValueError("Declare the immutable container image ID")
    if measurement["memory_limit_bytes"] >= hardware["memory_bytes"]:
        raise ValueError("Memory stop threshold must leave host headroom")


def validate_receipt(plan, receipt):
    """Require #75's target/readiness/isolation handoff before selected-model load."""
    if plan["evidence_kind"] == "fixture":
        return
    if not receipt or any(
        receipt.get(key) is not True
        for key in (
            "target_identity_verified",
            "readiness_verified",
            "selected_forecast_verified",
            "isolated_instance",
            "authorized_for_benchmark",
        )
    ):
        raise ValueError("#75 target identity, real readiness and isolation receipt required")
    for key in (
        "base_url",
        "source",
        "expected",
        "hardware",
        "startup",
        "resource",
        "measurement",
    ):
        if receipt.get(key) != plan[key]:
            raise ValueError(f"#75 receipt does not match declared {key}")
    if receipt.get("request_sha256") != digest(plan["request"]):
        raise ValueError("#75 receipt does not match declared request")


def validate_forecast(body, plan):
    """Check the API response identity/settings and finite ordered 48-value arrays."""
    if not isinstance(body, dict):
        return False
    if body.get("state") != plan["request"]["state"] or any(
        body.get(key) != value for key, value in plan["expected"].items()
    ):
        return False
    # Python considers True == 1; executed settings must retain JSON types too.
    try:
        if digest(body.get("settings")) != digest(plan["expected"]["settings"]):
            return False
    except (ValueError, TypeError):
        return False
    quantiles = [body.get(key) for key in ("p10", "p50", "p90")]
    if any(not isinstance(q, list) or len(q) != 48 for q in quantiles):
        return False
    if any(type(v) not in {int, float} or not math.isfinite(v) for q in quantiles for v in q):
        return False
    return all(low <= mid <= high for low, mid, high in zip(*quantiles))


@dataclass
class Observation:
    """One attempted request, including failures; probes are separate records."""

    phase: str
    concurrency: int
    index: int
    started_seconds: float
    elapsed_seconds: float
    status: int | None
    outcome: str
    response_sha256: str | None = None


class StopRun(Exception):
    """A recorded stop condition; partial evidence remains usable as partial evidence."""


class Budget:
    """One monotonic deadline covering startup, probes, first request and all phases."""

    def __init__(self, settings, clock=time.monotonic):
        self.settings = settings
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + settings["budget_seconds"] - 5
        self.timeouts = 0

    def remaining(self):
        """Reserve cleanup time and never start a request outside the global budget."""
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise StopRun("budget_exhausted")
        return remaining

    def observe(self, observation):
        """Stop on repeated timeout completions or an explicit memory failure."""
        self.timeouts = self.timeouts + 1 if observation.outcome == "timeout" else 0
        if self.timeouts >= self.settings["consecutive_timeout_limit"]:
            raise StopRun("repeated_timeouts")
        if observation.status == 507:
            raise StopRun("server_resource_error")
        if observation.outcome == "invalid_response":
            raise StopRun("identity_settings_or_contract_mismatch")


def percentile(values, fraction):
    """Linear sample quantile, matching the existing project's quantile convention."""
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(observations, elapsed):
    """Compute throughput from phase wall time; failures never enter warm quantiles."""
    successes = [o.elapsed_seconds for o in observations if o.outcome == "ok"]
    statuses = {}
    outcomes = {}
    for observation in observations:
        status = str(observation.status) if observation.status is not None else "no_response"
        statuses[status] = statuses.get(status, 0) + 1
        outcomes[observation.outcome] = outcomes.get(observation.outcome, 0) + 1
    return {
        "attempts": len(observations),
        "successes": len(successes),
        "errors": len(observations) - len(successes),
        "statuses": statuses,
        "outcomes": outcomes,
        "wall_seconds": elapsed,
        "p50_seconds": percentile(successes, 0.5),
        "p95_seconds": percentile(successes, 0.95),
        "successful_requests_per_second": len(successes) / elapsed if elapsed > 0 else None,
        "attempted_requests_per_second": len(observations) / elapsed if elapsed > 0 else None,
    }


def run_phase(phase, concurrency, count, request, budget, emit, check_resources):
    """Run bounded synchronized batches, draining every dispatched request on stop."""
    started = budget.clock()
    observations = []
    stopped = None
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for offset in range(0, count, concurrency):
            try:
                check_resources()
                timeout = min(budget.settings["request_timeout_seconds"], budget.remaining())
            except StopRun as error:
                stopped = str(error)
                break
            for index in range(offset, min(count, offset + concurrency)):
                emit(
                    {
                        "type": "dispatch",
                        "phase": phase,
                        "concurrency": concurrency,
                        "index": index,
                        "at_seconds": budget.clock() - budget.started,
                    }
                )
            futures = [
                pool.submit(request, phase, concurrency, index, timeout)
                for index in range(offset, min(count, offset + concurrency))
            ]
            for future in as_completed(futures):
                observation = future.result()
                observations.append(observation)
                emit({"type": "request", **asdict(observation)})
                try:
                    budget.observe(observation)
                    check_resources()
                    budget.remaining()
                except StopRun as error:
                    stopped = stopped or str(error)
            if stopped:
                break
    return summarize(observations, budget.clock() - started), stopped


def comparison_identity(plan):
    """Bind a confirmation to the same model, inputs, target and measurement protocol."""
    return digest(
        {
            key: plan[key]
            for key in (
                "evidence_kind",
                "base_url",
                "source",
                "expected",
                "hardware",
                "request",
                "measurement",
                "resource",
                "startup",
            )
        }
    )


def require_supervision(report, supervision):
    """A worker-written report is eligible only after successful parent supervision."""
    if not supervision or (
        supervision.get("status") != "success"
        or supervision.get("worker_returncode") != 0
        or supervision.get("cleanup_verified") is not True
        or supervision.get("service_launched") is not True
        or supervision.get("run_id") != report.get("run_id")
        or supervision.get("report_digest") != digest(report)
        or supervision.get("plan_sha256") != digest(report["plan"])
    ):
        raise ValueError("Successful supervisor outcome and verified cleanup are required")


def propose(report, report_sha256, concurrency, supervision=None):
    """Derive an unapproved NFR-7 candidate only from a complete real-target discovery."""
    if (
        report["plan"]["evidence_kind"] != "selected-target"
        or report["plan"]["purpose"] != "discovery"
        or report["status"] != "complete"
        or report["memory"]["peak_bytes"] is None
        or not report.get("cleanup_verified")
    ):
        raise ValueError(
            "NFR-7 proposal requires a complete target discovery with memory evidence"
        )
    require_supervision(report, supervision)
    if supervision["report_sha256"] != report_sha256:
        raise ValueError("Discovery report bytes do not match the supervisor outcome")
    phase = report["phases"][f"warm_c{concurrency}"]
    if phase["errors"] or phase["successes"] < 20:
        raise ValueError("Choose a warm level with at least 20 successes and no errors")
    margin = report["plan"]["measurement"]["proposal_margin_fraction"]
    return {
        "status": "proposed_pending_Didac_review",
        "discovery_run_id": report["run_id"],
        "discovery_report_sha256": report_sha256,
        "discovery_supervision": supervision,
        "comparison_identity": comparison_identity(report["plan"]),
        "concurrency": concurrency,
        "observed_p95_seconds": phase["p95_seconds"],
        "observed_throughput_rps": phase["successful_requests_per_second"],
        "margin_fraction": margin,
        "p95_ceiling_seconds": phase["p95_seconds"] * (1 + margin),
        "throughput_floor_rps": phase["successful_requests_per_second"] / (1 + margin),
        "allowed_errors": 0,
        "observed_peak_bytes": report["memory"]["peak_bytes"],
        "method": "Measured p95 * (1 + declared margin); measured throughput / (1 + margin); unapproved",
    }


def validate_confirmation_plan(plan, proposal):
    """Check discovery provenance and protocol before starting a new lifecycle."""
    proof = proposal.get("discovery_supervision") or {}
    if (
        proof.get("status") != "success"
        or proof.get("worker_returncode") != 0
        or proof.get("cleanup_verified") is not True
        or proof.get("service_launched") is not True
        or proof.get("run_id") != proposal.get("discovery_run_id")
        or proof.get("report_sha256") != proposal.get("discovery_report_sha256")
    ):
        raise ValueError("Confirmation requires successful discovery supervision")
    if (
        plan["purpose"] != "confirmation"
        or plan["evidence_kind"] != "selected-target"
        or proposal.get("status") != "proposed_pending_Didac_review"
    ) or (comparison_identity(plan) != proposal["comparison_identity"]):
        raise ValueError(
            "Confirmation must preserve discovery inputs, target, identity and protocol"
        )


def confirm(report, proposal, supervision=None):
    """Report a separately supervised comparison; never approve an NFR or deployment."""
    validate_confirmation_plan(report["plan"], proposal)
    if report.get("run_id") == proposal.get("discovery_run_id"):
        raise ValueError("Confirmation must use a separate run")
    require_supervision(report, supervision)
    phase = report["phases"].get(f"warm_c{proposal['concurrency']}", {})
    return {
        "proposal": proposal,
        "passed": bool(
            report["status"] == "complete"
            and report.get("cleanup_verified")
            and report.get("memory", {}).get("peak_bytes") is not None
            and phase.get("errors") == proposal["allowed_errors"]
            and phase.get("successes", 0) >= 20
            and phase["p95_seconds"] <= proposal["p95_ceiling_seconds"]
            and phase["successful_requests_per_second"] >= proposal["throughput_floor_rps"]
        ),
        "interpretation": "Separate bounded confirmation; human review and broader evidence pending",
    }
