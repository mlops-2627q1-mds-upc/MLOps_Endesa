"""Own an isolated benchmark lifecycle, HTTP observations and an external watchdog."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
from threading import Event, Lock, Thread, Timer
import time
from urllib.parse import urlsplit
from uuid import uuid4

from src.serving_benchmark import (
    Budget,
    Observation,
    StopRun,
    confirm,
    digest,
    propose,
    run_phase,
    synthetic_request,
    validate_forecast,
    validate_plan,
    validate_receipt,
    write_json,
)

DEFAULT_LOCK = (
    "/Users/dcayuela/Desktop/UPC/MLOPS/MLOps_Endesa/temporary/"
    "m4-m6-work-2026-10-23/m4-m6-compute.lock"
)


@contextmanager
def compute_lock(path):
    """Fail busy instead of overlapping #75/#76/#77 heavyweight subprocesses."""
    with Path(path).open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Journal:
    """Flush every observation so watchdog stops retain partial evidence."""

    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8")  # noqa: SIM115
        # Closed after the sampler joins; this journal spans the complete owned lifecycle.
        self.lock = Lock()

    def emit(self, value):
        """Keep request/resource records intact across the sampling thread."""
        with self.lock:
            self.stream.write(json.dumps(value, allow_nan=False) + "\n")
            self.stream.flush()

    def close(self):
        """Release the evidence descriptor after sampling stops."""
        self.stream.close()


def http_json(base_url, method, route, payload, timeout):
    """Use direct loopback HTTP, no proxies/redirects/retries; cap response size."""
    url = urlsplit(base_url)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    deadline = time.monotonic() + timeout
    expired = Event()
    timer = None
    try:
        connection.connect()
        connected_socket = connection.sock

        def expire():
            expired.set()
            # shutdown unblocks makefile reads too, even after Connection: close.
            try:
                connected_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = Timer(max(0, deadline - time.monotonic()), expire)
        timer.daemon = True
        timer.start()
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        connection.request(method, route, body=data, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read(1024 * 1024 + 1)
        if expired.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("request deadline")
        if len(body) > 1024 * 1024:
            return response.status, None, None
        try:
            parsed = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            parsed = None
        return response.status, parsed, hashlib.sha256(body).hexdigest()
    except (OSError, http.client.HTTPException) as error:
        if expired.is_set():
            raise TimeoutError("request deadline") from error
        raise
    finally:
        if timer is not None:
            timer.cancel()
            timer.join()
        connection.close()


def forecast_request(plan, budget, phase, concurrency, index, timeout):
    """Time HTTP serialization, transfer and validation; retain every attempt."""
    started = budget.clock()
    status, response_hash = None, None
    try:
        status, body, response_hash = http_json(
            plan["base_url"], "POST", "/forecast", plan["request"], timeout
        )
        if status == 200:
            outcome = "ok" if validate_forecast(body, plan) else "invalid_response"
        else:
            outcome = "http_error"
    except TimeoutError:
        outcome = "timeout"
    except (OSError, http.client.HTTPException):
        outcome = "transport_error"
    return Observation(
        phase,
        concurrency,
        index,
        started - budget.started,
        budget.clock() - started,
        status,
        outcome,
        response_hash,
    )


class MemoryMonitor:
    """Sample Linux leader RSS/HWM or an isolated cgroup v2; never report client RSS."""

    def __init__(self, plan, service, journal, started):
        self.plan, self.service, self.journal, self.started = plan, service, journal, started
        self.done = Event()
        self.reason = None
        self.peak = None
        self.samples = 0
        self.unavailable = 0
        self.armed = False
        self.thread = Thread(target=self._watch, daemon=True)

    def sample(self):
        """Read kernel high-water memory; container scope includes all cgroup processes."""
        resource = self.plan["resource"]
        try:
            if resource["scope"] == "container":
                directory = Path(resource["cgroup_path"])
                current = int((directory / "memory.current").read_text(encoding="utf-8"))
                peak = int((directory / "memory.peak").read_text(encoding="utf-8"))
                events = dict(
                    line.split()
                    for line in (directory / "memory.events")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                if int(events.get("oom", 0)) or int(events.get("oom_kill", 0)):
                    self.reason = "container_oom_event"
            else:
                fields = dict(
                    line.split(":", 1)
                    for line in Path(f"/proc/{self.service.process.pid}/status")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if ":" in line
                )
                current = int(fields["VmRSS"].split()[0]) * 1024
                peak = int(fields["VmHWM"].split()[0]) * 1024
        except (OSError, ValueError, KeyError):
            self.unavailable += 1
            if self.armed and self.plan["evidence_kind"] == "selected-target":
                self.reason = self.reason or "memory_telemetry_lost"
            return None
        self.samples += 1
        self.peak = max(self.peak or 0, current, peak)
        self.journal.emit(
            {
                "type": "memory",
                "at_seconds": time.monotonic() - self.started,
                "scope": resource["scope"],
                "current_bytes": current,
                "peak_bytes": peak,
            }
        )
        if self.peak >= self.plan["measurement"]["memory_limit_bytes"]:
            self.reason = self.reason or "memory_limit_reached"
        return current

    def _watch(self):
        while not self.done.is_set():
            self.sample()
            if self.reason:
                # Stop the actual owned service immediately, including a daemon container.
                self.service.stop()
                break
            self.done.wait(0.1)

    def check(self):
        """Stop launching work if memory failed or real-target telemetry is absent."""
        if self.reason:
            raise StopRun(self.reason)
        if not self.samples and self.plan["evidence_kind"] == "selected-target":
            raise StopRun("memory_telemetry_unavailable")

    def report(self):
        """Explicitly qualify scope and missing observations."""
        return {
            "scope": self.plan["resource"]["scope"],
            "peak_bytes": self.peak,
            "samples": self.samples,
            "unavailable_samples": self.unavailable,
            "method": "cgroup v2 memory.current/memory.peak"
            if self.plan["resource"]["scope"] == "container"
            else "Linux /proc owned leader VmRSS/VmHWM; excludes child processes and host memory",
            "host_memory": "unavailable",
            "sampling_interval_seconds": 0.1,
        }


def assert_port_free(base_url):
    """Refuse an already-running service, including an unrelated/non-HTTP listener."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", urlsplit(base_url).port))


def cleanup(startup, process):
    """Stop only the declared isolated lifecycle, with bounded cleanup."""
    clean = True
    if startup["stop_argv"]:
        try:
            result = subprocess.run(
                startup["stop_argv"],
                cwd=startup["cwd"],
                check=False,
                timeout=2,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            clean = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            clean = False
    if process is not None:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    clean = False
        clean = clean and process.poll() is not None
    return clean


class OwnedService:
    """Serialize sampler/finalizer cleanup and keep it idempotent."""

    def __init__(self, startup, process):
        self.startup, self.process = startup, process
        self.lock = Lock()
        self.cleaned = None

    def stop(self):
        """Stop the owned lifecycle exactly once."""
        with self.lock:
            if self.cleaned is None:
                self.cleaned = cleanup(self.startup, self.process)
            return self.cleaned


def runtime_identity(plan):
    """Record actual harness bytes/HEAD; require committed selected-model tooling."""
    root = Path(__file__).resolve().parents[1]
    files = ["src/benchmark_runtime.py", "src/serving_benchmark.py"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *files], cwd=root, text=True
    )
    if plan["evidence_kind"] == "selected-target":
        if status or head != plan["source"]["benchmark_commit"]:
            raise ValueError("Selected-target run requires the declared committed harness HEAD")
        provenance = json.loads(
            (root / "models/chronos-t5-small-fine-tuned.provenance.json").read_text(
                encoding="utf-8"
            )
        )
        if plan["expected"]["checkpoint"] != {
            "sha256": provenance["model_safetensors_sha256"],
            "training_run_id": provenance["training_run_id"],
        }:
            raise ValueError("Selected-target checkpoint differs from tracked provenance")
        if plan["hardware"]["host_identity"] != platform.node():
            raise ValueError("Run on the host identified by #75's receipt")
    return {
        "benchmark_commit": head,
        "benchmark_dirty": bool(status),
        "python": sys.version,
        "platform": platform.platform(),
        "host_identity": platform.node(),
        "module_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
        },
    }


def lifecycle_was_launched(output):
    """Read only complete flushed journal records after a forced worker stop."""
    raw = output / "observations.jsonl"
    if raw.exists():
        for line in raw.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("type") == "lifecycle":
                    return True
            except ValueError:
                continue
    return False


def wait_ready(plan, budget, process, journal, memory):
    """Measure launch to readiness; probe observations cannot contaminate warm latency."""
    deadline = min(budget.deadline, budget.started + plan["measurement"]["startup_seconds"])
    while budget.clock() < deadline:
        if memory.reason:
            raise StopRun(memory.reason)
        if process.poll() is not None:
            raise StopRun("service_exited_during_startup")
        timeout = min(1, deadline - budget.clock())
        try:
            status, body, _ = http_json(plan["base_url"], "GET", "/ready", None, timeout)
            journal.emit(
                {"type": "probe", "status": status, "at_seconds": budget.clock() - budget.started}
            )
            if status == 200 and body == {"status": "ready"}:
                return budget.clock() - budget.started
        except (OSError, http.client.HTTPException):
            journal.emit(
                {"type": "probe", "status": None, "at_seconds": budget.clock() - budget.started}
            )
        time.sleep(min(0.1, max(0, deadline - budget.clock())))
    raise StopRun("startup_deadline")


def execute(plan, output):
    """Worker: retain declarations, raw observations, partial phases and cleanup state."""
    budget = Budget(plan["measurement"])
    journal = Journal(output / "observations.jsonl")
    report = {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "started_unix_seconds": time.time(),
        "plan": plan,
        "plan_sha256": digest(plan),
        "status": "stopped",
        "stop_reason": None,
        "phases": {},
        "cold_start": {
            "scope": "owned service process/container restart; filesystem/model caches uncontrolled",
            "launch_to_ready_seconds": None,
            "first_forecast": None,
        },
    }
    process, memory, service = None, None, None
    try:
        assert_port_free(plan["base_url"])
        if (
            plan["resource"]["scope"] == "container"
            and (Path(plan["resource"]["cgroup_path"]) / "memory.current").exists()
        ):
            raise StopRun("container_cgroup_already_exists")
        env = os.environ.copy()
        env.update(
            {
                key: "2"
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "FORECAST_CPU_THREADS",
                )
            }
        )
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        launch_started = budget.clock()
        report["cold_start"]["launch_started_seconds"] = launch_started - budget.started
        with (output / "service.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(  # no shell, foreground owned lifecycle
                plan["startup"]["argv"],
                cwd=plan["startup"]["cwd"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        journal.emit(
            {
                "type": "lifecycle",
                "pid": process.pid,
                "at_seconds": budget.clock() - budget.started,
            }
        )
        service = OwnedService(plan["startup"], process)
        memory = MemoryMonitor(plan, service, journal, budget.started)
        memory.thread.start()
        ready_elapsed = wait_ready(plan, budget, process, journal, memory)
        report["cold_start"]["launch_to_ready_seconds"] = ready_elapsed - (
            launch_started - budget.started
        )
        # Sample synchronously before real forecast work as well as periodically.
        memory.sample()
        memory.armed = True

        def check():
            memory.check()
            if process.poll() is not None:
                raise StopRun("service_exited")

        def request(phase, concurrency, index, timeout):
            return forecast_request(plan, budget, phase, concurrency, index, timeout)

        settings = plan["measurement"]
        phases = [("first_forecast", 1, 1), ("warmup", 1, settings["warmup_requests"])]
        phases += [
            (f"warm_c{c}", c, settings["requests_per_level"]) for c in settings["concurrency"]
        ]
        phases += [("overload_c2", 2, settings["overload_requests"])]
        for phase, concurrency, count in phases:
            result, stopped = run_phase(
                phase, concurrency, count, request, budget, journal.emit, check
            )
            report["phases"][phase] = result
            if phase == "first_forecast":
                report["cold_start"]["first_forecast"] = result
            if stopped:
                raise StopRun(stopped)
            if phase in {"first_forecast", "warmup"} and result["errors"]:
                raise StopRun("first_request_or_warmup_failed")
        report["status"] = "complete"
    except StopRun as error:
        report["stop_reason"] = str(error)
    except (OSError, ValueError, http.client.HTTPException) as error:
        report["stop_reason"] = type(error).__name__  # exclude paths/credentials/raw server errors
    finally:
        if memory:
            memory.done.set()
            memory.thread.join(timeout=4)
            memory.sample()
        report["memory"] = (
            memory.report()
            if memory
            else {
                "scope": plan["resource"]["scope"],
                "peak_bytes": None,
                "samples": 0,
                "method": "unavailable; lifecycle did not start",
                "host_memory": "unavailable",
            }
        )
        # Never run a stop command when launch was refused (e.g. an occupied port).
        report["cleanup_verified"] = service.stop() if service else True
        if service:
            with socket.socket() as probe:
                probe.settimeout(0.2)
                if probe.connect_ex(("127.0.0.1", urlsplit(plan["base_url"]).port)) == 0:
                    report["cleanup_verified"] = False
        if memory and memory.reason:
            report.update(status="stopped", stop_reason=memory.reason)
        if not report["cleanup_verified"]:
            report.update(status="stopped", stop_reason="cleanup_failed")
        report["total_seconds"] = budget.clock() - budget.started
        report["limitations"] = [
            "Synthetic inputs check serving, not forecast quality; no dataset/test evaluation",
            "Warm quantiles include successful requests only; inspect attempts/errors/raw records",
            "Batched closed-loop concurrency 1/2 is not an arrival-rate capacity test",
            "Overload phase may show queuing without rejection; no inferred admission control",
            "Fixture timings are not selected-model or Virtech evidence"
            if plan["evidence_kind"] == "fixture"
            else "One isolated target and bounded sample; no production SLO or approval",
        ]
        write_json(output / "report.json", report)
        journal.close()
    return report["status"] == "complete"


def supervise(plan_path, output, receipt_path, lock_path, proposal_path=None):
    """Validate gates, lock the whole subprocess lifetime and enforce an external deadline."""
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    validate_plan(plan)
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8")) if receipt_path else None
    validate_receipt(plan, receipt)
    proposal = (
        json.loads(Path(proposal_path).read_text(encoding="utf-8")) if proposal_path else None
    )
    if plan["purpose"] == "confirmation":
        if proposal is None:
            raise ValueError("Confirmation requires a separately retained discovery proposal")
        confirm({"plan": plan, "status": "stopped", "phases": {}}, proposal)
    elif proposal is not None:
        raise ValueError("A discovery run cannot consume a confirmation proposal")
    with compute_lock(lock_path):
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "declaration.json", plan)
        if receipt:
            write_json(output / "target-receipt.json", receipt)
        write_json(output / "runtime-identity.json", runtime_identity(plan))
        argv = [sys.executable, "-m", "src.benchmark_runtime", "_worker", str(output)]
        process = subprocess.Popen(argv, start_new_session=True)
        interrupted = None
        try:
            process.wait(timeout=plan["measurement"]["budget_seconds"] - 3)
        except subprocess.TimeoutExpired:
            interrupted = "external_budget_deadline"
        except KeyboardInterrupt:
            interrupted = "operator_interrupt"
        finally:
            if interrupted:
                # Kill worker and its local descendants; container daemon needs explicit cleanup.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                cleaned = (
                    cleanup(plan["startup"], None) if lifecycle_was_launched(output) else True
                )
                write_json(
                    output / "watchdog.json",
                    {"stop_reason": interrupted, "cleanup_verified": cleaned},
                )
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        path = output / "report.json"
        if interrupted or not path.exists():
            return False
        report = json.loads(path.read_text(encoding="utf-8"))
        if proposal:
            confirmation = confirm(report, proposal)
            write_json(output / "confirmation.json", confirmation)
            if not confirmation["passed"]:
                return False
        return process.returncode == 0 and report["status"] == "complete"


def main():
    """Run/propose CLI; stdout contains only artifact locations and overall status."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("plan")
    run.add_argument("output")
    run.add_argument("--receipt")
    run.add_argument("--proposal")
    run.add_argument("--lock", default=DEFAULT_LOCK)
    request = sub.add_parser("request", help="Retain the 512-value synthetic request")
    request.add_argument("output")
    declare = sub.add_parser("declare", help="Bind #75's receipt and retained request to a plan")
    declare.add_argument("receipt")
    declare.add_argument("request")
    declare.add_argument("output")
    declare.add_argument("--purpose", choices=["discovery", "confirmation"], required=True)
    proposal = sub.add_parser("propose")
    proposal.add_argument("report")
    proposal.add_argument("output")
    proposal.add_argument("--concurrency", type=int, choices=[1, 2], required=True)
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("output")
    args = parser.parse_args()
    if args.command == "_worker":
        output = Path(args.output)
        plan = json.loads((output / "declaration.json").read_text(encoding="utf-8"))
        validate_plan(plan)
        receipt_path = output / "target-receipt.json"
        validate_receipt(
            plan,
            json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt_path.exists()
            else None,
        )
        ok = execute(plan, output)
    elif args.command == "request":
        write_json(args.output, synthetic_request())
        ok = True
    elif args.command == "declare":
        receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
        plan = {
            key: receipt[key]
            for key in (
                "base_url",
                "source",
                "expected",
                "hardware",
                "startup",
                "resource",
                "measurement",
            )
        }
        plan.update(
            evidence_kind="selected-target",
            purpose=args.purpose,
            request=json.loads(Path(args.request).read_text(encoding="utf-8")),
        )
        validate_plan(plan)
        validate_receipt(plan, receipt)
        write_json(args.output, plan)
        ok = True
    elif args.command == "propose":
        if (Path(args.report).parent / "watchdog.json").exists():
            raise ValueError("A forced-stop discovery cannot produce an NFR-7 proposal")
        content = Path(args.report).read_bytes()
        write_json(
            args.output,
            propose(json.loads(content), hashlib.sha256(content).hexdigest(), args.concurrency),
        )
        ok = True
    else:
        ok = supervise(args.plan, args.output, args.receipt, args.lock, args.proposal)
    print("complete" if ok else "stopped; inspect retained evidence")
    return 0 if ok else 1


if __name__ == "__main__":

    def terminate(_signum, _frame):
        """Route operator SIGTERM through the same owned-process cleanup as Ctrl-C."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    sys.exit(main())
