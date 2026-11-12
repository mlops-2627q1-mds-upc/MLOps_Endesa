"""Deterministic arithmetic, deadline, accounting, protocol and lifecycle contracts."""

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from src import benchmark_runtime as runtime
from src import serving_benchmark as benchmark
from src.api.app import create_app
from src.api.settings import Settings

SPEC = importlib.util.spec_from_file_location(
    "benchmark_http", Path(__file__).parent / "fixtures" / "benchmark_http.py"
)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


@pytest.fixture
def plan():
    """Use the same fixture declaration as the documented subprocess smoke."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return FIXTURE.fixture_plan(port, Path(__file__).resolve().parents[1])


class Clock:
    """A monotonic clock controlled only by deterministic requests."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def observation(outcome="ok", status=200, elapsed=1.0):
    """Build a deterministic completed request."""
    return benchmark.Observation("warm_c1", 1, 0, 0, elapsed, status, outcome)


def test_percentile_accounting_uses_successes_and_wall_time():
    values = [observation(elapsed=v) for v in (1, 2, 3, 4)]
    values += [observation("http_error", 503), observation("timeout", None)]
    result = benchmark.summarize(values, 4)
    assert result == {
        "attempts": 6,
        "successes": 4,
        "errors": 2,
        "statuses": {"200": 4, "503": 1, "no_response": 1},
        "outcomes": {"ok": 4, "http_error": 1, "timeout": 1},
        "wall_seconds": 4,
        "p50_seconds": 2.5,
        "p95_seconds": pytest.approx(3.85),
        "successful_requests_per_second": 1,
        "attempted_requests_per_second": 1.5,
    }
    assert benchmark.summarize([], 0)["p95_seconds"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(base_url="http://example.com:80"),
        lambda p: p.update(base_url="http://user:secret@127.0.0.1:80"),
        lambda p: p["request"].update(history=[1.0] * 511),
        lambda p: p["request"]["history"].__setitem__(0, True),
        lambda p: p["request"]["history"].__setitem__(0, float("nan")),
        lambda p: p["request"]["history"].__setitem__(0, 1e39),
        lambda p: p["measurement"].update(budget_seconds=1801),
        lambda p: p["measurement"].update(budget_seconds=901),
        lambda p: p["measurement"].update(request_timeout_seconds=100),
        lambda p: p["measurement"].update(warmup_requests=0),
        lambda p: p["measurement"].update(concurrency=[1, 2, 4]),
        lambda p: p["measurement"].update(requests_per_level=2),
        lambda p: p["measurement"].update(proposal_margin_fraction=0),
        lambda p: p["measurement"].update(memory_limit_bytes=p["hardware"]["memory_bytes"]),
        lambda p: p["expected"]["settings"].update(num_samples=1),
        lambda p: p["startup"].update(argv=[]),
        lambda p: p["resource"].update(scope="host"),
        lambda p: p["resource"].update(scope="container", cgroup_path="/tmp/group"),
    ],
)
def test_protocol_rejects_unsafe_or_changed_inputs(plan, mutate):
    mutate(plan)
    with pytest.raises(ValueError):
        benchmark.validate_plan(plan)


def test_selected_target_receipt_is_required_and_bound_to_identity(plan):
    benchmark.validate_plan(plan)
    benchmark.validate_receipt(plan, None)
    plan["evidence_kind"] = "selected-target"
    with pytest.raises(ValueError, match="#75"):
        benchmark.validate_receipt(plan, None)
    receipt = {
        **deepcopy(plan),
        **dict.fromkeys(
            [
                "target_identity_verified",
                "readiness_verified",
                "selected_forecast_verified",
                "isolated_instance",
                "authorized_for_benchmark",
            ],
            True,
        ),
    }
    receipt["request_sha256"] = benchmark.digest(plan["request"])
    benchmark.validate_receipt(plan, receipt)
    receipt["source"]["service_commit"] = "1" * 40
    with pytest.raises(ValueError, match="source"):
        benchmark.validate_receipt(plan, receipt)


def test_timeout_stop_resets_on_success_and_stops_new_batches(plan):
    clock = Clock()
    budget = benchmark.Budget(plan["measurement"], clock)
    emitted, calls = [], []
    sequence = iter(["timeout", "ok", "timeout", "timeout", "timeout", "ok"])

    def request(phase, concurrency, index, timeout):
        calls.append((phase, concurrency, index, timeout))
        clock.now += 1
        outcome = next(sequence)
        return observation(outcome, None if outcome == "timeout" else 200)

    result, stopped = benchmark.run_phase(
        "warm_c1", 1, 20, request, budget, emitted.append, lambda: None
    )
    assert stopped == "repeated_timeouts"
    assert (
        result["attempts"]
        == len(calls)
        == len([r for r in emitted if r["type"] == "request"])
        == 5
    )
    assert result["errors"] == 4


def test_budget_includes_prior_startup_and_warmup(plan):
    clock = Clock()
    budget = benchmark.Budget(plan["measurement"], clock)
    clock.now = 54
    emitted = []

    def request(_phase, _concurrency, _index, timeout):
        assert timeout == 1
        clock.now += timeout
        return observation()

    result, stopped = benchmark.run_phase(
        "warm_c1", 1, 20, request, budget, emitted.append, lambda: None
    )
    assert result["attempts"] == 1
    assert stopped == "budget_exhausted"


def test_concurrent_stop_accounts_for_all_inflight_requests(plan):
    emitted = []
    budget = benchmark.Budget(plan["measurement"])
    result, stopped = benchmark.run_phase(
        "overload_c2",
        2,
        20,
        lambda *_: observation("http_error", 507),
        budget,
        emitted.append,
        lambda: None,
    )
    assert result["attempts"] == len([r for r in emitted if r["type"] == "request"]) == 2
    assert result["errors"] == 2
    assert stopped == "server_resource_error"


def test_resource_stop_dispatches_nothing(plan):
    def exhausted():
        raise benchmark.StopRun("memory_limit_reached")

    result, stopped = benchmark.run_phase(
        "warm_c1",
        1,
        20,
        lambda *_: pytest.fail("must not dispatch"),
        benchmark.Budget(plan["measurement"]),
        lambda _: None,
        exhausted,
    )
    assert result["attempts"] == 0
    assert stopped == "memory_limit_reached"


@pytest.mark.parametrize(
    "status,content,outcome",
    [
        (503, b'{"detail":"unavailable"}', "http_error"),
        (429, b"{}", "http_error"),
        (200, b"not JSON", "invalid_response"),
        (200, b"[]", "invalid_response"),
    ],
)
def test_http_observations_account_errors_and_contract(plan, status, content, outcome):
    class Handler(BaseHTTPRequestHandler):
        """A real local socket with predetermined responses."""

        def do_POST(self):  # pylint: disable=invalid-name
            """Write one deterministic response."""
            self.send_response(status)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_args):
            """No fixture logs."""

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            plan["base_url"] = f"http://127.0.0.1:{server.server_port}"
            result = runtime.forecast_request(
                plan, benchmark.Budget(plan["measurement"]), "warm_c1", 1, 0, 1
            )
        finally:
            server.shutdown()
            thread.join()
    assert result.status == status
    assert result.outcome == outcome
    assert result.response_sha256 is not None


def test_forecast_validation_preserves_quantile_and_settings_contract(plan):
    body = {
        **deepcopy(plan["expected"]),
        "state": "TAS",
        "p10": [0.0] * 48,
        "p50": [1.0] * 48,
        "p90": [2.0] * 48,
    }
    assert benchmark.validate_forecast(body, plan)
    body["p10"][0] = 3
    assert not benchmark.validate_forecast(body, plan)


@pytest.mark.parametrize(
    "change",
    [None, "num_samples", "seed", "cpu_threads", "checkpoint", "training_run_id"],
)
def test_benchmark_checks_the_served_api_contract_without_a_model(plan, change):
    prediction = {
        **deepcopy(plan["expected"]),
        "p10": [-1.0] * 48,
        "p50": [0.0] * 48,
        "p90": [1.0] * 48,
    }
    if change in {"num_samples", "seed", "cpu_threads"}:
        prediction["settings"][change] += 1
    elif change == "checkpoint":
        prediction["checkpoint"]["sha256"] = "1" * 64
        prediction["model_revision"] = "local-sha256:" + "1" * 64
    elif change == "training_run_id":
        prediction["checkpoint"]["training_run_id"] = "1" * 32
    predictor = Mock()
    predictor.predict.return_value = prediction
    factory = Mock(return_value=predictor)
    settings = Settings()
    with TestClient(create_app(settings, factory)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        response = client.post("/forecast", json=plan["request"])
    assert response.status_code == 200
    assert benchmark.validate_forecast(response.json(), plan) is (change is None)
    predictor.predict.assert_called_once_with(plan["request"]["history"])
    factory.assert_called_once_with(settings)


def test_boolean_or_nonfinite_settings_cannot_pass(plan):
    body = {
        **deepcopy(plan["expected"]),
        "state": "TAS",
        "p10": [0.0] * 48,
        "p50": [1.0] * 48,
        "p90": [2.0] * 48,
    }
    body["settings"]["sampling"]["top_p"] = True
    assert not benchmark.validate_forecast(body, plan)
    body["settings"]["sampling"]["top_p"] = float("nan")
    assert not benchmark.validate_forecast(body, plan)
    body["p10"][0] = True
    assert not benchmark.validate_forecast(body, plan)
    body["p10"][0] = 0
    body["settings"]["seed"] = 8
    assert not benchmark.validate_forecast(body, plan)


def discovery_report(plan):
    """Synthetic report ONLY to test derivation arithmetic; no result is published."""
    return {
        "plan": plan,
        "run_id": "discovery-fixture-arithmetic-only",
        "cleanup_verified": True,
        "status": "complete",
        "memory": {"peak_bytes": 2000},
        "phases": {
            "warm_c1": {
                "errors": 0,
                "successes": 20,
                "p95_seconds": 2,
                "successful_requests_per_second": 0.5,
            }
        },
    }


def supervised_fixture(report, report_sha256="a" * 64):
    """Construct a parent-outcome double for isolated arithmetic/gate tests only."""
    return {
        "status": "success",
        "worker_returncode": 0,
        "cleanup_verified": True,
        "service_launched": True,
        "run_id": report["run_id"],
        "report_digest": benchmark.digest(report),
        "plan_sha256": benchmark.digest(report["plan"]),
        "report_sha256": report_sha256,
    }


def test_proposals_refuse_fixture_and_partial_evidence(plan):
    report = discovery_report(plan)
    with pytest.raises(ValueError, match="target discovery"):
        benchmark.propose(report, "a" * 64, 1, supervised_fixture(report))
    plan["evidence_kind"] = "selected-target"
    report["status"] = "stopped"
    with pytest.raises(ValueError):
        benchmark.propose(report, "a" * 64, 1, supervised_fixture(report))
    report["status"] = "complete"
    report["memory"]["peak_bytes"] = None
    with pytest.raises(ValueError):
        benchmark.propose(report, "a" * 64, 1, supervised_fixture(report))


def test_separate_confirmation_preserves_protocol_and_reports_regression(plan):
    plan["evidence_kind"] = "selected-target"
    report = discovery_report(plan)
    proposal = benchmark.propose(report, "a" * 64, 1, supervised_fixture(report))
    assert proposal["p95_ceiling_seconds"] == 2.4
    assert proposal["status"] == "proposed_pending_Didac_review"
    plan["purpose"] = "confirmation"
    report["run_id"] = "separate-confirmation-fixture-arithmetic-only"
    assert benchmark.confirm(report, proposal, supervised_fixture(report))["passed"]
    report["phases"]["warm_c1"]["p95_seconds"] = 2.5
    assert not benchmark.confirm(report, proposal, supervised_fixture(report))["passed"]
    plan["expected"]["settings"]["seed"] = 8
    with pytest.raises(ValueError, match="preserve"):
        benchmark.confirm(report, proposal, supervised_fixture(report))


def test_same_run_cannot_be_its_own_confirmation(plan):
    plan["evidence_kind"] = "selected-target"
    report = discovery_report(plan)
    proposal = benchmark.propose(report, "a" * 64, 1, supervised_fixture(report))
    plan["purpose"] = "confirmation"
    with pytest.raises(ValueError):
        benchmark.confirm(report, proposal, supervised_fixture(report))


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("source", "service_commit", "1" * 40),
        ("source", "benchmark_commit", "1" * 40),
        ("source", "image_id", "sha256:" + "1" * 64),
        ("source", "params_sha256", "1" * 64),
        ("expected", "checkpoint", {"sha256": "1" * 64, "training_run_id": "1" * 32}),
        ("expected", "settings", {"num_samples": 21, "seed": 8, "cpu_threads": 1}),
        ("hardware", "cpu_count", 1000),
        ("hardware", "memory_bytes", 1000),
        ("request", "history", [0.0] * 512),
        ("resource", "scope", "container"),
        ("startup", "stop_argv", ["docker", "stop", "--time", "20", "fixture-only"]),
        ("measurement", "requests_per_level", 50),
        ("measurement", "request_timeout_seconds", 30),
    ],
)
def test_confirmation_refuses_a_different_release_or_runtime(plan, section, field, value):
    plan["evidence_kind"] = "selected-target"
    report = discovery_report(plan)
    proposal = benchmark.propose(report, "a" * 64, 1, supervised_fixture(report))
    confirmation = deepcopy(plan)
    confirmation["purpose"] = "confirmation"
    confirmation[section][field] = value
    with pytest.raises(ValueError, match="preserve"):
        benchmark.validate_confirmation_plan(confirmation, proposal)


@pytest.mark.parametrize(
    "events,peak,reason",
    [
        ("oom 0\noom_kill 0\n", 2048, None),
        ("oom 0\noom_kill 0\n", 4096, "memory_limit_reached"),
        ("oom 1\noom_kill 1\n", 2048, "container_oom_event"),
    ],
)
def test_container_memory_uses_kernel_peak_and_oom_scope(plan, tmp_path, events, peak, reason):
    plan["resource"] = {"scope": "container", "cgroup_path": str(tmp_path), "cgroup_version": 2}
    plan["measurement"]["memory_limit_bytes"] = 4096
    (tmp_path / "memory.current").write_text("1024")
    (tmp_path / "memory.peak").write_text(str(peak))
    (tmp_path / "memory.events").write_text(events)
    (tmp_path / "memory.max").write_text("max")
    records = []
    monitor = runtime.MemoryMonitor(plan, None, SimpleNamespace(emit=records.append), 0)
    assert monitor.sample() == 1024
    assert monitor.report()["peak_bytes"] == peak
    assert monitor.report()["scope"] == "container"
    assert monitor.report()["host_memory"] == "unavailable"
    assert monitor.reason == reason
    assert records[0]["current_bytes"] == 1024


def test_missing_memory_stays_unavailable_and_blocks_selected_work(plan, tmp_path):
    plan["resource"] = {"scope": "container", "cgroup_path": str(tmp_path), "cgroup_version": 2}
    monitor = runtime.MemoryMonitor(plan, None, SimpleNamespace(emit=lambda _: None), 0)
    assert monitor.sample() is None
    assert monitor.report()["peak_bytes"] is None
    monitor.check()  # explicit fixture: unavailable is retained
    plan["evidence_kind"] = "selected-target"
    with pytest.raises(benchmark.StopRun, match="memory_telemetry_unavailable"):
        monitor.check()
    monitor.armed = True
    monitor.sample()
    assert monitor.reason == "memory_telemetry_lost"


@pytest.mark.parametrize(
    "failcnt,control,peak,reason",
    [
        (0, "oom_kill_disable 0\nunder_oom 0\noom_kill 0\n", 2048, None),
        (1, "oom_kill_disable 0\nunder_oom 0\noom_kill 0\n", 2048, "container_memory_limit_event"),
        (0, "oom_kill_disable 0\nunder_oom 1\n", 2048, "container_oom_event"),
        (0, "oom_kill_disable 0\nunder_oom 0\noom_kill 1\n", 2048, "container_oom_event"),
        (0, "oom_kill_disable 0\nunder_oom 0\n", 4096, "memory_limit_reached"),
    ],
)
def test_cgroup_v1_high_water_and_limit_events_are_distinct(
    plan, tmp_path, failcnt, control, peak, reason
):
    plan["resource"] = {"scope": "container", "cgroup_path": str(tmp_path), "cgroup_version": 1}
    plan["measurement"]["memory_limit_bytes"] = 4096
    for name, value in {
        "memory.usage_in_bytes": "1024",
        "memory.max_usage_in_bytes": str(peak),
        "memory.failcnt": str(failcnt),
        "memory.oom_control": control,
        "memory.limit_in_bytes": "8192",
    }.items():
        (tmp_path / name).write_text(value)
    records = []
    monitor = runtime.MemoryMonitor(plan, None, SimpleNamespace(emit=records.append), 0)
    assert monitor.sample() == 1024
    assert monitor.reason == reason
    report = monitor.report()
    assert report["peak_bytes"] == peak
    assert report["cgroup_version"] == 1
    assert report["container_limit_bytes"] == 8192
    assert report["cgroup_counters"]["failcnt"] == failcnt
    assert report["cgroup_counters"].get("oom_kill") == (
        1 if "oom_kill 1" in control else 0 if "oom_kill 0" in control else None
    )
    assert records[0]["scope"] == "container"
    assert not (tmp_path / "memory.current").exists()  # v2 is never substituted


def test_memory_stop_threshold_must_precede_container_limit(plan, tmp_path):
    plan["resource"] = {"scope": "container", "cgroup_path": str(tmp_path), "cgroup_version": 2}
    for name, value in {
        "memory.current": "1",
        "memory.peak": "2",
        "memory.max": "1000",
        "memory.events": "oom 0\noom_kill 0",
    }.items():
        (tmp_path / name).write_text(value)
    plan["measurement"]["memory_limit_bytes"] = 1000
    monitor = runtime.MemoryMonitor(plan, None, SimpleNamespace(emit=lambda _: None), 0)
    monitor.sample()
    assert monitor.reason == "memory_stop_not_below_container_limit"


def test_standalone_cli_imports_without_site_packages(tmp_path):
    result = subprocess.run(
        [sys.executable, "-S", runtime.__file__, "request", str(tmp_path / "request.json")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    request = json.loads((tmp_path / "request.json").read_text())
    assert request["state"] == "TAS"
    assert len(request["history"]) == 512
    assert min(request["history"]) < 0


def test_memory_stop_cleans_owned_lifecycle_once(plan, monkeypatch):
    stopped = []
    service = runtime.OwnedService(plan["startup"], object())
    monkeypatch.setattr(
        runtime, "cleanup", lambda startup, process: stopped.append((startup, process)) or True
    )
    monitor = runtime.MemoryMonitor(plan, service, None, 0)
    monitor.reason = "memory_limit_reached"
    monkeypatch.setattr(monitor, "sample", lambda: None)
    monitor._watch()  # pylint: disable=protected-access
    assert service.stop()
    assert len(stopped) == 1


def test_startup_clock_deadline_retains_probe_errors(plan, monkeypatch):
    clock = Clock()
    budget = benchmark.Budget(plan["measurement"], clock)
    records = []

    def unavailable(*_args):
        raise ConnectionRefusedError()

    monkeypatch.setattr(runtime, "http_json", unavailable)
    monkeypatch.setattr(
        runtime.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds)
    )
    with pytest.raises(benchmark.StopRun, match="startup_deadline"):
        runtime.wait_ready(
            plan,
            budget,
            SimpleNamespace(poll=lambda: None),
            SimpleNamespace(emit=records.append),
            SimpleNamespace(reason=None),
        )
    assert clock.now == plan["measurement"]["startup_seconds"]
    assert records and all(record["status"] is None for record in records)


def test_stalled_http_body_is_a_request_timeout(plan):
    done = Event()

    class Handler(BaseHTTPRequestHandler):
        """Stop after one byte so the full HTTP deadline must interrupt a body read."""

        def do_POST(self):  # pylint: disable=invalid-name
            """Retain an incomplete response until the client deadline."""
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"{")
            self.wfile.flush()
            done.wait(2)

        def log_message(self, *_args):
            """No fixture logs."""

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            plan["base_url"] = f"http://127.0.0.1:{server.server_port}"
            result = runtime.forecast_request(
                plan, benchmark.Budget(plan["measurement"]), "warm_c1", 1, 0, 0.1
            )
        finally:
            done.set()
            server.shutdown()
            thread.join()
    assert result.outcome == "timeout"
    assert result.response_sha256 is None


@pytest.mark.parametrize("launched", [False, True])
def test_external_watchdog_keeps_evidence_and_respects_launch_ownership(
    plan, tmp_path, monkeypatch, launched
):
    declaration = tmp_path / "plan.json"
    benchmark.write_json(declaration, plan)
    output = tmp_path / "run"
    signals, cleanups = [], []

    class HungWorker:
        """Deterministically simulate expiry, without a real hanging process."""

        pid, returncode = 12345, -9

        def __init__(self, _argv, **_kwargs):
            self.waits = 0
            if launched:
                (output / "observations.jsonl").write_text(
                    '{"type":"lifecycle","pid":42}\n{"partial":'
                )

        def wait(self, **_kwargs):
            """Time out on the first bounded wait, then reap after group termination."""
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("worker", 1)
            return 0

    monkeypatch.setattr(runtime.subprocess, "Popen", HungWorker)
    monkeypatch.setattr(runtime, "runtime_identity", lambda _: {"fixture": True})
    monkeypatch.setattr(runtime.os, "killpg", lambda *args: signals.append(args))
    monkeypatch.setattr(runtime, "cleanup", lambda *_args: cleanups.append(True) or True)
    assert not runtime.supervise(declaration, output, None, tmp_path / "lock")
    watchdog = json.loads((output / "watchdog.json").read_text())
    assert watchdog == {"stop_reason": "external_budget_deadline", "cleanup_verified": True}
    assert (output / "declaration.json").exists()
    assert len(signals) == 1
    assert len(cleanups) == int(launched)


@pytest.mark.parametrize("record", ["lifecycle", "lifecycle_intent", None])
@pytest.mark.parametrize("cleanup_ok", [False, True])
def test_nonzero_worker_exit_cleans_launched_daemon(
    plan, tmp_path, monkeypatch, record, cleanup_ok
):
    """An exited worker cannot leave its daemon-owned service after releasing the lock."""
    declaration = tmp_path / "plan.json"
    benchmark.write_json(declaration, plan)
    output = tmp_path / "run"
    cleanups = []

    class DeadWorker:
        """Simulate a crash after the lifecycle was flushed, before a report exists."""

        pid, returncode = 12345, 7

        def __init__(self, _argv, **_kwargs):
            if record:
                (output / "observations.jsonl").write_text(
                    json.dumps({"type": record, "pid": 42}) + "\n"
                )

        def wait(self, **_kwargs):
            """Return the abnormal terminal outcome."""
            return self.returncode

    monkeypatch.setattr(runtime.subprocess, "Popen", DeadWorker)
    monkeypatch.setattr(runtime, "runtime_identity", lambda _: {"fixture": True})
    monkeypatch.setattr(runtime.os, "killpg", lambda *_args: None)
    monkeypatch.setattr(runtime, "cleanup", lambda *_args: cleanups.append(True) or cleanup_ok)
    assert not runtime.supervise(declaration, output, None, tmp_path / "lock")
    assert len(cleanups) == int(record is not None)
    outcome = json.loads((output / "supervisor.json").read_text())
    assert outcome["status"] == "failed"
    assert outcome["worker_returncode"] == 7
    assert outcome["cleanup_verified"] == (cleanup_ok if record else True)


def test_nonzero_worker_cannot_emit_passing_confirmation(plan, tmp_path, monkeypatch):
    """A complete-looking worker report remains untrusted after exit 7."""
    plan["evidence_kind"] = "selected-target"
    report = discovery_report(plan)
    proposal = benchmark.propose(report, "a" * 64, 1, supervised_fixture(report))
    plan["purpose"] = "confirmation"
    report["run_id"] = "untrusted-worker-report-fixture-only"
    declaration, proposal_path = tmp_path / "plan.json", tmp_path / "proposal.json"
    benchmark.write_json(declaration, plan)
    benchmark.write_json(proposal_path, proposal)
    output = tmp_path / "run"

    class DeadWorker:
        """Flush a plausible report, then fail instead of completing successfully."""

        pid, returncode = 12345, 7

        def __init__(self, _argv, **_kwargs):
            (output / "observations.jsonl").write_text('{"type":"lifecycle","pid":42}\n')
            benchmark.write_json(output / "report.json", report)

        def wait(self, **_kwargs):
            """Return a failure after the report was written."""
            return self.returncode

    monkeypatch.setattr(runtime.subprocess, "Popen", DeadWorker)
    monkeypatch.setattr(runtime, "runtime_identity", lambda _: {"fixture": True})
    monkeypatch.setattr(benchmark, "validate_receipt", lambda *_args: None)
    monkeypatch.setattr(runtime.os, "killpg", lambda *_args: None)
    monkeypatch.setattr(runtime, "cleanup", lambda *_args: True)
    assert not runtime.supervise(declaration, output, None, tmp_path / "lock", proposal_path)
    assert not (output / "confirmation.json").exists()
    outcome = json.loads((output / "supervisor.json").read_text())
    assert outcome["status"] == "failed"
    assert outcome["worker_returncode"] == 7
    retained = json.loads((output / "report.json").read_text())
    assert retained == report  # never overwrite the negative evidence
    with pytest.raises(ValueError, match="Successful supervisor"):
        benchmark.confirm(report, proposal, outcome)


def test_propose_cli_requires_successful_supervision(plan, tmp_path):
    """A worker-only complete report cannot create a numerical candidate."""
    plan["evidence_kind"] = "selected-target"
    report = discovery_report(plan)  # deliberately untrusted synthetic worker report
    path, candidate = tmp_path / "report.json", tmp_path / "candidate.json"
    benchmark.write_json(path, report)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            runtime.__file__,
            "propose",
            str(path),
            str(candidate),
            "--concurrency",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not candidate.exists()


def test_nonzero_discovery_with_complete_report_cannot_generate_candidate(plan):
    plan["evidence_kind"] = "selected-target"
    report = discovery_report(plan)
    outcome = supervised_fixture(report)
    outcome.update(status="failed", worker_returncode=7)
    with pytest.raises(ValueError, match="Successful supervisor"):
        benchmark.propose(report, "a" * 64, 1, outcome)
    outcome.update(status="success", worker_returncode=0, cleanup_verified=False)
    with pytest.raises(ValueError, match="Successful supervisor"):
        benchmark.propose(report, "a" * 64, 1, outcome)
    outcome.update(cleanup_verified=True, report_digest="stale")
    with pytest.raises(ValueError, match="Successful supervisor"):
        benchmark.propose(report, "a" * 64, 1, outcome)


def test_early_service_exit_is_failed_startup_without_forecasts(plan, tmp_path):
    plan["startup"]["argv"] = [sys.executable, "-c", "raise SystemExit(3)"]
    assert not runtime.execute(plan, tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["stop_reason"] == "service_exited_during_startup"
    assert report["phases"] == {}
    assert report["cleanup_verified"]


def test_compute_lock_excludes_another_process(tmp_path):
    path = tmp_path / "compute.lock"
    with runtime.compute_lock(path):
        code = "from src.benchmark_runtime import compute_lock; import sys\nwith compute_lock(sys.argv[1]): pass"
        result = subprocess.run(
            [sys.executable, "-c", code, str(path)], check=False, capture_output=True
        )
    assert result.returncode != 0
    with runtime.compute_lock(path):
        pass


def test_occupied_port_refuses_launch_and_never_runs_cleanup(plan, tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        plan["base_url"] = f"http://127.0.0.1:{listener.getsockname()[1]}"
        marker = tmp_path / "must-not-exist"
        plan["startup"]["stop_argv"] = [
            sys.executable,
            "-c",
            f"open({str(marker)!r}, 'w').close()",
        ]
        assert not runtime.execute(plan, tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "stopped"
    assert report["cold_start"]["launch_to_ready_seconds"] is None
    assert not marker.exists()


def test_owned_http_lifecycle_report_and_cleanup(plan, tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    plan["base_url"] = f"http://127.0.0.1:{port}"
    plan["startup"]["argv"][-1] = str(port)
    declaration = tmp_path / "plan.json"
    benchmark.write_json(declaration, plan)
    output = tmp_path / "run"
    assert runtime.supervise(declaration, output, None, tmp_path / "lock")
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["cleanup_verified"]
    assert report["plan"]["expected"] == plan["expected"]
    assert report["cold_start"]["launch_to_ready_seconds"] > 0
    assert report["phases"]["warm_c1"]["attempts"] == 20
    assert report["phases"]["warm_c2"]["attempts"] == 20
    assert report["phases"]["overload_c2"]["attempts"] == 6
    records = [
        json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()
    ]
    assert len([r for r in records if r["type"] == "request"]) == 50
    assert report["memory"]["scope"] == "process"
    assert any("Fixture timings" in line for line in report["limitations"])
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0
    with pytest.raises(FileExistsError):
        runtime.supervise(declaration, output, None, tmp_path / "lock")
