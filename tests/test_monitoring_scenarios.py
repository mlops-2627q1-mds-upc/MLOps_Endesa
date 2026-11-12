"""Observable scenario transitions, budget fences and ownership refusal."""

from copy import deepcopy
import json
import subprocess
from types import SimpleNamespace

import pytest

from src.monitoring_scenarios import runner, service
from src.monitoring_scenarios.experiments import delayed_observation


@pytest.mark.parametrize("scenario", ["normal", "service-unavailable", "slow-request"])
def test_production_handlers_metrics_fault_and_recovery(tmp_path, scenario):
    result = service.fixture_service(tmp_path, scenario, runner.declarations()[scenario])
    assert result["procedure_passed"]
    assert result["recovery"]["http_status"] == 200
    metrics = (tmp_path / "fault-metrics.prom").read_text()
    assert 'endesa_http_requests_total{method="POST",route="/forecast"' in metrics
    if scenario == "service-unavailable":
        assert "endesa_model_ready 0.0" in metrics
        assert all(row["http_status"] == 503 for row in result["observed_signal"]["fault"])
    elif scenario == "slow-request":
        assert all(row["wall_seconds"] >= 0.25 for row in result["observed_signal"]["fault"])


def test_delay_public_sqlite_quality_and_exporter(tmp_path):
    result = delayed_observation(tmp_path, runner.declarations()["delayed-observation"])
    assert result["procedure_passed"]
    partial, recovered = result["observed_signal"]
    assert partial["clock_position"] == 516
    assert recovered["clock_position"] == 563
    assert partial["counts"]["matched_steps"] == 6
    assert partial["counts"]["pending_steps"] == 186
    assert recovered["counts"]["matched_steps"] == 192
    assert recovered["default_minimum_30_statuses"] == ["insufficient_samples"]
    assert {row["sample_status"] for row in partial["groups"]} == {
        "sufficient",
        "insufficient_samples",
        "pending_observations",
    }
    metrics = (tmp_path / "quality-recovery.prom").read_text()
    assert "endesa_quality_report_available 1.0" in metrics


@pytest.mark.parametrize("seconds", [True, 0, -1, float("inf"), float("nan"), 59, 1801])
def test_budgets_refused_without_output(tmp_path, seconds):
    with pytest.raises(ValueError, match="budget"):
        runner.supervise(tmp_path / "run", "normal", max_seconds=seconds)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"engine": "docker"},
        {"engine": "docker", "target_id": "shared", "fixture_image": "x"},
        {"engine": "alibi"},
        {"engine": "remote"},
    ],
)
def test_shared_or_incompatible_target_refused(tmp_path, kwargs):
    with pytest.raises(ValueError):
        runner.supervise(tmp_path / "run", "normal", **kwargs)
    assert not (tmp_path / "run").exists()


def test_reuse_directory_preserves_files(tmp_path):
    (tmp_path / "evidence").write_text("preserve")
    with pytest.raises(FileExistsError):
        runner.supervise(tmp_path, "normal")
    assert (tmp_path / "evidence").read_text() == "preserve"


def test_supervised_real_child_has_identity_result_and_no_invented_alert(tmp_path):
    directory = tmp_path / "control"
    summary = runner.supervise(directory, "normal")
    assert summary["status"] == "passed"
    assert not summary["cleanup"]
    request = json.loads((directory / "request.json").read_text())
    assert request["source"]["sha256"]["src/api/metrics.py"]
    result = json.loads((directory / "result.json").read_text())
    assert result["alerts"] == {"firing": None, "clearing": None, "delivery": "not_measured"}


def test_supervisor_retains_crash_without_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(
            wait=lambda **_kw: 7, poll=lambda: 7, returncode=7
        ),
    )
    monkeypatch.setattr(runner, "source_identity", dict)
    summary = runner.supervise(tmp_path / "failure", "normal")
    assert summary["status"] == "failed"
    assert summary["child_exit_code"] == 7
    assert summary["result"] is None
    assert (tmp_path / "failure" / "execution.log").exists()


def test_timeout_is_failed_and_owned_cleanup_still_runs(tmp_path, monkeypatch):
    process = SimpleNamespace(
        wait=lambda **_kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("scenario", 30)),
        returncode=-15,
    )
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kw: process)
    monkeypatch.setattr(runner, "source_identity", dict)
    stopped = []
    monkeypatch.setattr(runner, "_stop_process", lambda child: stopped.append(child))
    monkeypatch.setattr(runner, "cleanup", lambda directory: [{"owned": directory.name}])
    summary = runner.supervise(tmp_path / "timeout", "normal", max_seconds=60)
    assert summary["status"] == "failed"
    assert "TimeoutExpired" in summary["error"]
    assert stopped == [process]
    assert summary["cleanup"] == [{"owned": "timeout"}]


def test_cleanup_failure_cannot_be_a_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(
            wait=lambda **_kw: 0, poll=lambda: 0, returncode=0
        ),
    )
    monkeypatch.setattr(runner, "source_identity", dict)
    monkeypatch.setattr(
        runner, "cleanup", lambda _path: (_ for _ in ()).throw(ValueError("foreign identity"))
    )
    summary = runner.supervise(tmp_path / "cleanup-failure", "normal")
    assert summary["status"] == "failed"
    assert "Cleanup failed" in summary["error"]


def test_cleanup_refuses_foreign_container_and_does_not_stop(tmp_path, monkeypatch):
    (tmp_path / "runtime.json").write_text(json.dumps({"owner": "mine", "containers": ["a" * 64]}))
    monkeypatch.setattr(
        service,
        "inspect_container",
        lambda _id: {
            "Config": {"Labels": {service.OWNER_LABEL: "someone-else"}},
            "State": {"Running": True},
        },
    )
    commands = []
    monkeypatch.setattr(service, "docker", lambda *args: commands.append(args))
    with pytest.raises(ValueError, match="ownership"):
        service.cleanup(tmp_path)
    assert not commands


def test_cleanup_stops_exact_owned_id_and_keeps_stopped_resources(tmp_path, monkeypatch):
    identifier = "a" * 64
    (tmp_path / "runtime.json").write_text(
        json.dumps({"owner": "mine", "containers": [identifier]})
    )
    record = {"Config": {"Labels": {service.OWNER_LABEL: "mine"}}, "State": {"Running": True}}
    monkeypatch.setattr(service, "inspect_container", lambda _id: deepcopy(record))
    commands = []

    def stop(*args):
        commands.append(args)
        record["State"]["Running"] = False

    monkeypatch.setattr(service, "docker", stop)
    assert service.cleanup(tmp_path)[0]["state"]["Running"] is False
    assert commands == [("stop", "--time", "5", identifier)]
    assert service.cleanup(tmp_path)[0]["state"]["Running"] is False
    assert len(commands) == 1


def test_cleanup_recovers_created_id_before_runtime_write(tmp_path, monkeypatch):
    identifier = "c" * 64
    (tmp_path / "runtime.json").write_text(json.dumps({"owner": "mine", "containers": []}))
    (tmp_path / "api.cid").write_text(identifier)
    monkeypatch.setattr(
        service,
        "inspect_container",
        lambda _id: {
            "Config": {"Labels": {service.OWNER_LABEL: "mine"}},
            "State": {"Running": False},
        },
    )
    assert service.cleanup(tmp_path)[0]["id"] == identifier


def test_mutable_names_cannot_authorize_cleanup():
    with pytest.raises(ValueError, match="full container ID"):
        service.inspect_container("shared-api")


def test_remote_docker_daemon_refused(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.example:2376")
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda *_args, **_kw: SimpleNamespace(stdout="unix:///local/docker.sock"),
    )
    with pytest.raises(ValueError, match="local daemon"):
        service.docker("ps")


def test_cached_image_requires_exact_identity_before_creation(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        service.OwnedStack(tmp_path, "endesa-84-test", "mutable:latest").start()
    assert not (tmp_path / "runtime.json").exists()


def test_restart_refreshes_dynamic_binding_before_recovery_request(tmp_path, monkeypatch):
    owner = "endesa-84-rebind"
    stack = service.OwnedStack(tmp_path, owner, "sha256:" + "a" * 64)
    stack.api = "b" * 64
    stack.api_url = "http://127.0.0.1:12345"
    stack.runtime["api_urls"] = [stack.api_url]
    monkeypatch.setattr(
        service,
        "inspect_container",
        lambda _id: {"Config": {"Labels": {service.OWNER_LABEL: owner}}},
    )
    commands = []
    monkeypatch.setattr(service, "docker", lambda *args: commands.append(args))
    monkeypatch.setattr(stack, "url", lambda *_args: "http://127.0.0.1:54321")
    stack.change_running(True)
    assert commands == [("start", "b" * 64)]
    assert stack.api_url == "http://127.0.0.1:54321"
    assert stack.runtime["api_urls"] == ["http://127.0.0.1:12345", "http://127.0.0.1:54321"]
    assert json.loads((tmp_path / "runtime.json").read_text())["api_url"] == stack.api_url


def test_delay_publication_is_atomic_and_acknowledged(tmp_path, monkeypatch):
    service.publish_fault(tmp_path, 0)
    before = (tmp_path / "fault.json").stat().st_ino
    stack = service.OwnedStack(tmp_path, "endesa-84-fault", "sha256:" + "a" * 64)
    stack.api = "c" * 64
    monkeypatch.setattr(
        service,
        "inspect_container",
        lambda _id: {"Config": {"Labels": {service.OWNER_LABEL: stack.owner}}},
    )
    monkeypatch.setattr(service, "docker", lambda *_args: '{"delay_seconds": 0.25}')
    assert stack.set_delay(0.25) == {"delay_seconds": 0.25}
    assert (tmp_path / "fault.json").stat().st_ino != before
    assert not (tmp_path / "fault-next.json").exists()
    monkeypatch.setattr(service, "docker", lambda *_args: '{"delay_seconds": 0}')
    with pytest.raises(ValueError, match="acknowledge"):
        stack.set_delay(0.25)


def test_actual_ks_control_shift_recovery_when_isolated_backend_available(tmp_path):
    pytest.importorskip("alibi_detect")
    from src.monitoring_scenarios.experiments import (  # pylint: disable=import-outside-toplevel
        shifted_demand,
    )

    report = shifted_demand(tmp_path, runner.declarations()["shifted-demand"])
    assert report["evidence_kind"] == "fresh_synthetic_alibi"
    assert not report["missed_detections"]
    assert not report["control_or_recovery_alarms"]
    assert report["recovery"]["observed"]
    assert all(row["status"] == "drift" for row in report["observed_signal"][1]["states"].values())
