"""Check the image source boundary and HTTP harness without weights or Docker."""

from contextlib import nullcontext
from email.message import Message
from http.client import RemoteDisconnected
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tests.container import smoke
from tests.container.fixture_app import FixturePredictor
from tests.container.smoke import run_smoke

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("case", ["complete-linux", "missing-rss", "unavailable"])
def test_metrics_smoke_requires_linux_process_samples(monkeypatch, case):
    samples = [
        "endesa_model_ready 1",
        "endesa_forecasts_in_flight 0",
        'endesa_http_requests_total{method="POST",route="/forecast",status_class="2xx"} 1',
        'endesa_http_requests_total{method="POST",route="/forecast",status_class="4xx"} 1',
        'endesa_http_requests_total{method="POST",route="/forecast",status_class="5xx"} 0',
        'endesa_http_request_duration_seconds_count{method="POST",route="/forecast"} 2',
        f"endesa_process_metrics_available {int(case != 'unavailable')}",
    ]
    if case != "unavailable":
        samples.append("process_cpu_seconds_total 0.1")
    if case == "complete-linux":
        samples.append("process_resident_memory_bytes 1024")
    headers = Message()
    headers["Content-Type"] = "text/plain; version=1.0.0; charset=utf-8"
    response = SimpleNamespace(
        status=200, headers=headers, read=lambda: "\n".join(samples).encode()
    )
    monkeypatch.setattr(smoke, "urlopen", lambda *_args, **_kwargs: nullcontext(response))
    if case == "complete-linux":
        assert smoke.check_metrics("http://fixture", require_process_metrics=True) == {
            "ready": 1,
            "process_metrics_available": 1,
        }
    else:
        with pytest.raises((AssertionError, KeyError)):
            smoke.check_metrics("http://fixture", require_process_metrics=True)


@pytest.mark.parametrize("failure", [ConnectionResetError, RemoteDisconnected])
def test_startup_retries_transient_disconnect_then_checks_every_endpoint(monkeypatch, failure):
    payload = {"state": "TAS", "history": [-3, 0, 2.5, 4] * 128}
    forecast = {**FixturePredictor().predict(payload["history"]), "state": "TAS"}
    transport = Mock(
        side_effect=[
            failure("Port exposed before server startup"),
            (200, {"status": "ok"}),
            (200, {"status": "ready"}),
            (200, forecast),
            (422, {"detail": [{"type": "too_short"}]}),
        ]
    )
    sleep = Mock()
    monkeypatch.setattr(smoke, "request", transport)
    metrics = Mock(return_value={"ready": 1, "process_metrics_available": 0})
    monkeypatch.setattr(smoke, "check_metrics", metrics)
    monkeypatch.setattr(smoke.time, "monotonic", Mock(side_effect=[0.0, 0.1]))
    monkeypatch.setattr(smoke.time, "sleep", sleep)

    report = run_smoke("http://fixture", timeout=1)

    assert report["forecast"] == 200 and report["invalid_request"] == 422
    assert transport.call_args_list == [
        call("http://fixture", "/health"),
        call("http://fixture", "/health"),
        call("http://fixture", "/ready"),
        call("http://fixture", "/forecast", payload),
        call("http://fixture", "/forecast", {**payload, "history": [1]}),
    ]
    sleep.assert_called_once_with(0.1)
    metrics.assert_called_once_with("http://fixture", False)


def test_persistent_startup_reset_fails_at_deadline_without_further_checks(monkeypatch):
    transport = Mock(side_effect=ConnectionResetError(104, "Connection reset by peer"))
    sleep = Mock()
    monkeypatch.setattr(smoke, "request", transport)
    monkeypatch.setattr(smoke.time, "monotonic", Mock(side_effect=[10.0, 10.5, 11.0]))
    monkeypatch.setattr(smoke.time, "sleep", sleep)

    with pytest.raises(RuntimeError, match="did not start within the smoke deadline"):
        run_smoke("http://fixture", timeout=1)

    assert transport.call_args_list == [call("http://fixture", "/health")] * 2
    sleep.assert_called_once_with(0.1)


def test_startup_http_error_is_not_retried_or_treated_as_healthy(monkeypatch):
    transport = Mock(return_value=(503, {"detail": "Unavailable"}))
    sleep = Mock()
    monkeypatch.setattr(smoke, "request", transport)
    monkeypatch.setattr(smoke.time, "sleep", sleep)

    with pytest.raises(AssertionError):
        run_smoke("http://fixture")

    transport.assert_called_once_with("http://fixture", "/health")
    sleep.assert_not_called()


def test_forecast_connection_reset_is_not_retried_after_startup(monkeypatch):
    transport = Mock(
        side_effect=[
            (200, {"status": "ok"}),
            (200, {"status": "ready"}),
            ConnectionResetError(104, "Connection reset by peer"),
        ]
    )
    sleep = Mock()
    monkeypatch.setattr(smoke, "request", transport)
    monkeypatch.setattr(smoke.time, "sleep", sleep)

    with pytest.raises(ConnectionResetError):
        run_smoke("http://fixture")

    assert [args.args[1] for args in transport.call_args_list] == [
        "/health",
        "/ready",
        "/forecast",
    ]
    sleep.assert_not_called()


def _stage_runtime(directory):
    """Reproduce the Dockerfile's explicit local COPY inputs in an empty directory."""
    copied = set()
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        *sources, target = shlex.split(line)[1:]
        destination = directory / target
        destination.mkdir(parents=True, exist_ok=True)
        for source in sources:
            assert not any(character in source for character in "*?[")
            assert (ROOT / source).is_file(), source
            shutil.copyfile(ROOT / source, destination / Path(source).name)
            copied.add(source)
    return copied


def _environment(directory):
    return {
        **os.environ,
        "PYTHONPATH": str(directory),
        "PYTHONDONTWRITEBYTECODE": "1",
        "FORECAST_PARAMS_PATH": str(directory / "params.yaml"),
        "FORECAST_CHECKPOINT": str(directory / "models/chronos-t5-small-fine-tuned"),
        "FORECAST_CPU_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }


def test_build_context_admits_only_explicit_runtime_files(tmp_path):
    copied = _stage_runtime(tmp_path)
    rules = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert rules[0] == "**"
    allowed = set()
    for rule in rules[1:]:
        assert rule.startswith("!") and not any(char in rule for char in "*?[")
        path = rule[1:]
        # Re-including a directory could admit secrets and caches under that parent.
        assert (ROOT / path).is_file() and not (ROOT / path).is_symlink(), path
        allowed.add(path)
    assert allowed == copied | {"Dockerfile", ".dockerignore"}
    assert not (tmp_path / "src/modeling/train.py").exists()
    assert not (tmp_path / "tests").exists()
    assert list((tmp_path / "models").iterdir()) == [
        tmp_path / "models/chronos-t5-small-fine-tuned.provenance.json"
    ]


def test_production_factory_exits_without_selected_weights(tmp_path):
    _stage_runtime(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "uvicorn", "src.api.app:create_app", "--factory", "--port", "0"],
        cwd=tmp_path,
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "Could not verify local checkpoint" in result.stderr
    assert "Application startup failed" in result.stderr


def test_fixture_harness_exercises_actual_http_from_staged_runtime(tmp_path):
    _stage_runtime(tmp_path)
    environment = _environment(tmp_path)
    environment["PYTHONPATH"] += os.pathsep + str(ROOT / "tests/container")
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    with (tmp_path / "uvicorn.log").open("w+") as log:
        server = subprocess.Popen(  # pylint: disable=consider-using-with
            [
                sys.executable,
                "-m",
                "uvicorn",
                "fixture_app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
            ],
            cwd=tmp_path,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            report = run_smoke(f"http://127.0.0.1:{port}", timeout=10)
            assert report["fixture_only"]
            assert report["forecast"] == 200
            assert report["invalid_request"] == 422
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            log.seek(0)
            print(log.read())
