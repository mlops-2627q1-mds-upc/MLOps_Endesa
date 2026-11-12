"""Deployment safety and separate-client HTTP tests; all artifacts here are fixtures."""

from copy import deepcopy
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import socket
import subprocess
from tempfile import TemporaryDirectory
from threading import Thread

import pytest

from src.deployment import config as configuration
from src.deployment import runtime
from src.deployment import verify as client
from src.deployment.__main__ import main
from src.deployment.config import (
    COMMAND,
    DeploymentConfig,
    checkpoint_files,
    compute_lock,
    file_sha256,
    fingerprint,
    write_new,
)
from src.deployment.runtime import CONFIG_LABEL, ISSUE_LABEL, Deployment


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Tiny fixture artifacts and provenance, explicitly unrelated to the real model."""
    checkpoint = tmp_path / "incoming"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"deployment-unit-test-fixture")
    (checkpoint / "config.json").write_text("{}")
    params = tmp_path / "params.yaml"
    params.write_text("fixture: true\n")
    provenance = tmp_path / "fixture.provenance.json"
    weights = file_sha256(checkpoint / "model.safetensors")
    provenance.write_text(
        json.dumps({"model_safetensors_sha256": weights, "training_run_id": "1" * 32})
    )
    monkeypatch.setattr(configuration, "PROVENANCE", provenance)
    return DeploymentConfig(
        schema_version=1,
        deployment_id="endesa75-fixture",
        target_hostname=socket.getfqdn(),
        source_commit="a" * 40,
        image_id="sha256:" + "b" * 64,
        checkpoint=str(checkpoint),
        checkpoint_sha256=weights,
        checkpoint_tree_sha256=fingerprint(checkpoint_files(checkpoint)),
        training_run_id="1" * 32,
        params=str(params),
        params_sha256=file_sha256(params),
        cpu_threads=2,
        memory_mb=512,
        port=18075,
        docker_socket=str(tmp_path / "docker.sock"),
    )


def _settings():
    return {
        "num_samples": 20,
        "seed": 7,
        "device": "cpu",
        "cpu_threads": 2,
        "dtype": "float32",
        "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
        "quantile_method": "linear",
        "point_statistic": "median",
    }


def _verified(config):
    return {
        **config.verify_artifacts(),
        "settings": _settings(),
        "image": {"id": config.image_id, "revision": config.source_commit, "user": "10001:10001"},
        "target": {"CgroupDriver": "cgroupfs", "CgroupVersion": "1"},
    }


def _container(config):
    deployment = Deployment(config)
    return {
        "Id": "c" * 64,
        "Name": "/" + config.deployment_id,
        "Image": config.image_id,
        "Config": {
            "Labels": {ISSUE_LABEL: "75", CONFIG_LABEL: config.config_sha256},
            "User": "10001:10001",
            "Cmd": COMMAND,
            "Entrypoint": None,
            "Env": deployment.environment()[1::2],
        },
        "HostConfig": {
            "PortBindings": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(config.port)}]},
            "ReadonlyRootfs": True,
            "RestartPolicy": {"Name": "no"},
            "Memory": config.memory_mb * 1024 * 1024,
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": config.checkpoint,
                "Destination": runtime.MODEL_PATH,
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": config.params,
                "Destination": runtime.PARAMS_PATH,
                "RW": False,
            },
        ],
        "State": {
            "Running": True,
            "Pid": 1234,
            "ExitCode": 0,
            "OOMKilled": False,
            "StartedAt": "2026-11-02T12:00:00Z",
        },
        "RestartCount": 0,
    }


def _receipt(config, monkeypatch):
    deployment = Deployment(config)
    monkeypatch.setattr(deployment, "owned", lambda: _container(config))
    return deployment.receipt(_verified(config))


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_commit", "main"),
        ("image_id", "latest"),
        ("checkpoint", "relative/path"),
        ("checkpoint", "/tmp/check,point"),
        ("cpu_threads", True),
        ("cpu_threads", 3),
        ("port", 80),
        ("memory_mb", 0),
        ("training_run_id", "2" * 32),
        ("checkpoint_sha256", "0" * 64),
    ],
)
def test_config_refuses_implicit_invalid_or_different_selections(config, field, value):
    with pytest.raises(ValueError):
        replace(config, **{field: value}).validate()


@pytest.mark.parametrize("edit", ["missing", "extra"])
def test_config_requires_exact_fields(config, tmp_path, edit):
    raw = config.as_dict()
    if edit == "missing":
        del raw["target_hostname"]
    else:
        raw["password"] = "must-not-be-accepted"
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="exactly"):
        DeploymentConfig.load(path)


def test_changed_tokenizer_or_params_cannot_keep_old_identity(config):
    checkpoint = Path(config.checkpoint)
    (checkpoint / "tokenizer.json").write_text("changed")
    with pytest.raises(ValueError, match="Checkpoint bytes"):
        config.verify_artifacts()
    (checkpoint / "tokenizer.json").unlink()
    Path(config.params).write_text("changed: true")
    with pytest.raises(ValueError, match="Params bytes"):
        config.verify_artifacts()


def test_checkpoint_refuses_symlinks(config):
    (Path(config.checkpoint) / "external").symlink_to(config.params)
    with pytest.raises(ValueError, match="regular files"):
        checkpoint_files(Path(config.checkpoint))


def test_provision_creates_verified_read_only_slot_without_overwrite(config, tmp_path):
    destination = tmp_path / "selected"
    deployment = Deployment(replace(config, checkpoint=str(destination)))
    result = deployment.provision(Path(config.checkpoint))
    assert result["checkpoint"]["sha256"] == config.checkpoint_sha256
    assert (destination / "model.safetensors").stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        deployment.provision(Path(config.checkpoint))
    destination.chmod(0o755)  # Allow pytest to clean its own fixture directory.


def test_lock_excludes_second_owner_and_releases_on_error(tmp_path):
    path = tmp_path / "compute.lock"
    with compute_lock(path), pytest.raises(ValueError, match="busy"), compute_lock(path):
        pytest.fail("Second owner entered the lock")
    with pytest.raises(RuntimeError), compute_lock(path):
        raise RuntimeError("owned subprocess failed")
    with compute_lock(path):
        pass


def test_wrong_host_and_remote_daemon_fail_closed(config, monkeypatch):
    with pytest.raises(ValueError, match="hostname"):
        replace(config, target_hostname="another-course-host").verify_target()
    with (
        TemporaryDirectory(prefix="endesa75-", dir="/tmp") as directory,
        socket.socket(socket.AF_UNIX) as sock,
    ):
        config = replace(config, docker_socket=str(Path(directory) / "docker.sock"))
        sock.bind(config.docker_socket)
        deployment = Deployment(config)
        monkeypatch.setattr(
            deployment,
            "docker",
            lambda *args: json.dumps({"OSType": "linux", "Name": "another-daemon"}),
        )
        with pytest.raises(ValueError, match="local Linux host"):
            deployment.target()


@pytest.mark.parametrize("change", ["label", "image", "name"])
def test_stop_refuses_unrelated_container_without_mutation(config, monkeypatch, change):
    deployment = Deployment(config)
    container = _container(config)
    if change == "label":
        container["Config"]["Labels"][CONFIG_LABEL] = "someone-elses-config"
    elif change == "image":
        container["Image"] = "sha256:" + "0" * 64
    else:
        container["Name"] = "/unrelated"
    calls = []

    def docker(*args, **_kwargs):
        calls.append(args)
        assert args[:2] == ("container", "inspect")
        return json.dumps([container])

    monkeypatch.setattr(deployment, "target", dict)
    monkeypatch.setattr(deployment, "docker", docker)
    with pytest.raises(ValueError, match="ownership"):
        deployment.lifecycle("stop")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field,value", [("revision", "0" * 40), ("user", "root"), ("cmd", ["sleep", "600"])]
)
def test_image_refuses_mislabeled_or_replaced_entrypoint(config, monkeypatch, field, value):
    deployment = Deployment(config)
    image = {
        "Id": config.image_id,
        "Config": {
            "User": "10001:10001",
            "Cmd": COMMAND,
            "Entrypoint": None,
            "Labels": {"org.opencontainers.image.revision": config.source_commit},
        },
    }
    if field == "revision":
        image["Config"]["Labels"]["org.opencontainers.image.revision"] = value
    else:
        image["Config"][{"user": "User", "cmd": "Cmd"}[field]] = value
    monkeypatch.setattr(deployment, "docker", lambda *args: json.dumps([image]))
    with pytest.raises(ValueError, match="Image ID"):
        deployment.image()


def test_start_uses_immutable_private_isolated_spec(config, monkeypatch):
    deployment = Deployment(config)
    calls = []
    monkeypatch.setattr(deployment, "preflight", lambda: _verified(config))
    monkeypatch.setattr(deployment, "wait_ready", lambda: None)
    monkeypatch.setattr(deployment, "receipt", lambda verified: verified)
    monkeypatch.setattr(deployment, "docker", lambda *args, **kwargs: calls.append(args) or "")
    deployment.start()
    args = calls[-1]
    assert args[:2] == ("run", "--detach")
    assert args[-1] == config.image_id
    assert args[args.index("--publish") + 1] == "127.0.0.1:18075:8000"
    assert "--read-only" in args and "--cap-drop=ALL" in args
    assert all("readonly" in args[i + 1] for i, v in enumerate(args) if v == "--mount")
    assert "FORECAST_CPU_THREADS=2" in args and "--restart=no" in args


def test_receipt_rejects_public_port_binding(config, monkeypatch):
    deployment = Deployment(config)
    container = _container(config)
    container["HostConfig"]["PortBindings"]["8000/tcp"][0]["HostIp"] = "0.0.0.0"
    monkeypatch.setattr(deployment, "owned", lambda: container)
    with pytest.raises(ValueError, match="private container spec"):
        deployment.receipt(_verified(config))


def test_receipt_rejects_additional_writable_or_volume_mount(config, monkeypatch):
    deployment = Deployment(config)
    container = _container(config)
    container["Mounts"].append(
        {"Type": "volume", "Source": "other", "Destination": "/extra", "RW": True}
    )
    monkeypatch.setattr(deployment, "owned", lambda: container)
    with pytest.raises(ValueError, match="private container spec"):
        deployment.receipt(_verified(config))


@pytest.mark.parametrize("exit_code,oom", [(0, False), (137, False), (137, True), (1, False)])
def test_stop_records_clean_shutdown_and_preserves_failed_evidence(
    config, monkeypatch, tmp_path, exit_code, oom
):
    deployment = Deployment(config)
    running = _container(config)
    stopped = deepcopy(running)
    stopped["State"].update(Running=False, ExitCode=exit_code, OOMKilled=oom)
    inspections = iter((running, stopped))
    monkeypatch.setattr(deployment, "target", dict)
    monkeypatch.setattr(deployment, "owned", lambda: next(inspections))
    calls = []
    monkeypatch.setattr(deployment, "docker", lambda *args, **kwargs: calls.append(args))
    result = deployment.lifecycle("stop")
    clean = exit_code == 0 and not oom
    assert result["status"] == ("passed" if clean else "failed")
    assert result["clean_shutdown"] is clean
    assert result["exit_code"] == exit_code and result["oom_killed"] == oom
    assert calls == [("stop", "--time", "20", running["Id"])]
    path = tmp_path / "deployment.json"
    write_new(path, config.as_dict())
    monkeypatch.setattr(Deployment, "lifecycle", lambda self, action: result)
    output = tmp_path / "stop.json"
    assert main(
        [
            "--config",
            str(path),
            "--lock-file",
            str(tmp_path / "lock"),
            "--output",
            str(output),
            "stop",
        ]
    ) == (0 if clean else 1)
    assert json.loads(output.read_text())["status"] == result["status"]


def test_final_inspection_failure_stops_only_started_owned_instance(config, monkeypatch):
    deployment = Deployment(config)
    monkeypatch.setattr(deployment, "preflight", lambda: _verified(config))
    monkeypatch.setattr(deployment, "wait_ready", lambda: None)
    monkeypatch.setattr(deployment, "owned", lambda: _container(config))

    def reject(_verified):
        raise ValueError("Inspected runtime differs")

    monkeypatch.setattr(deployment, "receipt", reject)
    calls = []
    monkeypatch.setattr(deployment, "docker", lambda *args, **kwargs: calls.append(args) or "")
    with pytest.raises(ValueError, match="runtime differs"):
        deployment.start()
    assert calls[-1] == ("stop", "--time", "20", "c" * 64)


@pytest.mark.parametrize("failure", [None, "exit", "timeout"])
def test_preflight_parser_releases_only_its_owned_probe_on_failure(config, monkeypatch, failure):
    deployment = Deployment(config)
    container = _container(config)
    container["State"]["Running"] = failure == "timeout"
    name = config.deployment_id + "-preflight"
    inspected = []

    def owned(selected_name):
        inspected.append(selected_name)
        return container

    monkeypatch.setattr(deployment, "owned", owned)
    calls = []

    def docker(*args, **_kwargs):
        calls.append(args)
        if args[0] == "wait":
            if failure == "timeout":
                raise subprocess.TimeoutExpired("fixture-docker", 90)
            return "1" if failure == "exit" else "0"
        return json.dumps({"settings": _settings()}) if args[0] == "logs" else ""

    monkeypatch.setattr(deployment, "docker", docker)
    if failure:
        with pytest.raises((ValueError, subprocess.TimeoutExpired)):
            deployment.parse_settings("fixture parser")
    else:
        assert deployment.parse_settings("fixture parser") == {"settings": _settings()}
    assert inspected == [name]
    assert calls[0][:5] == ("run", "--detach", "--network=none", "--name", name)
    assert calls[-1] == ("rm", container["Id"])
    assert (
        ("stop", "--time", "10", container["Id"]) in calls
        if failure == "timeout"
        else not any(c[0] == "stop" for c in calls)
    )


@pytest.mark.parametrize("code,oom", [("0", False), ("137", True), ("3", False)])
def test_missing_model_uses_normal_entrypoint_and_cleans_its_probe(config, monkeypatch, code, oom):
    deployment = Deployment(config)
    calls = []
    container = _container(config)
    container["State"].update(Running=False, OOMKilled=oom)
    monkeypatch.setattr(deployment, "preflight", lambda: _verified(config))
    monkeypatch.setattr(deployment, "owned", lambda name: container)

    def docker(*args, **_kwargs):
        calls.append(args)
        if args[0] == "wait":
            return code
        if args[0] == "logs":
            return "Forecast model initialization failed"
        return ""

    monkeypatch.setattr(deployment, "docker", docker)
    if code == "3" and not oom:
        assert deployment.missing_model()["exit_code"] == 3
    else:
        with pytest.raises(ValueError, match="exit 3"):
            deployment.missing_model()
    assert calls[-1][0] == "rm"
    assert "--network=none" in calls[0]
    assert "--publish" not in calls[0] and "--entrypoint" not in calls[0]
    assert not any(runtime.MODEL_PATH in v for v in calls[0] if v.startswith("type=bind"))


def test_evidence_and_cli_refuse_overwrites_before_mutation(config, monkeypatch, tmp_path):
    output = tmp_path / "receipt.json"
    write_new(output, {"existing": True})
    cfg = tmp_path / "deployment.json"
    cfg.write_text(json.dumps(config.as_dict()))
    monkeypatch.setattr(Deployment, "start", lambda self: pytest.fail("Mutation happened"))
    assert (
        main(
            [
                "--config",
                str(cfg),
                "--lock-file",
                str(tmp_path / "lock"),
                "--output",
                str(output),
                "start",
            ]
        )
        == 1
    )
    assert json.loads(output.read_text()) == {"existing": True}


@pytest.fixture
def http_fixture(config, monkeypatch):
    """An isolated socket fixture for client behaviour, never target/model evidence."""
    receipt = _receipt(config, monkeypatch)
    payload = {"state": "TAS", "history": [-1.5] * 512}
    response = {
        "state": "TAS",
        "context_length": 512,
        "horizon": 48,
        "value_scale": "source",
        "model_id": "chronos-t5-small-fine-tuned",
        "model_revision": "local-sha256:" + config.checkpoint_sha256,
        "checkpoint": receipt["verified"]["checkpoint"],
        "settings": _settings(),
        "p10": [-2.0] * 48,
        "p50": [-1.5] * 48,
        "p90": [-1.0] * 48,
    }

    response["fixture_invalid_body"] = {"detail": [{"type": "too_short"}]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):  # pylint: disable=invalid-name
            self.respond(200, {"status": "ok" if self.path == "/health" else "ready"})

        def do_POST(self):  # pylint: disable=invalid-name
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if len(request["history"]) != 512:
                self.respond(422, response["fixture_invalid_body"])
            else:
                self.respond(200, response)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", receipt, payload, response
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_separate_http_client_checks_all_routes_and_records_actual_responses(http_fixture):
    url, receipt, payload, _ = http_fixture
    report = client.verify(url, receipt, payload)
    assert report["status"] == "passed"
    assert [c["status"] for c in report["checks"]] == [200, 200, 200, 422]
    assert report["inspection_receipt_sha256"] == fingerprint(receipt)
    assert report["checks"][2]["body"]["p50"] == [-1.5] * 48


@pytest.mark.parametrize("edit", ["model", "settings", "quantiles", "boolean", "short", "state"])
def test_client_rejects_wrong_identity_settings_and_invalid_forecasts(http_fixture, edit):
    url, receipt, payload, response = http_fixture
    if edit == "model":
        response["checkpoint"] = {"sha256": "0" * 64, "training_run_id": "0" * 32}
    elif edit == "settings":
        response["settings"]["seed"] += 1
    elif edit == "quantiles":
        response["p10"][0] = 100
    elif edit == "boolean":
        response["p50"][0] = True
    elif edit == "short":
        response["p90"].pop()
    else:
        response["state"] = "NSW"
    assert client.verify(url, receipt, payload)["status"] == "failed"


def test_client_rejects_incoherent_operator_receipt(http_fixture):
    url, receipt, payload, _ = http_fixture
    receipt = deepcopy(receipt)
    receipt["container"]["image"] = "sha256:" + "d" * 64
    with pytest.raises(ValueError, match="inspection receipt"):
        client.verify(url, receipt, payload)


@pytest.mark.parametrize(
    "url",
    [
        "http://user:secret@localhost",
        "http://localhost?token=secret",
        "file:///tmp/model",
        "http://localhost/other-api",
    ],
)
def test_client_refuses_ambiguous_or_credential_urls(url, http_fixture):
    _, receipt, payload, _ = http_fixture
    with pytest.raises(ValueError, match="explicit base"):
        client.verify(url, receipt, payload)


def test_shutdown_client_requires_connection_refusal_not_a_timeout(http_fixture, monkeypatch):
    url, receipt, payload, _ = http_fixture
    assert client.verify(url, receipt, payload, stopped=True)["status"] == "failed"

    monkeypatch.setattr(
        client,
        "request_json",
        lambda *args: {"status": None, "error": "TimeoutError", "connection_refused": False},
    )
    assert client.verify(url, receipt, payload, stopped=True)["status"] == "failed"


def test_readiness_wait_uses_real_http_and_keeps_owned_instance(http_fixture, config, monkeypatch):
    url, _, _, _ = http_fixture
    config = replace(config, port=int(url.rsplit(":", 1)[1]))
    deployment = Deployment(config)
    monkeypatch.setattr(deployment, "owned", lambda: _container(config))
    monkeypatch.setattr(deployment, "docker", lambda *args: pytest.fail("Unexpected stop"))
    deployment.wait_ready()


def test_readiness_failure_stops_only_its_owned_container(config, monkeypatch):
    deployment = Deployment(config)
    container = _container(config)
    container["State"]["Running"] = False
    monkeypatch.setattr(deployment, "owned", lambda: container)
    calls = []
    monkeypatch.setattr(deployment, "docker", lambda *args: calls.append(args) or "")
    with pytest.raises(RuntimeError, match="exited before readiness"):
        deployment.wait_ready()
    assert calls == [("stop", "--time", "10", container["Id"])]


@pytest.mark.parametrize("body", [None, [], "not-an-object", 42])
def test_non_object_invalid_response_is_a_failed_check_not_a_crash(http_fixture, body):
    url, receipt, payload, response = http_fixture
    response["fixture_invalid_body"] = body
    report = client.verify(url, receipt, payload)
    assert report["status"] == "failed"
    assert report["checks"][-1]["passed"] is False
    assert report["checks"][0]["method"] == "GET"
    assert report["checks"][2]["method"] == "POST"


def test_config_selection_cli_verifies_bytes_and_refuses_output_overwrite(
    config, monkeypatch, tmp_path
):
    from src.deployment import select  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(select, "PROVENANCE", configuration.PROVENANCE)
    output = tmp_path / "selected-config.json"
    argv = [
        "--deployment-id",
        config.deployment_id,
        "--target-hostname",
        config.target_hostname,
        "--source-commit",
        config.source_commit,
        "--image-id",
        config.image_id,
        "--docker-socket",
        config.docker_socket,
        "--checkpoint",
        str(tmp_path / "new-slot"),
        "--from-checkpoint",
        config.checkpoint,
        "--params",
        config.params,
        "--cpu-threads",
        "2",
        "--memory-mb",
        "512",
        "--port",
        "18075",
        "--output",
        str(output),
    ]
    assert select.main(argv) == 0
    actual = DeploymentConfig.load(output)
    assert actual.checkpoint_tree_sha256 == config.checkpoint_tree_sha256
    assert select.main(argv) == 1
    output.unlink()
    (Path(config.checkpoint) / "model.safetensors").write_bytes(b"different-fixture")
    assert select.main(argv) == 1
    assert not output.exists()


@pytest.mark.parametrize(
    "module", ["src.deployment", "src.deployment.select", "src.deployment.verify"]
)
def test_documented_entrypoints_use_only_declared_bootstrap_packages(module, tmp_path):
    """No site/training environment: reproduce the clean-target import path with python -S."""
    import importlib.util  # pylint: disable=import-outside-toplevel
    import os  # pylint: disable=import-outside-toplevel
    import shutil  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel

    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    for package in ("dotenv", "loguru"):
        location = Path(importlib.util.find_spec(package).origin).parent
        shutil.copytree(
            location, bootstrap / package, ignore=shutil.ignore_patterns("__pycache__")
        )
    environment = {**os.environ, "PYTHONPATH": str(bootstrap), "PYTHONNOUSERSITE": "1"}
    result = subprocess.run(
        [sys.executable, "-S", "-m", module, "--help"],
        cwd=configuration.ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout and "--output" in result.stdout
