"""Exercise #75 with #78's verifier and fabricated receipts, without model execution."""

from copy import deepcopy
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import socket
from threading import Thread

import pytest

from src import release_manifest
from src.deployment import release, runtime
from src.deployment import verify as client
from src.deployment.config import COMMAND, DeploymentConfig, file_sha256, fingerprint, write_new
from tests.test_release_manifest import manifest as fixture_manifest
from tests.test_release_manifest import (  # pylint: disable=unused-import
    receipts as release_receipts,  # noqa: F401
)


@pytest.fixture
def bound_release(release_receipts, tmp_path, monkeypatch):  # noqa: F811
    """Reuse the producer's complete evidence fixture; no claim in it is a real run."""
    bundle = release_receipts
    manifest = fixture_manifest(bundle)
    path = tmp_path / "manifest.json"
    write_new(path, manifest)
    monkeypatch.setattr(release, "ROOT", bundle.root)
    # The file map is an inspection fixture; no weights are created or downloaded.
    files = {"model.safetensors": manifest["model"]["weights_sha256"], "config.json": "0" * 64}
    config = DeploymentConfig(
        schema_version=2,
        deployment_id="endesa75-release-fixture",
        target_hostname=socket.getfqdn(),
        source_commit=bundle.sha,
        image_id="sha256:" + "b" * 64,
        checkpoint=str(tmp_path / "selected"),
        checkpoint_sha256=manifest["model"]["weights_sha256"],
        checkpoint_tree_sha256=fingerprint(files),
        training_run_id=manifest["model"]["training_run_id"],
        params=str(bundle.root / "params.yaml"),
        params_sha256=manifest["configuration"]["params_sha256"],
        cpu_threads=2,
        memory_mb=512,
        port=18075,
        docker_socket=str(tmp_path / "docker.sock"),
        release={
            "manifest": str(path),
            "manifest_sha256": file_sha256(path),
            "validation_dir": str(bundle.directory),
            "image_reference": bundle.reference,
        },
    )
    config.validate()
    verified = {
        "checkpoint": {
            "sha256": config.checkpoint_sha256,
            "training_run_id": config.training_run_id,
        },
        "checkpoint_files": files,
        "checkpoint_tree_sha256": config.checkpoint_tree_sha256,
        "params_sha256": config.params_sha256,
        "settings": release.prediction_settings(manifest),
        "image": {"id": config.image_id, "revision": config.source_commit, "user": "10001:10001"},
        "release": {"manifest_sha256": file_sha256(path), "manifest": manifest},
    }
    receipt = {
        "schema_version": 1,
        "evidence_kind": "container-inspection",
        "config": config.as_dict(),
        "config_sha256": config.config_sha256,
        "source_commit": config.source_commit,
        "host": {"hostname": config.target_hostname},
        "container": {"image": config.image_id, "running": True, "oom_killed": False},
        "verified": verified,
    }
    return bundle, config, receipt


def test_schema_roundtrip_and_candidate_hash_compatibility(bound_release, tmp_path):
    _, config, _ = bound_release
    path = tmp_path / "config.json"
    write_new(path, config.as_dict())
    assert DeploymentConfig.load(path) == config
    candidate = replace(config, schema_version=1, release=None)
    assert "release" not in candidate.as_dict()
    assert candidate.config_sha256 == fingerprint(
        {k: v for k, v in config.as_dict().items() if k != "release"} | {"schema_version": 1}
    )
    assert candidate.config_sha256 != config.config_sha256


def test_release_binding_reuses_complete_offline_verification_without_mutation(bound_release):
    bundle, config, receipt = bound_release
    before = {p: p.read_bytes() for p in bundle.directory.rglob("*") if p.is_file()}
    assert release.verify_release(config) == receipt["verified"]["release"]
    assert client.validate_receipt(receipt) == receipt["verified"]
    assert {p: p.read_bytes() for p in before} == before


def test_bad_release_fails_before_target_or_provisioning_mutations(bound_release, monkeypatch):
    _, config, _ = bound_release
    Path(config.release["manifest"]).write_text("{}", encoding="utf-8")
    deployment = runtime.Deployment(config)
    monkeypatch.setattr(deployment, "target", lambda: pytest.fail("Target was contacted"))
    with pytest.raises(ValueError, match="Release binding rejected"):
        deployment.preflight()
    with pytest.raises(ValueError, match="Release binding rejected"):
        deployment.provision(Path(config.checkpoint))
    assert not Path(config.checkpoint).exists()


@pytest.mark.parametrize("edit", ["extra", "missing", "path", "digest", "version"])
def test_release_config_requires_exact_explicit_fields(bound_release, tmp_path, edit):
    _, config, _ = bound_release
    raw = config.as_dict()
    if edit == "extra":
        raw["release"]["password"] = "fixture-only-must-not-be-accepted"
    elif edit == "missing":
        del raw["release"]["manifest_sha256"]
    elif edit == "path":
        raw["release"]["manifest"] = "relative/manifest.json"
    elif edit == "digest":
        raw["release"]["image_reference"] = "latest"
    else:
        raw["schema_version"] = True
    path = tmp_path / "invalid-config.json"
    write_new(path, raw)
    with pytest.raises(ValueError):
        DeploymentConfig.load(path)


@pytest.mark.parametrize(
    "change",
    [
        "report",
        "archive",
        "hidden-source",
        "manifest",
        "digest",
        "params",
        "threads",
        "source",
        "model",
    ],
)
def test_release_binding_rejects_mismatched_source_bundle_and_selection(bound_release, change):
    bundle, config, _ = bound_release
    if change in ("report", "archive"):
        (
            bundle.directory / ("report.json" if change == "report" else "validation-artifact.zip")
        ).write_bytes(b"altered fixture")
    elif change == "hidden-source":
        release_manifest._git(  # pylint: disable=protected-access
            bundle.root, "update-index", "--assume-unchanged", "src/__init__.py"
        )
        (bundle.root / "src/__init__.py").write_text(
            "# Different tracked bytes\n", encoding="utf-8"
        )
    elif change == "manifest":
        Path(config.release["manifest"]).write_text("{}", encoding="utf-8")
    elif change == "digest":
        config = replace(
            config,
            release={
                **config.release,
                "image_reference": bundle.repository + "@sha256:" + "f" * 64,
            },
        )
    else:
        fields = {
            "params": ("params_sha256", "f" * 64),
            "threads": ("cpu_threads", 1),
            "source": ("source_commit", "a" * 40),
            "model": ("checkpoint_sha256", "f" * 64),
        }
        name, value = fields[change]
        config = replace(config, **{name: value})
    with pytest.raises(ValueError, match="Release binding rejected"):
        release.verify_release(config)


@pytest.mark.parametrize("change", ["platform", "registry", "source-label"])
def test_release_image_checks_actual_registry_digest_and_platform(
    bound_release, monkeypatch, change
):
    bundle, config, receipt = bound_release
    image = deepcopy(bundle.image[0])
    image.update(Id=config.image_id)
    image["Config"].update(User="10001:10001", Cmd=COMMAND, Entrypoint=None)
    deployment = runtime.Deployment(config)
    monkeypatch.setattr(deployment, "docker", lambda *args: json.dumps([image]))
    assert deployment.image(receipt["verified"]["release"]) == receipt["verified"]["image"]
    if change == "platform":
        image["Architecture"] = "arm64"
    elif change == "registry":
        image["RepoDigests"] = []
    else:
        image["Config"]["Labels"]["org.opencontainers.image.source"] = (
            "https://example.invalid/other"
        )
    with pytest.raises(ValueError, match="platform/source/repository digest"):
        deployment.image(receipt["verified"]["release"])


@pytest.mark.parametrize("changed_settings", [False, True])
def test_preflight_compares_parsed_image_settings_with_verified_release(
    bound_release, monkeypatch, changed_settings
):
    _, config, receipt = bound_release
    expected = receipt["verified"]
    deployment = runtime.Deployment(config)
    monkeypatch.setattr(deployment, "target", dict)
    monkeypatch.setattr(DeploymentConfig, "verify_artifacts", lambda self: expected)
    monkeypatch.setattr(deployment, "image", lambda binding: expected["image"])
    observed = deepcopy({"checkpoint": expected["checkpoint"], "settings": expected["settings"]})
    if changed_settings:
        observed["settings"]["seed"] += 1
    monkeypatch.setattr(deployment, "parse_settings", lambda code: observed)
    if changed_settings:
        with pytest.raises(ValueError, match="executed prediction settings"):
            deployment.preflight()
    else:
        assert deployment.preflight()["release"] == expected["release"]


def test_separate_client_verifies_bundle_at_client_paths_and_actual_http(bound_release, tmp_path):
    _, config, receipt = bound_release
    path = tmp_path / "client-manifest.json"
    path.write_bytes(Path(config.release["manifest"]).read_bytes())
    payload = {"state": "TAS", "history": [-1.5] * 512}
    body = {
        "state": "TAS",
        "context_length": 512,
        "horizon": 48,
        "value_scale": "source",
        "model_id": "chronos-t5-small-fine-tuned",
        "model_revision": "local-sha256:" + config.checkpoint_sha256,
        "checkpoint": receipt["verified"]["checkpoint"],
        "settings": receipt["verified"]["settings"],
        "p10": [-2.0] * 48,
        "p50": [-1.5] * 48,
        "p90": [-1.0] * 48,
    }

    class Handler(BaseHTTPRequestHandler):
        """Answer with fixture identities while the client uses a separate socket."""

        def log_message(self, *_args):
            pass

        def respond(self, status, response):
            """Write one fixture response."""
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def do_GET(self):  # pylint: disable=invalid-name
            """Return fixture health/readiness."""
            self.respond(200, {"status": "ok" if self.path == "/health" else "ready"})

        def do_POST(self):  # pylint: disable=invalid-name
            """Reject a short request and otherwise return fixture quantiles."""
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if len(request["history"]) == 512:
                self.respond(200, body)
            else:
                self.respond(422, {"detail": [{"type": "too_short"}]})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        result = client.verify(
            f"http://127.0.0.1:{server.server_port}",
            receipt,
            payload,
            release_paths={
                "manifest_path": path,
                "validation_dir": Path(config.release["validation_dir"]),
            },
        )
        assert result["status"] == "passed"
        assert [r["status"] for r in result["checks"]] == [200, 200, 200, 422]
        assert result["release_manifest_sha256"] == config.release["manifest_sha256"]
        assert result["image_reference"] == config.release["image_reference"]
        receipt["verified"]["settings"]["seed"] += 1
        body["settings"]["seed"] += 1
        with pytest.raises(ValueError, match="release binding differ"):
            client.verify(f"http://127.0.0.1:{server.server_port}", receipt, payload)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
