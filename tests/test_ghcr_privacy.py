"""Fixture-only GHCR destination/privacy checks; no registry operation occurs."""

from copy import deepcopy
from io import BytesIO
import json
import os
import subprocess
import sys
from urllib.error import HTTPError, URLError

import pytest
import yaml

from src import ghcr_privacy as guard
from src.release_manifest import ROOT


@pytest.fixture
def api(monkeypatch):
    """Model authenticated metadata, including a private selected package."""
    organization = {"id": 10, "login": guard.ORGANIZATION, "type": "Organization"}
    responses = {
        f"orgs/{guard.ORGANIZATION}": (200, organization),
        f"repos/{guard.SOURCE_REPOSITORY}": (
            200,
            {"id": 20, "full_name": guard.SOURCE_REPOSITORY, "owner": organization},
        ),
        guard.PACKAGE_PATH: (
            200,
            {
                "id": 30,
                "name": guard.PACKAGE,
                "package_type": "container",
                "owner": organization,
                "visibility": "private",
            },
        ),
    }
    calls = []

    def get(path):
        calls.append(path)
        value = responses[path]
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)

    monkeypatch.setattr(guard, "_get", get)
    monkeypatch.setattr(guard.time, "sleep", lambda _seconds: None)
    return responses, calls


@pytest.mark.parametrize("phase", ["before", "after"])
def test_private_expected_package_succeeds(api, phase):
    receipt = guard.check_private_package(guard.DESTINATION, phase=phase)
    assert receipt["status"] == "passed" and receipt["package_state"] == "PRIVATE"
    assert receipt["package_id"] == 30 and receipt["visibility"] == "private"
    assert receipt["private_default_bootstrap"] is False


@pytest.mark.parametrize(
    "destination",
    [
        "ghcr.io/foreign/endesa-forecast",
        "ghcr.io/mlops-2627q1-mds-upc/other",
        "docker.io/example/endesa-forecast",
        guard.DESTINATION + ":latest",
    ],
)
def test_wrong_destination_fails_before_api_access(api, destination):
    receipt = guard.check_private_package(destination, phase="before", first_publication=True)
    assert receipt["status"] == "failed" and "selected" in receipt["error"]
    assert api[1] == []


@pytest.mark.parametrize("visibility", ["public", "internal", "unknown", None])
@pytest.mark.parametrize("phase", ["before", "after"])
def test_observed_nonprivate_or_malformed_visibility_always_fails(api, visibility, phase):
    api[0][guard.PACKAGE_PATH][1]["visibility"] = visibility
    receipt = guard.check_private_package(guard.DESTINATION, phase=phase, first_publication=True)
    assert receipt["status"] == "failed" and receipt["private_default_bootstrap"] is False


def test_404_default_fails_and_explicit_bootstrap_only_records_unobserved(api):
    api[0][guard.PACKAGE_PATH] = (404, None)
    blocked = guard.check_private_package(guard.DESTINATION, phase="before")
    assert blocked["status"] == "failed" and "normal publication" in blocked["error"]
    candidate = guard.check_private_package(
        guard.DESTINATION, phase="before", first_publication=True
    )
    assert candidate["status"] == "bootstrap-permitted"
    assert candidate["package_state"] == "UNOBSERVED" and candidate["private_default_bootstrap"]
    assert candidate["package_http_status"] == 404 and "visibility" not in candidate
    assert "reader_access" not in candidate and "absent" not in json.dumps(candidate)


@pytest.mark.parametrize("value", ["false", "true", 1])
def test_nonboolean_mode_cannot_authorize_bootstrap(api, value):
    receipt = guard.check_private_package(
        guard.DESTINATION, phase="before", first_publication=value
    )
    assert receipt["status"] == "failed" and api[1] == []


@pytest.mark.parametrize(
    "path,metadata",
    [
        (f"orgs/{guard.ORGANIZATION}", {"id": 99, "login": "foreign", "type": "Organization"}),
        (f"repos/{guard.SOURCE_REPOSITORY}", {"id": 20, "full_name": "foreign/repository"}),
        (guard.PACKAGE_PATH, {"id": 30, "name": "other", "package_type": "container"}),
        (guard.PACKAGE_PATH, {}),
    ],
)
def test_unverified_identity_or_malformed_metadata_never_authorizes_bootstrap(api, path, metadata):
    api[0][path] = (200, metadata)
    receipt = guard.check_private_package(
        guard.DESTINATION, phase="before", first_publication=True
    )
    assert receipt["status"] == "failed" and not receipt["private_default_bootstrap"]


@pytest.mark.parametrize(
    "path", [f"orgs/{guard.ORGANIZATION}", f"repos/{guard.SOURCE_REPOSITORY}"]
)
def test_identity_404_cannot_become_bootstrap_permission(api, path):
    api[0][path] = (404, None)
    receipt = guard.check_private_package(
        guard.DESTINATION, phase="before", first_publication=True
    )
    assert receipt["status"] == "failed" and not receipt["private_default_bootstrap"]
    assert guard.PACKAGE_PATH not in api[1]


def test_post_push_404_remains_unverified_with_bounded_retries_even_in_bootstrap_mode(api):
    api[0][guard.PACKAGE_PATH] = (404, None)
    receipt = guard.check_private_package(guard.DESTINATION, phase="after", first_publication=True)
    assert receipt["status"] == "failed" and "not observed after push" in receipt["error"]
    assert api[1].count(guard.PACKAGE_PATH) == guard.POST_ATTEMPTS


def test_post_push_observes_private_after_metadata_propagates(api, monkeypatch):
    get = guard._get
    count = 0

    def delayed(path):
        nonlocal count
        if path == guard.PACKAGE_PATH:
            count += 1
            if count < 3:
                return 404, None
        return get(path)

    monkeypatch.setattr(guard, "_get", delayed)
    assert guard.check_private_package(guard.DESTINATION, phase="after")["status"] == "passed"
    assert count == 3


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_package_api_error_cannot_become_bootstrap_permission(api, status):
    api[0][guard.PACKAGE_PATH] = guard.PrivacyError(f"GitHub metadata HTTP {status}; unverified")
    receipt = guard.check_private_package(
        guard.DESTINATION, phase="before", first_publication=True
    )
    assert receipt["status"] == "failed" and not receipt["private_default_bootstrap"]
    assert api[1] == [
        f"orgs/{guard.ORGANIZATION}",
        f"repos/{guard.SOURCE_REPOSITORY}",
        guard.PACKAGE_PATH,
    ]


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_http_authorization_and_other_errors_fail_without_exposing_body(monkeypatch, status):
    monkeypatch.setenv("GH_TOKEN", "never-print-token")
    calls = []

    def fail(request, *, timeout):
        calls.append((request, timeout))
        raise HTTPError(
            request.full_url, status, "never-print-secret", {}, BytesIO(b"secret body")
        )

    monkeypatch.setattr(guard, "urlopen", fail)
    receipt = guard.check_private_package(
        guard.DESTINATION, phase="before", first_publication=True
    )
    assert receipt["status"] == "failed" and str(status) in receipt["error"]
    assert "never-print" not in json.dumps(receipt) and "secret body" not in json.dumps(receipt)
    assert len(calls) == 1 and calls[0][1] == 10


def test_network_error_and_missing_token_do_not_grant_bootstrap(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert (
        guard.check_private_package(guard.DESTINATION, phase="before", first_publication=True)[
            "status"
        ]
        == "failed"
    )
    monkeypatch.setenv("GH_TOKEN", "never-print-token")
    monkeypatch.setattr(
        guard, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(URLError("never-print-secret"))
    )
    receipt = guard.check_private_package(
        guard.DESTINATION, phase="before", first_publication=True
    )
    assert receipt["status"] == "failed" and "never-print" not in json.dumps(receipt)


def test_cli_preserves_failed_receipt_and_exits_nonzero(api, tmp_path, monkeypatch):
    api[0][guard.PACKAGE_PATH] = (404, None)
    output = tmp_path / "before.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ghcr-privacy",
            "--repository",
            guard.DESTINATION,
            "--phase",
            "before",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        guard.main()
    assert error.value.code == 1 and json.loads(output.read_text())["status"] == "failed"


def test_cli_explicit_first_publication_records_unobserved_without_claiming_private(
    api, tmp_path, monkeypatch, capsys
):
    api[0][guard.PACKAGE_PATH] = (404, None)
    output = tmp_path / "before.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ghcr-privacy",
            "--repository",
            guard.DESTINATION,
            "--phase",
            "before",
            "--first-publication",
            "--output",
            str(output),
        ],
    )
    guard.main()
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "bootstrap-permitted" and receipt["package_state"] == "UNOBSERVED"
    assert json.loads(capsys.readouterr().out) == receipt


def test_failed_post_check_prevents_manifest_creation_in_actual_workflow_script(tmp_path):
    workflow = yaml.safe_load((ROOT / ".github/workflows/image-publication.yml").read_text())
    script = next(
        s["run"]
        for s in workflow["jobs"]["publish"]["steps"]
        if "docker push " in s.get("run", "")
    )
    commands = tmp_path / "bin"
    commands.mkdir()
    uv = commands / "uv"
    uv.write_text(
        '#!/bin/sh\nprintf "uv %s\\n" "$*" >> "$PRIVACY_TEST_TRACE"\ncase "$*" in *"--phase after"*) exit 19;; esac\n'
    )
    docker = commands / "docker"
    docker.write_text('#!/bin/sh\nprintf "docker %s\\n" "$*" >> "$PRIVACY_TEST_TRACE"\n')
    uv.chmod(0o755)
    docker.chmod(0o755)
    (tmp_path / "temporary/image-publication").mkdir(parents=True)
    trace = tmp_path / "commands.txt"
    result = subprocess.run(
        ["/bin/bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(commands) + os.pathsep + os.environ["PATH"],
            "PRIVACY_TEST_TRACE": str(trace),
            "VALIDATION_DIR": "fixture",
            "SOURCE_SHA": "a" * 40,
            "IMAGE_REPOSITORY": guard.DESTINATION,
            "IMAGE_TAG": "v1.0.0",
            "FIRST_PUBLICATION": "false",
        },
        check=False,
    )
    assert result.returncode == 19
    recorded = trace.read_text()
    assert "docker push " in recorded and "--phase after" in recorded
    assert "src.release_manifest create" not in recorded and "docker image inspect" not in recorded


def test_workflow_defaults_closed_and_checks_private_before_push_and_manifest():
    workflow = yaml.safe_load((ROOT / ".github/workflows/image-publication.yml").read_text())
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["first_publication"]["default"] is False
    assert inputs["image_repository"]["options"] == [guard.DESTINATION]
    steps = workflow["jobs"]["publish"]["steps"]
    mutation = next(s for s in steps if "docker push " in s.get("run", ""))
    script = mutation["run"]
    assert "set -euo pipefail" in script
    assert script.index("--phase before") < script.index("docker push ")
    assert (
        script.index("docker push ")
        < script.index("--phase after")
        < script.index("src.release_manifest create")
    )
    assert all("IMAGE_REGISTRY_PASSWORD" not in str(s) for s in steps)
