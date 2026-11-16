"""Synthetic receipts test the manifest contract without Docker, DVC or inference."""

from copy import deepcopy
from dataclasses import asdict
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
import yaml

from src import release_manifest as release
from src import release_validation as validation


def write_json(path, value):
    """Write a fixture receipt outside its clean source checkout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def seal_receipts(receipts):
    """Package fabricated test results using the real producer's handoff contract."""
    directory = receipts.directory
    report = json.loads((directory / "report.json").read_text())
    write_json(
        directory / "handoff.json", validation._public_handoff(directory / "report.json", report)
    )
    (directory / "handoff.sha256").write_text(
        f"{release._hash(directory / 'handoff.json')}  handoff.json\n"
    )
    archive = directory / "validation-artifact.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for pattern in (*validation.EVIDENCE_PATTERNS, "*.log", "handoff.json", "handoff.sha256"):
            for path in sorted(directory.glob(pattern)):
                bundle.write(path, report["run_id"] + "/" + path.relative_to(directory).as_posix())
    run = {
        "id": 123,
        "run_attempt": 1,
        "head_sha": receipts.sha,
        "head_branch": "main",
        "path": release.VALIDATION_WORKFLOW,
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "run_started_at": "2026-11-12T00:00:00Z",
        "updated_at": "2026-11-12T00:02:00Z",
        "repository": {"full_name": release.GITHUB_REPOSITORY, "id": 42},
        "head_repository": {"full_name": release.GITHUB_REPOSITORY, "id": 42},
    }
    artifact = {
        "id": 456,
        "name": "release-validation-123-1",
        "expired": False,
        "digest": "sha256:" + release._hash(archive),
        "size_in_bytes": archive.stat().st_size,
        "created_at": "2026-11-12T00:01:00Z",
        "updated_at": "2026-11-12T00:01:01Z",
        "workflow_run": {
            "id": 123,
            "repository_id": 42,
            "head_repository_id": 42,
            "head_sha": receipts.sha,
        },
    }
    selection = {
        "source_sha": receipts.sha,
        "run_id": "123",
        "run_attempt": "1",
        "artifact_id": "456",
        "artifact_digest": artifact["digest"],
    }
    write_json(
        directory / "actions-provenance.json",
        {"run": run, "artifact": artifact, "selection": selection},
    )
    receipts.api_run, receipts.api_artifact, receipts.selection = run, artifact, selection


@pytest.fixture
def receipts(tmp_path):
    """Use tracked configuration/identities with explicitly fabricated test outcomes."""
    root = tmp_path / "source"
    root.mkdir()
    names = (
        "params.yaml",
        "pyproject.toml",
        "uv.lock",
        "dvc.yaml",
        "dvc.lock",
        "data/raw.dvc",
        f"{release.MODEL}.dvc",
        f"{release.MODEL}.provenance.json",
        "tests/test_minimum_functionality_model.py",
        release.VALIDATION_WORKFLOW,
    )
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(release.ROOT / name, target)
    (root / "src").mkdir()
    (root / "src/__init__.py").write_text('"""Fixture source, never executed."""\n')
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Record fixture inputs",
        ],
        cwd=root,
        check=True,
    )
    sha = release._git(root, "rev-parse", "HEAD")
    provenance = json.loads((root / f"{release.MODEL}.provenance.json").read_text())
    model = {
        "path": "/fixture-only/checkpoint",
        "source": "local",
        "sha256": provenance["model_safetensors_sha256"],
        "training_run_id": provenance["training_run_id"],
        "training_metadata": [],
    }
    expected = {
        "data/raw": yaml.safe_load((root / "data/raw.dvc").read_text())["outs"][0]["md5"],
        release.MODEL: yaml.safe_load((root / f"{release.MODEL}.dvc").read_text())["outs"][0][
            "md5"
        ],
    }
    expected.update(
        {
            row["path"]: row["md5"]
            for row in yaml.safe_load((root / "dvc.lock").read_text())["stages"]["split"]["outs"]
        }
    )
    identities = {
        "model": model,
        "artifacts": {
            name: {"path": "/fixture-only/" + name, "md5": md5} for name, md5 in expected.items()
        },
    }
    directory = tmp_path / "validation"
    directory.mkdir()
    write_json(directory / "identities.json", identities)
    params = yaml.safe_load((root / "params.yaml").read_text())
    params["evaluation"].update(device="cpu", cpu_threads=2)
    (directory / "params-cpu2.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    source = {
        "git_commit": sha,
        "git_tree": release._git(root, "rev-parse", "HEAD^{tree}"),
        "git_dirty": False,
        "sha256": release._source_hashes(root),
        "github": {
            "GITHUB_SHA": sha,
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_REPOSITORY": release.GITHUB_REPOSITORY,
            "GITHUB_WORKFLOW_REF": f"{release.GITHUB_REPOSITORY}/{release.VALIDATION_WORKFLOW}@refs/heads/main",
        },
    }
    steps = []
    for name in (
        "retrieve",
        "verify",
        "configure-data",
        "data",
        "minimum-functionality",
        "directional",
    ):
        (directory / f"{name}.log").write_text("Synthetic test receipt; no model executed.\n")
        steps.append(
            {
                "name": name,
                "status": "passed",
                "exit_code": 0,
                "budget_seconds": release.BUDGETS[name],
                "wall_seconds": 1.0,
            }
        )
    report = {
        "schema_version": 1,
        "run_id": "a" * 32,
        "status": "passed",
        "exit_code": 0,
        "acceptance": release.ACCEPTANCE,
        "budgets_seconds": release.BUDGETS,
        "total_budget_seconds": release.TOTAL_SECONDS,
        "wall_seconds": 10.0,
        "started_utc": "2026-11-12T00:00:00Z",
        "finished_utc": "2026-11-12T00:00:10Z",
        "steps": steps,
        "source": source,
        "identities": identities,
        "executed_params_sha256": release._hash(directory / "params-cpu2.yaml"),
    }
    write_json(directory / "report.json", report)
    write_json(
        directory / "data-check/reports/validation.json",
        {
            "success": True,
            "evaluated_expectations": 118,
            "failed_expectations": 0,
            "failures": [],
        },
    )
    settings = {
        name: params["evaluation"][name]
        for name in ("num_samples", "seed", "device", "cpu_threads")
    }
    common = {"model": model, "settings": settings, "sampling": release.SAMPLING}
    write_json(
        directory / "minimum-functionality.json",
        {
            **common,
            "status": "completed",
            "exit_code": 0,
            "within_budget": True,
            "provenance": source,
            "protocol": {"cases": [asdict(c) for c in release.minimum_functionality.CASES]},
            "cases": [
                {"case_id": c.case_id, "status": "passed"}
                for c in release.minimum_functionality.CASES
            ],
        },
    )
    plan = {
        **common,
        "git_commit": sha,
        "git_dirty": False,
        "cases": [asdict(c) for c in release.directional.CASES],
        "minimum_response_fraction": release.directional.MIN_RESPONSE_FRACTION,
        "control_max_absolute_difference": release.directional.CONTROL_TOLERANCE,
    }
    write_json(directory / "directional/fixture/plan.json", plan)
    write_json(
        directory / "directional/fixture/report.json",
        {
            "status": "completed",
            "success": True,
            "plan": plan,
            "control": {"passed": True},
            "cases": [{"case_id": c.case_id, "passed": True} for c in release.directional.CASES],
        },
    )
    repository = "ghcr.io/example/forecast"
    digest = "sha256:" + "e" * 64
    image = [
        {
            "RepoDigests": [repository + "@" + digest],
            "RepoTags": [repository + ":v1.0.0"],
            "Architecture": "amd64",
            "Os": "linux",
            "Config": {
                "Labels": {
                    "org.opencontainers.image.revision": sha,
                    "org.opencontainers.image.source": f"https://github.com/{release.GITHUB_REPOSITORY}",
                }
            },
        }
    ]
    inspection = tmp_path / "image-inspect.json"
    inspection.write_text(json.dumps(image))
    result = SimpleNamespace(
        root=root,
        directory=directory,
        report=report,
        sha=sha,
        image=image,
        inspection=inspection,
        repository=repository,
        reference=repository + "@" + digest,
    )
    seal_receipts(result)
    return result


def manifest(receipts):
    """Create a manifest from synthetic passed reports and Docker metadata."""
    return release.create_manifest(
        receipts.directory,
        receipts.sha,
        receipts.inspection,
        repository=receipts.repository,
        tag="v1.0.0",
        architecture="amd64",
        root=receipts.root,
    )


def test_create_and_read_only_verify_bind_complete_evidence(receipts):
    result = manifest(receipts)
    before = {p: p.read_bytes() for p in receipts.directory.rglob("*") if p.is_file()}
    assert (
        release.verify_manifest(
            result, receipts.directory, receipts.reference, "amd64", root=receipts.root
        )
        == result
    )
    assert {p: p.read_bytes() for p in before} == before
    assert not release._git(receipts.root, "status", "--porcelain")
    assert result["source"]["git_commit"] == receipts.sha
    assert result["image"]["digest"] != result["model"]["weights_sha256"]
    assert len(result["validation"]["evidence_sha256"]) == 16
    assert "fixture-only" not in json.dumps(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "failed",
        "missing-step",
        "failed-step",
        "over-budget",
        "wrong-source",
        "dirty-source",
        "changed-hash",
        "missing-hash",
        "wrong-dvc",
        "wrong-checkpoint",
    ],
)
def test_reject_failed_incomplete_or_mismatched_summary(receipts, mutation):
    report = deepcopy(receipts.report)
    if mutation == "failed":
        report.update(status="failed", exit_code=1)
    elif mutation == "missing-step":
        report["steps"].pop()
    elif mutation == "failed-step":
        report["steps"][-1].update(status="failed", exit_code=7)
    elif mutation == "over-budget":
        report["steps"][-1]["wall_seconds"] = 1000
    elif mutation == "wrong-source":
        report["source"]["git_commit"] = "0" * 40
    elif mutation == "dirty-source":
        report["source"]["git_dirty"] = True
    elif mutation == "changed-hash":
        report["source"]["sha256"]["params.yaml"] = "0" * 64
    elif mutation == "missing-hash":
        report["source"]["sha256"].pop("uv.lock")
    elif mutation == "wrong-dvc":
        report["identities"]["artifacts"][release.MODEL]["md5"] = "0" * 32 + ".dir"
        write_json(receipts.directory / "identities.json", report["identities"])
    else:
        report["identities"]["model"]["sha256"] = "0" * 64
        write_json(receipts.directory / "identities.json", report["identities"])
    write_json(receipts.directory / "report.json", report)
    seal_receipts(receipts)
    with pytest.raises(ValueError):
        manifest(receipts)


@pytest.mark.parametrize(
    "name",
    [
        "report.json",
        "identities.json",
        "params-cpu2.yaml",
        "verify.log",
        "minimum-functionality.json",
        "directional/fixture/report.json",
        "directional/fixture/plan.json",
        "data-check/reports/validation.json",
    ],
)
def test_reject_missing_or_symlinked_retained_evidence(receipts, name):
    path = receipts.directory / name
    saved = path.with_name(path.name + ".saved")
    path.rename(saved)
    with pytest.raises(ValueError):
        manifest(receipts)
    path.symlink_to(saved)
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        manifest(receipts)


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("minimum-functionality.json", "status", "incomplete"),
        ("minimum-functionality.json", "cases", []),
        ("directional/fixture/report.json", "success", False),
        ("data-check/reports/validation.json", "failed_expectations", 1),
    ],
)
def test_reject_failed_detail_despite_successful_summary(receipts, name, field, value):
    path = receipts.directory / name
    report = json.loads(path.read_text())
    report[field] = value
    write_json(path, report)
    seal_receipts(receipts)
    with pytest.raises(ValueError):
        manifest(receipts)


@pytest.mark.parametrize(
    "mutation", ["untracked", "tracked", "other-commit", "wrong-parameters", "assume-unchanged"]
)
def test_source_and_executed_configuration_must_match(receipts, mutation):
    if mutation == "untracked":
        (receipts.root / "untracked.txt").write_text("dirty")
    elif mutation == "tracked":
        (receipts.root / "params.yaml").write_text("changed")
    elif mutation == "other-commit":
        receipts.sha = "0" * 40
    elif mutation == "assume-unchanged":
        subprocess.run(
            ["git", "update-index", "--assume-unchanged", "params.yaml"],
            cwd=receipts.root,
            check=True,
        )
        (receipts.root / "params.yaml").write_text("hidden mutation")
        assert not release._git(receipts.root, "status", "--porcelain")
    else:
        (receipts.directory / "params-cpu2.yaml").write_text("evaluation: {}\n")
    with pytest.raises(ValueError):
        manifest(receipts)


@pytest.mark.parametrize(
    "field,value",
    [
        ("Architecture", "arm64"),
        ("Os", "windows"),
        ("RepoDigests", []),
        ("RepoDigests", ["ghcr.io/foreign/image@sha256:" + "e" * 64]),
        ("RepoTags", ["ghcr.io/example/forecast:latest"]),
        ("Config", {"Labels": {"org.opencontainers.image.revision": "0" * 40}}),
    ],
)
def test_docker_identity_platform_and_version_must_match(receipts, field, value):
    receipts.image[0][field] = value
    receipts.inspection.write_text(json.dumps(receipts.image))
    with pytest.raises(ValueError):
        manifest(receipts)


@pytest.mark.parametrize(
    "field", ["model", "configuration", "validation", "source", "image", "extra"]
)
def test_verify_rejects_tampered_manifest(receipts, field):
    result = manifest(receipts)
    if field == "extra":
        result[field] = True
    elif field == "image":
        result[field]["digest"] = "sha256:" + "0" * 64
    elif field == "source":
        result[field]["git_tree"] = "0" * 40
    else:
        result[field]["tampered"] = True
    with pytest.raises(ValueError):
        release.verify_manifest(
            result, receipts.directory, receipts.reference, "amd64", root=receipts.root
        )


def test_verify_rejects_changed_evidence_and_expected_digest(receipts):
    result = manifest(receipts)
    with pytest.raises(ValueError, match="digest mismatch"):
        release.verify_manifest(
            result,
            receipts.directory,
            receipts.repository + "@sha256:" + "0" * 64,
            "amd64",
            root=receipts.root,
        )
    (receipts.directory / "verify.log").write_text("tampered log")
    with pytest.raises(ValueError, match="bytes mismatch"):
        release.verify_manifest(
            result, receipts.directory, receipts.reference, "amd64", root=receipts.root
        )


@pytest.mark.parametrize(
    "repository,version,digest",
    [
        ("example/forecast", "v1.0.0", "sha256:" + "e" * 64),
        ("https://user:secret@example.com/forecast", "v1.0.0", "sha256:" + "e" * 64),
        ("ghcr.io/example/forecast", "latest", "sha256:" + "e" * 64),
        ("ghcr.io/example/forecast", "v1.0.0", "e" * 64),
    ],
)
def test_reject_implicit_destination_mutable_alias_and_missing_digest(repository, version, digest):
    with pytest.raises(ValueError):
        release._image(repository, version, digest, "amd64", "0" * 40)


def test_cli_rejections_are_specific_and_do_not_echo_untrusted_values(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "manifest.json"
    path.write_text('{"secret":"never-print-this", "schema_version":1, "schema_version":2}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "release-manifest",
            "verify",
            "--manifest",
            str(path),
            "--validation-dir",
            str(tmp_path),
            "--image-reference",
            "unused",
            "--architecture",
            "amd64",
        ],
    )
    with pytest.raises(SystemExit) as error:
        release.main()
    captured = capsys.readouterr()
    assert error.value.code == 1
    assert "Duplicate JSON object key" in captured.err
    assert "never-print-this" not in captured.err + captured.out


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("head_sha", "0" * 40),
        ("path", ".github/workflows/ci.yml"),
        ("event", "pull_request"),
        ("run_attempt", 2),
        ("id", 999),
        ("head_branch", "unreviewed-branch"),
        ("repository", {"id": 1, "full_name": "foreign/repository"}),
        ("head_repository", {"id": 1, "full_name": "foreign/repository"}),
    ],
)
def test_reject_unsuccessful_or_different_api_selected_run(receipts, field, value):
    run = deepcopy(receipts.api_run)
    run[field] = value
    with pytest.raises(ValueError, match="run/source/workflow/event/attempt"):
        release.artifact_record(run, receipts.api_artifact, receipts.selection)


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 999),
        ("expired", True),
        ("name", "release-validation-123-2"),
        ("digest", "sha256:" + "0" * 64),
        ("size_in_bytes", release.ARCHIVE_LIMIT + 1),
        ("created_at", "2026-11-11T00:00:00Z"),
        ("updated_at", "2026-11-13T00:00:00Z"),
        (
            "workflow_run",
            {"id": 999, "repository_id": 42, "head_repository_id": 42, "head_sha": "0" * 40},
        ),
    ],
)
def test_reject_missing_expired_or_different_api_artifact(receipts, field, value):
    artifact = deepcopy(receipts.api_artifact)
    artifact[field] = value
    with pytest.raises(ValueError, match="artifact"):
        release.artifact_record(receipts.api_run, artifact, receipts.selection)


def test_ingest_checks_digest_before_unpacking_and_retains_only_selected_metadata(
    receipts, tmp_path
):
    output = tmp_path / "downloaded"
    receipts.api_run["untrusted_extra"] = "never-print-this"
    directory = release.ingest(
        receipts.directory / "validation-artifact.zip",
        receipts.api_run,
        receipts.api_artifact,
        receipts.selection,
        output,
    )
    assert directory.name == receipts.report["run_id"]
    assert "never-print-this" not in (directory / "actions-provenance.json").read_text()
    assert (
        release.candidate(directory, receipts.sha, root=receipts.root)["validation"]["artifact"][
            "id"
        ]
        == 456
    )
    with pytest.raises(FileExistsError):
        release.ingest(
            receipts.directory / "validation-artifact.zip",
            receipts.api_run,
            receipts.api_artifact,
            receipts.selection,
            output,
        )
    bad = tmp_path / "tampered.zip"
    bad.write_bytes((receipts.directory / "validation-artifact.zip").read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="digest mismatch"):
        release.ingest(
            bad, receipts.api_run, receipts.api_artifact, receipts.selection, tmp_path / "rejected"
        )
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize(
    "member", ["../escape.json", "/absolute.json", "a" * 32 + "/weights.safetensors"]
)
def test_ingest_rejects_unsafe_or_unexpected_members(receipts, tmp_path, member):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, b"not model weights")
    selection = {**receipts.selection, "artifact_digest": "sha256:" + release._hash(archive)}
    artifact = {
        **receipts.api_artifact,
        "digest": selection["artifact_digest"],
        "size_in_bytes": archive.stat().st_size,
    }
    with pytest.raises(ValueError, match="artifact member"):
        release.ingest(archive, receipts.api_run, artifact, selection, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("filename", ["handoff.json", "report.json", "validation-artifact.zip"])
def test_tampered_handoff_report_and_downloaded_archive_fail(receipts, filename):
    path = receipts.directory / filename
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        manifest(receipts)


def test_cli_create_check_verify_and_no_overwrite(receipts, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(release, "ROOT", receipts.root)
    output = tmp_path / "manifest.json"
    common = [
        "--validation-dir",
        str(receipts.directory),
        "--source-sha",
        receipts.sha,
        "--repository",
        receipts.repository,
        "--version",
        "v1.0.0",
        "--architecture",
        "amd64",
        "--image-inspect",
        str(receipts.inspection),
    ]
    monkeypatch.setattr(sys, "argv", ["release-manifest", "check", *common])
    release.main()
    assert json.loads(capsys.readouterr().out)["source"]["git_commit"] == receipts.sha
    monkeypatch.setattr(
        sys, "argv", ["release-manifest", "create", *common, "--output", str(output)]
    )
    release.main()
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text())
    retained = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        release.main()
    assert error.value.code == 1 and output.read_bytes() == retained
    capsys.readouterr()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "release-manifest",
            "verify",
            "--validation-dir",
            str(receipts.directory),
            "--manifest",
            str(output),
            "--image-reference",
            receipts.reference,
            "--architecture",
            "amd64",
        ],
    )
    release.main()
    assert json.loads(capsys.readouterr().out)["image"]["reference"] == receipts.reference


def test_manual_workflow_keeps_credentials_and_runtime_evidence_separate():
    workflow = yaml.safe_load(
        (release.ROOT / ".github/workflows/image-publication.yml").read_text()
    )
    assert set(workflow[True]) == {"workflow_dispatch"}
    job = workflow["jobs"]["publish"]
    assert job["if"] == "github.ref == 'refs/heads/main'"
    assert job["timeout-minutes"] <= 60
    steps = job["steps"]
    assert all("DVC_HTTP_PASSWORD" not in step.get("env", {}) for step in steps)
    assert job["permissions"]["actions"] == "read"
    retrieval = [step for step in steps if "gh api" in step.get("run", "")]
    assert len(retrieval) == 1 and "attempts/$VALIDATION_ATTEMPT" in retrieval[0]["run"]
    assert "src.release_manifest ingest" in retrieval[0]["run"]
    builds = [step["run"] for step in steps if "docker build --platform" in step.get("run", "")]
    assert len(builds) == 1 and "--build-arg GIT_SHA=" in builds[0]
    assert "PASSWORD" not in builds[0] and "TOKEN" not in builds[0]
    pushes = [i for i, step in enumerate(steps) if "docker push " in step.get("run", "")]
    assert len(pushes) == 1
    assert "src.release_manifest check" in steps[pushes[0]]["run"]
    assert all("latest" not in step.get("run", "") for step in steps)
