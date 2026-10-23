"""Release gates use fixture artifacts and subprocess doubles, never model inference."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from dvc.repo import Repo
import pytest
import yaml

from src import release_validation as release


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """Build tiny DVC-tracked artifacts with relocated weights and bound provenance."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    Repo.init(str(root)).close()
    data = root / "data"
    (data / "raw").mkdir(parents=True)
    (data / "processed").mkdir()
    (data / "raw/NSW.parquet").write_bytes(b"raw fixture")
    model = root / release.MODEL
    model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"fixture weights")
    (model / "config.json").write_text("{}")
    with Repo(str(root)) as repo:
        repo.add([str(data / "raw"), str(model)])
    outputs = []
    for block in release.BLOCKS:
        path = data / "processed" / f"{block}.parquet"
        path.write_bytes(block.encode())
        outputs.append(
            {
                "path": f"data/processed/{block}.parquet",
                "md5": hashlib.md5(block.encode()).hexdigest(),
            }
        )
    (root / "dvc.lock").write_text(yaml.safe_dump({"stages": {"split": {"outs": outputs}}}))
    (root / f"{release.MODEL}.provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "training_run_id": "a" * 32,
                "model_safetensors_sha256": hashlib.sha256(b"fixture weights").hexdigest(),
            }
        )
    )
    (root / "params.yaml").write_text(
        yaml.safe_dump(
            {
                "evaluation": {
                    "cpu_threads": 4,
                    "device": "cpu",
                    "seed": 7,
                    "num_samples": 20,
                }
            }
        )
    )
    relocated = tmp_path / "read-only-model"
    model.rename(relocated)
    monkeypatch.setattr(release, "ROOT", root)
    return SimpleNamespace(root=root, data=data, model=relocated, output=tmp_path / "evidence")


def test_verification_matches_dvc_directories_and_selected_provenance(artifacts):
    identities = release.verify_artifacts(artifacts.data, artifacts.model)
    assert len(identities["artifacts"]) == 6
    assert identities["model"]["training_run_id"] == "a" * 32
    assert identities["model"]["path"] == str(artifacts.model)


@pytest.mark.parametrize(
    "target,change",
    [
        ("data/processed/test.parquet", "corrupt"),
        ("data/raw/NSW.parquet", "missing"),
        ("model/config.json", "corrupt"),
        ("model/model.safetensors", "corrupt"),
        ("provenance", "missing"),
    ],
)
def test_verification_rejects_missing_or_changed_artifacts(artifacts, target, change):
    if target == "provenance":
        path = artifacts.root / f"{release.MODEL}.provenance.json"
    elif target.startswith("model/"):
        path = artifacts.model / target.removeprefix("model/")
    else:
        path = artifacts.root / target
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"changed")
    with pytest.raises((ValueError, OSError)):
        release.verify_artifacts(artifacts.data, artifacts.model)


@pytest.mark.parametrize("symlink", [False, True])
def test_verification_rejects_extra_files_and_member_symlinks(artifacts, symlink):
    extra = artifacts.model / "extra.json"
    if symlink:
        extra.symlink_to(artifacts.model / "config.json")
    else:
        extra.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="symlinks|identity mismatch"):
        release.verify_artifacts(artifacts.data, artifacts.model)


@pytest.fixture
def runner(artifacts, monkeypatch):
    """Keep real identity verification; replace only expensive child checks."""
    calls = []
    monkeypatch.setattr(release, "_source_record", lambda: {"fixture": True})
    monkeypatch.setattr(release, "_data_workspace", lambda *_: artifacts.root)

    def execute(command, *, cwd, log, seconds):
        calls.append((command, cwd, seconds))
        log.write_text("retained fixture output")
        if "--verify-only" in command:
            path = Path(command[command.index("--output-dir") + 1])
            path.write_text(json.dumps(release.verify_artifacts(artifacts.data, artifacts.model)))
        return {"status": "passed", "exit_code": 0, "command": command}

    monkeypatch.setattr(release, "run_step", execute)
    return calls


def test_success_propagates_exact_weights_sampling_reports_and_budgets(artifacts, runner):
    path, report = release.run_validation(
        output_dir=artifacts.output, data_dir=artifacts.data, checkpoint=artifacts.model
    )
    assert report["status"] == "passed" and report["exit_code"] == 0
    assert json.loads(path.read_text()) == report
    assert report["acceptance"] == release.ACCEPTANCE
    assert [step["name"] for step in report["steps"]] == [
        "verify",
        "configure-data",
        "data",
        "minimum-functionality",
        "directional",
    ]
    for command, _, seconds in runner[-2:]:
        assert command[:2] == [sys.executable, "-m"]
        assert command[command.index("--checkpoint") + 1] == str(artifacts.model)
        params_path = Path(command[command.index("--params-path") + 1])
        params = yaml.safe_load(params_path.read_text())
        assert params["evaluation"] == {
            "cpu_threads": 2,
            "device": "cpu",
            "seed": 7,
            "num_samples": 20,
        }
        assert seconds <= release.BUDGETS[command[2].rsplit(".", 1)[-1].replace("_", "-")]
    assert runner[-2][0][-1] == str(path.parent / "minimum-functionality.json")
    assert runner[-1][0][-1] == str(path.parent / "directional")
    assert runner[-2][0][runner[-2][0].index("--max-seconds") + 1] == "300"
    assert runner[-1][0][runner[-1][0].index("--max-seconds") + 1] == "180"


@pytest.mark.parametrize(
    "failed_step,status,code",
    [
        ("verify", "failed", 1),
        ("data", "failed", 1),
        ("minimum-functionality", "timeout", 124),
        ("directional", "failed", 7),
    ],
)
def test_failed_check_stops_later_checks_and_retains_report(
    artifacts, runner, monkeypatch, failed_step, status, code
):
    execute = release.run_step

    def fail(command, *, log, **kwargs):
        if log.stem == failed_step:
            log.write_text("controlled failure evidence")
            return {"status": status, "exit_code": code}
        return execute(command, log=log, **kwargs)

    monkeypatch.setattr(release, "run_step", fail)
    path, report = release.run_validation(
        output_dir=artifacts.output, data_dir=artifacts.data, checkpoint=artifacts.model
    )
    assert report["status"] == status and report["exit_code"] == code
    assert report["steps"][-1]["name"] == failed_step
    assert (path.parent / f"{failed_step}.log").read_text() == "controlled failure evidence"
    assert json.loads(path.read_text())["exit_code"] == code


def test_missing_artifact_prevents_any_data_or_model_checks(artifacts, runner):
    (artifacts.data / "processed/train.parquet").unlink()
    path, report = release.run_validation(
        output_dir=artifacts.output, data_dir=artifacts.data, checkpoint=artifacts.model
    )
    assert report["exit_code"] != 0 and len(runner) == 1
    assert "train.parquet" in json.loads(path.read_text())["error"]


def test_total_budget_caps_each_step_and_prevents_false_success(artifacts, runner, monkeypatch):
    clock = iter([0, 0, 0.8, 1.2, 1.3, 1.4, 1.5])
    monkeypatch.setattr(release.time, "monotonic", lambda: next(clock))
    _, report = release.run_validation(
        output_dir=artifacts.output,
        data_dir=artifacts.data,
        checkpoint=artifacts.model,
        max_seconds=1,
    )
    assert report["status"] == "timeout" and report["exit_code"] == 124
    assert runner[0][2] == 1
    assert runner[1][2] == pytest.approx(0.2)


def test_fresh_directory_preserves_previous_failure(artifacts, runner):
    first, _ = release.run_validation(
        output_dir=artifacts.output, data_dir=artifacts.data, checkpoint=artifacts.model
    )
    previous = first.read_bytes()
    second, _ = release.run_validation(
        output_dir=artifacts.output, data_dir=artifacts.data, checkpoint=artifacts.model
    )
    assert first != second and first.read_bytes() == previous


def test_source_capture_failure_is_reported_before_any_checks(artifacts, monkeypatch):
    def missing_source():
        raise FileNotFoundError("missing tracked source")

    monkeypatch.setattr(release, "_source_record", missing_source)
    path, report = release.run_validation(
        output_dir=artifacts.output, data_dir=artifacts.data, checkpoint=artifacts.model
    )
    assert report["status"] == "failed" and report["steps"] == []
    assert "missing tracked source" in path.read_text(encoding="utf-8")


def test_pull_rejects_relocated_artifacts_without_retrieving(artifacts, runner):
    _, report = release.run_validation(
        output_dir=artifacts.output, data_dir=artifacts.data, checkpoint=artifacts.model, pull=True
    )
    assert report["exit_code"] != 0 and not runner
    assert "repository-local" in report["error"]


def test_pull_occurs_before_verification(artifacts, runner):
    target = artifacts.root / release.MODEL
    artifacts.model.rename(target)
    artifacts.model = target
    _, report = release.run_validation(
        output_dir=artifacts.output, data_dir=artifacts.data, checkpoint=target, pull=True
    )
    assert report["exit_code"] == 0
    assert runner[0][0] == [sys.executable, "-m", "src.release_validation", "--retrieve-only"]
    assert report["steps"][0]["name"] == "retrieve"


@pytest.mark.skipif(os.name != "posix", reason="POSIX release runner")
def test_process_nonzero_exit_retains_output(tmp_path):
    log = tmp_path / "failed.log"
    result = release.run_step(
        [sys.executable, "-c", "print('failure details'); raise SystemExit(7)"],
        cwd=tmp_path,
        log=log,
        seconds=5,
    )
    assert result["exit_code"] == 7 and result["status"] == "failed"
    assert "failure details" in log.read_text()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_timeout_kills_nested_worker_and_retains_output(tmp_path, monkeypatch):
    real_killpg = os.killpg
    killed = []

    def kill_group(pid, sig):
        killed.append((pid, sig))
        real_killpg(pid, sig)

    monkeypatch.setattr(release.os, "killpg", kill_group)
    script = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "print('nested worker started',flush=True); time.sleep(30)"
    )
    log = tmp_path / "timeout.log"
    result = release.run_step([sys.executable, "-c", script], cwd=tmp_path, log=log, seconds=0.5)
    assert result["status"] == "timeout" and result["exit_code"] == 124
    assert len(killed) == 1 and killed[0][1] == release.signal.SIGKILL
    assert "nested worker started" in log.read_text()


def test_exhausted_budget_does_not_start_process(tmp_path, monkeypatch):
    monkeypatch.setattr(
        release.subprocess, "Popen", lambda *_a, **_k: pytest.fail("must not start")
    )
    result = release.run_step(["unused"], cwd=tmp_path, log=tmp_path / "log", seconds=0)
    assert result["exit_code"] == 124


def test_credentials_only_reach_retrieval(monkeypatch, tmp_path):
    monkeypatch.setenv("DVC_HTTP_USER", "fixture-user")
    monkeypatch.setenv("DVC_HTTP_PASSWORD", "fixture-password")
    assert "DVC_HTTP_PASSWORD" not in release._environment(tmp_path)
    assert release._environment(tmp_path, retrieval=True)["DVC_HTTP_USER"] == "fixture-user"


def test_retrieval_uses_only_declared_targets_and_in_memory_credentials(monkeypatch):
    calls = []

    class Repository:
        """Capture DVC calls without authentication or network access."""

        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def pull(self, **kwargs):
            """Capture exact artifact selection."""
            calls.append(kwargs)

    monkeypatch.setattr("dvc.repo.Repo", Repository)
    monkeypatch.setenv("DVC_HTTP_USER", "fixture-user")
    monkeypatch.setenv("DVC_HTTP_PASSWORD", "fixture-password")
    release.retrieve_artifacts()
    assert calls[0][1]["remote_config"] == {
        "auth": "basic",
        "user": "fixture-user",
        "password": "fixture-password",
    }
    assert calls[1] == {"targets": list(release.DVC_TARGETS), "remote": "origin", "jobs": 2}


def test_retrieval_rejects_an_expanded_stage_before_network_access(artifacts, monkeypatch):
    (artifacts.root / "dvc.yaml").write_text(
        yaml.safe_dump(
            {
                "stages": {
                    "split": {
                        "outs": [
                            *(f"data/processed/{block}.parquet" for block in release.BLOCKS),
                            "extra.bin",
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("dvc.repo.Repo", lambda *_a, **_k: pytest.fail("must not retrieve"))
    with pytest.raises(ValueError, match="exactly the four"):
        release.retrieve_artifacts()


def test_data_workspace_does_not_write_to_original_source_or_data(artifacts):
    (artifacts.root / "src").mkdir()
    (artifacts.root / "src/config.py").write_text("# fixture")
    artifacts.output.mkdir()
    workspace = release._data_workspace(artifacts.output, artifacts.data)
    (workspace / "src/config.py").write_text("# changed private copy")
    assert (artifacts.root / "src/config.py").read_text() == "# fixture"
    assert (workspace / "data").resolve() == artifacts.data


def test_cli_exits_nonzero_and_prints_retained_report(tmp_path, monkeypatch, capsys):
    path = tmp_path / "report.json"
    monkeypatch.setattr(
        release, "run_validation", lambda **_: (path, {"status": "failed", "exit_code": 7})
    )
    monkeypatch.setattr(sys, "argv", ["release-validation"])
    with pytest.raises(SystemExit) as error:
        release.main()
    assert error.value.code == 7 and str(path) in capsys.readouterr().out
