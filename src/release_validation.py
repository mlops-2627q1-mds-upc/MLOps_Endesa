"""Bounded, offline release checks for the tracked data and selected checkpoint."""

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from uuid import uuid4

import yaml

from src.modeling.checkpoints import describe_checkpoint

ROOT = Path(__file__).resolve().parents[1]
MODEL = "models/chronos-t5-small-fine-tuned"
BLOCKS = ("train", "validation", "test", "origins")
DVC_TARGETS = ("data/raw.dvc", "split", f"{MODEL}.dvc")
# Wall-clock caps include process startup and report generation, not just inference.
BUDGETS = {
    "retrieve": 180,
    "verify": 60,
    "configure-data": 30,
    "data": 120,
    "minimum-functionality": 330,
    "directional": 210,
}
TOTAL_SECONDS = 900
ACCEPTANCE = (
    "All tracked artifact identities must match, data validation must pass, and all "
    "10 minimum-functionality cases plus 12 directional cases and the unchanged "
    "control must pass their existing thresholds within every step and total budget."
)


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dvc_hash(path: Path) -> str:
    """Hash these unfiltered artifact directories using DVC's sorted MD5 manifest."""
    if path.is_symlink():
        raise ValueError(f"Artifact members must not be symlinks: {path}")
    if not path.is_dir():
        return _hash(path, "md5")
    children = sorted(path.rglob("*"))
    if any(child.is_symlink() for child in children):
        raise ValueError(f"Artifact members must not be symlinks: {path}")
    entries = [
        {"md5": _hash(child, "md5"), "relpath": child.relative_to(path).as_posix()}
        for child in children
        if child.is_file()
    ]
    return hashlib.md5(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest() + ".dir"


def verify_artifacts(data_dir: Path, checkpoint: Path) -> dict:
    """Require the selected weights/provenance and every data/model DVC output."""
    model = describe_checkpoint(checkpoint, provenance=ROOT / f"{MODEL}.provenance.json")
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text(encoding="utf-8"))
    expected = {
        "data/raw": yaml.safe_load((ROOT / "data/raw.dvc").read_text())["outs"][0]["md5"],
        MODEL: yaml.safe_load((ROOT / f"{MODEL}.dvc").read_text())["outs"][0]["md5"],
    }
    outputs = {row["path"]: row["md5"] for row in lock["stages"]["split"]["outs"]}
    expected.update(
        {f"data/processed/{b}.parquet": outputs[f"data/processed/{b}.parquet"] for b in BLOCKS}
    )
    artifacts = {}
    for name, wanted in expected.items():
        path = checkpoint if name == MODEL else data_dir / Path(name).relative_to("data")
        actual = _dvc_hash(path)
        if actual != wanted:
            raise ValueError(f"DVC identity mismatch for {name}: expected {wanted}, got {actual}")
        artifacts[name] = {"path": str(path), "md5": actual}
    return {"model": model, "artifacts": artifacts}


def _environment(root: Path, *, retrieval: bool = False) -> dict:
    inherited = {
        key: value
        for key, value in os.environ.items()
        if retrieval or key not in ("DVC_HTTP_USER", "DVC_HTTP_PASSWORD")
    }
    return {
        **inherited,
        "PYTHONPATH": str(root),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "GX_ANALYTICS_ENABLED": "false",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
    }


def run_step(command: list[str], *, cwd: Path, log: Path, seconds: float) -> dict:
    """Kill the whole POSIX process group on timeout, including nested pytest workers."""
    started = time.monotonic()
    result = {
        "command": command,
        "cwd": str(cwd),
        "budget_seconds": seconds,
        "log": str(log),
        "status": "failed",
        "exit_code": 1,
    }
    if seconds <= 0:
        return {**result, "status": "timeout", "exit_code": 124, "wall_seconds": 0.0}
    try:
        with (
            log.open("w", encoding="utf-8") as stream,
            subprocess.Popen(
                command,
                cwd=cwd,
                env=_environment(cwd, retrieval="--retrieve-only" in command),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ) as process,
        ):
            try:
                result["exit_code"] = process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                result.update(status="timeout", exit_code=124)
    except OSError as error:
        result["error"] = str(error)
    elapsed = time.monotonic() - started
    if elapsed > seconds:
        result.update(status="timeout", exit_code=124)
    elif result["exit_code"] == 0:
        result["status"] = "passed"
    result["wall_seconds"] = elapsed
    return result


def _source_record() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=5).strip()

    files = [
        "params.yaml",
        "uv.lock",
        "dvc.yaml",
        "dvc.lock",
        "data/raw.dvc",
        f"{MODEL}.dvc",
        f"{MODEL}.provenance.json",
        "tests/test_minimum_functionality_model.py",
    ]
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src").rglob("*.py"))]
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "sha256": {name: _hash(ROOT / name) for name in files},
        "github": {
            name: os.environ.get(name)
            for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA")
        },
    }


def _data_workspace(directory: Path, data_dir: Path) -> Path:
    # Existing data commands write to project-relative gx/ and reports/. Give them a
    # private source copy and read-only inputs instead of overwriting earlier evidence.
    workspace = directory / "data-check"
    workspace.mkdir()
    shutil.copytree(ROOT / "src", workspace / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(ROOT / "params.yaml", workspace / "params.yaml")
    (workspace / "data").symlink_to(data_dir, target_is_directory=True)
    return workspace


def run_validation(
    *,
    output_dir: Path,
    data_dir: Path,
    checkpoint: Path,
    pull: bool = False,
    max_seconds: int = TOTAL_SECONDS,
) -> tuple[Path, dict]:
    """Record the rule before checks; stop at the first failure and retain every log."""
    if not 0 < max_seconds <= TOTAL_SECONDS:
        raise ValueError(f"max_seconds must be between 1 and {TOTAL_SECONDS}")
    directory = output_dir.resolve() / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    report_path = directory / "report.json"
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "run_id": directory.name,
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "acceptance": ACCEPTANCE,
        "budgets_seconds": BUDGETS,
        "total_budget_seconds": max_seconds,
        "steps": [],
        "exit_code": 1,
    }
    _write(report_path, report)

    def step(name, command, cwd=ROOT):
        remaining = max_seconds - (time.monotonic() - started)
        result = run_step(
            command, cwd=cwd, log=directory / f"{name}.log", seconds=min(BUDGETS[name], remaining)
        )
        report["steps"].append({"name": name, **result})
        _write(report_path, report)
        if result["exit_code"]:
            report.update(status=result["status"], exit_code=result["exit_code"])
            raise RuntimeError(f"{name}: {result['status']}")

    try:
        if os.name != "posix":
            raise ValueError(
                "Release validation requires POSIX process-group timeouts (Linux/macOS)"
            )
        report["source"] = _source_record()
        data_dir, checkpoint = data_dir.resolve(), checkpoint.resolve()
        if pull:
            if data_dir != (ROOT / "data").resolve() or checkpoint != (ROOT / MODEL).resolve():
                raise ValueError("--pull requires repository-local artifact paths")
            step("retrieve", [sys.executable, "-m", "src.release_validation", "--retrieve-only"])
        identities = directory / "identities.json"
        step(
            "verify",
            [
                sys.executable,
                "-m",
                "src.release_validation",
                "--verify-only",
                "--data-dir",
                str(data_dir),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(identities),
            ],
        )
        report["identities"] = json.loads(identities.read_text(encoding="utf-8"))
        workspace = _data_workspace(directory, data_dir)
        step("configure-data", [sys.executable, "-m", "src.gx_context_configuration"], workspace)
        step("data", [sys.executable, "-m", "src.validate_data"], workspace)
        # Preserve sampling, seeds, cases and thresholds; bound only execution resources.
        params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
        params["evaluation"].update(device="cpu", cpu_threads=2)
        params_path = directory / "params-cpu2.yaml"
        params_path.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
        report["executed_params_sha256"] = _hash(params_path)
        common = ["--checkpoint", str(checkpoint), "--params-path", str(params_path)]
        step(
            "minimum-functionality",
            [
                sys.executable,
                "-m",
                "src.modeling.minimum_functionality",
                *common,
                "--max-seconds",
                "300",
                "--output-path",
                str(directory / "minimum-functionality.json"),
            ],
        )
        step(
            "directional",
            [
                sys.executable,
                "-m",
                "src.modeling.directional",
                *common,
                "--max-seconds",
                "180",
                "--output-dir",
                str(directory / "directional"),
            ],
        )
        report.update(status="passed", exit_code=0)
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        yaml.YAMLError,
        subprocess.SubprocessError,
    ) as error:
        report["error"] = str(error)
        if report["status"] == "running":
            report["status"] = "failed"
    finally:
        report["wall_seconds"] = time.monotonic() - started
        if report["wall_seconds"] > max_seconds:
            report.update(status="timeout", exit_code=124)
        report["finished_utc"] = datetime.now(UTC).isoformat()
        _write(report_path, report)
    return report_path, report


def main() -> None:
    """Run locally or from the manual workflow, without training or publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "temporary/release-validation")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / MODEL)
    parser.add_argument("--pull", action="store_true", help="Retrieve only declared DVC targets")
    parser.add_argument(
        "--max-seconds",
        type=int,
        choices=range(1, TOTAL_SECONDS + 1),
        default=TOTAL_SECONDS,
        metavar="1..900",
    )
    parser.add_argument("--verify-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--retrieve-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.retrieve_only:
        retrieve_artifacts()
        return
    if args.verify_only:
        _write(args.output_dir, verify_artifacts(args.data_dir, args.checkpoint))
        return
    path, report = run_validation(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        pull=args.pull,
        max_seconds=args.max_seconds,
    )
    print(f"{report['status']}; report: {path}")
    raise SystemExit(report["exit_code"])


def retrieve_artifacts() -> None:
    """Pass optional HTTP credentials in memory; never persist remote configuration."""
    from dvc.repo import Repo  # pylint: disable=import-outside-toplevel

    user, password = os.environ.get("DVC_HTTP_USER"), os.environ.get("DVC_HTTP_PASSWORD")
    if bool(user) != bool(password):
        raise ValueError("Set both DVC_HTTP_USER and DVC_HTTP_PASSWORD, or neither")
    # Select the stage once: repeated granular outputs of one stage can be collapsed
    # by DVC. Guard its output list so retrieval cannot expand to unrelated artifacts.
    pipeline = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    if set(pipeline["stages"]["split"]["outs"]) != {
        f"data/processed/{block}.parquet" for block in BLOCKS
    }:
        raise ValueError("The split stage must declare exactly the four release data outputs")
    remote_config = {"auth": "basic", "user": user, "password": password} if user else {}
    try:
        with Repo(str(ROOT), remote="origin", remote_config=remote_config) as repo:
            repo.pull(targets=list(DVC_TARGETS), remote="origin", jobs=2)
    except Exception:  # noqa: BLE001 - do not expose credential-bearing third-party errors
        # Third-party error details may include connection configuration.
        raise RuntimeError("DVC retrieval failed; check authorized remote access") from None


if __name__ == "__main__":
    main()
