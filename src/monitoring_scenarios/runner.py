"""A process deadline, retained evidence and owned cleanup for each declared scenario."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from src.monitoring_scenarios.service import ROOT, cleanup, timestamp

CATALOGUE = ROOT / "monitoring/scenarios/scenarios.json"
SERVICE_SCENARIOS = {"normal", "service-unavailable", "slow-request"}


def declarations():
    """Read the versioned schedule and reject durations outside the 30-minute cap."""
    value = json.loads(CATALOGUE.read_text())
    if value["schema_version"] != 1:
        raise ValueError("Unsupported scenario schema")
    for row in value["scenarios"].values():
        duration = row["max_seconds"]
        if type(duration) is not int or not 60 <= duration <= 1800:
            raise ValueError("Scenario max_seconds must be an integer in 60..1800")
    return value["scenarios"]


def source_identity():
    """Bind observations to exact source/config bytes, including uncommitted edits."""
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        CATALOGUE,
        ROOT / "monitoring/service/rules/alerts.yml",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
        ROOT / "params.yaml",
        ROOT / "tests/replay_fixture.py",
        ROOT / "monitoring/quality/exporter.py",
    ]
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
        ).strip(),
        "dirty": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).splitlines(),
        "sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }


def execute(directory, scenario, engine, target_id, fixture_image):
    """Run one child body; the public supervisor owns its absolute process deadline."""
    declaration = declarations()[scenario]
    if scenario in SERVICE_SCENARIOS:
        from src.monitoring_scenarios.service import (  # pylint: disable=import-outside-toplevel
            docker_service,
            fixture_service,
        )

        result = (
            docker_service(directory, scenario, declaration, target_id, fixture_image)
            if engine == "docker"
            else fixture_service(directory, scenario, declaration)
        )
    else:
        from src.monitoring_scenarios.experiments import (  # pylint: disable=import-outside-toplevel
            delayed_observation,
            shifted_demand,
        )

        result = (
            shifted_demand(directory, declaration)
            if scenario == "shifted-demand"
            else delayed_observation(directory, declaration)
        )
    result.setdefault("alerts", {"firing": None, "clearing": None, "delivery": "not_measured"})
    result.update(
        {
            "schema_version": 1,
            "scenario": scenario,
            "clock": declaration["clock"],
            "expected_signal": declaration["expected_signal"],
            "completed_utc": timestamp(),
        }
    )
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["procedure_passed"] else 1


def _stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)


def supervise(
    directory, scenario, engine="fixture", target_id=None, fixture_image=None, max_seconds=None
):
    """Refuse overwrite/remote targets and retain failures, even after child death."""
    declaration = declarations()[scenario]
    duration = declaration["max_seconds"] if max_seconds is None else max_seconds
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 60 <= duration <= declaration["max_seconds"]
    ):
        raise ValueError(
            "max_seconds may only reduce the declared budget, with a 60-second minimum"
        )
    if engine not in ("fixture", "docker", "alibi"):
        raise ValueError("Unsupported engine")
    if (scenario == "shifted-demand") != (engine == "alibi"):
        raise ValueError("shifted-demand requires the isolated Alibi engine")
    if engine == "docker" and (
        scenario not in SERVICE_SCENARIOS
        or not fixture_image
        or not re.fullmatch(r"endesa-84-[a-z0-9-]{1,45}", target_id or "")
    ):
        raise ValueError(
            "Docker requires a service scenario, cached image ID and explicit target ID"
        )
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    owner = f"endesa-84-{uuid.uuid4().hex[:16]}" if target_id is None else target_id
    started = time.monotonic()
    request = {
        "scenario": scenario,
        "engine": engine,
        "target_id": owner,
        "fixture_image": fixture_image,
        "max_seconds": duration,
        "started_utc": timestamp(),
        "source": source_identity(),
    }
    (directory / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    command = [
        sys.executable,
        "-m",
        "src.monitoring_scenarios",
        "_execute",
        "--directory",
        str(directory),
        "--scenario",
        scenario,
        "--engine",
        engine,
        "--target-id",
        owner,
    ]
    if fixture_image is not None:
        command += ["--fixture-image", fixture_image]
    status, process, error = "failed", None, None

    def cancel(_signum, _frame):
        raise KeyboardInterrupt("Scenario cancelled")

    original_signal = signal.signal(signal.SIGTERM, cancel)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    }
    with (directory / "execution.log").open("w") as stream:
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            code = process.wait(timeout=duration - 30)
            status = "passed" if code == 0 else "failed"
        except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if process is not None:
                _stop_process(process)
            try:
                stopped = cleanup(directory)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                stopped, status = [], "failed"
                error = f"Cleanup failed: {type(exc).__name__}: {exc}"
            finally:
                signal.signal(signal.SIGTERM, original_signal)
    result_path = directory / "result.json"
    if status == "passed" and not result_path.exists():
        status, error = "failed", "Child exited without result evidence"
    if error is not None:
        status = "failed"
    elapsed = time.monotonic() - started
    if elapsed > duration:
        status, error = "failed", "Scenario including cleanup exceeded the declared wall budget"
    summary = {
        "scenario": scenario,
        "status": status,
        "error": error,
        "elapsed_wall_seconds": elapsed,
        "max_seconds": duration,
        "child_exit_code": None if process is None else process.returncode,
        "cleanup": stopped,
        "completed_utc": timestamp(),
        "result": str(result_path) if result_path.exists() else None,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    """Default to an owned local fixture; external service execution is unsupported."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "run", "_execute"))
    parser.add_argument("--scenario", choices=tuple(declarations()), default="normal")
    parser.add_argument("--engine", choices=("fixture", "docker", "alibi"), default="fixture")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--target-id")
    parser.add_argument("--fixture-image")
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)
    if args.action == "list":
        print(json.dumps(declarations(), indent=2))
        return 0
    if args.directory is None:
        parser.error("--directory is required")
    if args.action == "_execute":
        return execute(
            args.directory, args.scenario, args.engine, args.target_id, args.fixture_image
        )
    try:
        summary = supervise(
            args.directory,
            args.scenario,
            args.engine,
            args.target_id,
            args.fixture_image,
            args.max_seconds,
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Scenario refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1
