"""Declared synthetic demand checks and a bounded runner for real model tests."""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import typer
import yaml

from src.config import PROJ_ROOT, REPORTS_DIR
from src.modeling.behaviour_source import describe_model
from src.modeling.evaluate import EvaluationConfig
from src.splits import PARAMS_PATH

CONTEXT_LENGTH = 512
HORIZON = 48
MAX_SECONDS = 300
LEVELS = (100.0, 1000.0, 10000.0)
app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass(frozen=True)
class DemandCase:
    """A known continuation and its error limits, set before any model inference."""

    case_id: str
    level: float
    amplitude: float
    mae_limit: float
    max_error_limit: float

    def series(self) -> tuple[np.ndarray, np.ndarray]:
        """Continue the same 48-step cycle across the context/target boundary."""
        cycle = self.level + self.amplitude * np.sin(2 * np.pi * np.arange(HORIZON) / HORIZON)
        values = np.resize(cycle, CONTEXT_LENGTH + HORIZON)
        return values[:CONTEXT_LENGTH].copy(), values[CONTEXT_LENGTH:].copy()


CASES = (
    *(
        DemandCase(f"constant-level-{level:g}", level, 0.0, 0.05 * level, 0.10 * level)
        for level in LEVELS
    ),
    DemandCase("zero-demand", 0.0, 0.0, 0.01, 0.02),
    *(
        DemandCase(
            f"pattern-level-{level:g}-amplitude-{level * ratio:g}",
            level,
            level * ratio,
            0.20 * level * ratio,
            0.50 * level * ratio,
        )
        for level in LEVELS
        for ratio in (0.10, 0.50)
    ),
)


def score_case(case: DemandCase, prediction: np.ndarray) -> dict:
    """Record both error limits; invalid predictions count as failures."""
    _, target = case.series()
    result = {**asdict(case), "target": target.tolist()}
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (HORIZON,) or not np.isfinite(prediction).all():
        return {**result, "status": "invalid", "error": "Expected 48 finite point forecasts"}
    errors = np.abs(prediction - target)
    mae, max_error = float(errors.mean()), float(errors.max())
    return {
        **result,
        "prediction": prediction.tolist(),
        "mae": mae,
        "max_absolute_error": max_error,
        "status": "passed"
        if mae <= case.mae_limit and max_error <= case.max_error_limit
        else "failed",
    }


def write_report(path: Path, report: dict) -> None:
    """Replace a report atomically so a timeout preserves the last complete write."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _provenance() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=PROJ_ROOT, text=True).strip()

    dependencies = {}
    for name in ("chronos-forecasting", "torch", "transformers", "numpy", "pytest"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "uv_lock_sha256": hashlib.sha256((PROJ_ROOT / "uv.lock").read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": dependencies,
    }


def run_model_tests(
    *,
    output_path: Path,
    params_path: Path = PARAMS_PATH,
    max_seconds: int = MAX_SECONDS,
    checkpoint: Path | None = None,
) -> int:
    """Run every real-model pytest case in one process, including a hard timeout."""
    if not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError("The model-test budget must be between 1 and 300 seconds")
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    if (
        params["forecast"]["context_length"] != CONTEXT_LENGTH
        or params["forecast"]["horizon"] != HORIZON
    ):
        raise ValueError("Minimum functionality checks require 512 context values and 48 targets")
    settings = replace(
        EvaluationConfig.from_params(params_path),
        device="cpu",
        cpu_threads=2,
        batch_size=1,
        max_seconds=max_seconds,
    )
    protocol = {
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "series": "level + amplitude * sin(2*pi*(t mod 48)/48), t=0..559",
        "target_start_phase": CONTEXT_LENGTH % HORIZON,
        "metrics": "mean and maximum absolute error over all 48 point forecasts",
        "cases": [asdict(case) for case in CASES],
    }
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "provenance": _provenance(),
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "settings": asdict(settings),
        "model": describe_model(settings, checkpoint),
        "dtype": "float32",
        "point_statistic": "median of sampled trajectories",
        "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
        "seed_scope": "set once before loading; cases run in the declared order, batch size 1",
        "cases": [{**asdict(case), "status": "not_run"} for case in CASES],
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite evidence from an earlier invocation.
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "--maxfail=0",
        "--run-model-tests",
        f"--model-test-report={output_path}",
        "tests/test_minimum_functionality_model.py",
    ]
    start = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=PROJ_ROOT, timeout=max_seconds, check=False)
        exit_code = completed.returncode
        report = json.loads(output_path.read_text(encoding="utf-8"))
        report["pytest_exit_code"] = exit_code
        if report["status"] != "completed":
            report["status"] = "incomplete"
            exit_code = exit_code or 2
        elif any(row["status"] != "passed" for row in report["cases"]):
            exit_code = exit_code or 1
    except subprocess.TimeoutExpired:
        report = json.loads(output_path.read_text(encoding="utf-8"))
        report["status"] = "timeout"
        exit_code = 124
    report["runtime_seconds"] = time.perf_counter() - start
    report["within_budget"] = report["runtime_seconds"] <= max_seconds
    if not report["within_budget"]:
        report["status"] = "timeout"
        exit_code = 124
    report["exit_code"] = exit_code
    write_report(output_path, report)
    return exit_code


@app.command()
def main(
    output_path: Path | None = None,
    params_path: Path = PARAMS_PATH,
    max_seconds: int = typer.Option(MAX_SECONDS, min=1, max=MAX_SECONDS),
    checkpoint: Path | None = None,
) -> None:
    """Check the real checkpoint once; save every result and return nonzero on failure."""
    output_path = output_path or REPORTS_DIR / "minimum-functionality" / f"{uuid4().hex}.json"
    exit_code = run_model_tests(
        output_path=output_path,
        params_path=params_path,
        max_seconds=max_seconds,
        checkpoint=checkpoint,
    )
    typer.echo(f"Model-test report: {output_path.resolve()}")
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
