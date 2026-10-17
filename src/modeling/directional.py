"""Bounded directional checks of synthetic demand histories for issue #56."""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Annotated
from uuid import uuid4

import numpy as np
import typer

from src.config import PROJ_ROOT, REPORTS_DIR
from src.modeling.evaluate import EvaluationConfig, chronos_forecast
from src.splits import PARAMS_PATH

CONTEXT_LENGTH = 512
HORIZON = 48
REFERENCE_LEVEL = 1000.0
MIN_RESPONSE_FRACTION = 0.1
CONTROL_TOLERANCE = 0.01
MAX_SECONDS = 180
app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass(frozen=True)
class DirectionalCase:
    """A sustained level change that preserves the recent daily pattern."""

    direction: str
    amplitude: float
    recent: int

    def __post_init__(self):
        if self.direction not in ("increase", "decrease"):
            raise ValueError("Choose increase or decrease")
        if not math.isfinite(self.amplitude) or not 0 < self.amplitude <= 0.5:
            raise ValueError("Amplitude must be a fraction in (0, 0.5]")
        if self.recent not in (48, 96, 192):
            raise ValueError("Recent observations must cover 1, 2 or 4 full daily cycles")

    @property
    def sign(self) -> int:
        """Return the expected sign of the forecast-level change."""
        return 1 if self.direction == "increase" else -1

    @property
    def case_id(self) -> str:
        """Give pytest and reports the same readable case identifier."""
        return f"{self.direction}-{self.amplitude:.0%}-last-{self.recent}"


CASES = tuple(
    DirectionalCase(direction, amplitude, recent)
    for direction in ("increase", "decrease")
    for amplitude in (0.05, 0.15)
    for recent in (48, 96, 192)
)


def paired_histories(case: DirectionalCase) -> tuple[np.ndarray, np.ndarray]:
    """Return a positive 512-value daily pattern and its controlled level change."""
    phase = 2 * np.pi * np.arange(CONTEXT_LENGTH) / HORIZON
    original = (REFERENCE_LEVEL + 150 * np.sin(phase) + 50 * np.cos(2 * phase)).astype(np.float32)
    changed = original.copy()
    changed[-case.recent :] += case.sign * case.amplitude * REFERENCE_LEVEL
    return original, changed


def compare_forecasts(case: DirectionalCase, original: np.ndarray, changed: np.ndarray) -> dict:
    """Check the mean of the 48-step sample median against a declared effect margin."""
    original, changed = np.asarray(original), np.asarray(changed)
    if any(
        values.shape != (HORIZON,) or not np.isfinite(values).all()
        for values in (original, changed)
    ):
        raise ValueError("Compare two finite 48-step median forecasts")
    delta = float(changed.astype(np.float64).mean() - original.astype(np.float64).mean())
    required_change = case.amplitude * REFERENCE_LEVEL * MIN_RESPONSE_FRACTION
    return {
        "case_id": case.case_id,
        "mean_forecast_change": delta,
        "signed_change": case.sign * delta,
        "minimum_signed_change": required_change,
        "passed": bool(case.sign * delta >= required_change),
        "median_forecast": changed.tolist(),
    }


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _plan(settings: EvaluationConfig) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=PROJ_ROOT, text=True).strip()

    packages = {}
    for name in ("chronos-forecasting", "torch", "transformers", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    original, _ = paired_histories(CASES[0])
    return {
        "started_utc": datetime.now(UTC).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "uv_lock_sha256": hashlib.sha256((PROJ_ROOT / "uv.lock").read_bytes()).hexdigest(),
        "settings": asdict(settings),
        "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
        "reset_seed_per_context": True,
        "dtype": "float32",
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "reference_level": REFERENCE_LEVEL,
        "summary": "arithmetic mean of the 48-step sample median forecast",
        "minimum_response_fraction": MIN_RESPONSE_FRACTION,
        "control_max_absolute_difference": CONTROL_TOLERANCE,
        "hypothesis": "A level shift lasting full daily cycles persists into the next cycle.",
        "original_history": original.tolist(),
        "cases": [asdict(case) for case in CASES],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "packages": packages,
        },
    }


def execute_plan(plan: dict) -> dict:
    """Run one checkpoint load for the reference, unchanged control and all cases."""
    settings = EvaluationConfig(**plan["settings"])
    cases = [DirectionalCase(**values) for values in plan["cases"]]
    original = np.asarray(plan["original_history"], dtype=np.float32)
    histories = [original, original.copy()]
    for case in cases:
        changed = original.copy()
        changed[-case.recent :] += case.sign * case.amplitude * plan["reference_level"]
        histories.append(changed)
    contexts = np.stack(histories)
    predictions, timing = chronos_forecast(
        contexts,
        HORIZON,
        settings,
        time.monotonic() + settings.max_seconds,
        reset_seed_per_context=True,
    )
    if predictions.shape != (len(histories), HORIZON) or not np.isfinite(predictions).all():
        raise ValueError("Expected a finite 48-step forecast for every history")
    control_difference = float(np.max(np.abs(predictions[1] - predictions[0])))
    results = [
        compare_forecasts(case, predictions[0], prediction)
        for case, prediction in zip(cases, predictions[2:], strict=True)
    ]
    control = {
        "case_id": "unchanged-control",
        "max_absolute_difference": control_difference,
        "tolerance": CONTROL_TOLERANCE,
        "passed": bool(control_difference <= CONTROL_TOLERANCE),
        "median_forecast": predictions[1].tolist(),
    }
    return {
        "status": "completed",
        "success": control["passed"] and all(result["passed"] for result in results),
        "plan": plan,
        "timing": timing,
        "history_sha256": [
            hashlib.sha256(values.astype("<f4").tobytes()).hexdigest() for values in contexts
        ],
        "original_median_forecast": predictions[0].tolist(),
        "original_forecast_mean": float(predictions[0].astype(np.float64).mean()),
        "control": control,
        "cases": results,
    }


def run_bounded(
    *,
    params_path: Path = PARAMS_PATH,
    output_dir: Path = REPORTS_DIR / "directional",
    max_seconds: int = MAX_SECONDS,
) -> tuple[Path, dict]:
    """Save the protocol before inference and enforce a wall-clock limit on the worker."""
    started = time.monotonic()
    settings = replace(
        EvaluationConfig.from_params(params_path),
        batch_size=1,
        cpu_threads=2,
        max_seconds=max_seconds,
    )
    plan = _plan(settings)
    directory = output_dir.resolve() / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "plan.json", plan)
    report_path = directory / "report.json"
    remaining = max_seconds - (time.monotonic() - started)
    try:
        if remaining <= 0:
            raise subprocess.TimeoutExpired("directional", max_seconds)
        subprocess.run(
            [sys.executable, "-m", "src.modeling.directional", "--worker-dir", str(directory)],
            cwd=PROJ_ROOT,
            check=False,
            timeout=remaining,
        )
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.exists()
            else {"status": "error", "success": False, "error": "Worker exited without a report"}
        )
    except subprocess.TimeoutExpired:
        report = {"status": "timeout", "success": False, "error": "Runtime cap exceeded"}
    report.update(
        plan=plan,
        wall_seconds=time.monotonic() - started,
        finished_utc=datetime.now(UTC).isoformat(),
    )
    _write_json(report_path, report)
    return report_path, report


@app.command()
def main(
    params_path: Path = PARAMS_PATH,
    output_dir: Path = REPORTS_DIR / "directional",
    max_seconds: int = MAX_SECONDS,
    worker_dir: Annotated[Path | None, typer.Option(hidden=True)] = None,
) -> None:
    """Check all declared cases; failures remain in the report and return exit code 1."""
    if worker_dir is not None:
        plan = json.loads((worker_dir / "plan.json").read_text(encoding="utf-8"))
        try:
            report = execute_plan(plan)
        except (ValueError, RuntimeError, OSError) as error:
            report = {"status": "error", "success": False, "error": str(error)}
        _write_json(worker_dir / "report.json", report)
        raise typer.Exit(0 if report["success"] else 1)
    path, report = run_bounded(
        params_path=params_path, output_dir=output_dir, max_seconds=max_seconds
    )
    for result in [report.get("control"), *report.get("cases", [])]:
        if result is not None:
            typer.echo(f"{result['case_id']}: {'PASS' if result['passed'] else 'FAIL'}")
    typer.echo(f"{report['status']}; {report['wall_seconds']:.2f}s; report: {path}")
    raise typer.Exit(0 if report["success"] else 1)


if __name__ == "__main__":
    app()
