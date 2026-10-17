"""Run a bounded paired comparison with one load of the pinned Chronos checkpoint."""

from dataclasses import asdict, replace
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time
from uuid import uuid4

import numpy as np
import typer

from src.config import PROCESSED_DATA_DIR, PROJ_ROOT, REPORTS_DIR
from src.invariance import (
    CONTEXT_LENGTH,
    HORIZON,
    NOISE_SEED,
    TRANSFORMATIONS,
    compare_pair,
    history_record,
    stored_histories,
    synthetic_histories,
)
from src.modeling.evaluate import EvaluationConfig, chronos_forecast
from src.splits import PARAMS_PATH, SplitConfig

app = typer.Typer(pretty_exceptions_show_locals=False)


def _write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _versions(params_path: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=PROJ_ROOT, text=True).strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "file_sha256": {
            name: hashlib.sha256(
                (params_path if name == "params.yaml" else PROJ_ROOT / name).read_bytes()
            ).hexdigest()
            for name in ("params.yaml", "uv.lock", "dvc.lock", "data/raw.dvc")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: version(name)
            for name in ("chronos-forecasting", "torch", "numpy", "transformers")
        },
    }


def run_checks(
    *,
    params_path: Path = PARAMS_PATH,
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = REPORTS_DIR / "invariance",
    max_seconds: int = 300,
    synthetic_only: bool = False,
) -> dict:
    """Save the protocol before inference and retain every completed pair, including failures."""
    start = time.monotonic()
    cfg = SplitConfig.from_params(params_path)
    if (cfg.context_length, cfg.horizon) != (CONTEXT_LENGTH, HORIZON):
        raise ValueError("Invariance checks require the agreed 512-value / 48-step protocol")
    settings = replace(
        EvaluationConfig.from_params(params_path), batch_size=1, max_seconds=max_seconds
    )
    histories = synthetic_histories()
    if not synthetic_only:
        histories += stored_histories(processed_dir, cfg, PROJ_ROOT / "dvc.lock")
    pairs = [(history, change) for history in histories for change in TRANSFORMATIONS]
    path = output_dir / uuid4().hex / "report.json"
    path.parent.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol_version": 1,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "prepared",
        "settings": asdict(settings),
        "split_config": asdict(cfg) | {"states": list(cfg.states)},
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
        "point_statistic": "median",
        "dtype": "float32",
        "reset_seed_before_each_history": True,
        "noise_seed": NOISE_SEED,
        "transformations": [asdict(change) | {"name": change.name} for change in TRANSFORMATIONS],
        "histories": [history_record(history) for history in histories],
        "versions": _versions(params_path),
        "data_inputs": {
            f"data/processed/{block}.parquet": hashlib.md5(
                (processed_dir / f"{block}.parquet").read_bytes()
            ).hexdigest()
            for block in (() if synthetic_only else ("train", "validation"))
        },
        "results": [],
    }
    _write_report(path, report)
    try:
        predictions, timing = chronos_forecast(
            np.stack(
                [
                    values
                    for history, change in pairs
                    for values in (history.values, change.apply(history))
                ]
            ),
            HORIZON,
            settings,
            start + max_seconds,
            reset_seed_per_context=True,
        )
        if time.monotonic() - start >= max_seconds:
            raise TimeoutError("Invariance checks exceeded max_seconds")
        if predictions.shape != (2 * len(pairs), HORIZON) or not np.isfinite(predictions).all():
            raise ValueError("Expected a finite 48-step forecast for every paired history")
        report["results"] = [
            compare_pair(history, change, predictions[2 * index], predictions[2 * index + 1])
            for index, (history, change) in enumerate(pairs)
        ]
        report.update(status="complete", timing=timing)
        report["counterexamples"] = [
            result["case_id"] for result in report["results"] if not result["within_tolerance"]
        ]
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["observed_seconds"] = time.monotonic() - start
        _write_report(path, report)
        typer.echo(f"Paired forecast evidence: {path}", err=True)
    return report


@app.command()
def main(
    params_path: Path = PARAMS_PATH,
    output_dir: Path = REPORTS_DIR / "invariance",
    max_seconds: int = 300,
    synthetic_only: bool = False,
) -> None:
    """Run declared checks; exit 1 after saving all results if any threshold is exceeded."""
    report = run_checks(
        params_path=params_path,
        output_dir=output_dir,
        max_seconds=max_seconds,
        synthetic_only=synthetic_only,
    )
    typer.echo(f"{len(report['results'])} pairs; {len(report['counterexamples'])} counterexamples")
    if report["counterexamples"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
