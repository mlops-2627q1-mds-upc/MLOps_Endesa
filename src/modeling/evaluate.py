"""Record comparable seasonal-naive and pinned Chronos validation runs."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
from uuid import uuid4

from loguru import logger
import numpy as np
import typer
import yaml

from src.config import PROCESSED_DATA_DIR, PROJ_ROOT, REPORTS_DIR
from src.evaluation import (
    BASELINE_LAGS,
    MASE_LAG,
    ValidationData,
    sample_median,
    seasonal_forecast,
)
from src.splits import PARAMS_PATH, SplitConfig
from src.tracking import tracked_run

METHODS = ("seasonal_lag48", "seasonal_lag336", "chronos")
app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass(frozen=True)
class EvaluationConfig:
    model_id: str
    model_revision: str
    seed: int
    num_samples: int
    batch_size: int
    device: str
    cpu_threads: int
    max_seconds: int

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("Pin the model to a full Hub commit revision")
        if min(self.num_samples, self.batch_size, self.cpu_threads, self.max_seconds) <= 0:
            raise ValueError("Sample count, batch size, threads and time budget must be positive")
        if self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("Choose an explicit cpu, mps or cuda device")

    @classmethod
    def from_params(cls, path: Path) -> "EvaluationConfig":
        return cls(**yaml.safe_load(path.read_text())["evaluation"])


def data_version() -> str:
    """Require the on-disk DVC artifacts to match their committed references."""
    status = json.loads(
        subprocess.check_output(["dvc", "status", "--json"], cwd=PROJ_ROOT, text=True)
    )
    if status:
        raise ValueError("DVC inputs have changed or are missing; restore them before evaluation")
    raw = yaml.safe_load((PROJ_ROOT / "data/raw.dvc").read_text())["outs"][0]["md5"]
    split = hashlib.sha256((PROJ_ROOT / "dvc.lock").read_bytes()).hexdigest()
    return f"raw:{raw};split-lock-sha256:{split}"


def chronos_forecast(
    contexts: np.ndarray, horizon: int, cfg: EvaluationConfig, deadline: float
) -> tuple[np.ndarray, dict[str, float]]:
    # Optional imports keep baseline reproduction and CI free of model dependencies.
    try:
        from chronos import ChronosPipeline
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Install forecasting dependencies with uv sync --extra forecast"
        ) from error

    torch.set_num_threads(cfg.cpu_threads)
    torch.manual_seed(cfg.seed)
    start = time.perf_counter()
    pipeline = ChronosPipeline.from_pretrained(
        cfg.model_id,
        revision=cfg.model_revision,
        device_map=cfg.device,
        torch_dtype=torch.float32,
        token=False,
    )
    pipeline.model.model.eval()
    load_seconds = time.perf_counter() - start
    predictions = []
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(contexts), cfg.batch_size):
            if time.monotonic() >= deadline:
                raise TimeoutError("Evaluation exceeded max_seconds; no complete result recorded")
            batch = torch.as_tensor(
                contexts[offset : offset + cfg.batch_size].copy(), dtype=torch.float32
            )
            samples = (
                pipeline.predict(
                    batch,
                    prediction_length=horizon,
                    num_samples=cfg.num_samples,
                    temperature=1.0,
                    top_k=50,
                    top_p=1.0,
                )
                .cpu()
                .numpy()
            )
            predictions.append(sample_median(samples, len(batch), horizon))
            typer.echo(
                f"Chronos: {min(offset + cfg.batch_size, len(contexts))}/{len(contexts)} origins",
                err=True,
            )
    return np.concatenate(predictions), {
        "model_load_seconds": load_seconds,
        "inference_seconds": time.perf_counter() - start,
    }


def evaluate(
    *,
    params_path: Path = PARAMS_PATH,
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = REPORTS_DIR / "evaluation",
    method: str = "all",
    tracking_uri: str | None = None,
) -> dict:
    """Evaluate the full validation block and give each invocation its own output directory."""
    if method not in (*METHODS, "all"):
        raise ValueError(f"Unknown method {method!r}; choose all or one of {METHODS}")
    if processed_dir.resolve() != (PROJ_ROOT / "data/processed").resolve():
        raise ValueError("Use this checkout's DVC-tracked data/processed directory")
    cfg = SplitConfig.from_params(params_path)
    settings = EvaluationConfig.from_params(params_path)
    if cfg.context_length < max(BASELINE_LAGS):
        raise ValueError("The agreed seasonal baselines need at least 336 context observations")
    version = data_version()
    data = ValidationData.load(processed_dir, cfg)
    group_id = uuid4().hex
    directory = output_dir / group_id
    directory.mkdir(parents=True, exist_ok=False)
    params = yaml.safe_load(params_path.read_text())
    target_hash = hashlib.sha256(
        data.predictions_frame(data.targets)
        .drop(columns="prediction")
        .to_csv(index=False, lineterminator="\n")
        .encode()
    ).hexdigest()
    common = {
        "evaluation_group": group_id,
        "block": "validation",
        "protocol": params,
        "mase_lag": MASE_LAG,
        "point_statistic": "median",
        "dtype": "float32",
        "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
        "targets_sha256": target_hash,
        "origin_order": "state order in params.yaml, then ascending origin",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
        "holiday_stratification": "unavailable: civil timestamps remain unverified in #6",
    }
    results = {}
    methods = METHODS if method == "all" else (method,)
    deadline = time.monotonic() + settings.max_seconds
    for name in methods:
        lag = None if name == "chronos" else int(name.removeprefix("seasonal_lag"))
        revision = settings.model_revision if lag is None else f"seasonal-naive-v1:lag-{lag}"
        config = {
            **common,
            "method": name,
            "seasonal_lag": lag,
            "forecast_device": settings.device if lag is None else "cpu",
        }
        output = directory / name
        output.mkdir()
        with tracked_run(
            name=f"validation-{name}",
            config=config,
            data_version=version,
            model_revision=revision,
            tracking_uri=tracking_uri,
        ) as run:
            start = time.perf_counter()
            if lag is None:
                predictions, timing = chronos_forecast(
                    data.contexts, cfg.horizon, settings, deadline
                )
            else:
                predictions = seasonal_forecast(data.contexts, cfg.horizon, lag)
                timing = {
                    "inference_seconds": time.perf_counter() - start,
                    "model_load_seconds": 0.0,
                }
            if time.monotonic() >= deadline:
                raise TimeoutError("Evaluation exceeded max_seconds; no complete result recorded")
            metrics = {
                **data.metrics(predictions),
                **timing,
                "inference_seconds_per_origin": timing["inference_seconds"] / len(data.origins),
                "forecast_rows": float(data.targets.size),
            }
            predictions_path = output / "predictions.parquet"
            data.predictions_frame(predictions).to_parquet(predictions_path, index=False)
            metrics_path = output / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
            run.log_metrics(metrics)
            run.log_artifact(predictions_path)
            run.log_artifact(metrics_path)
            run.client.log_dict(run.run_id, data.scales, "mase_scales.json")
            run.client.log_dict(run.run_id, params, "params.json")
            run.log_artifact(PROJ_ROOT / "data/raw.dvc")
            run.log_artifact(PROJ_ROOT / "dvc.lock")
        results[name] = {"run_id": run.run_id, "metrics": metrics}
        logger.info(f"{name}: mean MASE={metrics['mean.mase']:.6f}; run={run.run_id}")
    baselines = [name for name in results if name != "chronos"]
    primary = (
        min(baselines, key=lambda name: (results[name]["metrics"]["mean.mase"], name))
        if len(baselines) == len(BASELINE_LAGS)
        else None
    )
    comparison = {
        "evaluation_group": group_id,
        "block": "validation",
        "data_version": version,
        "targets_sha256": target_hash,
        "settings": asdict(settings),
        "primary_baseline_on_validation": primary,
        "results": results,
        "holiday_stratification": common["holiday_stratification"],
        "output_dir": str(directory),
    }
    (directory / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n"
    )
    return comparison


@app.command()
def main(
    method: str = "all",
    params_path: Path = PARAMS_PATH,
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = REPORTS_DIR / "evaluation",
    tracking_uri: str | None = None,
) -> None:
    """Evaluate all validation origins; all creates two baseline runs and one Chronos run."""
    result = evaluate(
        method=method,
        params_path=params_path,
        processed_dir=processed_dir,
        output_dir=output_dir,
        tracking_uri=tracking_uri,
    )
    typer.echo(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    app()
