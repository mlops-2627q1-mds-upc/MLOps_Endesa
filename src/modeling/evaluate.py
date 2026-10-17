"""Record comparable seasonal-naive and Chronos runs on a held-out block."""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, replace
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
from src.splits import HELD_OUT_BLOCKS, PARAMS_PATH, SplitConfig
from src.tracking import tracked_run

METHODS = ("seasonal_lag48", "seasonal_lag336", "chronos")
# The raw data and the split stage that produces the evaluated blocks.
DATA_DVC_TARGETS = ("data/raw.dvc", "dvc.yaml")
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
    """Require the on-disk DVC data to match its committed references.

    Only the raw data and the split stage are checked, so a DVC-tracked model that has
    not been pulled does not block an evaluation.
    """
    status = json.loads(
        subprocess.check_output(
            ["dvc", "status", "--json", *DATA_DVC_TARGETS], cwd=PROJ_ROOT, text=True
        )
    )
    if status:
        raise ValueError("DVC inputs have changed or are missing; restore them before evaluation")
    raw = yaml.safe_load((PROJ_ROOT / "data/raw.dvc").read_text())["outs"][0]["md5"]
    split = hashlib.sha256((PROJ_ROOT / "dvc.lock").read_bytes()).hexdigest()
    return f"raw:{raw};split-lock-sha256:{split}"


def _cuda_hardware() -> dict:
    """Record the GPU and the CUDA build that produced GPU forecasts."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; install the CUDA build of PyTorch")
    return {
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }


def checkpoint_sha256(checkpoint: Path) -> str:
    """Identify a local checkpoint by the hash of its weights."""
    digest = hashlib.sha256()
    with (checkpoint / "model.safetensors").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chronos_forecast(
    contexts: np.ndarray,
    horizon: int,
    cfg: EvaluationConfig,
    deadline: float,
    measure: Callable[[], AbstractContextManager[None]] = nullcontext,
    checkpoint: Path | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Forecast every context; `measure` wraps the forecast loop but not model loading.

    A local `checkpoint` replaces the pinned Hub model, for example a fine-tuned one.
    """
    # Optional imports keep baseline reproduction and CI free of model dependencies.
    try:
        from chronos import ChronosPipeline  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError(
            "Install forecasting dependencies with uv sync --extra forecast"
        ) from error

    torch.set_num_threads(cfg.cpu_threads)
    torch.manual_seed(cfg.seed)
    start = time.perf_counter()
    source = {"pretrained_model_name_or_path": str(checkpoint)}
    if checkpoint is None:
        source = {"pretrained_model_name_or_path": cfg.model_id, "revision": cfg.model_revision}
    pipeline = ChronosPipeline.from_pretrained(
        **source, device_map=cfg.device, torch_dtype=torch.float32, token=False
    )
    pipeline.model.model.eval()
    load_seconds = time.perf_counter() - start
    predictions = []
    with measure(), torch.inference_mode():
        start = time.perf_counter()
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
        inference_seconds = time.perf_counter() - start
    return np.concatenate(predictions), {
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
    }


def evaluate(
    *,
    params_path: Path = PARAMS_PATH,
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = REPORTS_DIR / "evaluation",
    method: str = "all",
    tracking_uri: str | None = None,
    checkpoint: Path | None = None,
    device: str | None = None,
    block: str = "validation",
) -> dict:
    """Evaluate a full held-out block and give each invocation its own output directory.

    `checkpoint` scores a local Chronos checkpoint instead of the pinned Hub model, and
    `device` overrides the configured device for this invocation only. The test block is
    for the final comparison only; select models on validation.
    """
    if method not in (*METHODS, "all"):
        raise ValueError(f"Unknown method {method!r}; choose all or one of {METHODS}")
    if block not in HELD_OUT_BLOCKS:
        raise ValueError(f"Choose a held-out block: {HELD_OUT_BLOCKS}")
    if checkpoint is not None and method != "chronos":
        raise ValueError("A checkpoint is scored with method chronos only")
    if processed_dir.resolve() != (PROJ_ROOT / "data/processed").resolve():
        raise ValueError("Use this checkout's DVC-tracked data/processed directory")
    cfg = SplitConfig.from_params(params_path)
    settings = EvaluationConfig.from_params(params_path)
    if device is not None:
        settings = replace(settings, device=device)
    checkpoint_record = None
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        if not (checkpoint / "model.safetensors").is_file():
            raise ValueError(f"No model.safetensors in checkpoint {checkpoint}")
        checkpoint_record = {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256(checkpoint),
            # Training saves checkpoints under <output_dir>/<training run ID>/.
            "training_run_id": checkpoint.parent.name,
        }
    if cfg.context_length < max(BASELINE_LAGS):
        raise ValueError("The agreed seasonal baselines need at least 336 context observations")
    version = data_version()
    data = ValidationData.load(processed_dir, cfg, block)
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
        "block": block,
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
            **(_cuda_hardware() if settings.device == "cuda" else {}),
        },
        "checkpoint": checkpoint_record,
        "holiday_stratification": "unavailable: civil timestamps remain unverified in #6",
    }
    results = {}
    methods = METHODS if method == "all" else (method,)
    deadline = time.monotonic() + settings.max_seconds
    for name in methods:
        lag = None if name == "chronos" else int(name.removeprefix("seasonal_lag"))
        revision = settings.model_revision if lag is None else f"seasonal-naive-v1:lag-{lag}"
        if checkpoint_record is not None:
            revision = f"local-sha256:{checkpoint_record['sha256']}"
        config = {
            **common,
            "method": name,
            "seasonal_lag": lag,
            "forecast_device": settings.device if lag is None else "cpu",
        }
        output = directory / name
        output.mkdir()
        with tracked_run(
            name=f"{block}-{name}" + ("-checkpoint" if checkpoint_record else ""),
            config=config,
            data_version=version,
            model_revision=revision,
            tracking_uri=tracking_uri,
        ) as run:
            if lag is None:
                predictions, timing = chronos_forecast(
                    data.contexts,
                    cfg.horizon,
                    settings,
                    deadline,
                    measure=run.measure_energy,
                    checkpoint=checkpoint,
                )
            else:
                with run.measure_energy():
                    start = time.perf_counter()
                    predictions = seasonal_forecast(data.contexts, cfg.horizon, lag)
                    inference_seconds = time.perf_counter() - start
                timing = {"inference_seconds": inference_seconds, "model_load_seconds": 0.0}
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
    # The primary baseline is chosen on validation only, never on test scores.
    primary = (
        min(baselines, key=lambda name: (results[name]["metrics"]["mean.mase"], name))
        if len(baselines) == len(BASELINE_LAGS) and block == "validation"
        else None
    )
    comparison = {
        "evaluation_group": group_id,
        "block": block,
        "data_version": version,
        "targets_sha256": target_hash,
        "settings": asdict(settings),
        "checkpoint": checkpoint_record,
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
    checkpoint: Path | None = None,
    device: str | None = None,
    block: str = "validation",
) -> None:
    """Evaluate all origins of a held-out block; all creates two baseline runs and one Chronos run.

    Use --method chronos --checkpoint <directory> to score a local checkpoint, --device
    cuda to forecast on the GPU, and --block test only for the final comparison.
    """
    result = evaluate(
        method=method,
        params_path=params_path,
        processed_dir=processed_dir,
        output_dir=output_dir,
        tracking_uri=tracking_uri,
        checkpoint=checkpoint,
        device=device,
        block=block,
    )
    typer.echo(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    app()
