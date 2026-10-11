"""Bounded Chronos fine-tuning on the training block for #13."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

from loguru import logger
from mlflow.entities import Metric
import numpy as np
import pandas as pd
import typer
import yaml

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT
from src.modeling.evaluate import EvaluationConfig, data_version
from src.splits import PARAMS_PATH, SplitConfig
from src.tracking import tracked_run

# The loss of every step is kept; MLflow accepts at most 1,000 metrics per batch.
_METRIC_BATCH = 1000
app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass(frozen=True)
class FineTuneConfig:
    learning_rate: float
    steps: int
    batch_size: int
    micro_batch_size: int
    seed: int
    window_start: int
    checkpoint_every: int
    device: str
    max_seconds: int

    def __post_init__(self):
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0):
            raise ValueError("The learning rate must be positive and finite")
        if min(self.steps, self.batch_size, self.checkpoint_every, self.max_seconds) <= 0:
            raise ValueError(
                "Steps, batch size, checkpoint interval and time budget must be positive"
            )
        if self.micro_batch_size <= 0 or self.batch_size % self.micro_batch_size:
            raise ValueError("micro_batch_size must be positive and divide batch_size")
        if self.window_start < 0:
            raise ValueError("window_start must not be negative")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Choose an explicit cpu or cuda device")

    @classmethod
    def from_params(cls, path: Path) -> "FineTuneConfig":
        return cls(**yaml.safe_load(path.read_text())["fine_tuning"])


def window_start_bounds(cfg: SplitConfig, start: int, prediction_length: int) -> tuple[int, int]:
    """Return the first and last start of a window that lies inside [start, train_end)."""
    last = cfg.train_end - cfg.context_length - prediction_length
    if not 0 <= start <= last:
        raise ValueError("No training window fits between window_start and train_end")
    return start, last


def sample_windows(
    training: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
    context_length: int,
    prediction_length: int,
    bounds: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Draw (context, label) windows with a random state and start position each."""
    first, last = bounds
    if last + context_length + prediction_length > len(training):
        raise ValueError("Windows must end inside the training block")
    states = rng.integers(0, training.shape[1], batch_size)
    starts = rng.integers(first, last + 1, batch_size)
    positions = starts[:, None] + np.arange(context_length + prediction_length)
    windows = training[positions, states[:, None]]
    return windows[:, :context_length], windows[:, context_length:]


def linear_decay(step: int, steps: int) -> float:
    """Learning-rate factor that falls linearly from 1 at step 0 to 0 at `steps`."""
    return max(0.0, 1.0 - step / steps)


def load_training_block(directory: Path, cfg: SplitConfig) -> np.ndarray:
    """Read only the training block; validation and test values never reach training."""
    frame = pd.read_parquet(directory / "train.parquet")
    start, end = cfg.bounds("train")
    if list(frame.columns) != list(cfg.states) or not np.array_equal(
        frame.index.to_numpy(), np.arange(start, end)
    ):
        raise ValueError("train columns or positions do not match params.yaml")
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("train contains nonfinite observations")
    return values


def tokenize(tokenizer, contexts: np.ndarray, labels: np.ndarray) -> dict:
    """Encode windows as the checkpoint does; padded label tokens are ignored by the loss."""
    import torch

    input_ids, attention_mask, scale = tokenizer.context_input_transform(
        torch.as_tensor(contexts, dtype=torch.float32)
    )
    label_ids, label_mask = tokenizer.label_input_transform(
        torch.as_tensor(labels, dtype=torch.float32), scale
    )
    label_ids[~label_mask] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}


def _hardware(device: str) -> dict:
    import torch

    hardware = {"device": device, "torch": torch.__version__, "cuda": torch.version.cuda}
    if device == "cuda":
        hardware["gpu"] = torch.cuda.get_device_name(0)
    return hardware


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fine_tune(
    *,
    params_path: Path = PARAMS_PATH,
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = MODELS_DIR / "fine-tuning",
    tracking_uri: str | None = None,
    max_steps: int | None = None,
    control: bool = False,
) -> dict:
    """Fine-tune the pinned checkpoint; `max_steps` stops early to measure step cost.

    `control` saves the pinned checkpoint untrained, so it is scored through the same
    local-checkpoint path as the fine-tuned models.
    """
    try:
        from chronos import ChronosPipeline
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Install forecasting dependencies with uv sync --extra forecast"
        ) from error

    if processed_dir.resolve() != (PROJ_ROOT / "data/processed").resolve():
        raise ValueError("Use this checkout's DVC-tracked data/processed directory")
    if max_steps is not None and (max_steps <= 0 or control):
        raise ValueError("max_steps must be positive and cannot be combined with control")
    cfg = SplitConfig.from_params(params_path)
    base = EvaluationConfig.from_params(params_path)
    settings = FineTuneConfig.from_params(params_path)
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; install the CUDA build of PyTorch")
    version = data_version()
    training = load_training_block(processed_dir, cfg)

    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    pipeline = ChronosPipeline.from_pretrained(
        base.model_id,
        revision=base.model_revision,
        device_map=settings.device,
        torch_dtype=torch.float32,
        token=False,
    )
    model = pipeline.model.model
    prediction_length = pipeline.model.config.prediction_length
    bounds = window_start_bounds(cfg, settings.window_start, prediction_length)
    steps = settings.steps if max_steps is None else min(max_steps, settings.steps)
    if control:
        steps = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=0.0)
    # The schedule always spans the configured steps, so a shortened run trains identically.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: linear_decay(step, settings.steps)
    )
    config = {
        "fine_tuning": asdict(settings),
        "base_model_id": base.model_id,
        "base_model_revision": base.model_revision,
        "context_length": cfg.context_length,
        "prediction_length": prediction_length,
        "window_start_bounds": list(bounds),
        "steps_run": steps,
        "dtype": "float32",
        "optimizer": "AdamW, weight decay 0, gradient norm clipped at 1.0",
        "hardware": _hardware(settings.device),
    }
    with tracked_run(
        name="control-untuned" if control else f"fine-tune-seed{settings.seed}",
        config=config,
        data_version=version,
        model_revision=base.model_revision,
        tracking_uri=tracking_uri,
        purpose="training" if max_steps is None else "smoke",
    ) as run:
        directory = output_dir / run.run_id
        directory.mkdir(parents=True, exist_ok=False)
        losses = []
        deadline = time.monotonic() + settings.max_seconds
        model.train()
        # The control trains nothing, so it has no training energy to measure.
        with run.measure_energy() if steps else nullcontext():
            start = time.perf_counter()
            for step in range(1, steps + 1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Training exceeded max_seconds; no checkpoint recorded")
                contexts, labels = sample_windows(
                    training,
                    rng,
                    settings.batch_size,
                    cfg.context_length,
                    prediction_length,
                    bounds,
                )
                encoded = tokenize(pipeline.tokenizer, contexts, labels)
                # Every window has the same number of label tokens, so the mean over equal
                # micro-batches equals the mean over the whole batch.
                parts = settings.batch_size // settings.micro_batch_size
                loss = 0.0
                for part in range(parts):
                    rows = slice(
                        part * settings.micro_batch_size, (part + 1) * settings.micro_batch_size
                    )
                    micro = {
                        name: tensor[rows].to(settings.device) for name, tensor in encoded.items()
                    }
                    part_loss = model(**micro).loss / parts
                    if not torch.isfinite(part_loss):
                        raise FloatingPointError(f"Non-finite training loss at step {step}")
                    part_loss.backward()
                    loss += float(part_loss.detach())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(loss)
                if step % settings.checkpoint_every == 0 and step < steps:
                    model.save_pretrained(directory / f"step-{step}")
            if settings.device == "cuda":
                torch.cuda.synchronize()
            train_seconds = time.perf_counter() - start
        model.save_pretrained(directory / "final")
        weights = directory / "final" / "model.safetensors"
        timestamp = int(time.time() * 1000)
        history = [
            Metric("train.loss", loss, timestamp, step) for step, loss in enumerate(losses, 1)
        ]
        for offset in range(0, len(history), _METRIC_BATCH):
            run.client.log_batch(
                run.run_id, metrics=history[offset : offset + _METRIC_BATCH], synchronous=True
            )
        summary = {"train.steps": float(steps)}
        if steps:
            summary |= {
                "train.seconds": train_seconds,
                "train.seconds_per_step": train_seconds / steps,
                "train.final_loss": float(np.mean(losses[-min(50, len(losses)) :])),
            }
        run.log_metrics(summary)
        checkpoint = {
            "path": str(directory / "final"),
            "model_safetensors_sha256": _sha256(weights),
        }
        run.client.log_dict(run.run_id, checkpoint, "checkpoint.json")
        run.client.log_dict(run.run_id, yaml.safe_load(params_path.read_text()), "params.json")
    logger.info(f"saved after {steps} training steps; run={run.run_id}")
    return {"run_id": run.run_id, **summary, "checkpoint": checkpoint}


@app.command()
def main(
    params_path: Path = PARAMS_PATH,
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = MODELS_DIR / "fine-tuning",
    tracking_uri: str | None = None,
    max_steps: int | None = None,
    control: bool = False,
) -> None:
    """Fine-tune the pinned Chronos checkpoint on training windows only.

    Use --control to save the untrained checkpoint for the control evaluation.
    """
    result = fine_tune(
        params_path=params_path,
        processed_dir=processed_dir,
        output_dir=output_dir,
        tracking_uri=tracking_uri,
        max_steps=max_steps,
        control=control,
    )
    typer.echo(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    app()
