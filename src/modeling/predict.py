"""Reusable CPU inference with the selected, weight-identified Chronos checkpoint."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from threading import Lock

import numpy as np
import typer
import yaml

from src.evaluation import sample_median
from src.modeling.checkpoints import describe_checkpoint
from src.splits import PARAMS_PATH

SAMPLING = {"temperature": 1.0, "top_k": 50, "top_p": 1.0}
SELECTED_CHECKPOINT = PARAMS_PATH.parent / "models" / "chronos-t5-small-fine-tuned"
SELECTED_PROVENANCE = SELECTED_CHECKPOINT.with_suffix(".provenance.json")
# Torch's CPU generator and thread setting are process-global, even across instances.
_INFERENCE_LOCK = Lock()
app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass(frozen=True)
class ForecastConfig:
    """CPU serving settings; None selects the pinned base instead of local weights."""

    model_id: str
    model_revision: str
    context_length: int
    horizon: int
    num_samples: int
    seed: int
    device: str
    cpu_threads: int
    checkpoint: Path | None = SELECTED_CHECKPOINT

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must name a checkpoint")
        if not isinstance(self.model_revision, str) or not re.fullmatch(
            r"[0-9a-f]{40}", self.model_revision
        ):
            raise ValueError("Pin model_revision to a full Hub commit")
        for name in ("context_length", "horizon", "num_samples", "cpu_threads"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer between 0 and 2**64 - 1")
        if self.device != "cpu":
            raise ValueError("This serving module currently supports device: cpu")
        if (self.context_length, self.horizon) != (512, 48):
            raise ValueError("Serving requires context_length=512 and horizon=48")
        if self.checkpoint is not None and not isinstance(self.checkpoint, Path):
            raise ValueError("checkpoint must be a local Path, or None for the pinned base model")

    @classmethod
    def from_params(
        cls,
        path: Path = PARAMS_PATH,
        *,
        checkpoint: Path | None = SELECTED_CHECKPOINT,
        cpu_threads: int = 2,
    ) -> "ForecastConfig":
        """Use evaluation sampling settings, with an explicit CPU serving budget."""
        try:
            params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            forecast, evaluation = params["forecast"], params["evaluation"]
            return cls(
                model_id=evaluation["model_id"],
                model_revision=evaluation["model_revision"],
                context_length=forecast["context_length"],
                horizon=forecast["horizon"],
                num_samples=evaluation["num_samples"],
                seed=evaluation["seed"],
                device=evaluation["device"],
                cpu_threads=cpu_threads,
                checkpoint=checkpoint,
            )
        except (KeyError, TypeError, yaml.YAMLError) as error:
            raise ValueError(
                "Expected forecast and evaluation settings in the params file"
            ) from error


def _load_pipeline(config: ForecastConfig, local_files_only: bool):
    checkpoint = None
    if config.checkpoint is not None:
        try:
            checkpoint = describe_checkpoint(config.checkpoint, provenance=SELECTED_PROVENANCE)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Could not verify local checkpoint {config.checkpoint}") from error
    # Importing this module and running its fixture tests needs no model dependencies.
    try:
        from chronos import ChronosPipeline  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError("Install model dependencies with uv sync --extra forecast") from error

    source = {"revision": config.model_revision}
    model_id = config.model_id
    if checkpoint is not None:
        model_id = checkpoint["path"]
        source = {}
        local_files_only = True
    try:
        with _INFERENCE_LOCK, torch.random.fork_rng(devices=[]):
            torch.set_num_threads(config.cpu_threads)
            torch.random.default_generator.manual_seed(config.seed)
            pipeline = ChronosPipeline.from_pretrained(
                model_id,
                **source,
                device_map=config.device,
                dtype=torch.float32,
                token=False,
                local_files_only=local_files_only,
            )
            pipeline.model.model.eval()
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"Could not load Chronos checkpoint {model_id}") from error
    return torch, pipeline, checkpoint


def _validated_history(history: Sequence[float] | np.ndarray, length: int) -> np.ndarray:
    try:
        values = np.asarray(history)
    except (TypeError, ValueError) as error:
        raise ValueError("History must be a one-dimensional numeric array") from error
    if values.shape != (length,):
        raise ValueError(f"History must contain exactly {length} values in one dimension")
    if values.dtype.kind not in "iuf" or any(isinstance(v, (bool, np.bool_)) for v in history):
        raise ValueError("History values must be numbers, not strings or booleans")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("History values must be finite and representable as float32")
    return values


class ChronosForecaster:
    """Create once at application startup and reuse for each history."""

    def __init__(self, config: ForecastConfig | None = None, *, local_files_only: bool = False):
        self.config = config if config is not None else ForecastConfig.from_params()
        self._torch, self._pipeline, self._checkpoint = _load_pipeline(
            self.config, local_files_only
        )

    def predict(self, history: Sequence[float] | np.ndarray) -> dict:
        """Return source-scale quantiles and the configuration for one forecast."""
        cfg = self.config
        values = _validated_history(history, cfg.context_length)
        # Serialize sampling and restore the caller's CPU RNG state after each request.
        with (
            _INFERENCE_LOCK,
            self._torch.inference_mode(),
            self._torch.random.fork_rng(devices=[]),
        ):
            self._torch.set_num_threads(cfg.cpu_threads)
            self._torch.random.default_generator.manual_seed(cfg.seed)
            try:
                output = self._pipeline.predict(
                    self._torch.as_tensor(values[None, :], dtype=self._torch.float32),
                    prediction_length=cfg.horizon,
                    num_samples=cfg.num_samples,
                    **SAMPLING,
                )
            except (OSError, ValueError, RuntimeError) as error:
                raise RuntimeError("Chronos prediction failed") from error
            try:
                samples = output.cpu().numpy()
                if (
                    not isinstance(samples, np.ndarray)
                    or samples.dtype.kind not in "iuf"
                    or samples.shape != (1, cfg.num_samples, cfg.horizon)
                ):
                    raise ValueError("Unexpected sample type or shape")
                with np.errstate(over="ignore", invalid="ignore"):
                    median = sample_median(samples, 1, cfg.horizon)[0]
                    lower, upper = np.quantile(samples[0], [0.1, 0.9], axis=0, method="linear")
                if not (
                    np.isfinite([lower, median, upper]).all()
                    and (lower <= median).all()
                    and (median <= upper).all()
                ):
                    raise ValueError("Nonfinite or unordered quantiles")
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                raise RuntimeError(
                    "Chronos must return finite samples with shape "
                    f"(1, {cfg.num_samples}, {cfg.horizon}) and finite ordered quantiles"
                ) from error

        return {
            "p10": lower.tolist(),
            "p50": median.tolist(),
            "p90": upper.tolist(),
            "model_id": "chronos-t5-small-fine-tuned" if self._checkpoint else cfg.model_id,
            "model_revision": (
                f"local-sha256:{self._checkpoint['sha256']}"
                if self._checkpoint
                else cfg.model_revision
            ),
            "checkpoint": (
                {key: self._checkpoint[key] for key in ("sha256", "training_run_id")}
                if self._checkpoint
                else None
            ),
            "value_scale": "source",
            "context_length": cfg.context_length,
            "horizon": cfg.horizon,
            "settings": {
                "num_samples": cfg.num_samples,
                "seed": cfg.seed,
                "device": cfg.device,
                "cpu_threads": cfg.cpu_threads,
                "dtype": "float32",
                "sampling": dict(SAMPLING),
                "quantile_method": "linear",
                "point_statistic": "median",
            },
        }


@app.command()
def main(
    history_path: Path,
    params_path: Path = PARAMS_PATH,
    output_path: Path | None = None,
    local_files_only: bool = False,
    checkpoint: Path | None = None,
    base_model: bool = False,
    cpu_threads: int = 2,
):
    """Forecast from a JSON array of history values."""
    try:
        history = json.loads(history_path.read_text())
        if base_model and checkpoint is not None:
            raise ValueError("Choose either --checkpoint or --base-model")
        config = ForecastConfig.from_params(
            params_path,
            checkpoint=None if base_model else checkpoint or SELECTED_CHECKPOINT,
            cpu_threads=cpu_threads,
        )
        _validated_history(history, config.context_length)
        forecaster = ChronosForecaster(config, local_files_only=local_files_only)
        result = json.dumps(forecaster.predict(history), indent=2, allow_nan=False) + "\n"
        if output_path is None:
            typer.echo(result, nl=False)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result)
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


if __name__ == "__main__":
    app()
