"""Validation windows, seasonal forecasts and point-error metrics for #12."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.splits import SplitConfig, forecast_origins, forecast_window

MASE_LAG = 336
BASELINE_LAGS = (48, 336)


def mase_scale(training: np.ndarray, lag: int = MASE_LAG) -> float:
    """Estimate the seasonal absolute-error scale using training values only."""
    training = np.asarray(training, dtype=np.float64)
    if lag <= 0 or len(training) <= lag or not np.isfinite(training).all():
        raise ValueError("MASE needs finite training values and more observations than its lag")
    scale = float(np.abs(training[lag:] - training[:-lag]).mean())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("MASE is undefined for a zero or nonfinite training scale")
    return scale


def seasonal_forecast(contexts: np.ndarray, horizon: int, lag: int) -> np.ndarray:
    """Repeat the last known seasonal cycle; never read forecast targets."""
    contexts = np.asarray(contexts)
    if contexts.ndim != 2 or not 0 < lag <= contexts.shape[1] or horizon <= 0:
        raise ValueError("Seasonal forecasts need a positive horizon and a full known cycle")
    if not np.isfinite(contexts).all():
        raise ValueError("Forecast contexts must be finite")
    return contexts[:, -lag:][:, np.arange(horizon) % lag].copy()


def sample_median(samples: np.ndarray, rows: int, horizon: int) -> np.ndarray:
    """Take the median over trajectories, including interpolation for even counts."""
    samples = np.asarray(samples)
    if (
        samples.ndim != 3
        or samples.shape[0] != rows
        or samples.shape[2] != horizon
        or samples.shape[1] == 0
        or not np.isfinite(samples).all()
    ):
        raise ValueError("Expected finite samples with shape (origins, samples, horizon)")
    return np.median(samples, axis=1)


@dataclass(frozen=True)
class ValidationData:
    contexts: np.ndarray
    targets: np.ndarray
    states: np.ndarray
    origins: np.ndarray
    scales: dict[str, float]

    @classmethod
    def load(cls, directory: Path, cfg: SplitConfig) -> "ValidationData":
        # Test targets are deliberately never opened by this evaluator.
        frames = {}
        for name in ("train", "validation"):
            frame = pd.read_parquet(directory / f"{name}.parquet")
            start, end = cfg.bounds(name)
            if list(frame.columns) != list(cfg.states) or not np.array_equal(
                frame.index.to_numpy(), np.arange(start, end)
            ):
                raise ValueError(f"{name} columns or positions do not match params.yaml")
            if not np.isfinite(frame.to_numpy()).all():
                raise ValueError(f"{name} contains nonfinite observations")
            frames[name] = frame
        contexts, targets, states, origins = [], [], [], []
        scales = {}
        for state in cfg.states:
            training = frames["train"][state].to_numpy()
            scales[state] = mase_scale(training)
            history = np.concatenate((training, frames["validation"][state].to_numpy()))
            for origin in forecast_origins("validation", cfg):
                context, target = forecast_window(history, int(origin), cfg)
                contexts.append(context)
                targets.append(target)
                states.append(state)
                origins.append(origin)
        return cls(
            np.stack(contexts), np.stack(targets), np.array(states), np.array(origins), scales
        )

    def predictions_frame(self, predictions: np.ndarray) -> pd.DataFrame:
        predictions = np.asarray(predictions, dtype=np.float64)
        if predictions.shape != self.targets.shape or not np.isfinite(predictions).all():
            raise ValueError("Predictions must be finite and match every validation target")
        horizon = self.targets.shape[1]
        return pd.DataFrame(
            {
                "state": np.repeat(self.states, horizon),
                "origin": np.repeat(self.origins, horizon),
                "timestep": (self.origins[:, None] + np.arange(horizon)).ravel(),
                "target": self.targets.ravel(),
                "prediction": predictions.ravel(),
            }
        )

    def metrics(self, predictions: np.ndarray) -> dict[str, float]:
        if predictions.shape != self.targets.shape or not np.isfinite(predictions).all():
            raise ValueError("Predictions must be finite and match every validation target")
        errors = np.abs(predictions.astype(np.float64) - self.targets)
        metrics = {}
        for state, scale in self.scales.items():
            per_origin = errors[self.states == state].mean(axis=1)
            mae = float(per_origin.mean())
            worst_count = max(1, math.ceil(0.05 * len(per_origin)))
            total = float(per_origin.sum())
            metrics.update(
                {
                    f"{state}.mae": mae,
                    f"{state}.mase": mae / scale,
                    f"{state}.worst_5pct_error_share": (
                        float(np.partition(per_origin, -worst_count)[-worst_count:].sum()) / total
                        if total
                        else 0.0
                    ),
                    f"{state}.worst_origin_count": float(worst_count),
                    f"{state}.origins": float(len(per_origin)),
                }
            )
        for metric in ("mae", "mase", "worst_5pct_error_share"):
            metrics[f"mean.{metric}"] = float(
                np.mean([metrics[f"{state}.{metric}"] for state in self.scales])
            )
        return metrics
