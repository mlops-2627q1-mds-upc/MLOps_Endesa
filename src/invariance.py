"""Paired histories and declared comparison rules for forecast behaviour checks."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.splits import SplitConfig

CONTEXT_LENGTH = 512
HORIZON = 48
NOISE_SEED = 7
PATTERNS = ("constant", "seasonal", "trend", "signed")


@dataclass(frozen=True)
class History:
    """One ordered history, identified by its source and positional forecast origin."""

    name: str
    values: np.ndarray
    block: str = "synthetic"
    origin: int | None = None

    def __post_init__(self):
        values = np.asarray(self.values, dtype=np.float32)
        if values.shape != (CONTEXT_LENGTH,) or not np.isfinite(values).all():
            raise ValueError("A history must contain exactly 512 finite values")
        object.__setattr__(self, "values", values.copy())
        if self.scale <= 0:
            raise ValueError("Relative comparisons require a positive mean absolute history")

    @property
    def scale(self) -> float:
        """Use the mean absolute history, without dividing by individual observations."""
        return float(np.abs(np.asarray(self.values, dtype=np.float64)).mean())


@dataclass(frozen=True)
class Transformation:
    """A fixed input change and its comparison thresholds, chosen before inference."""

    kind: str
    amount: float
    mean_tolerance: float
    max_tolerance: float

    def __post_init__(self):
        if self.kind not in ("scale", "noise") or not all(
            np.isfinite(value) and value > 0
            for value in (self.amount, self.mean_tolerance, self.max_tolerance)
        ):
            raise ValueError("Choose a scale or noise change with positive finite settings")

    @property
    def name(self) -> str:
        """Return a readable pytest and evidence identifier."""
        return f"{self.kind}-{self.amount:g}"

    def apply(self, history: History) -> np.ndarray:
        """Keep observation order; round changed inputs to the model's float32 dtype."""
        values = np.asarray(history.values, dtype=np.float64)
        if self.kind == "scale":
            changed = values * self.amount
        else:
            noise = np.random.default_rng(NOISE_SEED).uniform(-1, 1, CONTEXT_LENGTH)
            changed = values + self.amount * history.scale * noise
        return changed.astype(np.float32)


# Scale thresholds allow float32 roundoff; noise thresholds are robustness hypotheses.
TRANSFORMATIONS = (
    Transformation("scale", 0.5, 1e-5, 1e-4),
    Transformation("scale", 2.0, 1e-5, 1e-4),
    Transformation("scale", 10.0, 1e-5, 1e-4),
    Transformation("noise", 0.001, 0.05, 0.15),
    Transformation("noise", 0.01, 0.05, 0.15),
)


def synthetic_histories() -> list[History]:
    """Cover flat, cyclic, trending and signed inputs without calendar assumptions."""
    position = np.arange(CONTEXT_LENGTH)
    cycle = np.sin(2 * np.pi * position / 48)
    values = (
        np.full(CONTEXT_LENGTH, 1000.0),
        1000 + 200 * cycle,
        np.linspace(500, 1500, CONTEXT_LENGTH) + 100 * cycle,
        200 * cycle - 20,
    )
    return [History(name, series.astype(np.float32)) for name, series in zip(PATTERNS, values)]


def stored_histories(directory: Path, cfg: SplitConfig, lock_path: Path) -> list[History]:
    """Select one train and one validation history per state; never open the test file."""
    if (cfg.context_length, cfg.horizon) != (CONTEXT_LENGTH, HORIZON):
        raise ValueError("Invariance checks require the agreed 512-value / 48-step protocol")
    outputs = yaml.safe_load(lock_path.read_text(encoding="utf-8"))["stages"]["split"]["outs"]
    expected = {item["path"]: item["md5"] for item in outputs}
    histories = []
    for block in ("train", "validation"):
        path = directory / f"{block}.parquet"
        if not path.is_file():
            raise ValueError(f"Restore {path.name} with DVC, or explicitly choose synthetic-only")
        if hashlib.md5(path.read_bytes()).hexdigest() != expected[f"data/processed/{path.name}"]:
            raise ValueError(f"{path.name} does not match dvc.lock")
        frame = pd.read_parquet(path)
        start, end = cfg.bounds(block)
        if list(frame.columns) != list(cfg.states) or not np.array_equal(
            frame.index.to_numpy(), np.arange(start, end)
        ):
            raise ValueError(f"{block} columns or positions do not match params.yaml")
        if block == "train":
            origin = end - cfg.horizon
        else:
            origin = start + ((end - start) // (2 * cfg.origin_stride)) * cfg.origin_stride
        if origin - CONTEXT_LENGTH < start:
            raise ValueError(f"{block} is too short for the selected history")
        for state in cfg.states:
            values = frame[state].iloc[origin - CONTEXT_LENGTH - start : origin - start]
            histories.append(History(f"{state}-{block}", values.to_numpy(), block, origin))
    return histories


def history_record(history: History) -> dict:
    """Identify a window without putting source observations in the Git evidence."""
    return {
        "name": history.name,
        "block": history.block,
        "origin": history.origin,
        "context_start": None if history.origin is None else history.origin - CONTEXT_LENGTH,
        "mean_absolute_value": history.scale,
        "sha256_float32_le": hashlib.sha256(
            np.asarray(history.values, dtype="<f4").tobytes()
        ).hexdigest(),
    }


def compare_pair(
    history: History,
    change: Transformation,
    original: np.ndarray,
    transformed: np.ndarray,
) -> dict:
    """Compare all 48 steps, undoing a scale change before calculating differences."""
    original, transformed = np.asarray(original), np.asarray(transformed)
    if any(
        values.shape != (HORIZON,) or not np.isfinite(values).all()
        for values in (original, transformed)
    ):
        raise ValueError("Each forecast must contain exactly 48 finite values")
    factor = change.amount if change.kind == "scale" else 1.0
    restored = transformed.astype(np.float64) / factor
    errors = np.abs(original.astype(np.float64) - restored) / history.scale
    input_errors = (
        np.abs(change.apply(history).astype(np.float64) - history.values) / history.scale
    )
    mean_error, max_error = float(errors.mean()), float(errors.max())
    return {
        "case_id": f"{history.name}/{change.name}",
        "history": history.name,
        "transformation": change.name,
        "input_mean_absolute_change_relative": float(input_errors.mean()),
        "input_max_absolute_change_relative": float(input_errors.max()),
        "forecast_mean_absolute_change_relative": mean_error,
        "forecast_max_absolute_change_relative": max_error,
        "within_tolerance": mean_error <= change.mean_tolerance
        and max_error <= change.max_tolerance,
        "original_forecast": original.tolist(),
        "transformed_forecast": transformed.tolist(),
    }
