"""Chronological train/validation/test blocks and forecast origins.

Positions count 30-minute steps from the start of each series. The protocol values are
read from params.yaml; see docs/problem-definition.md for their rationale.
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer
import yaml

from src.config import PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR

PARAMS_PATH = PROJ_ROOT / "params.yaml"
BLOCKS = ("train", "validation", "test")
HELD_OUT_BLOCKS = ("validation", "test")

app = typer.Typer()


@dataclass(frozen=True)
class SplitConfig:
    states: tuple[str, ...]
    common_length: int
    train_end: int
    validation_end: int
    context_length: int
    horizon: int
    origin_stride: int

    def __post_init__(self):
        if min(self.context_length, self.horizon, self.origin_stride) <= 0:
            raise ValueError("context_length, horizon and origin_stride must be positive")
        if not 0 < self.train_end < self.validation_end < self.common_length:
            raise ValueError("expected 0 < train_end < validation_end < common_length")
        if self.context_length > self.train_end:
            raise ValueError("the first validation origin needs a full context")
        for name in HELD_OUT_BLOCKS:
            start, end = self.bounds(name)
            # Origins must tile the block so the last target ends exactly at its end.
            if end - start < self.horizon or (end - start - self.horizon) % self.origin_stride:
                raise ValueError(f"{name} length does not fit whole horizons at the stride")

    @classmethod
    def from_params(cls, path: Path = PARAMS_PATH) -> "SplitConfig":
        params = yaml.safe_load(Path(path).read_text())
        return cls(
            states=tuple(params["data"]["states"]),
            common_length=params["data"]["common_length"],
            train_end=params["split"]["train_end"],
            validation_end=params["split"]["validation_end"],
            context_length=params["forecast"]["context_length"],
            horizon=params["forecast"]["horizon"],
            origin_stride=params["forecast"]["origin_stride"],
        )

    def bounds(self, name: str) -> tuple[int, int]:
        """Return the [start, end) positions of a block."""
        bounds = {
            "train": (0, self.train_end),
            "validation": (self.train_end, self.validation_end),
            "test": (self.validation_end, self.common_length),
        }
        if name not in bounds:
            raise ValueError(f"unknown block {name!r}; expected one of {BLOCKS}")
        return bounds[name]


def truncate(series: np.ndarray, cfg: SplitConfig) -> np.ndarray:
    """Cut a state series to the common prefix shared by all states."""
    if len(series) < cfg.common_length:
        raise ValueError(f"series has {len(series)} steps, fewer than {cfg.common_length}")
    return series[: cfg.common_length]


def block(series: np.ndarray, name: str, cfg: SplitConfig) -> np.ndarray:
    """Return one block of a truncated series."""
    if len(series) != cfg.common_length:
        raise ValueError("truncate the series to common_length first")
    start, end = cfg.bounds(name)
    return series[start:end]


def forecast_origins(name: str, cfg: SplitConfig) -> np.ndarray:
    """Return the origin positions of a held-out block.

    The forecast made at an origin predicts positions [origin, origin + horizon).
    """
    if name not in HELD_OUT_BLOCKS:
        raise ValueError(f"forecast origins are defined for {HELD_OUT_BLOCKS} only")
    start, end = cfg.bounds(name)
    return np.arange(start, end - cfg.horizon + 1, cfg.origin_stride)


def forecast_window(
    series: np.ndarray, origin: int, cfg: SplitConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return (context, target) for an origin.

    The context ends just before the origin, so it never contains the target or any later
    value. It may include earlier held-out targets, as a rolling forecast would.
    """
    if origin < cfg.context_length or origin + cfg.horizon > len(series):
        raise ValueError(f"origin {origin} has no full context or target")
    context = series[origin - cfg.context_length : origin]
    target = series[origin : origin + cfg.horizon]
    return context, target


@app.command()
def main(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    params_path: Path = PARAMS_PATH,
):
    cfg = SplitConfig.from_params(params_path)
    series = {}
    for state in cfg.states:
        values = pd.read_parquet(raw_dir / f"{state}.parquet")[state].to_numpy()
        series[state] = truncate(values, cfg)
        logger.info(f"{state}: kept {cfg.common_length:,} of {len(values):,} steps")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in BLOCKS:
        start, end = cfg.bounds(name)
        frame = pd.DataFrame(
            {state: block(values, name, cfg) for state, values in series.items()},
            index=pd.RangeIndex(start, end, name="timestep"),
        )
        frame.to_parquet(output_dir / f"{name}.parquet")
        logger.info(f"{name}: positions [{start:,}, {end:,}) -> {output_dir / name}.parquet")

    origins = pd.DataFrame(
        [
            (state, name, origin, origin - cfg.context_length, origin + cfg.horizon)
            for name in HELD_OUT_BLOCKS
            for state in cfg.states
            for origin in forecast_origins(name, cfg)
        ],
        columns=["state", "block", "origin", "context_start", "target_end"],
    )
    origins.to_parquet(output_dir / "origins.parquet", index=False)
    logger.success(f"Wrote {len(origins):,} forecast origins.")


if __name__ == "__main__":
    app()
