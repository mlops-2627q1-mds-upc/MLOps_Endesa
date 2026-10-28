"""Validated historical positions; final-test files are never opened."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.schemas import ForecastRequest
from src.monitoring.types import ReplayConfig, canonical_json, sha256
from src.splits import SplitConfig, forecast_window


class ReplaySource:
    """Hold an immutable train/validation prefix and its exact source identity."""

    def __init__(
        self,
        frame: pd.DataFrame,
        split: SplitConfig,
        reference: str,
        file_hashes: dict[str, str] | None = None,
    ):
        if (split.context_length, split.horizon) != (512, 48):
            raise ValueError("Replay requires the API's 512 -> 48 positional protocol")
        if (
            len(frame) != split.validation_end
            or frame.index.dtype.kind not in "iu"
            or list(frame.columns) != list(split.states)
            or not np.array_equal(frame.index.to_numpy(), np.arange(split.validation_end))
        ):
            raise ValueError("Source columns/positions must match the train/validation prefix")
        if any(dtype.kind not in "iuf" for dtype in frame.dtypes):
            raise ValueError("Source columns must contain real numeric observations")
        values = frame.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Source observations must be finite")
        values.setflags(write=False)
        self._values = values
        self.split = split
        self.reference = canonical_json(
            {
                "reference": reference,
                "split": asdict(split),
                "file_hashes": file_hashes or {},
                "values_sha256": sha256(values.astype("<f8").tobytes()),
            }
        )
        self.source_sha256 = sha256(self.reference)

    @classmethod
    def from_validation(cls, directory: Path, params_path: Path) -> "ReplaySource":
        """Validate the same positional block boundaries as ValidationData.load."""
        params_hash = sha256(params_path.read_bytes())
        split = SplitConfig.from_params(params_path)
        if params_hash != sha256(params_path.read_bytes()):
            raise ValueError("Parameters changed while loading")
        frames = []
        hashes = {"params.yaml": params_hash}
        for name in ("train", "validation"):
            path = directory / f"{name}.parquet"
            before = sha256(path.read_bytes())
            frame = pd.read_parquet(path)
            if before != sha256(path.read_bytes()):
                raise ValueError("Source file changed while loading")
            start, end = split.bounds(name)
            if (
                len(frame) != end - start
                or frame.index.dtype.kind not in "iu"
                or list(frame.columns) != list(split.states)
                or not np.array_equal(frame.index.to_numpy(), np.arange(start, end))
            ):
                raise ValueError(f"{name} columns/positions do not match the split protocol")
            frames.append(frame)
            hashes[path.name] = before
        return cls(pd.concat(frames), split, "validation:train.parquet+validation.parquet", hashes)

    def validate_config(self, config: ReplayConfig) -> None:
        """Refuse unapproved origins or states before opening a journal or HTTP client."""
        # Check bounds arithmetically before constructing any origin collection.
        last = (
            config.origin_start
            + ((config.origin_stop - 1 - config.origin_start) // config.origin_stride)
            * config.origin_stride
        )
        if (
            config.origin_start < self.split.train_end
            or config.origin_stop > self.split.validation_end
            or last + self.split.horizon > self.split.validation_end
            or (config.origin_start - self.split.train_end) % self.split.origin_stride
            or config.origin_stride != self.split.origin_stride
            or not set(config.states) <= set(self.split.states)
        ):
            raise ValueError("Replay range/states must follow the validation origin protocol")

    def history(self, state: str, origin: int) -> list[float]:
        """Return exactly [origin-512, origin), with the existing window convention."""
        column = self.split.states.index(state)
        context, _ = forecast_window(self._values[:, column], origin, self.split)
        return ForecastRequest(state=state, history=context.tolist()).history

    def observation(self, state: str, position: int, clock: int, delay: int) -> float:
        """Gate every truth lookup by the positional release rule."""
        if not 0 <= position < len(self._values) or position + 1 + delay > clock:
            raise ValueError("Observation has not reached its release position")
        return float(self._values[position, self.split.states.index(state)])
