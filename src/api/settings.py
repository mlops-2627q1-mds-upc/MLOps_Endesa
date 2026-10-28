"""Explicit startup settings; no environment reads during requests."""

from dataclasses import dataclass
import os
from pathlib import Path

from src.modeling.predict import SELECTED_CHECKPOINT
from src.splits import PARAMS_PATH


@dataclass(frozen=True)
class Settings:
    """Paths and CPU budget for one application lifecycle."""

    params_path: Path = PARAMS_PATH
    checkpoint: Path = SELECTED_CHECKPOINT
    cpu_threads: int = 2

    def __post_init__(self):
        if not isinstance(self.params_path, Path) or not isinstance(self.checkpoint, Path):
            raise TypeError("params_path and checkpoint must be local Paths")
        if type(self.cpu_threads) is not int or self.cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive integer")

    @classmethod
    def from_env(cls) -> "Settings":
        """Capture the serving environment once when creating the application."""
        try:
            cpu_threads = int(os.environ.get("FORECAST_CPU_THREADS", "2"))
        except ValueError as error:
            raise ValueError("FORECAST_CPU_THREADS must be a positive integer") from error
        return cls(
            params_path=Path(os.environ.get("FORECAST_PARAMS_PATH", str(PARAMS_PATH))),
            checkpoint=Path(os.environ.get("FORECAST_CHECKPOINT", str(SELECTED_CHECKPOINT))),
            cpu_threads=cpu_threads,
        )
