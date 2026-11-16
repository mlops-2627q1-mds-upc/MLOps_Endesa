"""Explicit synthetic predictor through the production API lifecycle and metrics."""

import json
import os
from pathlib import Path
import time

from src.api.app import create_app as forecast_app


class ScenarioPredictor:  # pylint: disable=too-few-public-methods
    """Read a bounded delay from a run-owned file; never load model weights."""

    def __init__(self, fault: Path | None = None):
        self.fault = fault

    def predict(self, history):
        """Produce a constant synthetic interval with a fictional model identity."""
        delay = 0 if self.fault is None else json.loads(self.fault.read_text())["delay_seconds"]
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not 0 <= delay <= 1:
            raise ValueError("Fixture delay must be between zero and one second")
        time.sleep(delay)
        return {
            "p10": [history[-1] - 1] * 48,
            "p50": [history[-1]] * 48,
            "p90": [history[-1] + 1] * 48,
            "model_id": "chronos-t5-small-fine-tuned",
            "model_revision": "local-sha256:" + "0" * 64,
            "checkpoint": {"sha256": "0" * 64, "training_run_id": "0" * 32},
            "value_scale": "source",
            "context_length": 512,
            "horizon": 48,
            "settings": {
                "num_samples": 20,
                "seed": 7,
                "device": "cpu",
                "cpu_threads": 2,
                "dtype": "float32",
                "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
                "quantile_method": "linear",
                "point_statistic": "median",
            },
        }


def create_app():
    """Use only the explicitly mounted synthetic fault file in an owned container."""
    fault = Path(os.environ["ENDESA_SCENARIO_FAULT"])
    return forecast_app(predictor_factory=lambda _settings: ScenarioPredictor(fault))
