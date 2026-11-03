"""Explicit Uvicorn fixture factory, mounted only by the smoke harness."""

from src.api.app import create_app as create_forecast_app


class FixturePredictor:  # pylint: disable=too-few-public-methods
    """Return recognizable synthetic forecasts without importing a model runtime."""

    def predict(self, history: list[float]) -> dict:
        """Echo the last observation into a fixed interval for transport checks."""
        return {
            "p10": [history[-1] - 1.0] * 48,
            "p50": [history[-1]] * 48,
            "p90": [history[-1] + 1.0] * 48,
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
    """Enter the production lifecycle with an injected fixture predictor."""
    return create_forecast_app(predictor_factory=lambda _settings: FixturePredictor())
