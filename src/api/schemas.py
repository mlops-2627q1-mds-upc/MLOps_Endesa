"""Strict source-scale request and selected-artifact response contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

State = Literal["NSW", "VIC", "QLD", "SA", "TAS"]
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
# The inference module consumes float32; reject overflow before calling the model.
HistoryNumber = Annotated[
    FiniteNumber, Field(ge=-float.fromhex("0x1.fffffep+127"), le=float.fromhex("0x1.fffffep+127"))
]
Quantiles = Annotated[list[FiniteNumber], Field(min_length=48, max_length=48)]


class ForecastRequest(BaseModel):
    """The state labels a chronological history; it is not a model feature."""

    model_config = ConfigDict(strict=True, extra="forbid")

    state: State
    history: Annotated[list[HistoryNumber], Field(min_length=512, max_length=512)]


class CheckpointIdentity(BaseModel):
    """Public artifact identity, excluding internal paths and loader metadata."""

    model_config = ConfigDict(strict=True)

    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    training_run_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]


class SamplingSettings(BaseModel):
    """Sampling values reported by the executed predictor."""

    model_config = ConfigDict(strict=True)

    temperature: Annotated[FiniteNumber, Field(gt=0)]
    top_k: Annotated[int, Field(ge=0)]
    top_p: Annotated[FiniteNumber, Field(gt=0, le=1)]


class PredictionSettings(BaseModel):
    """Executed inference configuration, not additional request parameters."""

    model_config = ConfigDict(strict=True)

    num_samples: Annotated[int, Field(gt=0)]
    seed: Annotated[int, Field(ge=0, lt=2**64)]
    device: Literal["cpu"]
    cpu_threads: Annotated[int, Field(gt=0)]
    dtype: Literal["float32"]
    sampling: SamplingSettings
    quantile_method: Literal["linear"]
    point_statistic: Literal["median"]


class ForecastResponse(BaseModel):
    """48 positional quantiles in the original source scale."""

    model_config = ConfigDict(strict=True)

    state: State
    p10: Quantiles
    p50: Quantiles
    p90: Quantiles
    model_id: Literal["chronos-t5-small-fine-tuned"]
    model_revision: Annotated[str, Field(pattern=r"^local-sha256:[0-9a-f]{64}$")]
    checkpoint: CheckpointIdentity
    value_scale: Literal["source"]
    context_length: Literal[512]
    horizon: Literal[48]
    settings: PredictionSettings

    @model_validator(mode="after")
    def check_prediction(self) -> "ForecastResponse":
        """Reject inconsistent identity or unordered intervals at the HTTP boundary."""
        if self.model_revision != f"local-sha256:{self.checkpoint.sha256}":
            raise ValueError("Model revision does not match checkpoint identity")
        if any(not low <= mid <= high for low, mid, high in zip(self.p10, self.p50, self.p90)):
            raise ValueError("Forecast quantiles must be ordered")
        return self
