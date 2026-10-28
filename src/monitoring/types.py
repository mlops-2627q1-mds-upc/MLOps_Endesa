"""Public positional replay records and immutable run configuration."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from urllib.parse import urlsplit

from src.api.schemas import CheckpointIdentity, ForecastResponse, PredictionSettings


def canonical_json(value) -> str:
    """Serialize identities consistently, rejecting nonfinite values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(value: str | bytes) -> str:
    """Hash exact bytes or UTF-8 text."""
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


@dataclass(frozen=True)
class ModelIdentity:
    """Public artifact identity plus the predictor's executed settings."""

    model_id: str
    model_revision: str
    checkpoint_sha256: str
    training_run_id: str
    settings_json: str

    def __post_init__(self):
        CheckpointIdentity(sha256=self.checkpoint_sha256, training_run_id=self.training_run_id)
        if self.model_id != "chronos-t5-small-fine-tuned":
            raise ValueError("Replay requires the selected-model API identity")
        if self.model_revision != f"local-sha256:{self.checkpoint_sha256}":
            raise ValueError("Revision must match the declared checkpoint")
        settings = PredictionSettings.model_validate(json.loads(self.settings_json))
        object.__setattr__(self, "settings_json", canonical_json(settings.model_dump()))

    @classmethod
    def from_api(cls, value: dict) -> "ModelIdentity":
        """Read the identity fields of a declared API response or identity file."""
        checkpoint = CheckpointIdentity.model_validate(value["checkpoint"])
        return cls(
            value["model_id"],
            value["model_revision"],
            checkpoint.sha256,
            checkpoint.training_run_id,
            canonical_json(value["settings"]),
        )

    @classmethod
    def from_response(cls, response: ForecastResponse) -> "ModelIdentity":
        """Extract validated response identity without retaining quantiles."""
        return cls.from_api(response.model_dump())


@dataclass(frozen=True)
class ReplayConfig:
    """A journal binds one endpoint, positional range, model and retry policy."""

    origin_start: int
    origin_stop: int
    states: tuple[str, ...]
    model_identity: ModelIdentity
    endpoint: str = "http://127.0.0.1:8000"
    origin_stride: int = 48
    observation_delay: int = 0
    max_attempts: int = 3
    timeout_seconds: float = 10.0
    retry_delay_seconds: float = 0.1

    def __post_init__(self):
        for name in (
            "origin_start",
            "origin_stop",
            "origin_stride",
            "observation_delay",
            "max_attempts",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 512 <= self.origin_start < self.origin_stop or self.origin_stride <= 0:
            raise ValueError("Declare a positive origin range with a full 512-step history")
        if self.observation_delay < 0 or not 1 <= self.max_attempts <= 10:
            raise ValueError("Delay must be nonnegative and max_attempts must be 1..10")
        if (
            not isinstance(self.states, tuple)
            or not self.states
            or len(set(self.states)) != len(self.states)
        ):
            raise ValueError("Declare distinct states as an ordered tuple")
        if not set(self.states) <= {"NSW", "VIC", "QLD", "SA", "TAS"}:
            raise ValueError("Unsupported state")
        for name, upper, zero_allowed in (
            ("timeout_seconds", 60, False),
            ("retry_delay_seconds", 5, True),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= upper
                or (value == 0 and not zero_allowed)
            ):
                raise ValueError(f"{name} is outside its bounded range")
        endpoint = urlsplit(self.endpoint)
        if (
            endpoint.scheme not in ("http", "https")
            or not endpoint.hostname
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("Endpoint must be an HTTP base URL without credentials or query")
        object.__setattr__(self, "endpoint", self.endpoint.rstrip("/"))

    @property
    def origins(self) -> range:
        """Configured origins in chronological order."""
        return range(self.origin_start, self.origin_stop, self.origin_stride)

    @property
    def expected_forecasts(self) -> int:
        """One request per origin and state."""
        return len(self.origins) * len(self.states)

    def request_at(self, index: int) -> tuple[str, int]:
        """Resolve an origin-major durable request cursor."""
        return self.states[index % len(self.states)], self.origins[index // len(self.states)]

    def to_json(self) -> str:
        """Include protocol constants in the run identity."""
        return canonical_json(
            {
                **asdict(self),
                "mode": "historical_replay",
                "context_length": 512,
                "horizon": 48,
                "release_rule": "target+1+delay",
                "value_scale": "source",
            }
        )


@dataclass(frozen=True)
class RunInfo:
    """Identity shared by every exported row in a journal."""

    run_id: str
    config_sha256: str
    source_sha256: str
    source_reference: str
    config_json: str
    model_identity: ModelIdentity


@dataclass(frozen=True)
class ReplayCounts:
    """Counts distinguish unissued forecasts from unreleased committed steps."""

    expected_forecasts: int
    completed_forecasts: int
    forecast_steps: int
    matched_steps: int
    pending_steps: int
    failed_attempts: int
    clock_position: int
    next_origin: int | None
    complete: bool


@dataclass(frozen=True)
class ForecastStep:
    """One of 48 steps; horizon is 1-based and target positions are 0-based."""

    id: int
    forecast_id: str
    state: str
    origin: int
    horizon: int
    target_position: int
    p10: float
    p50: float
    p90: float
    seasonal_lag48: float
    model_identity: ModelIdentity
    source_reference: str
    input_sha256: str


@dataclass(frozen=True)
class MatchedObservation(ForecastStep):
    """A released forecast/observation join with its own monotonic match ID."""

    forecast_step_id: int
    observed_value: float
    released_position: int


@dataclass(frozen=True)
class ReleasedObservation:
    """A unique state/position sample, independent of forecast overlap."""

    id: int
    state: str
    position: int
    value: float
    released_position: int
    source_reference: str
    config_sha256: str
