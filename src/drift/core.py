"""Frozen weekly reference and a finite schedule of Alibi Detect KS comparisons.

Cycle means and residual standard deviations reduce intracycle dependence. They
do not make successive cycles iid; the nominal alpha is not a calibrated time
series false-alert guarantee. See docs/drift.md for the empirical assessment.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
import math
from numbers import Real
import time

import numpy as np
import pandas as pd

FEATURES = ("residual_mean", "residual_std")
STATES = ("NSW", "VIC", "QLD", "SA", "TAS")
DETECTOR = {
    "library": "alibi-detect",
    "version": "0.13.0",
    "method": "KSDrift",
    "alternative": "two-sided",
    "correction": "bonferroni",
    "update_x_ref": None,
}


def identity(value) -> str:
    """Hash canonical finite JSON, including versions and positional boundaries."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_integer(value) -> bool:
    """Require JSON-style integers, excluding booleans and numeric coercion."""
    return type(value) is int  # pylint: disable=unidiomatic-typecheck


def deadline_after(max_seconds) -> float:
    """Enforce the same finite 300-second upper limit for both bounded consumers."""
    if (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, Real)
        or not math.isfinite(max_seconds)
        or not 0 < max_seconds <= 300
    ):
        raise ValueError("max_seconds must be finite, positive and at most 300")
    return time.monotonic() + max_seconds


@dataclass(frozen=True)
class DriftConfig:  # pylint: disable=too-many-instance-attributes
    """One declared validation campaign; intervals use [start, end) positions."""

    train_end: int
    validation_end: int
    cycle_steps: int = 336
    reference_cycles: int = 104
    window_cycles: int = 8
    max_windows: int = 6
    family_alpha: float = 0.05
    states: tuple[str, ...] = STATES
    feature_version: str = "weekly-residual-v1"
    detector_version: str = "alibi-ks-0.13.0-v1"

    def __post_init__(self):
        integers = (
            self.train_end,
            self.validation_end,
            self.cycle_steps,
            self.reference_cycles,
            self.window_cycles,
            self.max_windows,
        )
        if any(not is_integer(value) or value <= 0 for value in integers):
            raise ValueError("boundaries, cycle sizes and budget must be positive integers")
        if self.states != STATES or self.feature_version != "weekly-residual-v1":
            raise ValueError("expected the five configured states and weekly-residual-v1")
        if self.detector_version != "alibi-ks-0.13.0-v1":
            raise ValueError("unknown detector version")
        if self.reference_cycles < 8 or self.window_cycles < 8:
            raise ValueError("at least eight complete cycles per reference and test are required")
        if not 0 < self.family_alpha < 1:
            raise ValueError("family_alpha must be between zero and one")
        if self.reference_end < self.reference_cycles * self.cycle_steps:
            raise ValueError("training block is too short for the reference")
        if self.schedule_start + self.max_windows * self.window_steps > self.validation_end:
            raise ValueError("the complete declared schedule must fit inside validation")

    @property
    def reference_end(self) -> int:
        """Last complete training cycle; no held-out observation is used."""
        return self.train_end // self.cycle_steps * self.cycle_steps

    @property
    def schedule_start(self) -> int:
        """First full cycle wholly inside validation."""
        return -(-self.train_end // self.cycle_steps) * self.cycle_steps

    @property
    def window_steps(self) -> int:
        """Positions in one nonoverlapping scheduled test window."""
        return self.window_cycles * self.cycle_steps

    def bounds(self, window: int) -> tuple[int, int]:
        """Reject unscheduled tests rather than silently renewing the test budget."""
        if not is_integer(window) or not 0 <= window < self.max_windows:
            raise ValueError("window is outside the declared testing budget")
        start = self.schedule_start + window * self.window_steps
        return start, start + self.window_steps


def validate_frame(frame: pd.DataFrame, cfg: DriftConfig) -> None:
    """Reject ambiguous positions, unknown columns and nonnumeric observations.

    Missing rows/columns and NaN values are counted by the window extractor.
    Infinity and booleans are malformed inputs, not observations.
    """
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("expected a demand DataFrame with unique state columns")
    if set(frame.columns) - set(cfg.states):
        raise ValueError("unknown demand state")
    positions = frame.index.to_numpy()
    if positions.dtype.kind not in "iu" or (positions < 0).any():
        raise ValueError("positions must be nonnegative integers")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("positions must be unique and strictly increasing")
    if any(frame[state].dtype.kind not in "fiu" for state in frame.columns):
        raise ValueError("demand values must be numeric, excluding booleans")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("infinite demand values are invalid")


def cycle_features(values: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Two interpretable source-scale features per whole positional cycle."""
    residual = values.reshape(-1, len(template)) - template
    features = np.column_stack((residual.mean(axis=1), residual.std(axis=1)))
    if not np.isfinite(features).all():
        raise ValueError("feature calculation produced nonfinite values")
    return features


@dataclass(frozen=True)
class Reference:
    """Training-derived pattern and features, with content/source identities."""

    config: DriftConfig
    source: dict
    template: np.ndarray
    features: np.ndarray
    sha256: str


def build_reference(train: pd.DataFrame, cfg: DriftConfig, source: dict) -> Reference:
    """Learn the phase template and reference from training positions only."""
    validate_frame(train, cfg)
    if len(train) and train.index[-1] >= cfg.train_end:
        raise ValueError("reference input must contain training positions only")
    if not source or source.get("kind") not in ("fixture", "dvc"):
        raise ValueError("reference requires an explicit fixture or verified DVC identity")
    end = cfg.reference_end
    start = end - cfg.reference_cycles * cfg.cycle_steps
    values = train.reindex(index=range(start, end), columns=cfg.states).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("reference needs complete finite training cycles for all five states")
    cycles = values.reshape(cfg.reference_cycles, cfg.cycle_steps, len(cfg.states))
    template = cycles.mean(axis=0).T.copy()
    features = np.stack(
        [cycle_features(values[:, index], pattern) for index, pattern in enumerate(template)]
    )
    payload = {
        "config": asdict(cfg),
        "detector": DETECTOR,
        "source": source,
        "bounds": [start, end],
        "template": template.tolist(),
        "features": features.tolist(),
    }
    template.setflags(write=False)
    features.setflags(write=False)
    return Reference(cfg, deepcopy(source), template, features, identity(payload))


def ks_compare(reference: np.ndarray, current: np.ndarray, alpha: float) -> dict:
    """Import Alibi only when a real, complete scheduled comparison is needed."""
    from alibi_detect.cd import KSDrift  # pylint: disable=import-outside-toplevel,import-error

    if version("alibi-detect") != DETECTOR["version"]:
        raise RuntimeError("KSDrift requires the declared Alibi Detect version")
    detector = KSDrift(
        reference, p_val=alpha, correction="bonferroni", alternative="two-sided", update_x_ref=None
    )
    result = detector.predict(current, drift_type="batch", return_p_val=True)["data"]
    return {
        "status": "drift" if result["is_drift"] else "no_drift",
        "p_values": np.asarray(result["p_val"], dtype=float).tolist(),
        "ks_distance": np.asarray(result["distance"], dtype=float).tolist(),
        "feature_threshold": float(result["threshold"]),
    }


class DriftMonitor:
    """A bounded campaign with retry identities and serializable tested results.

    Callers persist checkpoint() together with results. An incomplete state may
    be retried before moving on; a completed test accepts only identical values.
    Each state/window can consume at most one preallocated test slot.
    """

    def __init__(self, reference: Reference, campaign: dict, checkpoint: dict | None = None):
        self.reference = reference
        self.campaign = deepcopy(campaign)
        self.campaign_sha256 = identity({"reference": reference.sha256, "campaign": campaign})
        self._tested = {}
        if checkpoint is not None:
            if checkpoint.get("campaign_sha256") != self.campaign_sha256:
                raise ValueError(
                    "checkpoint belongs to a different reference/config/source campaign"
                )
            if checkpoint.get("tested_sha256") != identity(checkpoint["tested"]):
                raise ValueError("checkpoint test ledger is corrupt")
            for key, item in checkpoint["tested"].items():
                number, state = key.split(":")
                reference.config.bounds(int(number))
                if (
                    state not in reference.config.states
                    or item["input_sha256"] != item["result"]["input_sha256"]
                ):
                    raise ValueError("checkpoint contains invalid state/test identities")
            self._tested = deepcopy(checkpoint["tested"])

    def checkpoint(self) -> dict:
        """Return only immutable test identities and results; never raw observations."""
        return {
            "campaign_sha256": self.campaign_sha256,
            "tested": deepcopy(self._tested),
            "tested_sha256": identity(self._tested),
        }

    def _state_result(self, data, window, index):
        cfg = self.reference.config
        state = cfg.states[index]
        key = f"{window}:{state}"
        count = int(np.isfinite(data).sum())
        base = {
            "state": state,
            "reference_cycles": cfg.reference_cycles,
            "expected_positions": cfg.window_steps,
            "observed_positions": count,
            "missing_positions": cfg.window_steps - count,
        }
        if count != cfg.window_steps:
            if key in self._tested:
                raise ValueError("a tested window cannot become incomplete on retry")
            return {**base, "status": "insufficient_data", "test_cycles": 0}
        if key not in self._tested and any(
            int(item.split(":")[0]) > window for item in self._tested if item.endswith(f":{state}")
        ):
            raise ValueError("cannot fill an untested window after a later state window")
        input_hash = identity(data.tolist())
        if key in self._tested:
            if self._tested[key]["input_sha256"] != input_hash:
                raise ValueError("conflicting observation values for an already tested window")
            return deepcopy(self._tested[key]["result"])
        features = cycle_features(data, self.reference.template[index])
        alpha = cfg.family_alpha / (len(cfg.states) * cfg.max_windows)
        return {
            **base,
            **ks_compare(self.reference.features[index], features, alpha),
            "test_cycles": len(features),
            "feature_names": list(FEATURES),
            "reference_feature_mean": self.reference.features[index].mean(axis=0).tolist(),
            "current_feature_mean": features.mean(axis=0).tolist(),
            "input_sha256": input_hash,
        }

    def assess(self, frame: pd.DataFrame, window: int) -> dict:
        """Compare complete state windows, exposing missing data without testing it."""
        cfg = self.reference.config
        start, end = cfg.bounds(window)
        validate_frame(frame, cfg)
        if len(frame) and (frame.index[0] < start or frame.index[-1] >= end):
            raise ValueError("observations must lie inside the scheduled window")
        values = frame.reindex(index=range(start, end), columns=cfg.states)
        results = {}
        # Commit only after all states succeed, so dependency errors permit safe retries.
        tested = deepcopy(self._tested)
        for index, state in enumerate(cfg.states):
            result = self._state_result(values[state].to_numpy(dtype=float), window, index)
            results[state] = result
            if result["status"] != "insufficient_data":
                tested[f"{window}:{state}"] = {
                    "input_sha256": result["input_sha256"],
                    "result": deepcopy(result),
                }
        self._tested = tested
        return {
            "window": window,
            "bounds": [start, end],
            "reference_sha256": self.reference.sha256,
            "config_sha256": identity(asdict(cfg)),
            "campaign_sha256": self.campaign_sha256,
            "states": results,
        }
