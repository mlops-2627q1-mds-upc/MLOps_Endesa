"""Serve a bounded quality JSON report without loading a model or replay source."""

import hashlib
import json
import math
import os
from pathlib import Path
import time

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily

STATES = frozenset({"NSW", "VIC", "QLD", "SA", "TAS"})
STATUS_CODES = {
    "no_forecasts": 0,
    "pending_observations": 1,
    "insufficient_samples": 2,
    "sufficient": 3,
}
COUNTS = (
    "expected_forecasts",
    "forecast_count",
    "matched_forecast_count",
    "forecast_steps",
    "matched_steps",
    "pending_steps",
    "unique_matched_targets",
)
ROLE_METRICS = {
    "selected": ("mae", "mase", "empirical_coverage", "mean_interval_width"),
    "seasonal_lag48": ("mae", "mase"),
}
MAX_REPORT_BYTES = 1_048_576


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _validate_group(group, minimum):
    if group["sample_status"] not in STATUS_CODES or any(
        not _integer(group[name]) for name in COUNTS
    ):
        raise ValueError("Invalid quality sample status/counts")
    count = group["matched_steps"]
    status = (
        "no_forecasts"
        if group["forecast_steps"] == 0
        else "pending_observations"
        if count == 0
        else "insufficient_samples"
        if count < minimum
        else "sufficient"
    )
    if (
        group["sample_status"] != status
        or group["forecast_steps"] != count + group["pending_steps"]
        or group["matched_forecast_count"] > group["forecast_count"]
        or group["forecast_count"] > group["expected_forecasts"]
        or group["unique_matched_targets"] > count
    ):
        raise ValueError("Inconsistent quality sample counts")
    for role, names in ROLE_METRICS.items():
        for name in names:
            value = group[role][name]
            if status != "sufficient":
                if value is not None:
                    raise ValueError("Insufficient groups must have null metrics")
            elif not _number(value) or value < 0 or (name == "empirical_coverage" and value > 1):
                raise ValueError("Invalid quality metric value")


def _validate_report(report):
    if report["schema_version"] != 1:
        raise ValueError("Unsupported quality report schema")
    metadata = report["metadata"]
    if (
        _hash(metadata["config_json"]) != metadata["config_sha256"]
        or _hash(metadata["source_reference"]) != metadata["source_sha256"]
    ):
        raise ValueError("Quality report identity mismatch")
    config = json.loads(metadata["config_json"], parse_constant=_reject_constant)
    if (
        config["mode"] != "historical_replay"
        or config["value_scale"] != "source"
        or config["model_identity"] != metadata["model_identity"]
    ):
        raise ValueError("Conflicting quality model/settings identity or report mode")
    states = config["states"]
    minimum = metadata["policy"]["minimum_samples"]
    if (
        not states
        or len(set(states)) != len(states)
        or not set(states) <= STATES
        or not _integer(minimum)
        or minimum == 0
    ):
        raise ValueError("Invalid quality state set/sample policy")
    for collection, horizons in (("by_state", ("all",)), ("by_state_horizon", range(1, 49))):
        expected = {(state, str(horizon)) for state in states for horizon in horizons}
        observed = set()
        for group in report[collection]:
            if collection == "by_state" and "horizon" in group:
                raise ValueError("State summaries must not declare a horizon")
            key = (
                group["state"],
                str(group["horizon"]) if collection == "by_state_horizon" else "all",
            )
            if key not in expected or key in observed:
                raise ValueError("Unknown or duplicate quality state/horizon")
            _validate_group(group, minimum)
            observed.add(key)
        if observed != expected:
            raise ValueError("Missing quality state/horizon groups")
    return report


def _family(name, description, value=None, labels=None):
    return GaugeMetricFamily("endesa_quality_" + name, description, value=value, labels=labels)


class QualityCollector:  # pylint: disable=too-few-public-methods
    """Publish current valid measurements; never retain metrics after report failure."""

    def __init__(self, path: Path, max_age_seconds: int = 300):
        if not _integer(max_age_seconds) or not 1 <= max_age_seconds <= 86400:
            raise ValueError("Report max age must be an integer in 1..86400 seconds")
        self.path = path
        self.max_age_seconds = max_age_seconds

    def collect(self):
        """Read one bounded file version and expose explicit availability and freshness."""
        present, valid, fresh, age, report = 0, 0, 0, None, None
        try:
            with self.path.open("rb") as stream:
                present = 1
                stat = os.fstat(stream.fileno())
                age = time.time() - stat.st_mtime
                raw = stream.read(MAX_REPORT_BYTES + 1)
            if len(raw) > MAX_REPORT_BYTES or age < 0:
                raise ValueError("Report is oversized or has a future modification time")
            report = _validate_report(json.loads(raw, parse_constant=_reject_constant))
            valid = 1
            fresh = int(age <= self.max_age_seconds)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            OverflowError,
            AttributeError,
            RecursionError,
        ):
            pass
        available = int(bool(valid and fresh))
        for name, value in (
            ("report_present", present),
            ("report_valid", valid),
            ("report_fresh", fresh),
            ("report_available", available),
            ("report_max_age_seconds", self.max_age_seconds),
        ):
            yield _family(
                name, "Quality report export availability; historical replay only.", value
            )
        if age is not None and math.isfinite(age) and age >= 0:
            yield _family("report_age_seconds", "Age of report refresh, not observation age.", age)
        if not available:
            return
        yield _family(
            "minimum_samples",
            "Required matched rows per group.",
            report["metadata"]["policy"]["minimum_samples"],
        )
        families = {
            name: _family(
                name, "Historical replay quality sample count.", labels=("state", "horizon")
            )
            for name in COUNTS
        }
        families["sample_status"] = _family(
            "sample_status",
            "0 no forecasts; 1 pending; 2 insufficient; 3 sufficient.",
            labels=("state", "horizon"),
        )
        for name in ROLE_METRICS["selected"]:
            families[name] = _family(
                name,
                "Descriptive historical replay quality; source scale.",
                labels=("state", "horizon", "model_role"),
            )
        for group in report["by_state"] + report["by_state_horizon"]:
            labels = [group["state"], str(group.get("horizon", "all"))]
            for name in COUNTS:
                families[name].add_metric(labels, group[name])
            families["sample_status"].add_metric(labels, STATUS_CODES[group["sample_status"]])
            for role, names in ROLE_METRICS.items():
                for name in names:
                    if group[role][name] is not None:
                        families[name].add_metric([*labels, role], group[role][name])
        yield from families.values()


def create_app(path: Path | None = None, max_age_seconds: int | None = None) -> FastAPI:
    """Create the standalone exporter, configured by path and freshness environment."""
    path = path or Path(os.environ["ENDESA_QUALITY_REPORT"])
    max_age_seconds = (
        int(os.environ.get("ENDESA_QUALITY_MAX_AGE_SECONDS", "300"))
        if max_age_seconds is None
        else max_age_seconds
    )
    registry = CollectorRegistry()
    registry.register(QualityCollector(path, max_age_seconds))
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    return app
