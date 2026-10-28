"""Bounded forecast-quality reports over released, public replay records."""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

from src.evaluation import MASE_LAG, mase_scale
from src.monitoring import ForecastStep, JournalReader, ModelIdentity, ReplayConfig, RunInfo
from src.monitoring.source import ReplaySource


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class QualityPolicy:
    """Declare the sample threshold and the maximum number of exported step rows."""

    minimum_samples: int = 30
    max_steps: int = 100_000
    page_size: int = 1000

    def __post_init__(self):
        for name, upper in (
            ("minimum_samples", 1_000_000),
            ("max_steps", 1_000_000),
            ("page_size", 10_000),
        ):
            if not _integer(getattr(self, name)) or not 1 <= getattr(self, name) <= upper:
                raise ValueError(f"{name} must be an integer in 1..{upper}")


def _digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _configuration(info: RunInfo) -> ReplayConfig:
    value = json.loads(info.config_json)
    if (
        _digest(info.config_json) != info.config_sha256
        or _digest(info.source_reference) != info.source_sha256
        or value.get("mode") != "historical_replay"
        or value.get("value_scale") != "source"
        or value.get("context_length") != 512
        or value.get("horizon") != 48
        or value.get("release_rule") != "target+1+delay"
    ):
        raise ValueError("Journal identity or positional protocol does not match")
    arguments = {field.name: value[field.name] for field in fields(ReplayConfig)}
    arguments["states"] = tuple(arguments["states"])
    arguments["model_identity"] = ModelIdentity(**arguments["model_identity"])
    config = ReplayConfig(**arguments)
    if config.model_identity != info.model_identity:
        raise ValueError("Conflicting model/settings identity")
    return config


def _training_scales(info, config, directory, params_path):
    source = ReplaySource.from_validation(directory, params_path)
    if (source.source_sha256, source.reference) != (info.source_sha256, info.source_reference):
        raise ValueError("Local train/validation source identity does not match the journal")
    source.validate_config(config)
    # Parse the exact verified bytes, avoiding a second pathname read during decoding.
    training_bytes = (directory / "train.parquet").read_bytes()
    manifest = json.loads(source.reference)
    if _digest(training_bytes) != manifest["file_hashes"]["train.parquet"]:
        raise ValueError("Training source changed after identity verification")
    training = pd.read_parquet(BytesIO(training_bytes))
    scales = {state: mase_scale(training[state].to_numpy(), MASE_LAG) for state in config.states}
    return scales, source.split.train_end


def _read_rows(method, policy):
    rows, cursor = [], 0
    while True:
        page = method(after_id=cursor, limit=policy.page_size)
        if not page:
            return rows
        if len(page) > policy.page_size or len(rows) + len(page) > policy.max_steps:
            raise ValueError("Journal export exceeds max_steps; refusing a partial report")
        ids = [row.id for row in page]
        if any(not _integer(value) or value <= cursor for value in ids) or ids != sorted(ids):
            raise ValueError("Journal pagination did not advance monotonically")
        rows.extend(page)
        cursor = ids[-1]


def _valid_step(row, info, config, clock):
    for name in ("id", "origin", "horizon", "target_position"):
        if not _integer(getattr(row, name)):
            raise ValueError("Forecast positions and IDs must be integers")
    if (
        row.id <= 0
        or row.state not in config.states
        or row.origin not in config.origins
        or row.origin > clock
        or not 1 <= row.horizon <= 48
        or row.target_position != row.origin + row.horizon - 1
        or not isinstance(row.forecast_id, str)
        or not row.forecast_id
    ):
        raise ValueError("Forecast row does not follow the configured state/origin/horizon")
    if row.model_identity != info.model_identity or row.source_reference != info.source_reference:
        raise ValueError("Conflicting model/settings/source identity in forecast rows")
    if (
        not isinstance(row.input_sha256, str)
        or len(row.input_sha256) != 64
        or any(char not in "0123456789abcdef" for char in row.input_sha256)
    ):
        raise ValueError("Forecast input identity must be a SHA-256 digest")
    values = (row.p10, row.p50, row.p90, row.seasonal_lag48)
    if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
        raise ValueError("Forecast and baseline values must be finite")
    if not row.p10 <= row.p50 <= row.p90:
        raise ValueError("Forecast quantiles are not ordered")


def _unique_steps(rows, info, config, counts):
    by_id, by_key, forecasts, requests = {}, {}, {}, {}
    for row in rows:
        _valid_step(row, info, config, counts.clock_position)
        key = (row.forecast_id, row.horizon)
        value = {
            field.name: getattr(row, field.name)
            for field in fields(ForecastStep)
            if field.name != "id"
        }
        if row.id in by_id and by_id[row.id] != row:
            raise ValueError("Conflicting forecast step ID")
        if key in by_key and value != by_key[key]:
            raise ValueError("Conflicting duplicate forecast values")
        by_id[row.id], by_key[key] = row, value
        identity = (row.state, row.origin, row.input_sha256)
        if row.forecast_id in forecasts and forecasts[row.forecast_id] != identity:
            raise ValueError("Conflicting forecast identity across horizons")
        forecasts[row.forecast_id] = identity
        request = (row.state, row.origin)
        if request in requests and requests[request] != row.forecast_id:
            raise ValueError("Conflicting forecasts for the same state/origin")
        requests[request] = row.forecast_id
    if len(by_key) != counts.forecast_steps or len(forecasts) != counts.completed_forecasts:
        raise ValueError("Forecast rows do not match snapshot counts")
    if len(by_key) != 48 * len(forecasts):
        raise ValueError("A committed forecast must include all 48 horizons")
    expected_requests = {config.request_at(index) for index in range(counts.completed_forecasts)}
    if set(requests) != expected_requests:
        raise ValueError("Forecasts do not follow the committed chronological prefix")
    return by_id, by_key


def _unique_matches(rows, steps, info, config, counts):
    by_id, by_key, observations = {}, {}, {}
    for row in rows:
        _valid_step(row, info, config, counts.clock_position)
        if not _integer(row.forecast_step_id) or row.forecast_step_id not in steps:
            raise ValueError("Matched observation has no committed forecast step")
        step = steps[row.forecast_step_id]
        if any(
            getattr(row, field.name) != getattr(step, field.name)
            for field in fields(ForecastStep)
            if field.name != "id"
        ):
            raise ValueError("Conflicting join and forecast values/identities")
        if isinstance(row.observed_value, bool) or not math.isfinite(row.observed_value):
            raise ValueError("Observed values must be finite")
        if (
            not _integer(row.released_position)
            or row.released_position != row.target_position + 1 + config.observation_delay
            or row.released_position > counts.clock_position
        ):
            raise ValueError("Observation has not reached its configured release position")
        key = (row.forecast_id, row.horizon)
        if row.id in by_id and by_id[row.id] != row:
            raise ValueError("Conflicting match ID")
        if key in by_key and (row.observed_value, row.released_position) != (
            by_key[key].observed_value,
            by_key[key].released_position,
        ):
            raise ValueError("Conflicting duplicate observation join")
        target = (row.state, row.target_position)
        truth = (row.observed_value, row.released_position)
        if target in observations and observations[target] != truth:
            raise ValueError("Conflicting observations for an overlapping target")
        by_id[row.id], by_key[key], observations[target] = row, row, truth
    released = {
        (step.forecast_id, step.horizon)
        for step in steps.values()
        if step.target_position + 1 + config.observation_delay <= counts.clock_position
    }
    if len(by_key) != counts.matched_steps or set(by_key) != released:
        raise ValueError("Matched rows do not match the committed release snapshot")
    return by_key


def _validate_counts(counts, config, policy):
    for name, value in asdict(counts).items():
        if name not in ("next_origin", "complete") and (not _integer(value) or value < 0):
            raise ValueError("Journal counts must be nonnegative integers")
    completed = counts.completed_forecasts
    if counts.forecast_steps > policy.max_steps or counts.matched_steps > policy.max_steps:
        raise ValueError("Journal exceeds max_steps; refusing a partial report")
    if (
        counts.expected_forecasts != config.expected_forecasts
        or completed > config.expected_forecasts
        or counts.forecast_steps != completed * 48
        or not 0 <= counts.matched_steps <= counts.forecast_steps
        or counts.pending_steps != counts.forecast_steps - counts.matched_steps
        or counts.clock_position < config.origin_start
        or counts.next_origin
        != (config.request_at(completed)[1] if completed < config.expected_forecasts else None)
        or not isinstance(counts.complete, bool)
        or counts.complete
        != (completed == config.expected_forecasts and counts.pending_steps == 0)
    ):
        raise ValueError("Inconsistent replay counts/configuration")


def _finite_mean(values):
    result = float(np.mean(values))
    if not math.isfinite(result):
        raise ValueError("Metric arithmetic produced a nonfinite result")
    return result


def _group(steps, matches, scales, minimum_samples, expected_forecasts):
    count = len(matches)
    status = (
        "no_forecasts"
        if not steps
        else "pending_observations"
        if not count
        else "insufficient_samples"
        if count < minimum_samples
        else "sufficient"
    )
    selected = {"mae": None, "mase": None, "empirical_coverage": None, "mean_interval_width": None}
    baseline = {"mae": None, "mase": None}
    if status == "sufficient":
        for role, output in (("p50", selected), ("seasonal_lag48", baseline)):
            errors = [abs(getattr(row, role) - row.observed_value) for row in matches]
            output["mae"] = _finite_mean(errors)
            output["mase"] = _finite_mean([output["mae"] / scales[matches[0].state]])
        selected["empirical_coverage"] = _finite_mean(
            [row.p10 <= row.observed_value <= row.p90 for row in matches]
        )
        selected["mean_interval_width"] = _finite_mean([row.p90 - row.p10 for row in matches])
    return {
        "sample_status": status,
        "expected_forecasts": expected_forecasts,
        "forecast_count": len({row["forecast_id"] for row in steps}),
        "matched_forecast_count": len({row.forecast_id for row in matches}),
        "forecast_steps": len(steps),
        "matched_steps": count,
        "pending_steps": len(steps) - count,
        "unique_matched_targets": len({(row.state, row.target_position) for row in matches}),
        "selected": selected,
        "seasonal_lag48": baseline,
    }


def _summarize(info, config, counts, steps, matches, scales, train_end, policy):
    _validate_counts(counts, config, policy)
    by_id, unique_steps = _unique_steps(steps, info, config, counts)
    unique_matches = _unique_matches(matches, by_id, info, config, counts)
    state_steps, horizon_steps, state_matches, horizon_matches = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for row in unique_steps.values():
        state_steps[row["state"]].append(row)
        horizon_steps[row["state"], row["horizon"]].append(row)
    for row in unique_matches.values():
        state_matches[row.state].append(row)
        horizon_matches[row.state, row.horizon].append(row)
    per_state, per_horizon = [], []
    for state in config.states:
        per_state.append(
            {
                "state": state,
                **_group(
                    state_steps[state],
                    state_matches[state],
                    scales,
                    policy.minimum_samples,
                    len(config.origins),
                ),
            }
        )
        for horizon in range(1, 49):
            per_horizon.append(
                {
                    "state": state,
                    "horizon": horizon,
                    **_group(
                        horizon_steps[state, horizon],
                        horizon_matches[state, horizon],
                        scales,
                        policy.minimum_samples,
                        len(config.origins),
                    ),
                }
            )
    return {
        "schema_version": 1,
        "metadata": {
            **asdict(info),
            "policy": asdict(policy),
            "origin_interval": {
                "start": config.origin_start,
                "stop": config.origin_stop,
                "stride": config.origin_stride,
                "count": len(config.origins),
            },
            "target_interval": {"start": config.origin_start, "stop": config.origins[-1] + 48},
            "training_scale": {"lag": MASE_LAG, "start": 0, "stop": train_end, "by_state": scales},
            "coverage_interpretation": "descriptive inclusive p10-p90 coverage; no calibration SLO",
            "overlap_interpretation": "forecast errors include correlated overlapping targets",
        },
        "replay_counts": asdict(counts),
        "by_state": per_state,
        "by_state_horizon": per_horizon,
    }


def quality_report(
    reader: JournalReader,
    directory: Path,
    params_path: Path,
    policy: QualityPolicy | None = None,
) -> dict:
    """Verify source identity and score one complete, bounded committed read snapshot.

    Training/validation files establish identity and the training scale. All scored
    actuals come exclusively from released journal joins, never those local files.
    """
    policy = policy or QualityPolicy()
    info = reader.run_info()
    config = _configuration(info)
    scales, train_end = _training_scales(info, config, directory, params_path)
    with reader.snapshot():
        if reader.run_info() != info:
            raise ValueError("Journal identity changed while verifying the local source")
        counts = reader.counts()
        _validate_counts(counts, config, policy)
        steps = _read_rows(reader.forecast_steps, policy)
        matches = _read_rows(reader.matched_observations, policy)
    return _summarize(info, config, counts, steps, matches, scales, train_end, policy)


def main(argv: list[str] | None = None) -> int:
    """Print a deterministic JSON summary, or fail without emitting partial metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--minimum-samples", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--page-size", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        policy = QualityPolicy(args.minimum_samples, args.max_steps, args.page_size)
        with JournalReader(args.journal) as reader:
            report = quality_report(reader, args.data_dir, args.params, policy)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, OverflowError, sqlite3.Error) as exc:
        print(f"Forecast quality report failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
