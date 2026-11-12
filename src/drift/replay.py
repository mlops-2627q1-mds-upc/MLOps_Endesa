"""Consume #81's public unique released observations, never forecast histories."""

import math
from numbers import Real
import time

import pandas as pd

from src.drift.core import DriftMonitor, deadline_after, is_integer


def _validate_observation(observation, info, cfg):
    if (
        observation.config_sha256 != info.config_sha256
        or observation.source_reference != info.source_reference
    ):
        raise ValueError("released observation source/config identity differs")
    if not is_integer(observation.position) or not is_integer(observation.released_position):
        raise ValueError("released positions must be integers")
    if (
        observation.state not in cfg.states
        or not cfg.train_end <= observation.position < cfg.validation_end
        or observation.released_position <= observation.position
    ):
        raise ValueError("released observation has invalid state/position")
    if (
        isinstance(observation.value, bool)
        or not isinstance(observation.value, Real)
        or not math.isfinite(observation.value)
    ):
        raise ValueError("released observation has invalid value")


def _released_frame(reader, cfg, info, limits):
    max_pages, deadline = limits
    counts = reader.counts()
    if not is_integer(counts.clock_position) or counts.clock_position < 0:
        raise ValueError("replay clock must be a nonnegative integer")
    cursor, rows, seen = 0, [], set()
    for _ in range(max_pages):
        if time.monotonic() > deadline:
            raise TimeoutError("bounded replay drift read exceeded max_seconds")
        page = reader.released_observations(after_id=cursor, limit=1000)
        if not page:
            break
        for observation in page:
            if not is_integer(observation.id) or observation.id <= cursor:
                raise ValueError("released observation IDs must increase across pages")
            cursor = observation.id
            _validate_observation(observation, info, cfg)
            if observation.released_position > counts.clock_position:
                continue
            key = (observation.state, observation.position)
            if key in seen:
                raise ValueError("released observations must be unique by state and position")
            seen.add(key)
            rows.append((observation.position, observation.state, observation.value))
    else:
        raise ValueError("released observation read exceeded the declared pagination budget")
    if rows:
        frame = (
            pd.DataFrame(rows, columns=["position", "state", "value"])
            .pivot(index="position", columns="state", values="value")
            .sort_index()
        )
    else:
        frame = pd.DataFrame(index=pd.Index([], dtype="int64"))
    return frame, counts, cursor, len(rows)


def _check_snapshot(checkpoint, snapshot):
    if checkpoint is None:
        return
    if not isinstance(checkpoint, dict):
        raise TypeError("replay checkpoint must be a dictionary")
    if "replay_snapshot" not in checkpoint:
        # Ledgers without snapshot metadata rely on the CLI's saved report clock.
        return
    previous = checkpoint["replay_snapshot"]
    if (
        not isinstance(previous, dict)
        or not isinstance(previous.get("run_id"), str)
        or not previous["run_id"]
        or any(
            not is_integer(previous.get(key)) or previous[key] < 0
            for key in ("clock_position", "last_scanned_id")
        )
    ):
        raise ValueError("replay checkpoint snapshot is corrupt")
    if previous["run_id"] != snapshot["run_id"]:
        raise ValueError("replay checkpoint belongs to a different run")
    if snapshot["clock_position"] < previous["clock_position"]:
        raise ValueError("replay clock moved backwards from the saved checkpoint")
    if snapshot["last_scanned_id"] < previous["last_scanned_id"]:
        raise ValueError("replay observation cursor moved backwards from the saved checkpoint")


def assess_replay(
    reader, reference, source, checkpoint=None, *, max_pages=128, max_seconds=120
) -> dict:
    """Assess a bounded committed clock snapshot from a read-only JournalReader.

    `source` is the public ReplaySource created from the same verified artifacts
    as `reference`. Exact reader/source identities are required before paging.
    No future demand values are taken from that source; only released rows enter
    the monitored windows. Repeated calls reuse the returned checkpoint, which
    binds the run, clock and observation cursor even before any test completes.
    """
    deadline = deadline_after(max_seconds)
    cfg = reference.config
    if not is_integer(max_pages) or not 1 <= max_pages <= 128:
        raise ValueError("max_pages must be 1..128")
    with reader.snapshot():
        info = reader.run_info()
        if (
            info.source_sha256 != source.source_sha256
            or info.source_reference != source.reference
            or reference.source.get("replay_source_sha256") != source.source_sha256
            or reference.source.get("replay_source_reference") != source.reference
        ):
            raise ValueError("replay journal and verified reference source differ")
        frame, counts, cursor, row_count = _released_frame(
            reader, cfg, info, (max_pages, deadline)
        )
        snapshot = {
            "run_id": info.run_id,
            "clock_position": counts.clock_position,
            "last_scanned_id": cursor,
        }
        _check_snapshot(checkpoint, snapshot)
    campaign = {
        "kind": "historical-replay",
        "run_id": info.run_id,
        "source_sha256": info.source_sha256,
        "source_reference": info.source_reference,
        "replay_config_sha256": info.config_sha256,
    }
    monitor = DriftMonitor(reference, campaign, checkpoint)
    windows = []
    for window in range(cfg.max_windows):
        if time.monotonic() > deadline:
            raise TimeoutError("bounded replay drift assessment exceeded max_seconds")
        start, end = cfg.bounds(window)
        # Future windows have explicit insufficient-data counts without testing.
        result = monitor.assess(frame.loc[start : end - 1], window)
        result["window_end_reached"] = end <= counts.clock_position
        windows.append(result)
    return {
        "schema_version": 1,
        "evidence_kind": "historical-replay",
        "run_id": info.run_id,
        "snapshot_clock_position": counts.clock_position,
        "released_observations_read": row_count,
        "released_observations_used": sum(
            state["observed_positions"]
            for result in windows
            for state in result["states"].values()
        ),
        "last_scanned_id": cursor,
        "source_sha256": info.source_sha256,
        "replay_config_sha256": info.config_sha256,
        "reference_sha256": reference.sha256,
        "windows": windows,
        "checkpoint": {**monitor.checkpoint(), "replay_snapshot": snapshot},
    }
