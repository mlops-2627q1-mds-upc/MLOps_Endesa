"""Replay the retained historical assessment without reading data or running KS."""

from copy import deepcopy
import hashlib
import json
import time

from src.drift.core import DriftConfig, deadline_after, is_integer
from src.drift.data import ROOT

SUMMARY_PATH = ROOT / "monitoring/drift/assessment-summary.json"
SUMMARY_SHA256 = "53d1a53118832a1473a8deb47881268565e41673f0693f0f5922ee8cb1d54f66"
SCENARIOS = ("unchanged", "level", "variability")


def recorded_scenario(
    name: str, *, clock_position: int, observation_delay: int = 0, max_seconds=120
) -> dict:
    """Expose at most 30 recorded state/window signals through a positional clock.

    This is a playback of the fixed validation assessment, not a detector run or
    a replay journal. A window becomes available at end + observation_delay,
    matching #81's last-target + 1 + delay convention. No alert is delivered.
    """
    deadline = deadline_after(max_seconds)
    if name not in SCENARIOS:
        raise ValueError("scenario must be unchanged, level or variability")
    if not is_integer(observation_delay) or observation_delay < 0:
        raise ValueError("observation_delay must be a nonnegative integer")
    raw = SUMMARY_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SUMMARY_SHA256:
        raise ValueError("historical assessment summary differs from the frozen record")
    assessment = json.loads(raw)
    historical = assessment["results"]["validation"]
    settings = {**historical["configuration"]["settings"]}
    settings["states"] = tuple(settings["states"])
    cfg = DriftConfig(**settings)
    if (
        not is_integer(clock_position)
        or not 0 <= clock_position <= cfg.validation_end + observation_delay
    ):
        raise ValueError("clock_position must be within the declared historical schedule")
    summary = historical["scenarios"][name]
    onset = historical["interventions"]["start_position"] if name != "unchanged" else None
    first_signals = dict.fromkeys(cfg.states)
    windows = []
    for index in range(cfg.max_windows):
        if time.monotonic() > deadline:
            raise TimeoutError("bounded historical scenario exceeded max_seconds")
        start, end = cfg.bounds(index)
        available = end + observation_delay <= clock_position
        states = {}
        for state in cfg.states:
            alert = index in summary[state]["alert_windows"]
            status = ("drift" if alert else "no_drift") if available else "pending"
            states[state] = {
                "status": status,
                "missed_declared_shift": available
                and onset is not None
                and start >= onset
                and not alert,
            }
            if (
                available
                and alert
                and onset is not None
                and start >= onset
                and first_signals[state] is None
            ):
                first_signals[state] = end + observation_delay
        windows.append(
            {
                "window": index,
                "bounds": [start, end],
                "available_position": end + observation_delay,
                "states": states,
            }
        )
    return {
        "schema_version": 1,
        "evidence_kind": "recorded-historical-playback",
        "scenario": name,
        "clock_position": clock_position,
        "observation_delay_positions": observation_delay,
        "assessment_summary_sha256": SUMMARY_SHA256,
        "assessment_source_commit": assessment["assessment_source_commit"],
        "assessment_recorded_at": assessment["recorded_at"],
        "full_report_sha256": historical["full_report_sha256"],
        "source": deepcopy(historical["source"]),
        "reference_sha256": historical["reference_sha256"],
        "configuration": deepcopy(historical["configuration"]),
        "recorded_dependencies": deepcopy(historical["dependencies"]),
        "interventions": deepcopy(historical["interventions"]),
        "windows": windows,
        "recorded_summary": deepcopy(summary),
        "timing": {
            "nominal_minutes_per_position": 30,
            "intervention_start_position": onset,
            "replay_first_signal_position": first_signals,
            "replay_detection_delay_positions": {
                state: position - onset if position is not None else None
                for state, position in first_signals.items()
            },
            "wall_clock_alert_latency_seconds": None,
        },
        "recovery_assessed": False,
        "alert_delivery_verified": False,
        "interpretation": (
            "Playback preserves the recorded historical control alarms and missed shifts. "
            "Pending means the replay clock has not released a recorded window. "
            "No observations, p-values, forecast errors or runtime timings are recomputed. "
            "Shifts persist through the assessment; a later no_drift is a miss, not recovery."
        ),
    }
