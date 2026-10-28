"""One declared control/shift assessment on development data, without tuning."""

from importlib.metadata import version
import time

import numpy as np
import pandas as pd

from src.drift.core import DriftConfig, DriftMonitor, build_reference, deadline_after
from src.drift.data import config_record


def _intervention(validation, reference, scale, name):
    cfg = reference.config
    values = validation.astype(float)
    shifted = values.index >= cfg.schedule_start + cfg.window_steps
    if name == "level":
        values.loc[shifted] += scale
    elif name == "variability":
        phases = values.index[shifted].to_numpy() % cfg.cycle_steps
        pattern = reference.template[:, phases].T
        values.loc[shifted] = pattern + 2 * (values.loc[shifted].to_numpy() - pattern)
    return values


def _summaries(windows, cfg, name):
    onset = cfg.schedule_start + cfg.window_steps
    summaries = {}
    for state in cfg.states:
        alerts = [result for result in windows if result["states"][state]["status"] == "drift"]
        post_shift = [result for result in alerts if result["bounds"][0] >= onset]
        tested = sum(
            result["states"][state]["status"] != "insufficient_data" for result in windows
        )
        summaries[state] = {
            "tested_windows": tested,
            "insufficient_windows": [
                result["window"]
                for result in windows
                if result["states"][state]["status"] == "insufficient_data"
            ],
            "alert_windows": [result["window"] for result in alerts],
            "control_alert_fraction": len(alerts) / tested
            if name == "unchanged" and tested
            else None,
            "pre_shift_alerts": sum(result["bounds"][1] <= onset for result in alerts),
            "missed_post_shift_windows": (
                [
                    result["window"]
                    for result in windows
                    if result["bounds"][0] >= onset
                    and result["states"][state]["status"] == "no_drift"
                ]
                if name != "unchanged"
                else None
            ),
            "detected_after_shift": bool(post_shift) if name != "unchanged" else None,
            "first_detection_delay_positions": (
                post_shift[0]["bounds"][1] - onset if name != "unchanged" and post_shift else None
            ),
            "undetected_exposure_positions": (
                cfg.bounds(cfg.max_windows - 1)[1] - onset
                if name != "unchanged" and not post_shift
                else None
            ),
        }
    return summaries


def assess_controls(train, validation, source: dict, cfg: DriftConfig, max_seconds=120) -> dict:
    """Report original demand and two fixed synthetic interventions on validation.

    Interventions start after the first window. Level adds one training standard
    deviation; variability doubles residuals around the training phase pattern.
    These magnitudes are fixed before looking at development results.
    """
    deadline = deadline_after(max_seconds)
    reference = build_reference(train, cfg, source)
    onset = cfg.schedule_start + cfg.window_steps
    ref_start = cfg.reference_end - cfg.reference_cycles * cfg.cycle_steps
    scale = (
        train.loc[ref_start : cfg.reference_end - 1, list(cfg.states)].astype(float).std(ddof=0)
    )
    scenarios = {}
    for name in ("unchanged", "level", "variability"):
        values = _intervention(validation, reference, scale, name)
        monitor = DriftMonitor(reference, {"kind": "development-assessment", "scenario": name})
        windows = []
        for index in range(cfg.max_windows):
            if time.monotonic() > deadline:
                raise TimeoutError("bounded drift assessment exceeded max_seconds")
            start, end = cfg.bounds(index)
            windows.append(monitor.assess(values.loc[start : end - 1], index))
        scenarios[name] = {"summary": _summaries(windows, cfg, name), "windows": windows}
    return {
        "schema_version": 1,
        "evidence_kind": source["kind"],
        "source": source,
        "reference_sha256": reference.sha256,
        "configuration": config_record(cfg),
        "dependencies": {
            name: version(name) for name in ("alibi-detect", "numpy", "scipy", "pandas")
        },
        "interventions": {
            "start_position": onset,
            "level_offset_by_state": scale.to_dict(),
            "variability_residual_multiplier": 2,
            "selection": "single declared assessment; no parameter search",
        },
        "interpretation": (
            "Unchanged historical validation alarms are a false-alert proxy: natural drift may "
            "be present. Fixture controls have a known unchanged distribution. Detection "
            "delays use positional 30-minute steps, not civil timestamps or forecast error. "
            "Each scenario has a separate nominal budget; no joint iid guarantee is asserted."
        ),
        "scenarios": scenarios,
    }


def fixture_data() -> tuple:
    """Reproducible seasonal/dependent fixture with no imposed development drift."""
    cfg = DriftConfig(train_end=104 * 336, validation_end=(104 + 48) * 336)
    rng = np.random.default_rng(7)
    count = cfg.validation_end
    phase = np.arange(count) % cfg.cycle_steps
    pattern = 100 + 20 * np.sin(2 * np.pi * phase / 48) + 5 * np.cos(2 * np.pi * phase / 336)
    # Correlated observations inside each cycle; independent fixture cycle levels.
    levels = np.repeat(rng.normal(0, 2, count // cfg.cycle_steps), cfg.cycle_steps)
    noise = np.convolve(rng.normal(0, 1, count + 3), np.ones(4) / 2, mode="valid")
    frame = pd.DataFrame(
        {
            state: (index + 1) * (pattern + levels + noise)
            for index, state in enumerate(cfg.states)
        },
        index=pd.RangeIndex(count, name="timestep"),
    )
    source = {"kind": "fixture", "version": "seasonal-dependent-v1", "seed": 7}
    return frame.iloc[: cfg.train_end], frame.iloc[cfg.train_end :], source, cfg
