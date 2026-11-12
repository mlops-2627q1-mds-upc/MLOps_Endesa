"""Exercise public replay/quality and Alibi APIs without reading real datasets."""

from dataclasses import asdict
from importlib.metadata import version
import json


def delayed_observation(directory, declaration):
    """Use #81's declared SQLite fixture and score only released public joins."""
    from fastapi.testclient import TestClient  # pylint: disable=import-outside-toplevel
    from monitoring.quality.exporter import (  # pylint: disable=import-outside-toplevel
        create_app as quality_app,
    )
    from tests.replay_fixture import (  # pylint: disable=import-outside-toplevel
        create_replay_fixture,
    )

    from src.forecast_quality import (  # pylint: disable=import-outside-toplevel
        QualityPolicy,
        quality_report,
    )
    from src.monitoring import JournalReader  # pylint: disable=import-outside-toplevel
    from src.monitoring.replay import Replay  # pylint: disable=import-outside-toplevel

    fixture = create_replay_fixture(directory / "replay-fixture")
    phases = []
    for name, position in (
        ("delayed", declaration["partial_position"]),
        ("recovery", declaration["recovered_position"]),
    ):
        with Replay(fixture.journal, fixture.source, fixture.config) as replay:
            replay.release(position)
        with JournalReader(fixture.journal) as reader:
            report = quality_report(
                reader,
                fixture.directory,
                fixture.directory / "params.yaml",
                QualityPolicy(minimum_samples=declaration["minimum_samples"]),
            )
            conservative = quality_report(
                reader, fixture.directory, fixture.directory / "params.yaml", QualityPolicy()
            )
        path = directory / f"quality-{name}.json"
        path.write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
        with TestClient(quality_app(path)) as client:
            metrics = client.get("/metrics")
            metrics.raise_for_status()
            (directory / f"quality-{name}.prom").write_text(metrics.text)
        phases.append(
            {
                "phase": name,
                "clock_position": position,
                "counts": report["replay_counts"],
                "groups": report["by_state_horizon"],
                "default_minimum_30_statuses": sorted(
                    {row["sample_status"] for row in conservative["by_state_horizon"]}
                ),
                "metadata": report["metadata"],
            }
        )
    partial, final = phases
    passed = (
        partial["counts"]["matched_steps"] == 6
        and partial["counts"]["pending_steps"] == 186
        and final["counts"]["matched_steps"] == 192
        and final["counts"]["pending_steps"] == 0
        and {row["sample_status"] for row in final["groups"]} == {"sufficient"}
    )
    return {
        "evidence_kind": "deterministic_fixture",
        "procedure_passed": passed,
        "observed_signal": phases,
        "recovery": {"observed": final["counts"]["complete"]},
        "sample_sufficiency": {
            "display_minimum": declaration["minimum_samples"],
            "default_minimum": 30,
            "default_horizons_remain_insufficient": all(
                row["sample_status"] == "insufficient_samples"
                for row in conservative["by_state_horizon"]
            ),
            "overlapping_targets_are_dependent": True,
        },
        "missed_detections": [],
        "limitations": [
            "No observation-delay or quality alert policy/receiver exists; firing and clearing unmeasured",
            "Synthetic perfect forecasts and minimum two are display checks, not model performance",
        ],
    }


def shifted_demand(directory, declaration):
    """Run a fixed control/shift/recovery schedule through the actual Alibi backend."""
    from src.drift.assessment import fixture_data  # pylint: disable=import-outside-toplevel
    from src.drift.core import (  # pylint: disable=import-outside-toplevel
        DriftMonitor,
        build_reference,
    )

    train, validation, source, config = fixture_data()
    reference = build_reference(train, config, source)
    monitor = DriftMonitor(reference, {"kind": "fixture", "scenario": "issue84-level-recovery-v1"})
    scale = train.std(ddof=0) * declaration["shift_training_stddevs"]
    phases, misses, false_alarms = [], [], []
    for window, name in enumerate(("control", "shift", "recovery")):
        start, end = config.bounds(window)
        values = validation.loc[start : end - 1].copy()
        if name == "shift":
            values += scale
        result = monitor.assess(values, window)
        phases.append({"phase": name, "available_position": end, **result})
        for state, record in result["states"].items():
            if name == "shift" and record["status"] != "drift":
                misses.append({"state": state, "window": window, "status": record["status"]})
            if name != "shift" and record["status"] == "drift":
                false_alarms.append({"state": state, "phase": name, "window": window})
    raw = {"phases": phases, "checkpoint": monitor.checkpoint(), "config": asdict(config)}
    (directory / "drift.json").write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    return {
        "evidence_kind": "fresh_synthetic_alibi",
        "procedure_passed": True,
        "observed_signal": phases,
        "missed_detections": misses,
        "control_or_recovery_alarms": false_alarms,
        "recovery": {
            "observed": all(row["status"] == "no_drift" for row in phases[-1]["states"].values()),
            "position": phases[-1]["available_position"],
            "semantics": "next scheduled KS result, not an operational alert clearing",
        },
        "sample_sufficiency": {
            "reference_cycles": config.reference_cycles,
            "test_cycles": config.window_cycles,
            "expected_positions_per_state": config.window_steps,
        },
        "dependencies": {
            key: version(key) for key in ("alibi-detect", "numpy", "scipy", "pandas")
        },
        "limitations": [
            "Positional steps are nominal 30-minute observations; detection delay is not wall latency",
            "Synthetic level intervention fixed at one training standard deviation; no parameter search",
            "Fresh KS results do not establish forecast degradation or notification delivery",
        ],
    }
