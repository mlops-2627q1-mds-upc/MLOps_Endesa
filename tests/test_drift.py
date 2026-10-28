"""Pure positional/reference/retry behavior; real KS checks live separately."""

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from src.drift.assessment import assess_controls
from src.drift.core import DriftConfig, DriftMonitor, build_reference, identity, validate_frame
from src.drift.data import load_config


@pytest.fixture
def demand():
    """A short five-state source with complete cycles and a deliberate train tail."""
    cfg = DriftConfig(
        train_end=67, validation_end=264, cycle_steps=4, reference_cycles=8, max_windows=6
    )
    positions = np.arange(cfg.validation_end)
    frame = pd.DataFrame(
        {state: positions % 4 + index * 10.0 for index, state in enumerate(cfg.states)},
        index=pd.Index(positions, name="timestep"),
    )
    return cfg, frame


@pytest.fixture
def reference(demand):
    """An explicitly synthetic training-only reference."""
    cfg, frame = demand
    return build_reference(frame.loc[: cfg.train_end - 1], cfg, {"kind": "fixture", "version": 1})


@pytest.fixture
def fake_ks(monkeypatch):
    """Count comparisons in pure state tests, without requiring optional Alibi."""
    calls = []

    def compare(ref, current, alpha):
        calls.append((ref.copy(), current.copy(), alpha))
        return {
            "status": "no_drift",
            "p_values": [1.0, 1.0],
            "ks_distance": [0.0, 0.0],
            "feature_threshold": alpha / 2,
        }

    monkeypatch.setattr("src.drift.core.ks_compare", compare)
    return calls


def test_fixed_production_boundaries_and_nonoverlap():
    cfg = load_config()
    assert (cfg.reference_end - cfg.reference_cycles * 336, cfg.reference_end) == (160608, 195552)
    assert cfg.schedule_start == 195888
    assert cfg.bounds(5) == (209328, 212016)
    assert cfg.validation_end - cfg.bounds(5)[1] == 1200
    for index in range(5):
        assert cfg.bounds(index)[1] == cfg.bounds(index + 1)[0]


@pytest.mark.parametrize(
    "changes",
    [
        {"cycle_steps": 0},
        {"reference_cycles": 7},
        {"window_cycles": 7},
        {"family_alpha": 0},
        {"family_alpha": float("nan")},
        {"max_windows": True},
        {"validation_end": 100},
        {"states": ("NSW",)},
        {"feature_version": "unknown"},
    ],
)
def test_invalid_configuration(demand, changes):
    with pytest.raises(ValueError):
        replace(demand[0], **changes)


def test_training_only_and_frozen_template(demand, reference):
    cfg, frame = demand
    with pytest.raises(ValueError, match="training positions only"):
        build_reference(frame, cfg, {"kind": "fixture"})
    # Modifying the incomplete training tail cannot alter the reference.
    tail = frame.loc[: cfg.train_end - 1].copy()
    tail.loc[cfg.reference_end :] += 999
    ref = build_reference(tail, cfg, reference.source)
    assert ref.sha256 == reference.sha256
    assert not reference.template.flags.writeable
    assert not reference.features.flags.writeable
    assert np.array_equal(reference.features, np.zeros((5, 8, 2)))


def test_reference_hash_binds_values_config_and_source(demand, reference):
    cfg, frame = demand
    train = frame.loc[: cfg.train_end - 1].copy()
    assert build_reference(train, cfg, reference.source).sha256 == reference.sha256
    train.loc[40, "TAS"] -= 1
    assert build_reference(train, cfg, reference.source).sha256 != reference.sha256
    train = frame.loc[: cfg.train_end - 1]
    other = build_reference(train, cfg, {"kind": "fixture", "version": 2})
    assert other.sha256 != reference.sha256
    other = build_reference(train, replace(cfg, family_alpha=0.1), reference.source)
    assert other.sha256 != reference.sha256
    assert identity(asdict(cfg)) == identity(dict(reversed(list(asdict(cfg).items()))))


@pytest.mark.parametrize("problem", ["nan", "column", "position"])
def test_reference_requires_complete_finite_cycles(demand, problem):
    cfg, frame = demand
    train = frame.loc[: cfg.train_end - 1].copy()
    if problem == "nan":
        train.loc[40, "TAS"] = np.nan
    elif problem == "column":
        train = train.drop(columns="TAS")
    else:
        train = train.drop(index=40)
    with pytest.raises(ValueError, match="complete finite"):
        build_reference(train, cfg, {"kind": "fixture"})


@pytest.mark.parametrize(
    "problem", ["duplicates", "reverse", "float", "negative", "bool", "inf", "text"]
)
def test_malformed_frames_are_rejected(demand, problem):
    cfg, frame = demand
    values = frame.iloc[:4].copy()
    if problem == "duplicates":
        values.index = [0, 1, 1, 3]
    elif problem == "reverse":
        values = values.iloc[::-1]
    elif problem == "float":
        values.index = [0.0, 1.0, 2.0, 3.0]
    elif problem == "negative":
        values.index = [-1, 0, 1, 2]
    elif problem == "bool":
        values["NSW"] = True
    elif problem == "inf":
        values.loc[0, "NSW"] = np.inf
    else:
        values["NSW"] = "1"
    with pytest.raises(ValueError):
        validate_frame(values, cfg)


def test_missing_inputs_do_not_call_alibi_and_can_be_retried(demand, reference, fake_ks):
    cfg, frame = demand
    start, end = cfg.bounds(0)
    monitor = DriftMonitor(reference, {"fixture": 1})
    values = frame.loc[start : end - 1].drop(columns="TAS")
    result = monitor.assess(values, 0)
    assert result["states"]["TAS"]["status"] == "insufficient_data"
    assert result["states"]["TAS"]["missing_positions"] == cfg.window_steps
    assert len(fake_ks) == 4
    monitor.assess(frame.loc[start : end - 1], 0)
    assert len(fake_ks) == 5
    assert all(call[2] == cfg.family_alpha / 30 for call in fake_ks)


def test_missing_positions_and_nan_have_explicit_counts(demand, reference, fake_ks):
    cfg, frame = demand
    start, end = cfg.bounds(0)
    values = frame.loc[start : end - 1].drop(index=start)
    values.loc[start + 1, "TAS"] = np.nan
    result = DriftMonitor(reference, {"fixture": 1}).assess(values, 0)
    assert result["states"]["NSW"]["missing_positions"] == 1
    assert result["states"]["TAS"]["missing_positions"] == 2
    assert all(item["status"] == "insufficient_data" for item in result["states"].values())
    assert not fake_ks


def test_retries_checkpoint_and_budget(demand, reference, fake_ks):
    cfg, frame = demand
    start, end = cfg.bounds(0)
    values = frame.loc[start : end - 1].copy()
    monitor = DriftMonitor(reference, {"fixture": 1})
    first = monitor.assess(values, 0)
    checkpoint = monitor.checkpoint()
    restored = DriftMonitor(reference, {"fixture": 1}, checkpoint)
    assert restored.assess(values, 0) == first
    assert len(fake_ks) == 5
    values.loc[start, "TAS"] += 1
    with pytest.raises(ValueError, match="conflicting"):
        restored.assess(values, 0)
    with pytest.raises(ValueError, match="different"):
        DriftMonitor(reference, {"fixture": 2}, checkpoint)
    with pytest.raises(ValueError, match="budget"):
        restored.assess(values, 6)
    assert restored.checkpoint() == checkpoint


def test_outside_window_and_untested_out_of_order_window(demand, reference, fake_ks):
    cfg, frame = demand
    monitor = DriftMonitor(reference, {"fixture": 1})
    with pytest.raises(ValueError, match="inside"):
        monitor.assess(frame.loc[cfg.schedule_start - 1 : cfg.bounds(0)[1] - 1], 0)
    start, end = cfg.bounds(1)
    monitor.assess(frame.loc[start : end - 1], 1)
    start, end = cfg.bounds(0)
    with pytest.raises(ValueError, match="after a later"):
        monitor.assess(frame.loc[start : end - 1], 0)
    # An unchanged incomplete earlier slot remains reportable on every poll.
    partial = monitor.assess(frame.loc[start : end - 2], 0)
    assert all(item["status"] == "insufficient_data" for item in partial["states"].values())
    assert len(fake_ks) == 5


def test_corrupt_checkpoint_and_backend_failure_do_not_consume_slots(
    demand, reference, monkeypatch
):
    cfg, frame = demand
    monitor = DriftMonitor(reference, {"fixture": 1})
    checkpoint = monitor.checkpoint()
    checkpoint["tested_sha256"] = "corrupt"
    with pytest.raises(ValueError, match="corrupt"):
        DriftMonitor(reference, {"fixture": 1}, checkpoint)

    def failure(*_args):
        raise ImportError("fixture missing backend")

    monkeypatch.setattr("src.drift.core.ks_compare", failure)
    start, end = cfg.bounds(0)
    with pytest.raises(ImportError):
        monitor.assess(frame.loc[start : end - 1], 0)
    assert monitor.checkpoint()["tested"] == {}


@pytest.mark.parametrize("budget", [True, 0, -1, 301, float("nan"), float("inf"), "10"])
def test_control_assessment_rejects_unbounded_time_limits(demand, budget):
    cfg, frame = demand
    with pytest.raises(ValueError, match="max_seconds"):
        assess_controls(
            frame.loc[: cfg.train_end - 1],
            frame.loc[cfg.train_end :],
            {"kind": "fixture"},
            cfg,
            budget,
        )
