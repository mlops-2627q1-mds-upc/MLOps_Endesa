"""Public replay reader contract, release clock and unique-position semantics."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from src.drift.core import DriftConfig, build_reference
from src.drift.replay import assess_replay


@pytest.fixture
def replay_case():
    """Protocol fixture independent of SQLite implementation details."""
    cfg = DriftConfig(train_end=64, validation_end=256, cycle_steps=4, reference_cycles=8)
    source = SimpleNamespace(source_sha256="fixture-source", reference="fixture-reference")
    provenance = {
        "kind": "fixture",
        "replay_source_sha256": source.source_sha256,
        "replay_source_reference": source.reference,
    }
    train = pd.DataFrame({state: [1.0] * 64 for state in cfg.states}, index=range(64))
    reference = build_reference(train, cfg, provenance)
    info = SimpleNamespace(
        source_sha256=source.source_sha256,
        source_reference=source.reference,
        config_sha256="replay-config",
        run_id="fixture-run",
    )
    rows = [
        SimpleNamespace(
            id=number + 1,
            state=state,
            position=position,
            value=1.0,
            released_position=position + 3,
            source_reference=source.reference,
            config_sha256=info.config_sha256,
        )
        for number, (position, state) in enumerate(
            (position, state) for position in range(64, 96) for state in cfg.states
        )
    ]
    counts = SimpleNamespace(clock_position=98)
    reader = SimpleNamespace(
        snapshot=nullcontext,
        run_info=lambda: info,
        counts=lambda: counts,
        released_observations=lambda after_id, limit: [row for row in rows if row.id > after_id][
            :limit
        ],
    )
    return cfg, reader, reference, source, rows, counts


@pytest.fixture
def compare_calls(monkeypatch):
    """State-only checks do not require the optional real backend."""
    calls = []

    def compare(_reference, _current, _alpha):
        calls.append(1)
        return {"status": "no_drift"}

    monkeypatch.setattr("src.drift.core.ks_compare", compare)
    return calls


def test_release_snapshot_and_durable_retry(replay_case, compare_calls):
    _, reader, reference, source, _, counts = replay_case
    counts.clock_position = 97
    first = assess_replay(reader, reference, source)
    assert first["released_observations_used"] == 31 * 5
    assert all(
        state["status"] == "insufficient_data" for state in first["windows"][0]["states"].values()
    )
    assert not compare_calls
    counts.clock_position = 98
    complete = assess_replay(reader, reference, source, first["checkpoint"])
    assert complete["released_observations_used"] == 32 * 5
    assert len(compare_calls) == 5
    assert assess_replay(reader, reference, source, complete["checkpoint"]) == complete
    assert len(compare_calls) == 5
    assert all(
        item["status"] == "insufficient_data" for item in complete["windows"][1]["states"].values()
    )


@pytest.mark.parametrize(
    "problem", ["source", "config", "duplicate", "ids", "infinite", "test_position"]
)
def test_malformed_replay_inputs_fail_closed(replay_case, compare_calls, problem):
    _, reader, reference, source, rows, counts = replay_case
    if problem == "source":
        source.source_sha256 = "other-source"
    elif problem == "config":
        rows[0].config_sha256 = "other-config"
    elif problem == "duplicate":
        rows[1].state = rows[0].state
    elif problem == "ids":
        rows[1].id = rows[0].id
    elif problem == "infinite":
        rows[0].value = float("inf")
    else:
        rows[0].position = 256
        rows[0].released_position = 257
        counts.clock_position = 257
    with pytest.raises(ValueError):
        assess_replay(reader, reference, source)
    assert not compare_calls


def test_reference_source_binding_and_pagination_budget(replay_case):
    _, reader, reference, source, _, _ = replay_case
    with pytest.raises(ValueError, match="source differ"):
        assess_replay(reader, replace(reference, source={"kind": "fixture"}), source)
    with pytest.raises(ValueError, match="pagination budget"):
        assess_replay(reader, reference, source, max_pages=1)


@pytest.mark.parametrize("budget", [True, 0, -1, 301, float("nan"), float("inf"), "10"])
def test_replay_rejects_unbounded_time_limits(replay_case, budget):
    _, reader, reference, source, _, _ = replay_case
    with pytest.raises(ValueError, match="max_seconds"):
        assess_replay(reader, reference, source, max_seconds=budget)
