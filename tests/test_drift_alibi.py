"""Real Alibi checks, executed by the isolated drift project and its CI job."""

import importlib.util
import json

import numpy as np
import pytest

from src.drift.assessment import assess_controls, fixture_data
from src.drift.core import DriftConfig, build_reference, ks_compare
from src.drift.replay import assess_replay

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("alibi_detect") is None,
    reason="real KSDrift requires the locked monitoring/drift environment",
)


def test_real_ks_unchanged_and_constant_data():
    reference = np.column_stack((np.arange(104), np.arange(104)))
    unchanged = ks_compare(reference, reference.copy(), 0.05 / 30)
    assert unchanged["status"] == "no_drift"
    assert unchanged["p_values"] == [1.0, 1.0]
    constant = np.zeros((104, 2))
    assert ks_compare(constant, np.zeros((8, 2)), 0.05 / 30)["status"] == "no_drift"
    assert ks_compare(constant, np.ones((8, 2)), 0.05 / 30)["status"] == "drift"


def test_real_fixed_seasonal_controls_and_declared_shifts():
    report = assess_controls(*fixture_data())
    unchanged = report["scenarios"]["unchanged"]["summary"]
    assert all(state["tested_windows"] == 6 for state in unchanged.values())
    assert all(not state["alert_windows"] for state in unchanged.values())
    for scenario in ("level", "variability"):
        for state in report["scenarios"][scenario]["summary"].values():
            assert state["detected_after_shift"]
            assert state["first_detection_delay_positions"] >= 2688
            assert state["pre_shift_alerts"] == 0


def test_real_journal_release_snapshot_and_restart(tmp_path):
    """Actual #81 SQLite exports plus actual Alibi, with an explicit HTTP fixture."""
    import httpx  # pylint: disable=import-outside-toplevel
    import pandas as pd  # pylint: disable=import-outside-toplevel

    from src.monitoring import (  # pylint: disable=import-outside-toplevel
        JournalReader,
        ModelIdentity,
        ReplayConfig,
    )
    from src.monitoring.replay import Replay  # pylint: disable=import-outside-toplevel
    from src.monitoring.source import ReplaySource  # pylint: disable=import-outside-toplevel
    from src.splits import SplitConfig  # pylint: disable=import-outside-toplevel

    cfg = DriftConfig(train_end=512, validation_end=704, cycle_steps=4, reference_cycles=8)
    frame = pd.DataFrame({state: [1.0] * 704 for state in cfg.states}, index=range(704))
    split = SplitConfig(cfg.states, 896, 512, 704, 512, 48, 48)
    source = ReplaySource(frame, split, "constant-drift-integration-fixture")
    reference = build_reference(
        frame.iloc[:512],
        cfg,
        {
            "kind": "fixture",
            "replay_source_sha256": source.source_sha256,
            "replay_source_reference": source.reference,
        },
    )
    payload = {
        "model_id": "chronos-t5-small-fine-tuned",
        "model_revision": "local-sha256:" + "a" * 64,
        "checkpoint": {"sha256": "a" * 64, "training_run_id": "b" * 32},
        "settings": {
            "num_samples": 20,
            "seed": 7,
            "device": "cpu",
            "cpu_threads": 2,
            "dtype": "float32",
            "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
            "quantile_method": "linear",
            "point_statistic": "median",
        },
        "value_scale": "source",
        "context_length": 512,
        "horizon": 48,
        "p10": [0.0] * 48,
        "p50": [1.0] * 48,
        "p90": [2.0] * 48,
    }
    config = ReplayConfig(
        512,
        704,
        cfg.states,
        ModelIdentity.from_api(payload),
        observation_delay=5,
    )
    path = tmp_path / "drift.sqlite"

    def response(request):
        return httpx.Response(200, json={**payload, "state": json.loads(request.content)["state"]})

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        with Replay(path, source, config, client=client) as replay:
            replay.run(max_forecasts=5)
            replay.release(548)
            with JournalReader(path) as reader:
                first = assess_replay(reader, reference, source)
            assert all(
                item["missing_positions"] == 1 for item in first["windows"][0]["states"].values()
            )
            replay.release(549)
            with JournalReader(path) as reader:
                second = assess_replay(reader, reference, source, first["checkpoint"])
            assert all(
                item["status"] == "no_drift" for item in second["windows"][0]["states"].values()
            )
        # Reopen both the real journal writer and the detector's persisted checkpoint.
        with Replay(path, source, config, client=client) as replay:
            replay.run(max_forecasts=15)
            assert replay.release(709).complete
        with JournalReader(path) as reader:
            final = assess_replay(reader, reference, source, second["checkpoint"])
            assert final["released_observations_used"] == 5 * 192
            assert len(final["checkpoint"]["tested"]) == 30
            assert assess_replay(reader, reference, source, final["checkpoint"]) == final
