"""Historical signal playback preserves negative evidence and positional timing."""

import json

import pytest

from src.drift import scenario
from src.drift.__main__ import main
from src.drift.scenario import recorded_scenario


def test_historical_control_retains_eight_alarms_in_thirty_comparisons():
    report = recorded_scenario("unchanged", clock_position=212016)
    results = [state for window in report["windows"] for state in window["states"].values()]
    assert len(results) == 30
    assert sum(state["status"] == "drift" for state in results) == 8
    assert not any(state["missed_declared_shift"] for state in results)
    assert report["timing"]["replay_first_signal_position"] == {
        "NSW": 206640,
        "VIC": 209328,
        "QLD": 203952,
        "SA": 206640,
        "TAS": 203952,
    }
    assert all(
        value is None for value in report["timing"]["replay_detection_delay_positions"].values()
    )
    assert report["evidence_kind"] == "recorded-historical-playback"
    assert not report["recovery_assessed"]
    assert not report["alert_delivery_verified"]
    assert report["timing"]["wall_clock_alert_latency_seconds"] is None
    assert report == recorded_scenario("unchanged", clock_position=212016)


def test_delayed_clock_preserves_variability_misses_and_56_to_112_day_delays():
    before = recorded_scenario("variability", clock_position=201270, observation_delay=7)
    assert all(state["status"] == "pending" for state in before["windows"][1]["states"].values())
    first = recorded_scenario("variability", clock_position=201271, observation_delay=7)
    assert first["windows"][1]["states"]["NSW"] == {
        "status": "no_drift",
        "missed_declared_shift": True,
    }
    assert first["timing"]["replay_first_signal_position"]["NSW"] is None
    assert first["timing"]["replay_detection_delay_positions"]["VIC"] == 2688 + 7
    later = recorded_scenario("variability", clock_position=203959, observation_delay=7)
    assert later["timing"]["replay_detection_delay_positions"]["NSW"] == 5376 + 7
    assert later["timing"]["replay_detection_delay_positions"]["SA"] == 5376 + 7
    assert later["recorded_summary"]["NSW"]["first_detection_delay_positions"] == 5376
    final = recorded_scenario("variability", clock_position=212016)
    assert final["windows"][5]["states"]["QLD"]["missed_declared_shift"]
    assert not final["recovery_assessed"]


def test_persistent_level_shift_no_drift_results_are_misses_without_recovery():
    report = recorded_scenario("level", clock_position=212016)
    assert report["recorded_summary"]["SA"]["missed_post_shift_windows"] == [3, 4, 5]
    assert report["recorded_summary"]["VIC"]["missed_post_shift_windows"] == [5]
    assert report["windows"][5]["states"]["VIC"] == {
        "status": "no_drift",
        "missed_declared_shift": True,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {"name": "recovery"},
        {"clock_position": True},
        {"clock_position": -1},
        {"clock_position": 213217},
        {"observation_delay": True},
        {"observation_delay": -1},
        {"max_seconds": 301},
        {"max_seconds": float("nan")},
    ],
)
def test_scenario_rejects_undeclared_or_unbounded_inputs(arguments):
    with pytest.raises(ValueError):
        recorded_scenario(**{"name": "level", "clock_position": 212016, **arguments})


def test_scenario_rejects_a_changed_historical_record(tmp_path, monkeypatch):
    path = tmp_path / "changed.json"
    path.write_bytes(scenario.SUMMARY_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(scenario, "SUMMARY_PATH", path)
    with pytest.raises(ValueError, match="frozen record"):
        recorded_scenario("level", clock_position=212016)


def test_scenario_cli_writes_without_data_access_and_preserves_output(tmp_path, monkeypatch):
    def unexpected(*_args):
        pytest.fail("Historical playback must not load data or run KS")

    monkeypatch.setattr("src.drift.__main__.load_data", unexpected)
    monkeypatch.setattr("src.drift.core.ks_compare", unexpected)
    path = tmp_path / "scenario.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "drift",
            "scenario",
            "--scenario",
            "variability",
            "--clock-position",
            "203952",
            "--output",
            str(path),
        ],
    )
    main()
    assert json.loads(path.read_text()) == recorded_scenario("variability", clock_position=203952)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert path.read_bytes() == original
