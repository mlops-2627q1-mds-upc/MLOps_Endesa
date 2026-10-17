"""DST placebo windows must exclude every declared mainland and TAS transition."""

import numpy as np
import pandas as pd
import pytest

from src.dst import placebo_sampling_pool, transitions
from src.splits import SplitConfig


@pytest.mark.parametrize("state", ["NSW", "TAS"])
def test_declared_calendar_preserves_exception_and_rule_change(state):
    dates = dict(transitions(state, [2006]))
    assert dates["DST ends"] == pd.Timestamp("2006-04-02")
    assert dict(transitions(state, [2008])) == {
        "DST ends": pd.Timestamp("2008-04-06"),
        "DST starts": pd.Timestamp("2008-10-05"),
    }


def test_pool_excludes_both_inclusive_window_edges_and_transition_day():
    date = pd.Timestamp("2005-03-27")
    candidates = pd.date_range(date - pd.Timedelta(days=11), periods=23)
    pool = placebo_sampling_pool(candidates, [date])
    assert pool.equals(candidates[[0, -1]])


def test_pool_uses_tas_start_before_the_mainland_start():
    candidates = pd.date_range("2005-09-20", "2005-11-11")
    mainland = [date for _, date in transitions("NSW", [2005])]
    tas = [date for _, date in transitions("TAS", [2005])]
    assert pd.Timestamp("2005-10-02") in placebo_sampling_pool(candidates, mainland)
    pool = placebo_sampling_pool(candidates, mainland + tas)
    expected = pd.DatetimeIndex(
        [
            "2005-09-20",
            "2005-09-21",
            "2005-10-13",
            "2005-10-14",
            "2005-10-15",
            "2005-10-16",
            "2005-10-17",
            "2005-10-18",
            "2005-10-19",
            "2005-11-10",
            "2005-11-11",
        ]
    )
    assert pool.equals(expected)


def test_seeded_training_sample_has_complete_transition_free_windows():
    cfg = SplitConfig.from_params()
    nominal_days = pd.date_range("2002-01-01", periods=cfg.train_end // 48)
    years = range(nominal_days[0].year, nominal_days[-1].year + 1)
    dates = [date for state in ("NSW", "TAS") for _, date in transitions(state, years)]
    candidates = nominal_days[15:-15]
    pool = placebo_sampling_pool(candidates, dates)
    sample = pd.DatetimeIndex(np.random.default_rng(0).choice(pool, 300))
    repeated = pd.DatetimeIndex(np.random.default_rng(0).choice(pool, 300))
    assert sample.equals(repeated)
    assert len(sample) == 300
    assert sample.isin(pool).all()
    assert sample.min() - pd.Timedelta(days=10) >= nominal_days[0]
    assert sample.max() + pd.Timedelta(days=10) <= nominal_days[-1]
    for date in pool:
        window = pd.date_range(date - pd.Timedelta(days=10), periods=21)
        assert not window.isin(dates).any()
    for date in sample:
        window = pd.date_range(date - pd.Timedelta(days=10), periods=21)
        assert not window.isin(dates).any()

    # The old seed included actual transitions; the corrected pool removes them.
    original = pd.DatetimeIndex(np.random.default_rng(0).choice(candidates, 300))
    for contaminated in ("2005-03-27", "2012-10-07"):
        assert pd.Timestamp(contaminated) in original
        assert pd.Timestamp(contaminated) not in sample
