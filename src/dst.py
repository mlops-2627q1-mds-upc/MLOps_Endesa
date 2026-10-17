"""Declared DST dates and transition-free placebo windows for the demand EDA."""

import numpy as np
import pandas as pd


def _last_sunday(year, month):
    date = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return date - pd.Timedelta(days=(date.dayofweek + 1) % 7)


def _first_sunday(year, month):
    date = pd.Timestamp(year, month, 1)
    return date + pd.Timedelta(days=(6 - date.dayofweek) % 7)


def transitions(state, years):
    """Return the notebook's declared mainland or TAS dates, without clock inference."""
    dates = []
    for year in years:
        if year == 2006:
            end = _first_sunday(2006, 4)  # Commonwealth Games extension
        else:
            end = _last_sunday(year, 3) if year <= 2007 else _first_sunday(year, 4)
        if year <= 2007:
            start = _first_sunday(year, 10) if state == "TAS" else _last_sunday(year, 10)
        else:
            start = _first_sunday(year, 10)
        dates += [("DST ends", end), ("DST starts", start)]
    return dates


def placebo_sampling_pool(candidate_dates, transition_dates, window_days=10):
    """Exclude inclusive before/after windows touching any declared transition.

    Candidates must already have complete windows inside the training block.
    Excluding the centre and weekends too keeps the entire calendar window clear,
    even though the ramp comparison only uses weekdays on either side.
    """
    candidates = pd.DatetimeIndex(candidate_dates)
    margin = pd.Timedelta(days=window_days)
    eligible = np.ones(len(candidates), dtype=bool)
    for date in pd.DatetimeIndex(transition_dates):
        eligible &= (candidates < date - margin) | (candidates > date + margin)
    return candidates[eligible]
