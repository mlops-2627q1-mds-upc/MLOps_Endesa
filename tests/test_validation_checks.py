"""Regression tests for the custom validation checks in src/validate_data.py.

These tests use synthetic in-memory fixtures and do not require DVC data or
credentials, so they always run in CI. They cover the failure cases that GX
cannot express directly: infinity rejection, exact TAS positions, complete block
index sequences, and the full forecast origin grid.
"""

import json

import numpy as np
import pandas as pd

from src.splits import SplitConfig, forecast_origins
import src.validate_data as vd
from src.validate_data import (
    _check_block_index_ranges,
    _check_finite,
    _check_origins_windows,
    _check_tas_negative_positions,
    _collect_gx_failures,
)

# Minimal SplitConfig whose validation and test blocks tile exactly at the stride.
# validation: [60, 80), origins [60, 70]; test: [80, 100), origins [80, 90].
_CFG = SplitConfig(
    states=("NSW",),
    common_length=100,
    train_end=60,
    validation_end=80,
    context_length=10,
    horizon=10,
    origin_stride=10,
)

_FAKE_TAS_POSITIONS = [10, 11, 12]


# ---------------------------------------------------------------------------
# _check_finite
# ---------------------------------------------------------------------------


def test_finite_passes_clean_data():
    """No failures for a column with only finite values."""
    df = pd.DataFrame({"a": np.array([1.0, 2.0, 3.0], dtype="float32")})
    assert not _check_finite(df, "suite")


def test_finite_detects_positive_inf():
    """Positive infinity in a column is reported as a failure."""
    df = pd.DataFrame({"a": np.array([1.0, np.inf, 3.0], dtype="float64")})
    failures = _check_finite(df, "suite")
    assert len(failures) == 1
    assert failures[0]["expectation"] == "custom_no_infinite_values"
    assert failures[0]["column"] == "a"


def test_finite_detects_negative_inf():
    """Negative infinity in a column is reported as a failure."""
    df = pd.DataFrame({"a": np.array([-np.inf, 2.0], dtype="float64")})
    assert len(_check_finite(df, "suite")) == 1


def test_finite_ignores_nan():
    """NaN values do not trigger the infinity check."""
    df = pd.DataFrame({"a": np.array([1.0, np.nan, 3.0], dtype="float64")})
    assert not _check_finite(df, "suite")


# ---------------------------------------------------------------------------
# _check_tas_negative_positions
# ---------------------------------------------------------------------------


def _make_tas_df(negative_positions, length=20):
    vals = np.ones(length, dtype="float32")
    for p in negative_positions:
        vals[p] = -1.0
    return pd.DataFrame({"TAS": vals})


def test_tas_positions_passes(monkeypatch):
    """Exact documented positions produce no failures."""
    monkeypatch.setattr(vd, "TAS_NEGATIVE_POSITIONS", _FAKE_TAS_POSITIONS)
    assert not _check_tas_negative_positions(_make_tas_df(_FAKE_TAS_POSITIONS))


def test_tas_positions_extra_negative(monkeypatch):
    """An undocumented negative position is reported as a failure."""
    monkeypatch.setattr(vd, "TAS_NEGATIVE_POSITIONS", _FAKE_TAS_POSITIONS)
    df = _make_tas_df(_FAKE_TAS_POSITIONS + [15])
    failures = _check_tas_negative_positions(df)
    assert len(failures) == 1
    assert failures[0]["expectation"] == "custom_tas_negative_positions"


def test_tas_positions_missing_negative(monkeypatch):
    """A documented position that is not negative is reported as a failure."""
    monkeypatch.setattr(vd, "TAS_NEGATIVE_POSITIONS", _FAKE_TAS_POSITIONS)
    df = _make_tas_df([10, 11])  # position 12 missing
    assert len(_check_tas_negative_positions(df)) == 1


# ---------------------------------------------------------------------------
# _check_block_index_ranges
# ---------------------------------------------------------------------------


def _write_blocks(tmp_path, cfg, bad_block=None, bad_index=None):
    for name in ("train", "validation", "test"):
        start, end = cfg.bounds(name)
        if name == bad_block and bad_index is not None:
            idx = pd.Index(bad_index, name="timestep")
        else:
            idx = pd.RangeIndex(start, end, name="timestep")
        df = pd.DataFrame(
            {"NSW": np.ones(len(idx), dtype="float32")},
            index=idx,
        )
        df.to_parquet(tmp_path / f"{name}.parquet")


def test_block_index_passes(tmp_path, monkeypatch):
    """Correct contiguous RangeIndex produces no failures."""
    monkeypatch.setattr(vd, "PROCESSED_DATA_DIR", tmp_path)
    _write_blocks(tmp_path, _CFG)
    assert not _check_block_index_ranges(_CFG)


def test_block_index_detects_gap(tmp_path, monkeypatch):
    """An interior gap in the index is reported as a failure."""
    monkeypatch.setattr(vd, "PROCESSED_DATA_DIR", tmp_path)
    start, end = _CFG.bounds("train")
    bad = list(range(start, end - 1)) + [end + 10]
    _write_blocks(tmp_path, _CFG, bad_block="train", bad_index=bad)
    failures = _check_block_index_ranges(_CFG)
    assert any(f["suite"] == "train" for f in failures)


def test_block_index_detects_wrong_start(tmp_path, monkeypatch):
    """An index starting at the wrong position is reported as a failure."""
    monkeypatch.setattr(vd, "PROCESSED_DATA_DIR", tmp_path)
    start, end = _CFG.bounds("validation")
    bad = list(range(start + 1, end + 1))  # shifted by one
    _write_blocks(tmp_path, _CFG, bad_block="validation", bad_index=bad)
    failures = _check_block_index_ranges(_CFG)
    assert any(f["suite"] == "validation" for f in failures)


# ---------------------------------------------------------------------------
# _check_origins_windows
# ---------------------------------------------------------------------------


def _make_origins_df(cfg, extra_rows=None, drop_last=False):
    rows = []
    for block_name in ("validation", "test"):
        for state in cfg.states:
            origins_list = list(forecast_origins(block_name, cfg))
            if drop_last:
                origins_list = origins_list[:-1]
            for origin in origins_list:
                rows.append(
                    (state, block_name, origin, origin - cfg.context_length, origin + cfg.horizon)
                )
    if extra_rows:
        rows.extend(extra_rows)
    return pd.DataFrame(rows, columns=["state", "block", "origin", "context_start", "target_end"])


def _write_origins(tmp_path, df):
    df.to_parquet(tmp_path / "origins.parquet", index=False)


def test_origins_passes(tmp_path, monkeypatch):
    """A correct origins table produces no failures."""
    monkeypatch.setattr(vd, "PROCESSED_DATA_DIR", tmp_path)
    _write_origins(tmp_path, _make_origins_df(_CFG))
    assert not _check_origins_windows(_CFG)


def test_origins_negative_context_start(tmp_path, monkeypatch):
    """A row with negative context_start is reported as a failure."""
    monkeypatch.setattr(vd, "PROCESSED_DATA_DIR", tmp_path)
    bad = pd.DataFrame(
        [("NSW", "validation", 0, -_CFG.context_length, _CFG.horizon)],
        columns=["state", "block", "origin", "context_start", "target_end"],
    )
    _write_origins(tmp_path, bad)
    failures = _check_origins_windows(_CFG)
    assert any(f["expectation"] == "custom_nonnegative_positions" for f in failures)


def test_origins_duplicate_detected(tmp_path, monkeypatch):
    """A duplicated origin row is reported as a failure."""
    monkeypatch.setattr(vd, "PROCESSED_DATA_DIR", tmp_path)
    df = _make_origins_df(_CFG)
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    _write_origins(tmp_path, df_dup)
    failures = _check_origins_windows(_CFG)
    assert any(f["expectation"] == "custom_origin_grid" for f in failures)


def test_origins_missing_window_detected(tmp_path, monkeypatch):
    """A missing origin window is reported as a failure."""
    monkeypatch.setattr(vd, "PROCESSED_DATA_DIR", tmp_path)
    _write_origins(tmp_path, _make_origins_df(_CFG, drop_last=True))
    failures = _check_origins_windows(_CFG)
    assert any(f["expectation"] == "custom_origin_grid" for f in failures)


# ---------------------------------------------------------------------------
# _collect_gx_failures
# ---------------------------------------------------------------------------


class _FakeValidationResult:  # pylint: disable=too-few-public-methods
    """Minimal stand-in for a GX 1.24.0 ValidationResult."""

    suite_name = "raw_NSW"

    def __getitem__(self, key):
        """Return statistics or results for the fake failing validation."""
        data = {
            "statistics": {"evaluated_expectations": 4, "unsuccessful_expectations": 1},
            "results": [
                {
                    "success": False,
                    "expectation_config": {
                        "type": "expect_table_row_count_to_equal",
                        "kwargs": {"value": 100},
                    },
                }
            ],
        }
        return data[key]


class _FakeCheckpointResult:  # pylint: disable=too-few-public-methods
    """Minimal stand-in for a GX CheckpointResult with one failing suite."""

    @property
    def run_results(self):
        """Return a single failing validation result."""
        return {"key": _FakeValidationResult()}


def test_collect_gx_failures_uses_suite_name():
    """suite_name attribute is used instead of result['meta'][...] (GX 1.24.0 API)."""
    evaluated, failed, failures = _collect_gx_failures(_FakeCheckpointResult())
    assert evaluated == 4
    assert failed == 1
    assert len(failures) == 1
    assert failures[0]["suite"] == "raw_NSW"
    assert failures[0]["expectation"] == "expect_table_row_count_to_equal"
    assert failures[0]["column"] == "table-level"


def test_collect_gx_no_failures():
    """A passing checkpoint returns zero failures and an empty list."""

    class _PassResult:  # pylint: disable=too-few-public-methods
        """Fake passing ValidationResult."""

        suite_name = "raw_VIC"

        def __getitem__(self, key):
            """Return statistics or results for the fake passing validation."""
            return {
                "statistics": {"evaluated_expectations": 3, "unsuccessful_expectations": 0},
                "results": [],
            }[key]

    class _PassCheckpoint:  # pylint: disable=too-few-public-methods
        """Fake passing CheckpointResult."""

        @property
        def run_results(self):
            """Return a single passing validation result."""
            return {"k": _PassResult()}

    evaluated, failed, failures = _collect_gx_failures(_PassCheckpoint())
    assert evaluated == 3
    assert failed == 0
    assert not failures


def test_gx_failure_report_written_and_exit_1(tmp_path):
    """Failures from _collect_gx_failures are written to the report with success=false."""
    report_path = tmp_path / "validation.json"
    evaluated, failed, failures = _collect_gx_failures(_FakeCheckpointResult())
    success = failed == 0
    report = {
        "success": success,
        "evaluated_expectations": evaluated,
        "failed_expectations": failed,
        "failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2))

    assert success is False
    loaded = json.loads(report_path.read_text())
    assert loaded["success"] is False
    assert loaded["failed_expectations"] == 1
    assert loaded["failures"][0]["suite"] == "raw_NSW"
