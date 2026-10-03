"""Split boundaries, forecast origins and leakage checks for src.splits."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from src import splits

CFG = splits.SplitConfig.from_params()
# Each value equals its position, so a window's values show which positions it used.
POSITIONS = np.arange(CFG.common_length)


def small_config(**changes):
    cfg = splits.SplitConfig(
        states=("A", "B"),
        common_length=40,
        train_end=20,
        validation_end=30,
        context_length=8,
        horizon=2,
        origin_stride=2,
    )
    return replace(cfg, **changes)


def test_params_hold_the_proposed_protocol():
    assert CFG.states == ("NSW", "VIC", "QLD", "SA", "TAS")
    assert CFG.bounds("train") == (0, 195_696)
    assert CFG.bounds("validation") == (195_696, 213_216)
    assert CFG.bounds("test") == (213_216, 230_736)
    assert (CFG.context_length, CFG.horizon, CFG.origin_stride) == (512, 48, 48)


def test_blocks_partition_the_common_prefix_in_order():
    pieces = [splits.block(POSITIONS, name, CFG) for name in splits.BLOCKS]
    np.testing.assert_array_equal(np.concatenate(pieces), POSITIONS)
    for name in splits.HELD_OUT_BLOCKS:
        assert len(splits.block(POSITIONS, name, CFG)) == 365 * 48


@pytest.mark.parametrize("name", splits.HELD_OUT_BLOCKS)
def test_origins_are_daily_and_targets_stay_in_their_block(name):
    start, end = CFG.bounds(name)
    origins = splits.forecast_origins(name, CFG)
    assert len(origins) == 365
    assert origins[0] == start
    assert set(np.diff(origins)) == {48}
    assert origins[-1] + CFG.horizon == end


@pytest.mark.parametrize("name", splits.HELD_OUT_BLOCKS)
def test_context_never_contains_the_origin_or_later_targets(name):
    for origin in splits.forecast_origins(name, CFG):
        context, target = splits.forecast_window(POSITIONS, int(origin), CFG)
        assert len(context) == CFG.context_length
        assert context.max() == origin - 1
        np.testing.assert_array_equal(target, np.arange(origin, origin + CFG.horizon))


def test_first_test_context_uses_only_earlier_validation_targets():
    first = int(splits.forecast_origins("test", CFG)[0])
    context, _ = splits.forecast_window(POSITIONS, first, CFG)
    validation_start, validation_end = CFG.bounds("validation")
    assert context.min() >= validation_start
    assert context.max() < validation_end == first


def test_training_block_has_no_forecast_origins():
    with pytest.raises(ValueError):
        splits.forecast_origins("train", CFG)


@pytest.mark.parametrize(
    "changes",
    [
        {"validation_end": 20},  # empty validation block
        {"validation_end": 45},  # beyond the common prefix
        {"context_length": 21},  # first validation origin lacks a full context
        {"origin_stride": 3},  # origins do not tile the validation block
        {"horizon": 0},
    ],
)
def test_invalid_configs_are_rejected(changes):
    with pytest.raises(ValueError):
        small_config(**changes)


def test_truncate_rejects_a_short_series():
    with pytest.raises(ValueError):
        splits.truncate(np.arange(39), small_config())


def test_window_rejects_an_origin_without_full_context():
    with pytest.raises(ValueError):
        splits.forecast_window(np.arange(40), 7, small_config())


def test_main_writes_blocks_and_origins(tmp_path):
    cfg = small_config()
    params = {
        "data": {"states": list(cfg.states), "common_length": cfg.common_length},
        "split": {"train_end": cfg.train_end, "validation_end": cfg.validation_end},
        "forecast": {
            "context_length": cfg.context_length,
            "horizon": cfg.horizon,
            "origin_stride": cfg.origin_stride,
        },
    }
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(params))
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir()
    for state, length in (("A", 40), ("B", 44)):  # B has extra steps to drop
        pd.DataFrame({state: np.arange(length, dtype=np.float32)}).to_parquet(
            raw_dir / f"{state}.parquet"
        )

    splits.main(raw_dir=raw_dir, output_dir=out_dir, params_path=params_path)

    for name in splits.BLOCKS:
        frame = pd.read_parquet(out_dir / f"{name}.parquet")
        start, end = cfg.bounds(name)
        assert list(frame.columns) == ["A", "B"]
        np.testing.assert_array_equal(frame.index, np.arange(start, end))
        np.testing.assert_array_equal(frame["B"], np.arange(start, end))

    origins = pd.read_parquet(out_dir / "origins.parquet")
    assert len(origins) == 2 * 2 * 5  # blocks x states x origins per block
    assert (origins["origin"] - origins["context_start"] == cfg.context_length).all()
    assert (origins["target_end"] - origins["origin"] == cfg.horizon).all()
    assert origins.groupby("block")["target_end"].max().to_dict() == {
        "validation": 30,
        "test": 40,
    }
