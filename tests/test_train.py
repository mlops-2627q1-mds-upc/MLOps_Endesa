import numpy as np
import pandas as pd
import pytest

from src.modeling import train
from src.splits import SplitConfig

CFG = SplitConfig(
    states=("A", "B"),
    common_length=400,
    train_end=200,
    validation_end=300,
    context_length=20,
    horizon=4,
    origin_stride=4,
)


def settings(**changes):
    values = {
        "learning_rate": 0.001,
        "steps": 10,
        "batch_size": 4,
        "micro_batch_size": 2,
        "seed": 7,
        "window_start": 0,
        "checkpoint_every": 5,
        "device": "cpu",
        "max_seconds": 60,
    }
    return train.FineTuneConfig(**{**values, **changes})


@pytest.mark.parametrize(
    "change",
    [
        {"learning_rate": 0.0},
        {"learning_rate": float("nan")},
        {"steps": 0},
        {"batch_size": 0},
        {"micro_batch_size": 0},
        {"micro_batch_size": 3},
        {"window_start": -1},
        {"device": "mps"},
    ],
)
def test_invalid_fine_tuning_settings_are_rejected(change):
    with pytest.raises(ValueError):
        settings(**change)


def test_windows_never_leave_the_training_block():
    # Positions are encoded in the values, so each window shows where it came from.
    training = np.stack([np.arange(200), np.arange(200) + 1000], axis=1).astype(np.float32)
    bounds = train.window_start_bounds(CFG, 50, prediction_length=8)
    assert bounds == (50, 200 - 20 - 8)
    contexts, labels = train.sample_windows(training, np.random.default_rng(0), 500, 20, 8, bounds)
    windows = np.concatenate((contexts, labels), axis=1) % 1000
    assert contexts.shape == (500, 20) and labels.shape == (500, 8)
    assert windows.min() >= 50 and windows.max() < CFG.train_end
    # Each window is one contiguous stretch of a single state.
    assert (np.diff(windows, axis=1) == 1).all()


def test_window_start_must_leave_room_for_a_window():
    with pytest.raises(ValueError, match="No training window"):
        train.window_start_bounds(CFG, 200 - 20 - 8 + 1, prediction_length=8)


def test_sampling_is_reproducible_for_a_seed():
    training = np.random.default_rng(1).normal(size=(200, 2))
    bounds = train.window_start_bounds(CFG, 0, prediction_length=8)
    first = train.sample_windows(training, np.random.default_rng(7), 8, 20, 8, bounds)
    second = train.sample_windows(training, np.random.default_rng(7), 8, 20, 8, bounds)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_learning_rate_decays_linearly_to_zero():
    assert [train.linear_decay(step, 4) for step in range(6)] == [1.0, 0.75, 0.5, 0.25, 0.0, 0.0]


def test_only_the_training_block_is_read(tmp_path):
    frame = pd.DataFrame(
        {"A": np.arange(200.0), "B": np.arange(200.0)},
        index=pd.RangeIndex(0, 200, name="timestep"),
    )
    frame.to_parquet(tmp_path / "train.parquet")
    assert train.load_training_block(tmp_path, CFG).shape == (200, 2)
    frame.iloc[:150].to_parquet(tmp_path / "train.parquet")
    with pytest.raises(ValueError, match="positions"):
        train.load_training_block(tmp_path, CFG)


def test_padded_label_tokens_are_ignored_by_the_loss():
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        def context_input_transform(self, context):
            return context.long(), torch.ones_like(context, dtype=torch.bool), torch.ones(1)

        def label_input_transform(self, label, scale):
            mask = torch.ones_like(label, dtype=torch.bool)
            mask[:, -1] = False
            return label.long(), mask

    batch = train.tokenize(FakeTokenizer(), np.ones((2, 3)), np.full((2, 4), 5.0))
    assert batch["labels"][:, :-1].eq(5).all()
    assert batch["labels"][:, -1].eq(-100).all()
