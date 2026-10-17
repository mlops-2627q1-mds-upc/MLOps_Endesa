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


def tiny_t5():
    transformers = pytest.importorskip("transformers")
    config = transformers.T5Config(
        vocab_size=32,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_heads=2,
        decoder_start_token_id=0,
    )
    return transformers.T5ForConditionalGeneration(config)


def test_full_fine_tuning_trains_every_weight_and_saves_no_adapter(tmp_path):
    model = tiny_t5()
    model, trainable = train.prepare_model(model, settings())
    assert len(trainable) == len(list(model.parameters()))
    saved = train.save_final(model, tmp_path, lora=False)
    assert (tmp_path / "final" / "model.safetensors").is_file()
    assert "adapter_path" not in saved


def test_lora_trains_only_adapters_and_saves_a_loadable_merged_checkpoint(tmp_path):
    pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model, trainable = train.prepare_model(tiny_t5(), settings(lora_rank=2))
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert names and all("lora_" in name for name in names)
    # Names end in <module>.lora_<A|B>.default.weight.
    assert {name.split(".")[-4] for name in names} == set(train.LORA_TARGET_MODULES)
    assert len(trainable) == len(names)
    # Move the adapters away from their zero start, so merging changes the weights.
    with torch.no_grad():
        for parameter in trainable:
            parameter.add_(0.1)
    inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "decoder_input_ids": torch.tensor([[0, 4]])}
    model.eval()
    expected = model(**inputs).logits

    saved = train.save_final(model, tmp_path, lora=True)
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()
    assert len(saved["adapter_safetensors_sha256"]) == 64
    reloaded = transformers.T5ForConditionalGeneration.from_pretrained(tmp_path / "final").eval()
    torch.testing.assert_close(reloaded(**inputs).logits, expected)


def test_invalid_lora_settings_are_rejected():
    with pytest.raises(ValueError, match="lora_rank"):
        settings(lora_rank=-1)
    with pytest.raises(ValueError, match="lora_alpha"):
        settings(lora_rank=8, lora_alpha=0)


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
