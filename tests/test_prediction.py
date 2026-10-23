"""Selected-checkpoint serving validation without model weights or Torch."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import json
from random import Random
import sys
from threading import Barrier
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from typer.testing import CliRunner

from src.evaluation import sample_median
from src.modeling import predict
from src.modeling.checkpoints import METADATA_FILENAME, write_training_metadata
from src.modeling.predict import ChronosForecaster, ForecastConfig, app


@pytest.fixture
def backend(monkeypatch, tmp_path):
    """Use tiny weight fixtures and a fake Chronos/Torch backend."""
    checkpoint = tmp_path / "selected"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"selected weights")
    metadata = write_training_metadata(checkpoint, "a" * 32)
    provenance = checkpoint.with_suffix(".provenance.json")
    (checkpoint / METADATA_FILENAME).rename(provenance)
    monkeypatch.setattr(predict, "SELECTED_PROVENANCE", provenance)
    config = ForecastConfig.from_params(checkpoint=checkpoint)
    samples = (
        np.arange(48, dtype=np.float32)[None, None, :]
        + np.arange(20, dtype=np.float32)[None, :, None]
    )
    output = Mock()
    output.cpu.return_value.numpy.return_value = samples
    pipeline = Mock()
    pipeline.predict.return_value = output
    factory = Mock(return_value=pipeline)
    torch = SimpleNamespace(
        float32=np.float32,
        set_num_threads=Mock(),
        as_tensor=lambda values, dtype: np.asarray(values, dtype=dtype),
        inference_mode=nullcontext,
        random=SimpleNamespace(
            fork_rng=lambda devices: nullcontext(),
            default_generator=SimpleNamespace(manual_seed=Mock()),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "chronos",
        SimpleNamespace(ChronosPipeline=SimpleNamespace(from_pretrained=factory)),
    )
    return SimpleNamespace(
        samples=samples,
        output=output,
        pipeline=pipeline,
        factory=factory,
        config=config,
        metadata=metadata,
        torch=torch,
        provenance=provenance,
    )


def test_forecaster_reuses_selected_model_and_returns_traceable_quantiles(backend):
    config = backend.config
    forecaster = ChronosForecaster(config, local_files_only=True)
    history = np.linspace(-3, 20, 512)
    original = history.copy()
    result = forecaster.predict(history)
    assert forecaster.predict(history) == result
    backend.factory.assert_called_once_with(
        str(config.checkpoint),
        device_map="cpu",
        dtype=np.float32,
        token=False,
        local_files_only=True,
    )
    assert backend.pipeline.predict.call_count == 2
    context = backend.pipeline.predict.call_args.args[0]
    np.testing.assert_array_equal(context, original.astype(np.float32)[None, :])
    np.testing.assert_array_equal(history, original)
    assert backend.pipeline.predict.call_args.kwargs == {
        "prediction_length": 48,
        "num_samples": 20,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
    }
    assert (config.context_length, config.horizon) == (512, 48)
    assert result["model_id"] == "chronos-t5-small-fine-tuned"
    assert (
        result["model_revision"] == f"local-sha256:{backend.metadata['model_safetensors_sha256']}"
    )
    assert result["checkpoint"]["training_run_id"] == "a" * 32
    assert result["value_scale"] == "source"
    assert result["settings"]["cpu_threads"] == 2
    assert (result["context_length"], result["horizon"]) == (512, 48)
    assert result["settings"]["seed"] == 7
    assert result["settings"]["dtype"] == "float32"
    np.testing.assert_array_equal(result["p50"], sample_median(backend.samples, 1, 48)[0])
    np.testing.assert_allclose(result["p10"], np.arange(48) + 1.9)
    np.testing.assert_allclose(result["p90"], np.arange(48) + 17.1)
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize(
    "history",
    [
        [],
        [1.0] * 511,
        [1.0] * 513,
        [[1.0] * 512],
        ["1"] * 512,
        [True] + [1.0] * 511,
        [None] * 512,
        [1 + 2j] * 512,
        np.array([np.bool_(True)] * 512),
        [1.0] * 511 + ["2"],
        [float("nan")] + [1.0] * 511,
        [float("inf")] + [1.0] * 511,
        [1e40] + [1.0] * 511,
    ],
)
def test_invalid_history_fails_before_prediction(backend, history):
    forecaster = ChronosForecaster(backend.config)
    with pytest.raises(ValueError, match="History"):
        forecaster.predict(history)
    backend.pipeline.predict.assert_not_called()


@pytest.mark.parametrize(
    "samples",
    [
        None,
        [[[1.0] * 48] * 20],
        np.zeros((1, 20, 48), dtype=bool),
        np.full((1, 20, 48), "1"),
        np.zeros((1, 20, 48), dtype=complex),
        np.zeros((1, 20, 48), dtype=object),
        np.zeros((20, 48)),
        np.zeros((2, 20, 48)),
        np.zeros((1, 19, 48)),
        np.zeros((1, 20, 47)),
        np.full((1, 20, 48), np.nan),
        np.full((1, 20, 48), np.inf),
        np.full((1, 20, 48), float.fromhex("0x1.fffffep+127"), dtype=np.float32),
    ],
)
def test_invalid_model_samples_fail_clearly(backend, samples):
    backend.output.cpu.return_value.numpy.return_value = samples
    with pytest.raises(RuntimeError, match="finite samples with shape"):
        ChronosForecaster(backend.config).predict([1.0] * 512)


def test_model_loading_failure_keeps_the_cause(backend):
    backend.factory.side_effect = OSError("checkpoint unavailable")
    with pytest.raises(RuntimeError, match="Could not load") as caught:
        ChronosForecaster(backend.config, local_files_only=True)
    assert isinstance(caught.value.__cause__, OSError)


def test_prediction_failure_keeps_the_cause(backend):
    backend.pipeline.predict.side_effect = RuntimeError("inference unavailable")
    with pytest.raises(RuntimeError, match="Chronos prediction failed") as caught:
        ChronosForecaster(backend.config).predict([1.0] * 512)
    assert str(caught.value.__cause__) == "inference unavailable"


def test_missing_optional_dependencies_fail_clearly(monkeypatch, backend):
    monkeypatch.setitem(sys.modules, "chronos", None)
    with pytest.raises(RuntimeError, match="uv sync --extra forecast"):
        ChronosForecaster(backend.config)


@pytest.mark.parametrize(
    "change",
    [
        {"model_revision": "main"},
        {"num_samples": 0},
        {"cpu_threads": True},
        {"context_length": 0},
        {"context_length": 511},
        {"horizon": 47},
        {"checkpoint": "missing-directory"},
        {"horizon": 1.5},
        {"device": "cuda"},
        {"seed": -1},
    ],
)
def test_invalid_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        replace(ForecastConfig.from_params(), **change)


def test_cli_writes_forecast_json_and_rejects_bad_history_before_loading(backend, tmp_path):
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps([1.0] * 512))
    output_path = tmp_path / "output" / "forecast.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            str(history_path),
            "--checkpoint",
            str(backend.config.checkpoint),
            "--output-path",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(json.loads(output_path.read_text())["p50"]) == 48
    backend.factory.reset_mock()
    history_path.write_text(json.dumps([1.0] * 511))
    result = runner.invoke(app, [str(history_path)])
    assert result.exit_code == 1
    assert "exactly 512" in result.output
    backend.factory.assert_not_called()


def test_default_configuration_selects_the_promoted_local_checkpoint():
    config = ForecastConfig.from_params()
    assert config.checkpoint == predict.SELECTED_CHECKPOINT
    assert config.checkpoint.name == "chronos-t5-small-fine-tuned"
    assert config.cpu_threads == 2


def test_explicit_base_model_keeps_its_pinned_hub_identity(backend):
    config = replace(backend.config, checkpoint=None)
    result = ChronosForecaster(config, local_files_only=True).predict([1.0] * 512)
    backend.factory.assert_called_once_with(
        config.model_id,
        revision=config.model_revision,
        device_map="cpu",
        dtype=np.float32,
        token=False,
        local_files_only=True,
    )
    assert result["model_id"] == config.model_id
    assert result["model_revision"] == config.model_revision
    assert result["checkpoint"] is None


@pytest.mark.parametrize(
    "damage",
    [
        "missing-weights",
        "changed-weights",
        "missing-provenance",
        "malformed-provenance",
        "wrong-run",
        "conflicting-run",
    ],
)
def test_selected_identity_failure_never_loads_or_falls_back(backend, damage):
    weights = backend.config.checkpoint / "model.safetensors"
    if damage == "missing-weights":
        weights.unlink()
    elif damage == "changed-weights":
        weights.write_bytes(b"another checkpoint")
        # Even self-consistent metadata for replacement weights cannot override selection.
        write_training_metadata(backend.config.checkpoint, "b" * 32)
    elif damage == "missing-provenance":
        write_training_metadata(backend.config.checkpoint, "a" * 32)
        backend.provenance.unlink()
    elif damage == "malformed-provenance":
        backend.provenance.write_text("{")
    elif damage == "wrong-run":
        backend.provenance.write_text(json.dumps({**backend.metadata, "training_run_id": "bad"}))
    else:
        write_training_metadata(backend.config.checkpoint, "b" * 32)
    with pytest.raises(RuntimeError, match="Could not verify local checkpoint") as caught:
        ChronosForecaster(backend.config)
    assert isinstance(caught.value.__cause__, ValueError)
    backend.factory.assert_not_called()


def test_public_identity_has_no_paths_and_cannot_be_changed_by_a_caller(backend):
    forecaster = ChronosForecaster(backend.config)
    result = forecaster.predict([1.0] * 512)
    assert str(backend.config.checkpoint) not in json.dumps(result)
    result["checkpoint"]["sha256"] = "changed by caller"
    assert (
        forecaster.predict([1.0] * 512)["checkpoint"]["sha256"]
        == (backend.metadata["model_safetensors_sha256"])
    )


@pytest.mark.parametrize("failure", ["no-tensor", "tensor-conversion", "unordered-quantiles"])
def test_malformed_outputs_report_the_inference_contract(backend, monkeypatch, failure):
    if failure == "no-tensor":
        backend.pipeline.predict.return_value = None
    elif failure == "tensor-conversion":
        backend.output.cpu.side_effect = RuntimeError("invalid tensor")
    else:
        monkeypatch.setattr(predict, "sample_median", lambda *_: np.full((1, 48), -100))
    with pytest.raises(RuntimeError, match="finite samples with shape"):
        ChronosForecaster(backend.config).predict([1.0] * 512)


def test_cpu_rng_is_restored_and_sampling_serialized_across_instances(backend):
    # Model Torch's process-wide generator without importing optional model dependencies.
    rng = Random(123)
    active = []

    @contextmanager
    def fork_rng(devices):
        assert devices == []
        assert not active, "overlapping access to the shared CPU generator"
        active.append(True)
        state = rng.getstate()
        try:
            yield
        finally:
            rng.setstate(state)
            active.pop()

    backend.torch.random.fork_rng = fork_rng
    backend.torch.random.default_generator.manual_seed = rng.seed
    original_state = rng.getstate()

    def sample(*_args, **_kwargs):
        time.sleep(0.01)  # Release the GIL while another instance is ready to predict.
        output = Mock()
        output.cpu.return_value.numpy.return_value = np.array(
            [rng.gauss(0, 1) for _ in range(20 * 48)]
        ).reshape((1, 20, 48))
        return output

    backend.pipeline.predict.side_effect = sample
    backend.factory.side_effect = lambda *_args, **_kwargs: (rng.random(), backend.pipeline)[1]
    forecasters = [ChronosForecaster(backend.config), ChronosForecaster(backend.config)]
    barrier = Barrier(4)

    def forecast(index):
        barrier.wait(timeout=5)
        return forecasters[index % 2].predict([1.0] * 512)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(forecast, range(4)))
    assert all(result == results[0] for result in results)
    assert rng.getstate() == original_state
    assert backend.factory.call_count == 2

    def fail(*_args, **_kwargs):
        rng.random()
        raise RuntimeError("failed after consuming randomness")

    backend.pipeline.predict.side_effect = fail
    with pytest.raises(RuntimeError, match="Chronos prediction failed"):
        forecasters[0].predict([1.0] * 512)
    assert rng.getstate() == original_state


def test_cli_base_model_is_explicit_and_conflicting_sources_fail(backend, tmp_path):
    history = tmp_path / "history.json"
    history.write_text(json.dumps([1.0] * 512))
    runner = CliRunner()
    result = runner.invoke(app, [str(history), "--base-model", "--local-files-only"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["model_id"] == backend.config.model_id
    backend.factory.reset_mock()
    result = runner.invoke(app, [str(history), "--base-model", "--checkpoint", "somewhere"])
    assert result.exit_code == 1
    assert "Choose either" in result.output
    backend.factory.assert_not_called()


def test_cli_default_cannot_fall_back_when_selected_provenance_is_absent(backend, tmp_path):
    history = tmp_path / "history.json"
    history.write_text(json.dumps([1.0] * 512))
    backend.provenance.unlink()
    result = CliRunner().invoke(app, [str(history)])
    assert result.exit_code == 1
    assert "Could not verify local checkpoint" in result.output
    backend.factory.assert_not_called()


@pytest.mark.parametrize("text", ["null", "{", "forecast: {}\nevaluation: {}"])
def test_invalid_params_fail_clearly(tmp_path, text):
    params = tmp_path / "params.yaml"
    params.write_text(text)
    with pytest.raises(ValueError, match="forecast and evaluation settings"):
        ForecastConfig.from_params(params)
