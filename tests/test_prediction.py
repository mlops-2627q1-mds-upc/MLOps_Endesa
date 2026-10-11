from contextlib import nullcontext
from dataclasses import replace
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from typer.testing import CliRunner

from src.evaluation import sample_median
from src.modeling.predict import ChronosForecaster, ForecastConfig, app


@pytest.fixture
def backend(monkeypatch):
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
    return SimpleNamespace(samples=samples, output=output, pipeline=pipeline, factory=factory)


def test_forecaster_reuses_pinned_model_and_returns_traceable_quantiles(backend):
    config = ForecastConfig.from_params()
    forecaster = ChronosForecaster(config, local_files_only=True)
    history = np.linspace(-3, 20, 512)
    original = history.copy()
    result = forecaster.predict(history)
    assert forecaster.predict(history) == result
    backend.factory.assert_called_once_with(
        config.model_id,
        revision=config.model_revision,
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
    assert result["model_id"] == config.model_id
    assert result["model_revision"] == config.model_revision
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
        [float("nan")] + [1.0] * 511,
        [float("inf")] + [1.0] * 511,
        [1e40] + [1.0] * 511,
    ],
)
def test_invalid_history_fails_before_prediction(backend, history):
    forecaster = ChronosForecaster()
    with pytest.raises(ValueError, match="History"):
        forecaster.predict(history)
    backend.pipeline.predict.assert_not_called()


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros((20, 48)),
        np.zeros((1, 19, 48)),
        np.zeros((1, 20, 47)),
        np.full((1, 20, 48), np.nan),
        np.full((1, 20, 48), np.inf),
        np.full((1, 20, 48), np.finfo(np.float32).max, dtype=np.float32),
    ],
)
def test_invalid_model_samples_fail_clearly(backend, samples):
    backend.output.cpu.return_value.numpy.return_value = samples
    with pytest.raises(RuntimeError, match="finite samples with shape"):
        ChronosForecaster().predict([1.0] * 512)


def test_model_loading_failure_keeps_the_cause(backend):
    backend.factory.side_effect = OSError("checkpoint unavailable")
    with pytest.raises(RuntimeError, match="Could not load") as caught:
        ChronosForecaster(local_files_only=True)
    assert isinstance(caught.value.__cause__, OSError)


def test_prediction_failure_keeps_the_cause(backend):
    backend.pipeline.predict.side_effect = RuntimeError("inference unavailable")
    with pytest.raises(RuntimeError, match="Chronos prediction failed") as caught:
        ChronosForecaster().predict([1.0] * 512)
    assert str(caught.value.__cause__) == "inference unavailable"


def test_missing_optional_dependencies_fail_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "chronos", None)
    with pytest.raises(RuntimeError, match="uv sync --extra forecast"):
        ChronosForecaster()


@pytest.mark.parametrize(
    "change",
    [
        {"model_revision": "main"},
        {"num_samples": 0},
        {"cpu_threads": True},
        {"context_length": 0},
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
    result = runner.invoke(app, [str(history_path), "--output-path", str(output_path)])
    assert result.exit_code == 0, result.output
    assert len(json.loads(output_path.read_text())["p50"]) == 48
    backend.factory.reset_mock()
    history_path.write_text(json.dumps([1.0] * 511))
    result = runner.invoke(app, [str(history_path)])
    assert result.exit_code == 1
    assert "exactly 512" in result.output
    backend.factory.assert_not_called()
