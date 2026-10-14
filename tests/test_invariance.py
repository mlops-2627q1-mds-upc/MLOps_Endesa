"""Check paired-comparison logic; opt in separately for real model behaviour."""

from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation import seasonal_forecast
from src.invariance import (
    HORIZON,
    PATTERNS,
    TRANSFORMATIONS,
    History,
    compare_pair,
    stored_histories,
    synthetic_histories,
)
from src.modeling import evaluate as evaluator
from src.modeling import invariance as runner
from src.splits import SplitConfig


@pytest.mark.parametrize("history", synthetic_histories(), ids=PATTERNS)
@pytest.mark.parametrize("change", TRANSFORMATIONS, ids=lambda change: change.name)
def test_seasonal_reference_obeys_declared_relationships(history, change):
    original = history.values.copy()
    changed = change.apply(history)
    forecasts = seasonal_forecast(np.stack((original, changed)), HORIZON, lag=48)
    result = compare_pair(history, change, *forecasts)
    assert result["within_tolerance"]
    np.testing.assert_array_equal(history.values, original)
    if change.kind == "noise":
        assert result["input_max_absolute_change_relative"] <= change.amount + 1e-6
        assert result["forecast_max_absolute_change_relative"] <= change.amount + 1e-6
        np.testing.assert_array_equal(change.apply(history), changed)
    else:
        assert result["forecast_max_absolute_change_relative"] < 1e-6


@pytest.mark.parametrize(
    "values",
    [np.ones(511), np.ones((1, 512)), np.full(512, np.nan), np.full(512, np.inf), np.zeros(512)],
    ids=["short", "matrix", "nan", "infinity", "zero-scale"],
)
def test_invalid_histories_are_rejected(values):
    with pytest.raises(ValueError):
        History("invalid", values)


def test_comparison_checks_the_last_step_and_retains_a_counterexample():
    history = synthetic_histories()[0]
    original = np.zeros(HORIZON)
    changed = original.copy()
    changed[-1] = 0.2 * history.scale
    result = compare_pair(history, TRANSFORMATIONS[-1], original, changed)
    assert result["forecast_mean_absolute_change_relative"] == pytest.approx(0.2 / HORIZON)
    assert result["forecast_max_absolute_change_relative"] == pytest.approx(0.2)
    assert not result["within_tolerance"]
    assert result["transformed_forecast"][-1] == 200.0


@pytest.mark.parametrize("forecast", [np.ones(47), np.full(48, np.nan)], ids=["short", "nan"])
def test_incomplete_or_nonfinite_forecasts_are_rejected(forecast):
    with pytest.raises(ValueError, match="48 finite"):
        compare_pair(synthetic_histories()[0], TRANSFORMATIONS[0], np.ones(48), forecast)


@pytest.fixture
def stored_files(tmp_path):
    """Create versioned train/validation fixtures with an observable positional pattern."""
    cfg = SplitConfig(("NSW", "TAS"), 3456, 1152, 2304, 512, 48, 48)
    outputs = []
    for block in ("train", "validation"):
        start, end = cfg.bounds(block)
        positions = np.arange(start, end, dtype=np.float32)
        frame = pd.DataFrame(
            {"NSW": positions, "TAS": -positions},
            index=pd.RangeIndex(start, end, name="timestep"),
        )
        path = tmp_path / f"{block}.parquet"
        frame.to_parquet(path)
        outputs.append(
            {
                "path": f"data/processed/{path.name}",
                "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            }
        )
    lock = tmp_path / "dvc.lock"
    lock.write_text(yaml.safe_dump({"stages": {"split": {"outs": outputs}}}))
    return tmp_path, cfg, lock


def test_stored_windows_use_only_observations_before_their_origin(stored_files):
    directory, cfg, lock = stored_files
    histories = stored_histories(directory, cfg, lock)
    assert [history.name for history in histories] == [
        "NSW-train",
        "TAS-train",
        "NSW-validation",
        "TAS-validation",
    ]
    for history in histories:
        expected = np.arange(history.origin - 512, history.origin)
        if history.name.startswith("TAS"):
            expected = -expected
        np.testing.assert_array_equal(history.values, expected)
    # Change later validation observations, while retaining valid DVC metadata.
    path = directory / "validation.parquet"
    frame = pd.read_parquet(path)
    frame.loc[histories[-1].origin :] += 100000
    frame.to_parquet(path)
    metadata = yaml.safe_load(lock.read_text())
    metadata["stages"]["split"]["outs"][1]["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    lock.write_text(yaml.safe_dump(metadata))
    for original, updated in zip(histories, stored_histories(directory, cfg, lock)):
        np.testing.assert_array_equal(original.values, updated.values)


def test_changed_stored_inputs_stop_before_inference(stored_files):
    directory, cfg, lock = stored_files
    (directory / "validation.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match dvc.lock"):
        stored_histories(directory, cfg, lock)


def test_missing_stored_inputs_require_an_explicit_synthetic_only_choice(stored_files):
    directory, cfg, lock = stored_files
    (directory / "validation.parquet").unlink()
    with pytest.raises(ValueError, match="Restore validation.parquet"):
        stored_histories(directory, cfg, lock)


def test_runner_records_protocol_before_inference_and_keeps_all_outcomes(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "_versions", lambda _: {"fixture": True})
    calls = []

    def fake_forecast(contexts, horizon, cfg, deadline, *, reset_seed_per_batch):
        prepared = json.loads(next(tmp_path.glob("*/report.json")).read_text())
        assert prepared["status"] == "prepared"
        assert prepared["results"] == []
        assert prepared["transformations"][0]["mean_tolerance"] == 1e-5
        assert cfg.batch_size == 1 and reset_seed_per_batch
        assert deadline > time.monotonic()
        calls.append(contexts.copy())
        predictions = seasonal_forecast(contexts, horizon, 48)
        predictions[1, -1] += 1000  # Deliberately violate one relationship.
        return predictions, {"model_load_seconds": 0.01, "inference_seconds": 0.02}

    monkeypatch.setattr(runner, "chronos_forecast", fake_forecast)
    result = runner.run_checks(output_dir=tmp_path, synthetic_only=True)
    assert len(calls) == 1
    assert calls[0].shape == (40, 512)
    assert result["status"] == "complete"
    assert len(result["results"]) == 20
    assert result["counterexamples"] == ["constant/scale-0.5"]
    saved = json.loads(next(tmp_path.glob("*/report.json")).read_text())
    assert saved == result


@pytest.mark.parametrize("failure", ["timeout", "shape"], ids=["timeout", "incomplete-forecasts"])
def test_runner_never_labels_an_incomplete_run_complete(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(runner, "_versions", lambda _: {"fixture": True})

    def fail(*_args, **_kwargs):
        if failure == "timeout":
            raise TimeoutError("fixture deadline")
        return np.zeros((1, 48)), {}

    monkeypatch.setattr(runner, "chronos_forecast", fail)
    with pytest.raises((TimeoutError, ValueError)):
        runner.run_checks(output_dir=tmp_path, synthetic_only=True)
    saved = json.loads(next(tmp_path.glob("*/report.json")).read_text())
    assert saved["status"] == "failed"
    assert saved["results"] == []
    assert "error" in saved


@pytest.mark.parametrize("reset_seed", [False, True], ids=["ordinary-stream", "matched-draws"])
def test_evaluator_loads_once_and_can_match_random_draws(monkeypatch, reset_seed):
    loads, seeds = [], []
    random = [None]

    def seed(value):
        seeds.append(value)
        random[0] = np.random.default_rng(value)

    def predict(context, *, prediction_length, num_samples, **_kwargs):
        samples = random[0].normal(size=(len(context), num_samples, prediction_length))
        return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: samples))

    def load(*_args, **kwargs):
        loads.append(kwargs)
        return SimpleNamespace(
            model=SimpleNamespace(model=SimpleNamespace(eval=lambda: None)), predict=predict
        )

    monkeypatch.setitem(
        sys.modules,
        "chronos",
        SimpleNamespace(ChronosPipeline=SimpleNamespace(from_pretrained=load)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            manual_seed=seed,
            set_num_threads=lambda _: None,
            inference_mode=nullcontext,
            as_tensor=np.asarray,
            float32=np.float32,
        ),
    )
    cfg = replace(evaluator.EvaluationConfig.from_params(runner.PARAMS_PATH), batch_size=1)
    forecasts, _ = evaluator.chronos_forecast(
        np.ones((2, 512)),
        HORIZON,
        cfg,
        time.monotonic() + 30,
        reset_seed_per_batch=reset_seed,
    )
    assert len(loads) == 1
    assert loads[0]["revision"] == cfg.model_revision
    assert seeds == [cfg.seed] * (3 if reset_seed else 1)
    assert np.array_equal(forecasts[0], forecasts[1]) == reset_seed


def test_matched_sampling_rejects_multi_history_batches():
    cfg = evaluator.EvaluationConfig.from_params(runner.PARAMS_PATH)
    with pytest.raises(ValueError, match="batch_size=1"):
        evaluator.chronos_forecast(
            np.ones((2, 512)), HORIZON, cfg, time.monotonic() + 30, reset_seed_per_batch=True
        )


@pytest.fixture(scope="module")
def real_results(request):
    """One optional real run serves every parametrized assertion and saves all outcomes."""
    if not request.config.getoption("--run-forecast-invariance"):
        pytest.skip("Real checkpoint checks require --run-forecast-invariance")
    report = runner.run_checks()
    return {result["case_id"]: result for result in report["results"]}


REAL_HISTORY_NAMES = [*PATTERNS] + [
    f"{state}-{block}"
    for block in ("train", "validation")
    for state in SplitConfig.from_params().states
]


@pytest.mark.parametrize("history_name", REAL_HISTORY_NAMES)
@pytest.mark.parametrize("change", TRANSFORMATIONS, ids=lambda change: change.name)
def test_real_model_relationship(real_results, history_name, change):
    result = real_results[f"{history_name}/{change.name}"]
    assert result["within_tolerance"], (
        f"{result['case_id']}: mean change={result['forecast_mean_absolute_change_relative']:.6g} "
        f"(limit {change.mean_tolerance:g}); "
        f"max change={result['forecast_max_absolute_change_relative']:.6g} "
        f"(limit {change.max_tolerance:g}); see the saved report for both full forecasts"
    )
