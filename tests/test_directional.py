"""Synthetic case construction, reporting and opt-in real-checkpoint expectations."""

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.modeling import directional, evaluate


@pytest.mark.parametrize("case", directional.CASES, ids=lambda case: case.case_id)
def test_paired_histories_preserve_pattern_and_change_only_recent_level(case):
    original, changed = directional.paired_histories(case)
    assert original.shape == changed.shape == (512,)
    assert original.dtype == changed.dtype == np.float32
    assert np.isfinite(changed).all() and (changed > 0).all()
    np.testing.assert_array_equal(changed[: -case.recent], original[: -case.recent])
    np.testing.assert_allclose(
        changed[-case.recent :] - original[-case.recent :],
        case.sign * case.amplitude * 1000,
        atol=0.0002,
    )
    np.testing.assert_allclose(
        np.diff(changed[-case.recent :]), np.diff(original[-case.recent :]), atol=0.0002
    )


@pytest.mark.parametrize("direction", ["increase", "decrease"])
def test_direction_is_a_horizon_summary_not_a_stepwise_requirement(direction):
    case = directional.DirectionalCase(direction, 0.05, 48)
    original = np.full(48, 1000.0)
    changed = original + case.sign * 20
    changed[0] = original[0] - case.sign * 30
    result = directional.compare_forecasts(case, original, changed)
    assert result["passed"]
    assert result["signed_change"] > result["minimum_signed_change"] == 5
    for delta in (0, case.sign * 4, -case.sign * 50):
        assert not directional.compare_forecasts(case, original, original + delta)["passed"]
    assert directional.compare_forecasts(case, original, original + case.sign * 5)["passed"]


@pytest.mark.parametrize(
    "forecast",
    [np.ones(47), np.full(48, np.nan), np.full(48, np.inf)],
    ids=["wrong-horizon", "nan", "infinity"],
)
def test_invalid_forecasts_cannot_support_an_expectation(forecast):
    with pytest.raises(ValueError, match="finite 48-step"):
        directional.compare_forecasts(directional.CASES[0], np.ones(48), forecast)


@pytest.mark.parametrize(
    "values",
    [
        ("spike", 0.05, 48),
        ("increase", 0, 48),
        ("increase", np.nan, 48),
        ("decrease", 0.6, 48),
        ("decrease", 0.05, 1),
    ],
    ids=["unknown-direction", "zero-change", "nan-change", "excessive-change", "spike-window"],
)
def test_invalid_case_definitions_are_rejected(values):
    with pytest.raises(ValueError):
        directional.DirectionalCase(*values)


def test_report_keeps_every_counterexample_and_detects_control_drift(monkeypatch):
    settings = replace(
        evaluate.EvaluationConfig.from_params(directional.PARAMS_PATH), batch_size=1
    )
    plan = directional._plan(settings)  # pylint: disable=protected-access
    calls = []

    def forecast(contexts, horizon, cfg, deadline, *, reset_seed_per_context):
        calls.append(contexts.copy())
        assert horizon == 48 and cfg == settings and reset_seed_per_context
        assert deadline > time.monotonic()
        np.testing.assert_array_equal(contexts[0], contexts[1])
        # Deliberately return no response to any intervention and let the control drift.
        predictions = np.full((len(contexts), horizon), 1000.0)
        predictions[1, 0] += 1
        predictions[1, 1] -= 1
        return predictions, {"model_load_seconds": 0.1, "inference_seconds": 0.2}

    monkeypatch.setattr(directional, "chronos_forecast", forecast)
    report = directional.execute_plan(plan)
    assert len(calls) == 1
    assert calls[0].shape == (14, 512)
    assert report["status"] == "completed" and not report["success"]
    assert len(report["cases"]) == 12
    assert all(not result["passed"] for result in report["cases"])
    assert not report["control"]["passed"]
    assert report["control"]["max_absolute_difference"] == 1
    assert report["history_sha256"][0] == report["history_sha256"][1]
    assert len(set(report["history_sha256"])) == 13


@pytest.mark.parametrize("outcome", ["counterexample", "timeout", "worker-error"])
def test_bounded_runner_saves_plan_and_preserves_failed_outcomes(tmp_path, monkeypatch, outcome):
    settings = replace(
        evaluate.EvaluationConfig.from_params(directional.PARAMS_PATH),
        batch_size=1,
        cpu_threads=2,
        max_seconds=180,
    )
    plan = directional._plan(settings)  # pylint: disable=protected-access
    monkeypatch.setattr(directional, "_plan", lambda _: plan)

    def worker(command, *, cwd, check, timeout):
        directory = Path(command[-1])
        plan = json.loads((directory / "plan.json").read_text())
        assert plan["settings"]["batch_size"] == 1
        assert plan["minimum_response_fraction"] == 0.1
        assert plan["settings"]["max_seconds"] == 180
        assert 0 < timeout <= 180 and not check and cwd == directional.PROJ_ROOT
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        if outcome == "counterexample":
            (directory / "report.json").write_text(
                json.dumps({"status": "completed", "success": False, "cases": [{"passed": False}]})
            )

    monkeypatch.setattr(directional.subprocess, "run", worker)
    path, report = directional.run_bounded(output_dir=tmp_path)
    assert json.loads(path.read_text()) == report
    assert not report["success"]
    assert report["wall_seconds"] >= 0
    assert (
        report["status"]
        == {"counterexample": "completed", "timeout": "timeout", "worker-error": "error"}[outcome]
    )
    if outcome == "counterexample":
        assert report["cases"] == [{"passed": False}]


def test_adapter_loads_once_and_resets_seed_before_each_context(monkeypatch):
    seeds, loads, predictions = [], [], []

    class Tensor:
        """The three tensor operations used by the inference adapter."""

        def __init__(self, values):
            self.values = values

        def cpu(self):
            """Stand in for result transfer to CPU."""
            return self

        def numpy(self):
            """Expose samples for the shared median calculation."""
            return self.values

    def predict(batch, **sampling):
        predictions.append((batch, sampling, seeds[-1]))
        return Tensor(np.repeat(batch[:, -1:, None], 20, axis=1).repeat(48, axis=2))

    pipeline = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(eval=lambda: None)), predict=predict
    )

    def load(model_id, **kwargs):
        loads.append((model_id, kwargs))
        return pipeline

    torch = SimpleNamespace(
        set_num_threads=lambda _: None,
        manual_seed=seeds.append,
        as_tensor=lambda values, **_: values,
        float32=np.float32,
        inference_mode=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "chronos",
        SimpleNamespace(ChronosPipeline=SimpleNamespace(from_pretrained=load)),
    )
    settings = replace(
        evaluate.EvaluationConfig.from_params(directional.PARAMS_PATH), batch_size=1
    )
    contexts = np.ones((3, 512))
    medians, _ = evaluate.chronos_forecast(
        contexts, 48, settings, time.monotonic() + 30, reset_seed_per_context=True
    )
    assert medians.shape == (3, 48)
    assert len(loads) == 1 and len(predictions) == 3
    assert loads[0][1]["revision"] == settings.model_revision
    assert seeds == [settings.seed] * 4
    assert all(values[2] == settings.seed for values in predictions)
    assert all(
        values[1]
        == {
            "prediction_length": 48,
            "num_samples": 20,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
        }
        for values in predictions
    )
    with pytest.raises(ValueError, match="batch_size=1"):
        evaluate.chronos_forecast(
            contexts,
            48,
            replace(settings, batch_size=2),
            time.monotonic() + 30,
            reset_seed_per_context=True,
        )


@pytest.fixture(scope="session")
def real_report(request):
    """Share one bounded real-checkpoint run across all parameterized expectations."""
    if not request.config.getoption("--run-directional"):
        pytest.skip("Use --run-directional with the forecast extra for the real checkpoint")
    path, report = directional.run_bounded()
    print(f"\nReal directional report: {path}")
    assert report["status"] == "completed", report.get("error", report["status"])
    return report


def test_real_unchanged_control(real_report):
    assert real_report["control"]["passed"], real_report["control"]


@pytest.mark.parametrize("case", directional.CASES, ids=lambda case: case.case_id)
def test_real_directional_expectation(case, real_report):
    result = next(result for result in real_report["cases"] if result["case_id"] == case.case_id)
    assert result["passed"], result
