"""Fast checks for known continuations, error limits, and result preservation."""

import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation import seasonal_forecast
from src.modeling import minimum_functionality as checks
from src.modeling.minimum_functionality import CASES, CONTEXT_LENGTH, HORIZON, score_case
from src.splits import PARAMS_PATH


@pytest.mark.parametrize("case", CASES, ids=[case.case_id for case in CASES])
def test_known_continuation_repeats_the_last_observed_cycle(case):
    context, target = case.series()
    assert context.shape == (CONTEXT_LENGTH,)
    assert target.shape == (HORIZON,)
    # 512 is not a multiple of 48; resetting the cycle at the boundary is wrong.
    np.testing.assert_array_equal(target, context[-HORIZON:])
    prediction = seasonal_forecast(context[None, :], HORIZON, lag=48)[0]
    result = score_case(case, prediction)
    assert result["status"] == "passed"
    assert result["mae"] == result["max_absolute_error"] == 0.0


@pytest.mark.parametrize("case", CASES[4:], ids=[case.case_id for case in CASES[4:]])
def test_flat_forecast_fails_for_a_repeating_pattern(case):
    result = score_case(case, np.full(HORIZON, case.level))
    assert result["status"] == "failed"
    assert result["mae"] > case.mae_limit


def test_zero_demand_has_absolute_limits_and_checks_a_single_bad_step():
    case = CASES[3]
    assert score_case(case, np.full(HORIZON, 0.005))["status"] == "passed"
    assert score_case(case, np.full(HORIZON, 0.015))["status"] == "failed"
    prediction = np.zeros(HORIZON)
    prediction[-1] = 0.03
    result = score_case(case, prediction)
    assert result["mae"] < case.mae_limit
    assert result["max_absolute_error"] > case.max_error_limit
    assert result["status"] == "failed"


@pytest.mark.parametrize(
    "prediction",
    [np.zeros(47), np.zeros((1, 48)), np.full(48, np.nan), np.full(48, np.inf)],
    ids=["short-horizon", "wrong-dimensions", "nan", "infinity"],
)
def test_invalid_forecasts_are_recorded_as_failures(prediction):
    result = score_case(CASES[0], prediction)
    assert result["status"] == "invalid"
    json.dumps(result, allow_nan=False)


def test_signed_errors_cannot_cancel_out():
    case = CASES[3]
    prediction = np.tile([-0.02, 0.02], HORIZON // 2)
    assert prediction.mean() == 0.0
    assert score_case(case, prediction)["mae"] == pytest.approx(0.02)
    assert score_case(case, prediction)["status"] == "failed"


@pytest.mark.parametrize("outcome", ["passed", "failed", "timeout", "incomplete"])
def test_runner_records_protocol_before_inference_and_preserves_outcomes(
    tmp_path, monkeypatch, outcome
):
    path = tmp_path / "results.json"

    def fake_pytest(command, *, cwd, timeout, check):
        assert "--maxfail=0" in command
        assert "--run-model-tests" in command
        assert command[-1] == "tests/test_minimum_functionality_model.py"
        assert cwd == checks.PROJ_ROOT
        assert timeout == 5
        assert check is False
        report = json.loads(path.read_text())
        assert report["status"] == "running"
        assert len(report["cases"]) == 10
        assert all(row["status"] == "not_run" for row in report["cases"])
        assert report["protocol"]["cases"] == [
            {key: value for key, value in row.items() if key != "status"}
            for row in report["cases"]
        ]
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        if outcome == "incomplete":
            # An interrupted/empty pytest run must never count as a model pass.
            return SimpleNamespace(returncode=0)
        results = []
        for case in CASES:
            _, target = case.series()
            results.append(score_case(case, target + (100000 if outcome == "failed" else 0)))
        report.update(status="completed", cases=results)
        checks.write_report(path, report)
        return SimpleNamespace(returncode=1 if outcome == "failed" else 0)

    monkeypatch.setattr(
        checks, "_provenance", lambda: {"git_commit": "fixture", "git_dirty": False}
    )
    monkeypatch.setattr(checks.subprocess, "run", fake_pytest)
    exit_code = checks.run_model_tests(output_path=path, max_seconds=5)
    report = json.loads(path.read_text())
    assert report["protocol_sha256"]
    assert report["settings"]["batch_size"] == 1
    assert report["settings"]["cpu_threads"] == 2
    assert report["settings"]["max_seconds"] == 5
    assert report["runtime_seconds"] >= 0
    assert report["status"] == ("completed" if outcome in ("passed", "failed") else outcome)
    assert exit_code == {"passed": 0, "failed": 1, "timeout": 124, "incomplete": 2}[outcome]
    if outcome == "failed":
        assert len(report["cases"]) == 10
        assert all(row["status"] == "failed" for row in report["cases"])


def test_existing_evidence_is_not_overwritten(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("existing evidence\n")
    with pytest.raises(FileExistsError):
        checks.run_model_tests(output_path=path)
    assert path.read_text() == "existing evidence\n"


def test_different_forecast_contract_is_rejected_before_inference(tmp_path):
    params = tmp_path / "params.yaml"
    params.write_text(
        PARAMS_PATH.read_text().replace("context_length: 512", "context_length: 400")
    )
    with pytest.raises(ValueError, match="512 context"):
        checks.run_model_tests(output_path=tmp_path / "report.json", params_path=params)


@pytest.mark.parametrize("budget", [0, 301])
def test_unbounded_runtime_is_rejected(tmp_path, budget):
    with pytest.raises(ValueError, match="between 1 and 300"):
        checks.run_model_tests(output_path=tmp_path / "report.json", max_seconds=budget)
