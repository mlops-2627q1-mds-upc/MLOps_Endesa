"""Opt-in behavioural checks against the real, pinned Chronos checkpoint."""

import json
from pathlib import Path
import time

import numpy as np
import pytest

from src.modeling.behaviour_source import checkpoint_for_model
from src.modeling.evaluate import EvaluationConfig, chronos_forecast
from src.modeling.minimum_functionality import CASES, HORIZON, score_case, write_report

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def model_results(request):
    """Load once, forecast all cases, and save all outcomes before assertions."""
    if not request.config.getoption("--run-model-tests"):
        pytest.skip("Run python -m src.modeling.minimum_functionality for bounded real inference")
    report_path = request.config.getoption("--model-test-report")
    if report_path is None:
        pytest.fail(
            "Use python -m src.modeling.minimum_functionality to set the budget and report"
        )
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    settings = EvaluationConfig(**report["settings"])
    checkpoint = checkpoint_for_model(report["model"])
    contexts = np.stack([case.series()[0] for case in CASES])
    predictions, timing = chronos_forecast(
        contexts,
        HORIZON,
        settings,
        deadline=time.monotonic() + settings.max_seconds,
        **({"checkpoint": checkpoint} if checkpoint is not None else {}),
    )
    results = [
        score_case(case, prediction) for case, prediction in zip(CASES, predictions, strict=True)
    ]
    report.update(
        status="completed",
        model_timing_seconds=timing,
        cases=results,
        counts={
            status: sum(row["status"] == status for row in results)
            for status in ("passed", "failed", "invalid")
        },
    )
    write_report(path, report)
    return {row["case_id"]: row for row in results}


@pytest.mark.parametrize("case", CASES, ids=[case.case_id for case in CASES])
def test_real_forecaster_continues_simple_demand(case, model_results):
    result = model_results[case.case_id]
    assert result["status"] == "passed", (
        f"{case.case_id}: MAE={result.get('mae')} (limit {case.mae_limit}), "
        f"maximum error={result.get('max_absolute_error')} (limit {case.max_error_limit}); "
        f"status={result['status']}"
    )
