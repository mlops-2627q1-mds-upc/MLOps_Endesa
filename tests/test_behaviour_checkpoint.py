"""Route local artifacts through each real-check harness without model inference."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation import seasonal_forecast
from src.modeling import directional, invariance, minimum_functionality
from src.modeling.behaviour_source import checkpoint_for_model, describe_model
from src.modeling.evaluate import EvaluationConfig
from tests import test_minimum_functionality_model as minimum_model


@pytest.fixture
def checkpoint(tmp_path):
    """Use small identifiable bytes; no real model or Hub access is needed."""
    directory = tmp_path / "selected-checkpoint"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"local checkpoint fixture")
    return directory


@pytest.mark.parametrize("suite", ["minimum", "directional", "invariance"])
def test_local_artifact_reaches_inference_and_is_identified_in_report(
    suite, checkpoint, tmp_path, monkeypatch
):
    calls = []

    def forecast(contexts, horizon, settings, deadline, **kwargs):
        calls.append(kwargs)
        assert kwargs["checkpoint"] == checkpoint.resolve()
        assert settings.batch_size == 1 and settings.num_samples == 20 and settings.seed == 7
        assert deadline > 0
        return seasonal_forecast(contexts, horizon, 48), {"inference_seconds": 0.0}

    monkeypatch.setattr(minimum_model, "chronos_forecast", forecast)
    monkeypatch.setattr(directional, "chronos_forecast", forecast)
    monkeypatch.setattr(invariance, "chronos_forecast", forecast)
    monkeypatch.setattr(minimum_functionality, "_provenance", lambda: {"fixture": True})
    monkeypatch.setattr(invariance, "_versions", lambda _: {"fixture": True})
    monkeypatch.setattr(directional.subprocess, "check_output", lambda *_args, **_kwargs: "")

    def worker(command, **_kwargs):
        if suite == "minimum":
            report_path = next(
                value.split("=", 1)[1]
                for value in command
                if value.startswith("--model-test-report=")
            )
            options = {"--run-model-tests": True, "--model-test-report": report_path}
            request = SimpleNamespace(config=SimpleNamespace(getoption=options.get))
            minimum_model.model_results.__wrapped__(request)
        else:
            directory = Path(command[-1])
            plan = json.loads((directory / "plan.json").read_text())
            report = directional.execute_plan(plan)
            (directory / "report.json").write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(minimum_functionality.subprocess, "run", worker)
    if suite == "minimum":
        path = tmp_path / "minimum.json"
        assert (
            minimum_functionality.run_model_tests(
                output_path=path, checkpoint=checkpoint, max_seconds=5
            )
            == 0
        )
        report = json.loads(path.read_text())
        model, results = report["model"], report["cases"]
        assert len(results) == 10 and all(row["status"] == "passed" for row in results)
    elif suite == "directional":
        _, report = directional.run_bounded(
            output_dir=tmp_path / "directional", checkpoint=checkpoint, max_seconds=5
        )
        model, results = report["plan"]["model"], report["cases"]
        assert len(results) == 12 and report["control"]["passed"]
    else:
        report = invariance.run_checks(
            output_dir=tmp_path / "invariance", checkpoint=checkpoint, synthetic_only=True
        )
        model = report["model"]
        assert len(report["results"]) == 20
    assert len(calls) == 1
    assert model["source"] == "local"
    assert model["path"] == str(checkpoint.resolve())
    assert model["sha256"] == hashlib.sha256(b"local checkpoint fixture").hexdigest()
    assert "revision" not in model


def test_worker_rejects_weights_replaced_after_plan_was_written(checkpoint):
    settings = EvaluationConfig.from_params(minimum_functionality.PARAMS_PATH)
    model = describe_model(settings, checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="weights changed"):
        checkpoint_for_model(model)
