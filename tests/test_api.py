"""HTTP contract and lifecycle tests; no model weights, network or Torch required."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from src.api import app as api
from src.api.settings import Settings
from src.modeling.predict import SELECTED_CHECKPOINT
from src.splits import PARAMS_PATH


@pytest.fixture
def prediction():
    """A complete selected-artifact result, including private fields to filter."""
    return {
        "p10": [-2.0] * 48,
        "p50": [0.0] * 48,
        "p90": [2.0] * 48,
        "model_id": "chronos-t5-small-fine-tuned",
        "model_revision": "local-sha256:" + "a" * 64,
        "checkpoint": {"sha256": "a" * 64, "training_run_id": "b" * 32, "path": "/private"},
        "value_scale": "source",
        "context_length": 512,
        "horizon": 48,
        "settings": {
            "num_samples": 20,
            "seed": 7,
            "device": "cpu",
            "cpu_threads": 2,
            "dtype": "float32",
            "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
            "quantile_method": "linear",
            "point_statistic": "median",
            "private": "secret",
        },
    }


@pytest.fixture
def service(prediction):
    """Run the production lifecycle with one injected predictor factory."""
    predictor = Mock()
    predictor.predict.return_value = prediction
    factory = Mock(return_value=predictor)
    application = api.create_app(Settings(), factory)
    with TestClient(application) as client:
        yield SimpleNamespace(client=client, predictor=predictor, factory=factory)


@pytest.mark.parametrize("state", ["NSW", "VIC", "QLD", "SA", "TAS"])
def test_forecast_preserves_history_state_and_artifact_identity(service, state):
    history = [-3, 0, 2.5, 4] * 128
    body = {"state": state, "history": history}
    response = service.client.post("/forecast", json=body)
    assert response.status_code == 200
    result = response.json()
    assert result["state"] == state
    assert result["value_scale"] == "source"
    assert (result["context_length"], result["horizon"]) == (512, 48)
    assert result["model_id"] == "chronos-t5-small-fine-tuned"
    assert result["model_revision"] == "local-sha256:" + "a" * 64
    assert result["checkpoint"] == {"sha256": "a" * 64, "training_run_id": "b" * 32}
    assert result["p10"] == [-2.0] * 48
    assert result["p50"] == [0.0] * 48
    assert result["p90"] == [2.0] * 48
    assert result["settings"] == {
        key: value
        for key, value in service.predictor.predict.return_value["settings"].items()
        if key != "private"
    }
    assert "private" not in response.text and "secret" not in response.text
    service.predictor.predict.assert_called_once_with(history)
    assert body == {"state": state, "history": [-3, 0, 2.5, 4] * 128}
    assert service.client.post("/forecast", json=body).json() == result
    service.factory.assert_called_once_with(Settings())
    assert service.predictor.predict.call_count == 2


@pytest.mark.parametrize(
    "change",
    [
        {"state": "WA"},
        {"state": "tas"},
        {"state": None},
        {"state": 1},
        {"history": []},
        {"history": [1] * 511},
        {"history": [1] * 513},
        {"history": [[1] * 512]},
        {"history": "1"},
        {"history": None},
        {"history": {"value": 1}},
        {"history": ["1"] + [1] * 511},
        {"history": [True] + [1] * 511},
        {"history": [False] + [1] * 511},
        {"history": [None] + [1] * 511},
        {"history": [float("nan")] + [1] * 511},
        {"history": [float("inf")] + [1] * 511},
        {"history": [-float("inf")] + [1] * 511},
        {"history": [1e40] + [1] * 511},
        {"history": [10**400] + [1] * 511},
        {"unexpected": "secret"},
    ],
)
def test_invalid_requests_are_useful_json_errors_without_inference(service, change):
    body = {"state": "TAS", "history": [1] * 512, **change}
    response = service.client.post(
        "/forecast", content=json.dumps(body), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    details = response.json()["detail"]
    assert details and all(set(error) == {"loc", "msg", "type"} for error in details)
    assert all(error["loc"][0] == "body" for error in details)
    assert "secret" not in response.text
    service.predictor.predict.assert_not_called()


@pytest.mark.parametrize("body", ["{", "null", "[]", "{}", '{"state":"TAS"}'])
def test_missing_fields_and_malformed_json_are_controlled(service, body):
    response = service.client.post(
        "/forecast", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]
    service.predictor.predict.assert_not_called()


def test_lifecycle_loads_once_and_clears_model_on_shutdown(prediction):
    predictor = Mock()
    predictor.predict.return_value = prediction
    factory = Mock(return_value=predictor)
    application = api.create_app(Settings(), factory)
    client = TestClient(application)
    factory.assert_not_called()
    for _ in range(2):
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 503
        response = client.post("/forecast", json={"state": "TAS", "history": [0] * 512})
        assert response.status_code == 503
        assert response.json() == {"detail": "Model unavailable"}
        with client:
            assert client.get("/health").status_code == 200
            assert client.get("/ready").json() == {"status": "ready"}
            assert (
                client.post("/forecast", json={"state": "NSW", "history": [0] * 512}).status_code
                == 200
            )
        assert application.state.predictor is None
        assert client.get("/ready").status_code == 503
    assert factory.call_count == 2
    assert predictor.predict.call_count == 2


@pytest.mark.parametrize("failure", [RuntimeError, OSError, ValueError, ImportError])
def test_model_initialization_failure_aborts_startup(failure):
    factory = Mock(side_effect=failure("/private/model token=secret"))
    application = api.create_app(Settings(), factory)
    with (
        pytest.raises(RuntimeError, match="^Forecast model initialization failed$"),
        TestClient(application),
    ):
        pytest.fail("A failed model must never finish application startup")
    assert application.state.predictor is None
    factory.assert_called_once()


def test_empty_factory_result_cannot_mark_application_ready():
    with (
        pytest.raises(RuntimeError, match="Forecast model initialization failed"),
        TestClient(api.create_app(Settings(), lambda _settings: None)),
    ):
        pytest.fail("Missing predictor must abort startup")


def test_real_loader_rejects_missing_selected_artifact_without_fallback(tmp_path):
    application = api.create_app(Settings(checkpoint=tmp_path / "missing"))
    with (
        pytest.raises(RuntimeError, match="Forecast model initialization failed"),
        TestClient(application),
    ):
        pytest.fail("Missing selected weights must abort startup without a download")
    assert application.state.predictor is None


@pytest.mark.parametrize("failure", [RuntimeError, ValueError, KeyError])
def test_inference_failure_does_not_leak_details_or_reload_model(service, prediction, failure):
    service.predictor.predict.side_effect = [failure("/private/model token=secret"), prediction]
    body = {"state": "NSW", "history": [1] * 512}
    response = service.client.post("/forecast", json=body)
    assert response.status_code == 503
    assert response.json() == {"detail": "Forecast unavailable"}
    assert service.client.get("/health").status_code == 200
    assert service.client.get("/ready").status_code == 200
    assert service.client.post("/forecast", json=body).status_code == 200
    service.factory.assert_called_once()


@pytest.mark.parametrize(
    "change",
    [
        {"p10": []},
        {"p50": [0.0] * 47},
        {"p90": [2.0] * 49},
        {"p10": [float("nan")] * 48},
        {"p50": [float("inf")] * 48},
        {"p90": ["2"] * 48},
        {"p50": [True] * 48},
        {"p10": [[-2.0] * 48]},
        {"p10": [1.0] * 48},
        {"p90": [-1.0] * 48},
        {"horizon": 47},
        {"context_length": 511},
        {"value_scale": "MW"},
        {"model_id": "/private/model"},
        {"model_revision": "/private/revision"},
        {"model_revision": "local-sha256:" + "c" * 64},
        {"checkpoint": None},
        {"checkpoint": {"sha256": "a" * 64, "training_run_id": "secret"}},
    ],
)
def test_malformed_prediction_is_controlled_and_never_returned(service, prediction, change):
    service.predictor.predict.return_value = {**prediction, **change}
    response = service.client.post("/forecast", json={"state": "TAS", "history": [1] * 512})
    assert response.status_code == 500
    assert response.json() == {"detail": "Invalid forecast output"}
    assert service.client.get("/health").status_code == 200


@pytest.mark.parametrize("output", [None, [], {}, "secret"])
def test_missing_prediction_is_controlled(service, output):
    service.predictor.predict.return_value = output
    response = service.client.post("/forecast", json={"state": "TAS", "history": [0] * 512})
    assert response.status_code == 500
    assert response.json() == {"detail": "Invalid forecast output"}


@pytest.mark.parametrize(
    "change",
    [
        {"num_samples": "20"},
        {"cpu_threads": True},
        {"seed": -1},
        {"device": "cuda"},
        {"dtype": "float64"},
        {"sampling": {"temperature": 1.0, "top_k": 50, "top_p": float("nan")}},
        {"quantile_method": "secret"},
        {"point_statistic": "mean"},
    ],
)
def test_invalid_executed_settings_fail_closed(service, prediction, change):
    prediction["settings"].update(change)
    response = service.client.post("/forecast", json={"state": "TAS", "history": [0] * 512})
    assert response.status_code == 500
    assert response.json() == {"detail": "Invalid forecast output"}


def test_health_and_readiness_do_not_wait_for_running_inference(service, prediction):
    started, release = Event(), Event()

    def predict(_history):
        started.set()
        assert release.wait(5), "Test did not release its blocked predictor"
        return prediction

    service.predictor.predict.side_effect = predict
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(
            service.client.post, "/forecast", json={"state": "NSW", "history": [1] * 512}
        )
        try:
            assert started.wait(2)
            for path in ("/health", "/ready"):
                response = executor.submit(service.client.get, path).result(timeout=1)
                assert response.status_code == 200
            assert not future.done()
        finally:
            release.set()
        assert future.result(timeout=2).status_code == 200


def test_default_loader_uses_selected_artifact_offline_and_explicit_cpu_budget(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(api, "ChronosForecaster", constructor)
    settings = Settings(checkpoint=Path("/relocated/selected"), cpu_threads=1)
    with TestClient(api.create_app(settings)) as client:
        assert client.get("/ready").status_code == 200
    constructor.assert_called_once()
    config = constructor.call_args.args[0]
    assert config.checkpoint == settings.checkpoint
    assert config.cpu_threads == 1
    assert constructor.call_args.kwargs == {"local_files_only": True}


def test_environment_settings_are_captured_once(monkeypatch, prediction):
    for name in ("FORECAST_PARAMS_PATH", "FORECAST_CHECKPOINT", "FORECAST_CPU_THREADS"):
        monkeypatch.delenv(name, raising=False)
    assert Settings.from_env() == Settings(PARAMS_PATH, SELECTED_CHECKPOINT, 2)
    monkeypatch.setenv("FORECAST_PARAMS_PATH", "/configured/params.yaml")
    monkeypatch.setenv("FORECAST_CHECKPOINT", "/configured/model")
    monkeypatch.setenv("FORECAST_CPU_THREADS", "1")
    factory = Mock(return_value=SimpleNamespace(predict=lambda _history: deepcopy(prediction)))
    application = api.create_app(predictor_factory=factory)
    monkeypatch.setenv("FORECAST_CPU_THREADS", "invalid")
    monkeypatch.setenv("FORECAST_CHECKPOINT", "/later/model")
    with TestClient(application) as client:
        assert (
            client.post("/forecast", json={"state": "TAS", "history": [0] * 512}).status_code
            == 200
        )
    factory.assert_called_once_with(
        Settings(Path("/configured/params.yaml"), Path("/configured/model"), 1)
    )


@pytest.mark.parametrize("value", ["invalid", "1.5", "0", "-1"])
def test_invalid_environment_fails_before_serving(monkeypatch, value):
    monkeypatch.setenv("FORECAST_CPU_THREADS", value)
    with pytest.raises(ValueError, match="positive integer"):
        api.create_app()


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_invalid_explicit_cpu_budget(threads):
    with pytest.raises(ValueError, match="positive integer"):
        Settings(cpu_threads=threads)


@pytest.mark.parametrize("change", [{"params_path": "params.yaml"}, {"checkpoint": None}])
def test_settings_require_paths_and_cannot_select_base_model(change):
    with pytest.raises(TypeError, match="local Paths"):
        Settings(**change)


def test_openapi_documents_request_limits_and_response_identity(service):
    response = service.client.get("/openapi.json")
    assert response.status_code == 200
    schemas = response.json()["components"]["schemas"]
    request = schemas["ForecastRequest"]["properties"]
    assert set(request["state"]["enum"]) == {"NSW", "VIC", "QLD", "SA", "TAS"}
    assert request["history"]["minItems"] == request["history"]["maxItems"] == 512
    forecast = schemas["ForecastResponse"]["properties"]
    for name in ("p10", "p50", "p90"):
        assert forecast[name]["minItems"] == forecast[name]["maxItems"] == 48
    assert forecast["horizon"]["const"] == 48
    assert "checkpoint" in forecast
