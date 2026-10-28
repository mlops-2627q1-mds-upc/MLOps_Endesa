"""Observable service instrumentation, cardinality, lifecycle and config checks."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, ProcessCollector
from prometheus_client.parser import text_string_to_metric_families
import pytest
import yaml

from src.api import metrics
from src.api.app import create_app
from src.api.settings import Settings
from tests.container.fixture_app import FixturePredictor

ROOT = Path(__file__).resolve().parents[1]


def _samples(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    return [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    ]


def _value(client, name, labels=None):
    return next(
        (s.value for s in _samples(client) if s.name == name and s.labels == (labels or {})),
        None,
    )


@pytest.fixture
def service():
    """Exercise the real API with an explicit synthetic predictor."""
    predictor = Mock(wraps=FixturePredictor())
    app = create_app(Settings(), lambda _settings: predictor)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, predictor=predictor, app=app)


def test_counts_latency_validation_and_controlled_failures(service):
    body = {"state": "TAS", "history": [1] * 512}
    assert service.client.post("/forecast", json=body).status_code == 200
    assert service.client.post("/forecast", json={**body, "history": []}).status_code == 422
    service.predictor.predict.side_effect = RuntimeError("private model hash and history")
    assert service.client.post("/forecast", json=body).status_code == 503
    service.predictor.predict.side_effect = None
    service.predictor.predict.return_value = {}
    assert service.client.post("/forecast", json=body).status_code == 500

    for status_class, expected in (("2xx", 1), ("4xx", 1), ("5xx", 2)):
        assert (
            _value(
                service.client,
                "endesa_http_requests_total",
                {"method": "POST", "route": "/forecast", "status_class": status_class},
            )
            == expected
        )
    assert _value(service.client, "endesa_http_errors_total", {"status_class": "4xx"}) == 1
    assert _value(service.client, "endesa_http_errors_total", {"status_class": "5xx"}) == 2
    labels = {"method": "POST", "route": "/forecast"}
    assert _value(service.client, "endesa_http_request_duration_seconds_count", labels) == 4
    assert _value(service.client, "endesa_http_request_duration_seconds_sum", labels) > 0
    assert _value(service.client, "endesa_forecasts_in_flight") == 0
    assert _value(service.client, "endesa_model_ready") == 1


def test_histogram_observes_elapsed_time_and_scrapes_do_not_count(service, monkeypatch):
    monkeypatch.setattr(metrics, "perf_counter", Mock(side_effect=[10.0, 10.25]))
    assert service.client.get("/health").status_code == 200
    labels = {"method": "GET", "route": "/health"}
    for _ in range(2):
        assert _value(service.client, "endesa_http_request_duration_seconds_sum", labels) == 0.25
        assert (
            _value(
                service.client,
                "endesa_http_request_duration_seconds_bucket",
                {**labels, "le": "0.25"},
            )
            == 1
        )
        assert all(s.labels.get("route") != "/metrics" for s in _samples(service.client))


def test_unknown_routes_methods_and_request_values_cannot_expand_labels(service):
    for number in range(25):
        response = service.client.request(
            f"PRIVATE{number}", f"/unknown-secret-{number}?forecast_id=private-{number}"
        )
        assert response.status_code == 404
        response = service.client.post(
            "/forecast", json={"state": f"private-state-{number}", "history": [number]}
        )
        assert response.status_code == 422
    assert service.client.get("/forecast?private=raw-url").status_code == 405
    samples = _samples(service.client)
    for sample in samples:
        if "route" in sample.labels:
            assert sample.labels["route"] in metrics.ROUTES | {"__unmatched__"}
        if "method" in sample.labels:
            assert sample.labels["method"] in metrics.METHODS | {"OTHER"}
        assert set(sample.labels) <= {"method", "route", "status_class", "le"}
    unknown_counts = [
        s
        for s in samples
        if s.name == "endesa_http_requests_total" and s.labels["method"] == "OTHER"
    ]
    assert len(unknown_counts) == 1
    assert unknown_counts[0].value == 25
    text = service.client.get("/metrics").text
    assert "private" not in text and "raw-url" not in text
    assert "forecast_id" not in text and "model_revision" not in text


@pytest.mark.parametrize("result", ["success", "failure", "invalid-output"])
def test_in_flight_cleanup_and_probes_remain_responsive(service, result):
    started, release = Event(), Event()

    def predict(history):
        started.set()
        assert release.wait(5), "Blocked fixture was not released"
        if result == "failure":
            raise RuntimeError("Controlled inference failure")
        return {} if result == "invalid-output" else FixturePredictor().predict(history)

    service.predictor.predict.side_effect = predict
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(
            service.client.post, "/forecast", json={"state": "NSW", "history": [1] * 512}
        )
        try:
            assert started.wait(2)
            assert (
                executor.submit(_value, service.client, "endesa_forecasts_in_flight").result(
                    timeout=1
                )
                == 1
            )
            for path in ("/health", "/ready"):
                assert (
                    executor.submit(service.client.get, path).result(timeout=1).status_code == 200
                )
            assert not future.done()
        finally:
            release.set()
        assert (
            future.result(timeout=2).status_code
            == {
                "success": 200,
                "failure": 503,
                "invalid-output": 500,
            }[result]
        )
    assert _value(service.client, "endesa_forecasts_in_flight") == 0


def test_readiness_reflects_lifecycle_even_without_readiness_requests():
    app = create_app(Settings(), lambda _settings: FixturePredictor())
    client = TestClient(app)
    assert _value(client, "endesa_model_ready") == 0
    with client:
        assert _value(client, "endesa_model_ready") == 1
    assert _value(client, "endesa_model_ready") == 0
    assert client.post("/forecast", json={"state": "NSW", "history": [1] * 512}).status_code == 503
    assert _value(client, "endesa_forecasts_in_flight") == 0


def test_initialization_failure_keeps_readiness_false():
    app = create_app(Settings(), Mock(side_effect=RuntimeError("Controlled startup failure")))
    with (
        pytest.raises(RuntimeError, match="Forecast model initialization failed"),
        TestClient(app),
    ):
        pytest.fail("A failed startup cannot finish")
    assert _value(TestClient(app), "endesa_model_ready") == 0


def test_apps_have_independent_metrics_and_preserve_global_registry():
    before = {metric.name for metric in REGISTRY.collect()}
    first = TestClient(create_app(Settings(), lambda _settings: FixturePredictor()))
    second = TestClient(create_app(Settings(), lambda _settings: FixturePredictor()))
    assert first.get("/unknown").status_code == 404
    assert _value(first, "endesa_http_errors_total", {"status_class": "4xx"}) == 1
    assert _value(second, "endesa_http_errors_total", {"status_class": "4xx"}) == 0
    assert before == {metric.name for metric in REGISTRY.collect()}


def test_unhandled_error_before_response_start_is_counted(service):
    @service.app.get("/unhandled-private-route")
    async def unhandled():
        raise RuntimeError("Controlled unexpected handler failure")

    with TestClient(service.app, raise_server_exceptions=False) as client:
        assert client.get("/unhandled-private-route").status_code == 500
        assert _value(client, "endesa_http_errors_total", {"status_class": "5xx"}) == 1
        assert (
            _value(
                client,
                "endesa_http_requests_total",
                {"method": "GET", "route": "__unmatched__", "status_class": "5xx"},
            )
            == 1
        )


@pytest.mark.parametrize("supported", [False, True])
def test_process_collector_with_unavailable_and_linux_proc_files(tmp_path, supported):
    if supported:
        (tmp_path / "stat").write_text("btime 100\n")
        (tmp_path / "self").mkdir()
        fields = ["0"] * 22
        for index, value in ((11, "200"), (12, "100"), (19, "50"), (20, "1000"), (21, "2")):
            fields[index] = value
        (tmp_path / "self/stat").write_text("1 (fixture process) " + " ".join(fields))
    collector = metrics.ProcessMetrics()
    collector.collector = ProcessCollector(registry=None, proc=str(tmp_path))
    samples = {s.name: s.value for family in collector.collect() for s in family.samples}
    assert samples["endesa_process_metrics_available"] == int(supported)
    for name in ("process_cpu_seconds_total", "process_resident_memory_bytes"):
        if supported:
            assert samples[name] > 0
        else:
            assert name not in samples


def test_current_platform_process_samples_are_present_or_explicitly_unavailable(service):
    supported = platform.system() == "Linux"
    assert _value(service.client, "endesa_process_metrics_available") == int(supported)
    for name in ("process_cpu_seconds_total", "process_resident_memory_bytes"):
        value = _value(service.client, name)
        assert value is not None if supported else value is None


def test_monitoring_configuration_has_private_ports_and_resolvable_provisioning():
    directory = ROOT / "monitoring/service"
    compose = yaml.safe_load((directory / "compose.yml").read_text())
    assert compose["networks"]["monitoring"]["internal"]
    for service_config in compose["services"].values():
        assert service_config["networks"] == ["monitoring"]
        assert all(port.startswith("127.0.0.1:") for port in service_config.get("ports", []))
    prometheus = yaml.safe_load((directory / "prometheus.yml").read_text())
    assert prometheus["scrape_configs"][0]["static_configs"][0]["targets"] == ["api:8000"]
    for path in prometheus["rule_files"]:
        assert (directory / path).is_file()
    datasource = yaml.safe_load(
        (directory / "grafana/provisioning/datasources/prometheus.yml").read_text()
    )["datasources"][0]
    assert datasource["url"] == "http://prometheus:9090"
    dashboard = json.loads((directory / "grafana/dashboards/service.json").read_text())
    assert dashboard["uid"] == "endesa-service"
    for panel in dashboard["panels"]:
        if "targets" in panel:
            assert panel["datasource"]["uid"] == datasource["uid"]
            assert panel["fieldConfig"]["defaults"]["noValue"] == "No telemetry"
