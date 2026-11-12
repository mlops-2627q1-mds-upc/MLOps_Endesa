"""Quality exporter availability, bounded labels and monitoring configuration."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from fastapi.testclient import TestClient
from monitoring.quality.exporter import MAX_REPORT_BYTES, STATUS_CODES, create_app
from prometheus_client.parser import text_string_to_metric_families
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def report_file(tmp_path):
    """A minimal complete report with explicit missing horizon samples."""
    config = {
        "mode": "historical_replay",
        "value_scale": "source",
        "states": ["NSW"],
        "model_identity": {"private_model_identity": "fixture-only"},
    }
    config_json, source_reference = json.dumps(config), "private source identity"
    group = {
        "state": "NSW",
        "expected_forecasts": 2,
        "forecast_count": 2,
        "matched_forecast_count": 2,
        "forecast_steps": 96,
        "matched_steps": 96,
        "pending_steps": 0,
        "unique_matched_targets": 96,
        "sample_status": "sufficient",
        "selected": {
            "mae": 0.0,
            "mase": 0.0,
            "empirical_coverage": 1.0,
            "mean_interval_width": 4.0,
        },
        "seasonal_lag48": {"mae": 48.0, "mase": 48 / 336},
    }
    horizons = []
    for horizon in range(1, 49):
        horizons.append(
            {
                **deepcopy(group),
                "horizon": horizon,
                "forecast_steps": 2,
                "matched_steps": 2,
                "unique_matched_targets": 2,
                "sample_status": "insufficient_samples",
                "selected": {key: None for key in group["selected"]},
                "seasonal_lag48": {"mae": None, "mase": None},
            }
        )
    report = {
        "schema_version": 1,
        "metadata": {
            "config_json": config_json,
            "source_reference": source_reference,
            "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
            "source_sha256": hashlib.sha256(source_reference.encode()).hexdigest(),
            "model_identity": config["model_identity"],
            "policy": {"minimum_samples": 30},
        },
        "by_state": [group],
        "by_state_horizon": horizons,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return path


def _samples(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    return [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    ]


def _value(client, name, **labels):
    return next(
        (
            sample.value
            for sample in _samples(client)
            if sample.name == "endesa_quality_" + name and sample.labels == labels
        ),
        None,
    )


def test_valid_export_and_actual_zero_error(report_file):
    with TestClient(create_app(report_file)) as client:
        for name in ("report_present", "report_valid", "report_fresh", "report_available"):
            assert _value(client, name) == 1
        assert _value(client, "mae", state="NSW", horizon="all", model_role="selected") == 0
        assert _value(client, "mae", state="NSW", horizon="all", model_role="seasonal_lag48") == 48
        assert _value(client, "matched_steps", state="NSW", horizon="all") == 96
        assert _value(client, "pending_steps", state="NSW", horizon="all") == 0
        assert _value(client, "sample_status", state="NSW", horizon="1") == 2
        assert _value(client, "mae", state="NSW", horizon="1", model_role="selected") is None
        for sample in _samples(client):
            assert set(sample.labels) <= {"state", "horizon", "model_role"}
            assert sample.labels.get("state", "NSW") == "NSW"
            assert sample.labels.get("horizon", "all") in {"all", *(str(n) for n in range(1, 49))}
            assert sample.labels.get("model_role", "selected") in {"selected", "seasonal_lag48"}
        text = client.get("/metrics").text
        assert "private" not in text and "fixture-only" not in text
        assert client.get("/forecast").status_code == 404


def test_missing_report_is_not_zero_quality(tmp_path):
    with TestClient(create_app(tmp_path / "absent.json")) as client:
        assert _value(client, "report_present") == 0
        assert _value(client, "report_available") == 0
        assert _value(client, "report_age_seconds") is None
        assert _value(client, "matched_steps", state="NSW", horizon="all") is None


def test_stale_report_is_suppressed_without_losing_export_status(report_file):
    with TestClient(create_app(report_file, 60)) as client:
        assert _value(client, "report_available") == 1
        old = time.time() - 120
        os.utime(report_file, (old, old))
        assert _value(client, "report_valid") == 1
        assert _value(client, "report_present") == 1
        assert _value(client, "report_fresh") == 0
        assert _value(client, "report_available") == 0
        assert _value(client, "report_age_seconds") >= 120
        assert _value(client, "matched_steps", state="NSW", horizon="all") is None
        report_file.write_text(report_file.read_text())
        assert _value(client, "report_available") == 1
        report_file.unlink()
        assert _value(client, "report_available") == 0


@pytest.mark.parametrize(
    "content",
    ["invalid JSON", "[]", "null", '{"schema_version":NaN}', " " * (MAX_REPORT_BYTES + 1)],
    ids=["malformed", "array", "null", "nonfinite", "oversized"],
)
def test_invalid_or_oversized_report_is_suppressed(report_file, content):
    report_file.write_text(content)
    with TestClient(create_app(report_file)) as client:
        assert _value(client, "report_present") == 1
        assert _value(client, "report_valid") == 0
        assert _value(client, "report_available") == 0
        assert not any(sample.labels for sample in _samples(client))


@pytest.mark.parametrize(
    "change",
    [
        {"state": "arbitrary-private-label"},
        {"horizon": 49},
        {"matched_steps": 1},
        {"sample_status": "invented"},
        {"selected": {"mae": 1.0}},
    ],
)
def test_conflicting_groups_cannot_expand_labels(report_file, change):
    report = json.loads(report_file.read_text())
    report["by_state_horizon"][0].update(change)
    report_file.write_text(json.dumps(report))
    with TestClient(create_app(report_file)) as client:
        assert _value(client, "report_available") == 0
        assert not any(sample.labels for sample in _samples(client))


def test_conflicting_report_identity_is_suppressed(report_file):
    report = json.loads(report_file.read_text())
    report["metadata"]["source_sha256"] = "0" * 64
    report_file.write_text(json.dumps(report))
    with TestClient(create_app(report_file)) as client:
        assert _value(client, "report_valid") == 0


def test_duplicate_json_keys_are_invalid_even_when_last_value_is_valid(report_file):
    raw = report_file.read_text()
    report_file.write_text('{"schema_version":999,' + raw[1:])
    with TestClient(create_app(report_file)) as client:
        assert _value(client, "report_valid") == 0
        assert not any(sample.labels for sample in _samples(client))


@pytest.mark.parametrize("field,value", [("schema_version", True), ("horizon", "1")])
def test_report_schema_and_horizon_require_integers(report_file, field, value):
    report = json.loads(report_file.read_text())
    if field == "schema_version":
        report[field] = value
    else:
        report["by_state_horizon"][0][field] = value
    report_file.write_text(json.dumps(report))
    with TestClient(create_app(report_file)) as client:
        assert _value(client, "report_valid") == 0


@pytest.mark.parametrize("kind", ["forecast_dimensions", "aggregate_matches"])
def test_inconsistent_state_and_horizon_counts_are_suppressed(report_file, kind):
    report = json.loads(report_file.read_text())
    if kind == "forecast_dimensions":
        report["by_state"][0]["expected_forecasts"] = 3
        report["by_state"][0]["forecast_count"] = 3
    else:
        group = report["by_state_horizon"][0]
        group.update(
            matched_steps=1, pending_steps=1, matched_forecast_count=1, unique_matched_targets=1
        )
    report_file.write_text(json.dumps(report))
    with TestClient(create_app(report_file)) as client:
        assert _value(client, "report_valid") == 0


@pytest.mark.parametrize("horizon", ["arbitrary-private-horizon", "all", 1])
def test_state_summary_cannot_declare_a_horizon_label(report_file, horizon):
    report = json.loads(report_file.read_text())
    report["by_state"][0]["horizon"] = horizon
    report_file.write_text(json.dumps(report))
    with TestClient(create_app(report_file)) as client:
        assert _value(client, "report_available") == 0
        assert _value(client, "report_valid") == 0
        assert not any(sample.labels for sample in _samples(client))
        assert "arbitrary-private-horizon" not in client.get("/metrics").text


def test_future_report_time_is_not_fresh(report_file):
    future = time.time() + 60
    os.utime(report_file, (future, future))
    with TestClient(create_app(report_file)) as client:
        assert _value(client, "report_available") == 0
        assert _value(client, "report_age_seconds") is None


@pytest.mark.parametrize("maximum", [0, 86401, True])
def test_freshness_policy_bounds(report_file, maximum):
    with pytest.raises(ValueError, match="max age"):
        create_app(report_file, maximum)


def test_quality_compose_preserves_loopback_service_monitoring():
    service = yaml.safe_load((ROOT / "monitoring/service/compose.yml").read_text())
    quality = yaml.safe_load((ROOT / "monitoring/quality/compose.yml").read_text())
    exporter = quality["services"]["quality"]
    assert exporter["networks"] == ["monitoring"]
    assert "ports" not in exporter
    assert service["networks"]["monitoring"] == {"driver": "bridge", "internal": False}
    assert exporter["image"] == service["services"]["api"]["image"]
    assert exporter["platform"] == "linux/amd64"
    assert exporter["read_only"] is True
    assert exporter["mem_limit"] == "256m"
    assert all(
        port.startswith("127.0.0.1:")
        for name in ("api", "prometheus", "grafana")
        for port in service["services"][name]["ports"]
    )
    assert all("ro" in volume or volume["read_only"] for volume in exporter["volumes"])
    assert not any("model" in str(volume) for volume in exporter["volumes"])
    scrape = yaml.safe_load((ROOT / "monitoring/quality/prometheus.yml").read_text())
    jobs = {job["job_name"]: job for job in scrape["scrape_configs"]}
    assert set(jobs) == {"forecast-api", "prometheus", "forecast-quality"}
    assert jobs["forecast-quality"]["static_configs"] == [{"targets": ["quality:9101"]}]
    assert jobs["forecast-quality"]["metrics_path"] == "/metrics"


def test_quality_provisioning_replaces_an_existing_readonly_parent_file():
    override = yaml.safe_load((ROOT / "monitoring/quality/compose.yml").read_text())
    volumes = override["services"]["grafana"]["volumes"]
    provision_mount = next(volume for volume in volumes if "/etc/grafana/provisioning/" in volume)
    source, target, mode = provision_mount.split(":")
    assert mode == "ro"
    # The base stack mounts this parent read-only; Docker cannot create a child there.
    relative = Path(target).relative_to("/etc/grafana/provisioning")
    assert (ROOT / "monitoring/service/grafana/provisioning" / relative).is_file()
    providers = yaml.safe_load((ROOT / "monitoring/service" / source).read_text())["providers"]
    base = yaml.safe_load(
        (ROOT / "monitoring/service/grafana/provisioning" / relative).read_text()
    )
    assert all(provider in providers for provider in base["providers"])
    assert {provider["options"]["path"] for provider in providers} == {
        "/var/lib/grafana/dashboards",
        "/var/lib/grafana/quality",
    }


def test_quality_dashboard_uses_availability_gates_and_existing_datasource():
    path = ROOT / "monitoring/quality/grafana/dashboards/quality.json"
    dashboard = json.loads(path.read_text())
    assert dashboard["uid"] == "endesa-quality"
    expressions = []
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            assert panel["datasource"]["uid"] == "endesa-prometheus"
            expression = target["expr"]
            assert 'job="forecast-quality"' in expression
            assert "or vector(0)" not in expression
            if any(
                name in expression
                for name in (
                    "_mae{",
                    "_mase{",
                    "_empirical_coverage{",
                    "_mean_interval_width{",
                    "_matched_steps{",
                )
            ):
                assert "endesa_quality_report_available" in expression
                assert "up{" in expression
            expressions.append(expression)
    assert any("seasonal_lag48" in expression for expression in expressions)
    assert any("_sample_status{" in expression for expression in expressions)
    assert set(STATUS_CODES.values()) == {0, 1, 2, 3}


def test_promtool_scenarios_cover_every_dashboard_expression():
    dashboard = json.loads(
        (ROOT / "monitoring/quality/grafana/dashboards/quality.json").read_text()
    )
    expressions = {
        target["expr"].replace("$state", "NSW")
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    }
    cases = yaml.safe_load((ROOT / "monitoring/quality/dashboard-tests.yml").read_text())
    assert len(cases["tests"]) == 6
    for case in cases["tests"]:
        assert {check["expr"] for check in case["promql_expr_test"]} == expressions
    panels = {panel["id"]: panel for panel in dashboard["panels"]}
    for panel_id in (2, 3):
        defaults = panels[panel_id]["fieldConfig"]["defaults"]
        mappings = defaults["mappings"][0]["options"]
        assert mappings["0"]["color"] == "red"
        assert mappings["1"]["color"] == "green"
    assert "minimum_samples" in panels[15]["targets"][0]["expr"]
    for panel in dashboard["panels"]:
        if panel["type"] == "timeseries":
            assert panel["options"]["legend"]["calcs"] == ["last"]
            assert panel["fieldConfig"]["defaults"]["custom"]["spanNulls"] is False
        if panel["type"] == "table":
            horizon = panel["fieldConfig"]["overrides"][0]
            assert horizon["matcher"] == {"id": "byName", "options": "Horizon"}
            properties = {value["id"]: value["value"] for value in horizon["properties"]}
            assert properties["unit"] == "none" and properties["decimals"] == 0
            assert properties["mappings"] == []
