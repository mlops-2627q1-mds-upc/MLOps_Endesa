import hashlib
import json
from pathlib import Path
import subprocess

import mlflow
from mlflow import MlflowClient
import pytest

from src import tracking
from src.modeling.tracking_smoke import smoke_run


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Use a real local store and a small Git checkout; never contact DagsHub."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    root = tmp_path / "project"
    root.mkdir()
    (root / "uv.lock").write_text("version = 1\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "uv.lock"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Tracking test",
            "-c",
            "user.email=tracking@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    monkeypatch.setattr(tracking, "PROJ_ROOT", root)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    return uri, client, root


def open_run(uri, **overrides):
    arguments = {
        "name": "test-run",
        "config": {"seed": 7, "horizon": 2, "split": {"train": [0, 8], "validation": [8, 12]}},
        "data_version": "fixture-data-v1",
        "model_revision": "fixture-model-v1",
        "experiment_name": "test",
        "tracking_uri": uri,
    }
    return tracking.tracked_run(**(arguments | overrides))


def read_json(client, run_id, name, directory):
    return json.loads(Path(client.download_artifacts(run_id, name, str(directory))).read_text())


def test_run_round_trip_records_versions_config_environment_and_artifacts(
    store, tmp_path, monkeypatch
):
    uri, client, root = store
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret-that-must-not-be-logged")
    monkeypatch.setenv("UNRELATED_PRIVATE_SETTING", "another-secret")
    output = tmp_path / "predictions.csv"
    output.write_text("target,prediction\n1,2\n")
    with open_run(uri) as run:
        run.log_metrics({"mae": 1, "NSW.mae": 1}, step=0)
        run.log_metrics({"mae": 0.5}, step=1)
        run.log_artifact(output)

    recorded = client.get_run(run.run_id)
    assert recorded.info.status == "FINISHED"
    assert recorded.data.params["seed"] == "7"
    assert recorded.data.params["split"] == '{"train": [0, 8], "validation": [8, 12]}'
    assert recorded.data.metrics["mae"] == 0.5
    assert recorded.data.metrics["run_body_seconds"] >= 0
    assert [
        (metric.step, metric.value) for metric in client.get_metric_history(run.run_id, "mae")
    ] == [(0, 1), (1, 0.5)]
    assert recorded.data.tags["git.dirty"] == "false"
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    assert recorded.data.tags["mlflow.source.git.commit"] == expected_commit
    assert recorded.data.tags["data.version"] == "fixture-data-v1"
    assert recorded.data.tags["model.revision"] == "fixture-model-v1"
    assert (
        recorded.data.tags["uv.lock.sha256"]
        == hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    )

    config = read_json(client, run.run_id, "config.json", tmp_path)
    assert config["split"] == {"train": [0, 8], "validation": [8, 12]}
    provenance = read_json(client, run.run_id, "provenance.json", tmp_path)
    assert provenance["git_commit"] == expected_commit
    assert provenance["git_dirty"] is False
    environment = read_json(client, run.run_id, "environment.json", tmp_path)
    assert environment["packages"]["mlflow"] == mlflow.__version__
    assert environment["python"].startswith("3.11.")
    for payload in (config, provenance, environment, recorded.data.tags):
        assert "secret-that-must-not-be-logged" not in json.dumps(payload)
        assert "another-secret" not in json.dumps(payload)
    downloaded = Path(client.download_artifacts(run.run_id, "predictions.csv", str(tmp_path)))
    assert downloaded.read_bytes() == output.read_bytes()
    lock = Path(client.download_artifacts(run.run_id, "uv.lock", str(tmp_path)))
    assert lock.read_bytes() == (root / "uv.lock").read_bytes()


def test_untracked_source_changes_are_recorded_as_dirty(store):
    uri, client, root = store
    (root / "new_code.py").write_text("print('changed')\n")
    with open_run(uri) as run:
        pass
    assert client.get_run(run.run_id).data.tags["git.dirty"] == "true"


def test_execution_failure_is_preserved_and_run_is_failed(store):
    uri, client, _ = store
    with pytest.raises(RuntimeError, match="forecast failed"), open_run(uri) as run:
        raise RuntimeError("forecast failed")
    assert client.get_run(run.run_id).info.status == "FAILED"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_metrics_fail_without_partially_logging_the_batch(store, invalid):
    uri, client, _ = store
    with pytest.raises(ValueError, match="must be finite"), open_run(uri) as run:
        run.log_metrics({"valid": 1, "invalid": invalid})
    recorded = client.get_run(run.run_id)
    assert recorded.info.status == "FAILED"
    assert "valid" not in recorded.data.metrics


def test_explicit_local_store_works_without_credentials_and_preserves_active_run(
    store, monkeypatch
):
    uri, client, _ = store
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/pauadal03/MLOps_Endesa.mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        experiment_id = client.get_experiment_by_name("test").experiment_id
        with mlflow.start_run(experiment_id=experiment_id) as outer:
            with open_run(uri) as run:
                assert mlflow.active_run().info.run_id == outer.info.run_id
            assert mlflow.active_run().info.run_id == outer.info.run_id
        assert client.get_run(run.run_id).info.status == "FINISHED"
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_missing_tracking_uri_does_not_fall_back_to_local_storage(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        tracking.tracking_client()


def test_dagshub_requires_the_callers_credentials(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "didicayu")
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_PASSWORD"):
        tracking.tracking_client("https://dagshub.com/pauadal03/MLOps_Endesa.mlflow")


def test_smoke_command_round_trips_in_a_local_store(store):
    uri, client, _ = store
    run_id = smoke_run(uri, "test")
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["run.purpose"] == "smoke"
    assert run.data.tags["data.version"] == "synthetic-fixture-v1"
    assert run.data.metrics["smoke.mae"] == 0.75


def test_new_experiment_is_created_and_reused(store):
    uri, client, _ = store
    with open_run(uri, experiment_name="new-experiment") as first:
        pass
    with open_run(uri, experiment_name="new-experiment") as second:
        pass
    assert (
        client.get_run(first.run_id).info.experiment_id
        == client.get_run(second.run_id).info.experiment_id
    )
