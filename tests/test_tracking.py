import hashlib
import json
import logging
from pathlib import Path
import subprocess
from types import SimpleNamespace

import mlflow
from mlflow import MlflowClient
import pytest

from src import tracking
from src.modeling.tracking_smoke import smoke_run
from tests.conftest import FakeEmissionsTracker


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


def test_energy_of_the_measured_block_is_logged_with_its_method(store, tmp_path, fake_codecarbon):
    uri, client, _ = store
    with open_run(uri) as run, run.measure_energy():
        pass

    (tracker,) = fake_codecarbon
    assert tracker.started and tracker.stopped
    recorded = client.get_run(run.run_id)
    assert recorded.info.status == "FINISHED"
    assert recorded.data.metrics["energy.kwh"] == 0.002
    assert recorded.data.metrics["energy.cpu_kwh"] == 0.0015
    assert recorded.data.metrics["energy.emissions_kg"] == 0.0003
    assert recorded.data.metrics["energy.duration_seconds"] == 1.5
    assert recorded.data.params["energy.country_iso_code"] == "ESP"
    assert recorded.data.params["energy.tracking_mode"] == "process"
    assert recorded.data.params["energy.cpu_method"] == "TDP constant"
    assert recorded.data.params["energy.gpu_model"] == "none"
    artifacts = {item.path for item in client.list_artifacts(run.run_id, "energy")}
    assert artifacts == {"energy/emissions.csv", "energy/codecarbon.log"}
    log = Path(client.download_artifacts(run.run_id, "energy/codecarbon.log", str(tmp_path)))
    assert "CPU Tracking Method: TDP constant" in log.read_text()


def test_failure_inside_measured_block_stops_tracker_and_logs_no_energy(store, fake_codecarbon):
    uri, client, _ = store
    with (
        pytest.raises(RuntimeError, match="forecast failed"),
        open_run(uri) as run,
        run.measure_energy(),
    ):
        raise RuntimeError("forecast failed")

    (tracker,) = fake_codecarbon
    assert tracker.stopped
    recorded = client.get_run(run.run_id)
    assert recorded.info.status == "FAILED"
    assert "energy.kwh" not in recorded.data.metrics


def test_missing_codecarbon_result_fails_the_run(store, monkeypatch):
    class SilentlyFailingTracker:
        """CodeCarbon suppresses its errors and then reports no data."""

        def __init__(self, output_dir, country_iso_code):
            pass

        def start(self):
            pass

        def stop(self):
            return None

    monkeypatch.setattr(tracking, "_emissions_tracker", SilentlyFailingTracker)
    uri, client, _ = store
    with (
        pytest.raises(RuntimeError, match="no measurement"),
        open_run(uri) as run,
        run.measure_energy(),
    ):
        pass
    assert client.get_run(run.run_id).info.status == "FAILED"


def test_power_samples_are_integrated_over_the_measured_interval():
    # 100 W held from before the start, then 50 W; the interval is 2 s of each.
    samples = [(0, 100.0), (3_000_000, 50.0)]
    kwh = tracking._integrate_power_kwh(samples, start_us=1_000_000, stop_us=5_000_000)
    assert kwh == pytest.approx((100 * 2 + 50 * 2) / 3.6e6)


class FakeNvml:
    """Stands in for pynvml's power-sample buffer of a single GPU."""

    NVML_TOTAL_POWER_SAMPLES = "total-power"

    class NVMLError_NotFound(Exception):
        pass

    def __init__(self, buffer):
        self.buffer = buffer

    def nvmlDeviceGetSamples(self, handle, kind, since_us):
        newer = [
            SimpleNamespace(timeStamp=t, sampleValue=SimpleNamespace(uiVal=watts * 1000))
            for t, watts in self.buffer
            if t > since_us
        ]
        if not newer:
            raise self.NVMLError_NotFound()
        return kind, newer


def test_gpu_sampler_measures_the_board_power_samples():
    now = 1_000_000_000_000_000
    clock = iter([now / 1e6, (now + 2_000_000) / 1e6]).__next__
    nvml = FakeNvml([(now - 500_000, 50.0), (now - 100_000, 50.0)])
    sampler = tracking._GpuPowerSampler(nvml, ["gpu0"], poll_seconds=3600, clock=clock)
    sampler.start()
    nvml.buffer += [(now + 1_000_000, 110.0)]
    sampler.stop()
    assert sampler.kwh == pytest.approx((50 * 1 + 110 * 1) / 3.6e6)


def test_gpu_sampler_rejects_samples_lost_from_the_buffer():
    now = 1_000_000_000_000_000
    clock = iter([now / 1e6, (now + 5_000_000) / 1e6]).__next__
    nvml = FakeNvml([(now - 200_000, 50.0), (now - 100_000, 50.0)])
    sampler = tracking._GpuPowerSampler(nvml, ["gpu0"], poll_seconds=3600, clock=clock)
    sampler.start()
    # The two-sample buffer was fully replaced before the next read.
    nvml.buffer = [(now + 4_000_000, 110.0), (now + 4_500_000, 110.0)]
    with pytest.raises(RuntimeError, match="overwritten"):
        sampler.stop()


class FakeGpuSampler:
    def __init__(self, kwh):
        self.kwh = kwh
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_gpu_samples_replace_codecarbons_zero_and_add_their_emissions(store, monkeypatch):
    sampler = FakeGpuSampler(kwh=0.004)
    monkeypatch.setattr(tracking, "_gpu_power_sampler", lambda: sampler)
    monkeypatch.setattr(tracking, "_grid_emissions_kg", lambda kwh, country: kwh * 0.2)
    uri, client, _ = store
    with open_run(uri) as run, run.measure_energy():
        pass

    assert sampler.started and sampler.stopped
    recorded = client.get_run(run.run_id)
    assert recorded.data.metrics["energy.gpu_kwh"] == 0.004
    assert recorded.data.metrics["energy.kwh"] == pytest.approx(0.006)
    assert recorded.data.metrics["energy.emissions_kg"] == pytest.approx(0.0003 + 0.0008)
    assert recorded.data.params["energy.gpu_method"] == tracking.GPU_SAMPLES_METHOD


def test_unmeasurable_gpu_is_recorded_instead_of_a_zero(store, fake_codecarbon, monkeypatch):
    start = FakeEmissionsTracker.start

    def start_with_failing_counter(self):
        start(self)
        logging.getLogger("codecarbon").warning(tracking._GPU_COUNTER_UNSUPPORTED)

    monkeypatch.setattr(FakeEmissionsTracker, "start", start_with_failing_counter)
    uri, client, _ = store
    with open_run(uri) as run, run.measure_energy():
        pass

    recorded = client.get_run(run.run_id)
    assert "energy.gpu_kwh" not in recorded.data.metrics
    assert recorded.data.metrics["energy.kwh"] == 0.002
    assert recorded.data.params["energy.gpu_method"] == tracking.GPU_UNMEASURED_METHOD


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
