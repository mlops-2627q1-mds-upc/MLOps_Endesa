import hashlib
import json
from pathlib import Path
import subprocess

from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest
import yaml

from src import tracking
from src.evaluation import ValidationData, mase_scale, sample_median, seasonal_forecast
from src.modeling import evaluate as evaluator
from src.modeling.checkpoints import write_training_metadata
from src.splits import SplitConfig


def test_seasonal_forecast_uses_known_values_even_beyond_one_cycle():
    contexts = np.array([[10, 11, 12, 13], [20, 21, 22, 23]])
    np.testing.assert_array_equal(
        seasonal_forecast(contexts, horizon=5, lag=2),
        [[12, 13, 12, 13, 12], [22, 23, 22, 23, 22]],
    )
    with pytest.raises(ValueError, match="full known cycle"):
        seasonal_forecast(contexts, horizon=5, lag=5)


def test_sample_median_interpolates_even_sample_counts():
    samples = np.array([[[1, 10], [3, 20]], [[4, 30], [8, 40]]])
    np.testing.assert_array_equal(sample_median(samples, 2, 2), [[2, 15], [6, 35]])
    with pytest.raises(ValueError, match="finite samples"):
        sample_median(np.full((2, 2, 2), np.nan), 2, 2)


def test_mase_scale_and_equal_state_weighting():
    assert mase_scale(np.array([1, 3, 8, 11]), lag=2) == 7.5
    data = ValidationData(
        contexts=np.zeros((3, 2)),
        targets=np.zeros((3, 2)),
        states=np.array(["A", "B", "B"]),
        origins=np.array([2, 2, 4]),
        scales={"A": 2.0, "B": 1.0},
    )
    metrics = data.metrics(np.array([[4, 4], [1, 1], [1, 1]]))
    assert metrics["A.mae"] == 4
    assert metrics["A.mase"] == 2
    assert metrics["B.mase"] == 1
    assert metrics["mean.mase"] == 1.5
    with pytest.raises(ValueError, match="zero or nonfinite"):
        mase_scale(np.ones(400))


def test_worst_origin_share_and_zero_error_case():
    data = ValidationData(
        np.zeros((20, 2)), np.zeros((20, 2)), np.full(20, "A"), np.arange(20), {"A": 1.0}
    )
    predictions = np.ones((20, 2))
    predictions[-1] = 19
    metrics = data.metrics(predictions)
    assert metrics["A.worst_5pct_error_share"] == 0.5
    assert metrics["A.worst_origin_count"] == 1
    assert data.metrics(np.zeros((20, 2)))["A.worst_5pct_error_share"] == 0


@pytest.fixture
def validation_files(tmp_path):
    root = tmp_path / "project"
    directory = root / "data/processed"
    directory.mkdir(parents=True)
    params = {
        "data": {"states": ["A", "B"], "common_length": 592},
        "split": {"train_end": 400, "validation_end": 496},
        "forecast": {"context_length": 400, "horizon": 48, "origin_stride": 48},
        "evaluation": {
            "model_id": "amazon/chronos-t5-small",
            "model_revision": "a" * 40,
            "seed": 7,
            "num_samples": 20,
            "batch_size": 16,
            "device": "cpu",
            "cpu_threads": 4,
            "max_seconds": 3600,
        },
    }
    params_path = root / "params.yaml"
    params_path.write_text(yaml.safe_dump(params))
    training = pd.DataFrame(
        {"A": np.arange(400), "B": 2 * np.arange(400)},
        index=pd.RangeIndex(0, 400, name="timestep"),
    )
    validation = pd.DataFrame(
        {"A": np.arange(400, 496), "B": 2 * np.arange(400, 496)},
        index=pd.RangeIndex(400, 496, name="timestep"),
    )
    training.to_parquet(directory / "train.parquet")
    validation.to_parquet(directory / "validation.parquet")
    return root, params_path, directory


def test_validation_windows_and_training_scale_ignore_held_out_values(validation_files):
    _, params, directory = validation_files
    cfg = SplitConfig.from_params(params)
    original = ValidationData.load(directory, cfg)
    assert original.origins.tolist() == [400, 448, 400, 448]
    np.testing.assert_array_equal(original.contexts[0], np.arange(400))
    np.testing.assert_array_equal(original.targets[0], np.arange(400, 448))
    # No test file exists; loading validation still succeeds.
    changed = pd.read_parquet(directory / "validation.parquet") + 10000
    changed.to_parquet(directory / "validation.parquet")
    updated = ValidationData.load(directory, cfg)
    assert original.scales == updated.scales == {"A": 336.0, "B": 672.0}
    np.testing.assert_array_equal(updated.contexts[0], original.contexts[0])
    assert updated.contexts[1, -1] == 10447
    assert updated.targets[1, 0] == 10448


def write_test_block(directory):
    test = pd.DataFrame(
        {"A": np.arange(496, 592), "B": 2 * np.arange(496, 592)},
        index=pd.RangeIndex(496, 592, name="timestep"),
    )
    test.to_parquet(directory / "test.parquet")


def test_test_windows_use_test_targets_and_training_only_scales(validation_files):
    _, params, directory = validation_files
    cfg = SplitConfig.from_params(params)
    validation = ValidationData.load(directory, cfg)
    write_test_block(directory)
    test = ValidationData.load(directory, cfg, "test")
    assert test.origins.tolist() == [496, 544, 496, 544]
    # The context ends just before the origin and may include validation values.
    np.testing.assert_array_equal(test.contexts[0], np.arange(96, 496))
    np.testing.assert_array_equal(test.targets[0], np.arange(496, 544))
    np.testing.assert_array_equal(test.contexts[1][-48:], np.arange(496, 544))
    assert test.scales == validation.scales == {"A": 336.0, "B": 672.0}


def test_validation_is_unchanged_by_test_values(validation_files):
    _, params, directory = validation_files
    cfg = SplitConfig.from_params(params)
    write_test_block(directory)
    before = ValidationData.load(directory, cfg)
    (pd.read_parquet(directory / "test.parquet") + 10000).to_parquet(directory / "test.parquet")
    after = ValidationData.load(directory, cfg)
    np.testing.assert_array_equal(before.contexts, after.contexts)
    np.testing.assert_array_equal(before.targets, after.targets)


def test_only_held_out_blocks_can_be_scored(validation_files):
    _, params, directory = validation_files
    with pytest.raises(ValueError, match="held-out block"):
        ValidationData.load(directory, SplitConfig.from_params(params), "train")


def test_validation_rejects_wrong_positions_and_nonfinite_predictions(validation_files):
    _, params, directory = validation_files
    cfg = SplitConfig.from_params(params)
    data = ValidationData.load(directory, cfg)
    with pytest.raises(ValueError, match="finite"):
        data.predictions_frame(np.full_like(data.targets, np.nan, dtype=float))
    validation = pd.read_parquet(directory / "validation.parquet").reset_index(drop=True)
    validation.to_parquet(directory / "validation.parquet")
    with pytest.raises(ValueError, match="positions"):
        ValidationData.load(directory, cfg)


@pytest.fixture
def tracked_project(validation_files, monkeypatch, tmp_path):
    """A committed fixture project with a local MLflow store and a fake Chronos."""
    root, params, directory = validation_files
    (root / "uv.lock").write_text("version = 1\n")
    (root / "dvc.lock").write_text("schema: '2.0'\n")
    (root / "data/raw.dvc").write_text("outs:\n- md5: fixture.dir\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    monkeypatch.setattr(evaluator, "PROJ_ROOT", root)
    monkeypatch.setattr(tracking, "PROJ_ROOT", root)
    monkeypatch.setattr(evaluator, "data_version", lambda: "fixture-data")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)

    calls = []

    def fake_chronos(contexts, horizon, cfg, deadline, measure, checkpoint=None):
        calls.append({"device": cfg.device, "checkpoint": checkpoint})
        with measure():
            predictions = np.repeat(contexts[:, -1:], horizon, axis=1)
        return predictions, {"inference_seconds": 0.1, "model_load_seconds": 0.2}

    monkeypatch.setattr(evaluator, "chronos_forecast", fake_chronos)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment("MLOps_Endesa", artifact_location=(tmp_path / "artifacts").as_uri())
    return params, directory, uri, client, calls


@pytest.mark.parametrize("native_linesep", ["\n", "\r\n"], ids=["lf", "windows-crlf"])
def test_evaluator_records_comparable_runs_and_artifacts(
    tracked_project, monkeypatch, tmp_path, native_linesep
):
    monkeypatch.setattr(evaluator.os, "linesep", native_linesep)
    params, directory, uri, client, _ = tracked_project
    comparison = evaluator.evaluate(
        params_path=params,
        processed_dir=directory,
        output_dir=tmp_path / "outputs",
        tracking_uri=uri,
    )
    expected_csv = "state,origin,timestep,target\n" + "".join(
        f"{state},{origin},{position},{multiplier * position}\n"
        for state, multiplier in (("A", 1), ("B", 2))
        for origin in (400, 448)
        for position in range(origin, origin + 48)
    )
    assert comparison["targets_sha256"] == hashlib.sha256(expected_csv.encode()).hexdigest()
    assert comparison["primary_baseline_on_validation"] == "seasonal_lag48"
    target_tables = []
    for name, result in comparison["results"].items():
        run = client.get_run(result["run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.tags["data.version"] == "fixture-data"
        assert run.data.params["block"] == "validation"
        assert run.data.params["targets_sha256"] == comparison["targets_sha256"]
        path = client.download_artifacts(run.info.run_id, "predictions.parquet", str(tmp_path))
        frame = pd.read_parquet(path)
        assert len(frame) == 192
        target_tables.append(frame.drop(columns="prediction"))
        if name == "chronos":
            assert run.data.tags["model.revision"] == "a" * 40
        assert run.data.metrics["mean.mase"] == result["metrics"]["mean.mase"]
        # Every method's forecast step is measured with the same settings.
        assert run.data.metrics["energy.kwh"] == 0.002
        assert run.data.params["energy.country_iso_code"] == "ESP"
    for frame in target_tables[1:]:
        pd.testing.assert_frame_equal(frame, target_tables[0])
    saved = json.loads((Path(comparison["output_dir"]) / "comparison.json").read_text())
    assert saved["results"] == comparison["results"]


def test_test_block_runs_are_labelled_and_select_no_baseline(tracked_project, tmp_path):
    params, directory, uri, client, _ = tracked_project
    write_test_block(directory)
    comparison = evaluator.evaluate(
        params_path=params,
        processed_dir=directory,
        output_dir=tmp_path / "outputs",
        tracking_uri=uri,
        block="test",
    )
    assert comparison["block"] == "test"
    # The primary baseline comes from validation; test scores never choose it.
    assert comparison["primary_baseline_on_validation"] is None
    for name, result in comparison["results"].items():
        run = client.get_run(result["run_id"])
        assert run.info.run_name == f"test-{name}"
        assert run.data.params["block"] == "test"
        assert result["metrics"]["A.origins"] == 2


def test_local_checkpoint_is_scored_and_identified_by_its_weights(tracked_project, tmp_path):
    params, directory, uri, client, calls = tracked_project
    checkpoint = tmp_path / "models/fine-tuning/training-run-1/final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"fine-tuned weights")
    write_training_metadata(checkpoint, "a" * 32)
    comparison = evaluator.evaluate(
        params_path=params,
        processed_dir=directory,
        output_dir=tmp_path / "outputs",
        tracking_uri=uri,
        method="chronos",
        checkpoint=checkpoint,
    )
    weights_hash = hashlib.sha256(b"fine-tuned weights").hexdigest()
    run = client.get_run(comparison["results"]["chronos"]["run_id"])
    assert calls == [{"device": "cpu", "checkpoint": checkpoint.resolve()}]
    assert run.info.run_name == "validation-chronos-checkpoint"
    assert run.data.tags["model.revision"] == f"local-sha256:{weights_hash}"
    recorded = json.loads(run.data.params["checkpoint"])
    assert recorded["sha256"] == weights_hash
    assert recorded["training_run_id"] == "a" * 32
    assert comparison["checkpoint"] == recorded


def test_checkpoint_provenance_mismatch_stops_before_inference(tracked_project, tmp_path):
    params, directory, uri, client, calls = tracked_project
    checkpoint = tmp_path / "models/selected-checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"trained weights")
    write_training_metadata(checkpoint, "a" * 32)
    (checkpoint / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="does not match checkpoint weights"):
        evaluator.evaluate(
            params_path=params,
            processed_dir=directory,
            tracking_uri=uri,
            method="chronos",
            checkpoint=checkpoint,
        )
    assert calls == []
    experiment = client.get_experiment_by_name("MLOps_Endesa")
    assert client.search_runs([experiment.experiment_id]) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"method": "all", "checkpoint": Path("final")}, "method chronos only"),
        ({"method": "chronos", "checkpoint": Path("missing")}, "No model.safetensors"),
        ({"method": "chronos", "device": "tpu"}, "explicit cpu, mps or cuda"),
    ],
)
def test_invalid_checkpoint_or_device_stops_before_a_run(tracked_project, changes, message):
    params, directory, uri, _, calls = tracked_project
    with pytest.raises(ValueError, match=message):
        evaluator.evaluate(
            params_path=params, processed_dir=directory, tracking_uri=uri, **changes
        )
    assert calls == []


def test_model_revision_must_be_pinned(validation_files):
    _, params, _ = validation_files
    config = yaml.safe_load(params.read_text())["evaluation"]
    with pytest.raises(ValueError, match="full Hub commit"):
        evaluator.EvaluationConfig(**{**config, "model_revision": "main"})


def test_missing_or_changed_dvc_inputs_stop_before_a_run(monkeypatch):
    monkeypatch.setattr(
        evaluator.subprocess, "check_output", lambda *args, **kwargs: '{"split": ["changed"]}'
    )
    with pytest.raises(ValueError, match="DVC inputs have changed or are missing"):
        evaluator.data_version()


def test_data_version_checks_only_the_data_not_tracked_models(monkeypatch):
    commands = []

    def fake_check_output(command, **kwargs):
        commands.append(command)
        return "{}"

    monkeypatch.setattr(evaluator.subprocess, "check_output", fake_check_output)
    evaluator.data_version()
    assert commands == [["dvc", "status", "--json", "data/raw.dvc", "dvc.yaml"]]


def test_untracked_input_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="DVC-tracked"):
        evaluator.evaluate(processed_dir=tmp_path)
