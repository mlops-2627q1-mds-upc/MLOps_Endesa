"""Hand-computable forecast metrics using the public replay reader contract."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
import yaml

from src.forecast_quality import QualityPolicy, main, quality_report
from src.monitoring import JournalReader, ModelIdentity, ReplayConfig
from src.monitoring.replay import Replay
from src.monitoring.source import ReplaySource


@pytest.fixture
def quality_case(tmp_path):
    """Two states and two origins; training lag336 MAE is exactly 336 per state."""
    params = tmp_path / "params.yaml"
    params.write_text(
        yaml.safe_dump(
            {
                "data": {"states": ["NSW", "TAS"], "common_length": 656},
                "split": {"train_end": 512, "validation_end": 608},
                "forecast": {"context_length": 512, "horizon": 48, "origin_stride": 48},
            }
        )
    )
    frame = pd.DataFrame({"NSW": np.arange(608), "TAS": -np.arange(608) - 1})
    frame.iloc[:512].to_parquet(tmp_path / "train.parquet")
    frame.iloc[512:].to_parquet(tmp_path / "validation.parquet")
    # A report must never open a final-test artifact, even when one is present.
    (tmp_path / "test.parquet").write_bytes(b"deliberately invalid final test")
    payload = {
        "model_id": "chronos-t5-small-fine-tuned",
        "model_revision": "local-sha256:" + "a" * 64,
        "checkpoint": {"sha256": "a" * 64, "training_run_id": "b" * 32},
        "settings": {
            "num_samples": 20,
            "seed": 7,
            "device": "cpu",
            "cpu_threads": 2,
            "dtype": "float32",
            "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
            "quantile_method": "linear",
            "point_statistic": "median",
        },
    }
    source = ReplaySource.from_validation(tmp_path, params)
    config = ReplayConfig(512, 608, ("NSW", "TAS"), ModelIdentity.from_api(payload))

    def handler(request):
        body = json.loads(request.content)
        direction = 1 if body["state"] == "NSW" else -1
        truth = body["history"][-1] + direction * np.arange(1, 49)
        medians = truth + np.tile([0, 2, -4, 6], 12)
        return httpx.Response(
            200,
            json={
                **deepcopy(payload),
                "state": body["state"],
                "value_scale": "source",
                "context_length": 512,
                "horizon": 48,
                "p10": (medians - 2).tolist(),
                "p50": medians.tolist(),
                "p90": (medians + 2).tolist(),
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        yield SimpleNamespace(
            path=tmp_path / "run.sqlite",
            directory=tmp_path,
            params=params,
            source=source,
            config=config,
            client=client,
        )


def _report(case, minimum=1, **kwargs):
    with JournalReader(case.path) as reader:
        return quality_report(
            reader, case.directory, case.params, QualityPolicy(minimum, **kwargs)
        )


def _complete(case):
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        replay.run(max_forecasts=4)
        replay.release(case.config.origins[-1] + 48 + case.config.observation_delay)


def _reader_proxy(reader, **overrides):
    return SimpleNamespace(
        **{
            "run_info": reader.run_info,
            "counts": reader.counts,
            "snapshot": reader.snapshot,
            "forecast_steps": reader.forecast_steps,
            "matched_observations": reader.matched_observations,
            **overrides,
        }
    )


def test_exact_metrics_baseline_and_source_scale(quality_case):
    case = quality_case
    _complete(case)
    report = _report(case, minimum=2, page_size=7)
    for group in report["by_state"]:
        assert group["selected"] == {
            "mae": 3.0,
            "mase": 3 / 336,
            "empirical_coverage": 0.5,
            "mean_interval_width": 4.0,
        }
        assert group["seasonal_lag48"] == {"mae": 48.0, "mase": 48 / 336}
        assert group["sample_status"] == "sufficient"
        assert (group["forecast_count"], group["matched_steps"], group["pending_steps"]) == (
            2,
            96,
            0,
        )
        assert group["unique_matched_targets"] == 96
    first = report["by_state_horizon"][:4]
    assert [group["selected"]["mae"] for group in first] == [0, 2, 4, 6]
    assert [group["selected"]["empirical_coverage"] for group in first] == [1, 1, 0, 0]
    assert report["metadata"]["training_scale"] == {
        "lag": 336,
        "start": 0,
        "stop": 512,
        "by_state": {"NSW": 336.0, "TAS": 336.0},
    }
    assert report["metadata"]["origin_interval"] == {
        "start": 512,
        "stop": 608,
        "stride": 48,
        "count": 2,
    }
    assert report["metadata"]["target_interval"] == {"start": 512, "stop": 608}
    assert report == _report(case, minimum=2, page_size=7)
    assert "NaN" not in json.dumps(report, allow_nan=False)


def test_training_scale_excludes_validation_values(quality_case):
    case = quality_case
    frame = pd.read_parquet(case.directory / "validation.parquet")
    frame += 100_000
    frame.to_parquet(case.directory / "validation.parquet")
    case.source = ReplaySource.from_validation(case.directory, case.params)
    with Replay(case.path, case.source, case.config, client=case.client):
        pass
    assert _report(case)["metadata"]["training_scale"]["by_state"] == {"NSW": 336.0, "TAS": 336.0}


def test_no_forecast_returns_every_declared_state_and_horizon(quality_case):
    case = quality_case
    with Replay(case.path, case.source, case.config, client=case.client):
        pass
    report = _report(case)
    assert len(report["by_state_horizon"]) == 96
    for group in report["by_state"] + report["by_state_horizon"]:
        assert group["sample_status"] == "no_forecasts"
        assert group["expected_forecasts"] == 2
        assert group["forecast_steps"] == group["pending_steps"] == 0
        assert all(value is None for value in group["selected"].values())
        assert all(value is None for value in group["seasonal_lag48"].values())


def test_delayed_partial_and_insufficient_observations(quality_case):
    case = quality_case
    case.config = replace(case.config, observation_delay=5)
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        replay.run()
        pending = _report(case)["by_state"][0]
        assert pending["sample_status"] == "pending_observations"
        assert pending["matched_steps"] == 0 and pending["pending_steps"] == 48
        replay.run()
        replay.release(522)
        report = _report(case, minimum=6)
        group = report["by_state"][0]
        assert group["sample_status"] == "insufficient_samples"
        assert group["matched_steps"] == 5 and group["pending_steps"] == 43
        assert all(value is None for value in group["selected"].values())
        assert all(value is None for value in group["seasonal_lag48"].values())
        group = _report(case, minimum=5)["by_state"][0]
        assert group["selected"]["mae"] == 12 / 5
        assert group["selected"]["empirical_coverage"] == 3 / 5
        assert report["by_state"][1]["sample_status"] == "insufficient_samples"


@pytest.mark.parametrize("filename", ["params.yaml", "train.parquet", "validation.parquet"])
def test_source_mismatch_is_rejected(quality_case, filename):
    case = quality_case
    _complete(case)
    path = case.directory / filename
    if filename.endswith("yaml"):
        path.write_text(path.read_text() + "\n")
    else:
        frame = pd.read_parquet(path)
        frame.iloc[0, 0] += 1
        frame.to_parquet(path)
    with pytest.raises(ValueError, match="source identity"):
        _report(case)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_source_is_rejected(quality_case, value):
    case = quality_case
    _complete(case)
    path = case.directory / "train.parquet"
    frame = pd.read_parquet(path).astype(float)
    frame.iloc[0, 0] = value
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="finite"):
        _report(case)


def test_duplicate_exports_do_not_double_count(quality_case):
    case = quality_case
    _complete(case)
    expected = _report(case)
    with JournalReader(case.path) as reader:

        def duplicate(method):
            def page(*, after_id, limit):
                return [
                    row for row in method(after_id=after_id, limit=limit // 2) for _ in range(2)
                ]

            return page

        proxy = _reader_proxy(
            reader,
            forecast_steps=duplicate(reader.forecast_steps),
            matched_observations=duplicate(reader.matched_observations),
        )
        actual = quality_report(proxy, case.directory, case.params, QualityPolicy(1))
    assert actual == expected


@pytest.mark.parametrize(
    "change, message",
    [
        ({"observed_value": float("nan")}, "finite"),
        ({"observed_value": float("inf")}, "finite"),
        ({"p50": 999.0}, "ordered|Conflicting"),
        ({"seasonal_lag48": 999.0}, "Conflicting join"),
        ({"source_reference": "another-source"}, "identity"),
        ({"released_position": 9999}, "release"),
        ({"forecast_step_id": 9999}, "committed forecast"),
    ],
)
def test_conflicting_or_invalid_join_is_rejected(quality_case, change, message):
    case = quality_case
    _complete(case)
    with JournalReader(case.path) as reader:

        def corrupted(*, after_id, limit):
            rows = reader.matched_observations(after_id=after_id, limit=limit)
            return [replace(rows[0], **change), *rows[1:]] if rows else rows

        with pytest.raises(ValueError, match=message):
            quality_report(
                _reader_proxy(reader, matched_observations=corrupted), case.directory, case.params
            )


def test_conflicting_duplicate_join_is_rejected(quality_case):
    case = quality_case
    _complete(case)
    with JournalReader(case.path) as reader:

        def corrupted(*, after_id, limit):
            rows = reader.matched_observations(after_id=after_id, limit=limit // 2)
            if rows:
                return [rows[0], replace(rows[0], observed_value=0), *rows[1:]]
            return rows

        with pytest.raises(ValueError, match="Conflicting match ID"):
            quality_report(
                _reader_proxy(reader, matched_observations=corrupted), case.directory, case.params
            )


def test_conflicting_model_settings_are_rejected(quality_case):
    case = quality_case
    _complete(case)
    with JournalReader(case.path) as reader:

        def corrupted(*, after_id, limit):
            rows = reader.forecast_steps(after_id=after_id, limit=limit)
            if rows:
                identity = replace(rows[0].model_identity, training_run_id="c" * 32)
                rows[0] = replace(rows[0], model_identity=identity)
            return rows

        with pytest.raises(ValueError, match="identity"):
            quality_report(
                _reader_proxy(reader, forecast_steps=corrupted), case.directory, case.params
            )


def test_truncated_export_is_rejected(quality_case):
    case = quality_case
    _complete(case)
    with pytest.raises(ValueError, match="max_steps"):
        _report(case, max_steps=191)
    with JournalReader(case.path) as reader:

        def missing(*, after_id, limit):
            return [
                row for row in reader.forecast_steps(after_id=after_id, limit=limit) if row.id != 1
            ]

        with pytest.raises(ValueError, match="snapshot counts"):
            quality_report(
                _reader_proxy(reader, forecast_steps=missing), case.directory, case.params
            )


def test_report_keeps_one_snapshot_while_replay_releases_observations(quality_case):
    case = quality_case
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        replay.run(max_forecasts=2)
        with JournalReader(case.path) as reader:

            def concurrent_counts():
                counts = reader.counts()
                replay.release(520)
                return counts

            report = quality_report(
                _reader_proxy(reader, counts=concurrent_counts), case.directory, case.params
            )
        assert report["replay_counts"]["matched_steps"] == 0
        assert report["by_state"][0]["selected"]["mae"] is None
        assert _report(case)["replay_counts"]["matched_steps"] == 16


def test_overlap_counts_forecast_errors_and_unique_targets_separately(quality_case):
    case = quality_case
    value = yaml.safe_load(case.params.read_text())
    value["forecast"]["origin_stride"] = 1
    case.params.write_text(yaml.safe_dump(value))
    case.source = ReplaySource.from_validation(case.directory, case.params)
    case.config = replace(case.config, origin_stop=514, origin_stride=1, states=("NSW",))
    _complete(case)
    group = _report(case)["by_state"][0]
    assert group["forecast_count"] == group["matched_forecast_count"] == 2
    assert group["matched_steps"] == 96 and group["unique_matched_targets"] == 49
    assert group["seasonal_lag48"]["mae"] == 48


def test_cli_reports_valid_json_and_fails_without_partial_output(quality_case, capsys):
    case = quality_case
    _complete(case)
    args = [
        "--journal",
        str(case.path),
        "--data-dir",
        str(case.directory),
        "--params",
        str(case.params),
        "--minimum-samples",
        "2",
        "--page-size",
        "1",
    ]
    assert main(args) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["replay_counts"]["matched_steps"] == 192
    assert output.err == ""
    assert main([*args, "--max-steps", "1"]) == 1
    output = capsys.readouterr()
    assert output.out == "" and "max_steps" in output.err


@pytest.mark.parametrize(
    "kwargs",
    [{"minimum_samples": 0}, {"max_steps": 0}, {"page_size": 10_001}, {"minimum_samples": True}],
)
def test_policy_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        QualityPolicy(**kwargs)
