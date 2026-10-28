"""Hand-checkable historical replay, durable progress and HTTP contract tests."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import json
import socket
import sqlite3
from threading import Thread
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
import uvicorn
import yaml

from src.api.app import create_app
from src.api.settings import Settings
from src.monitoring import JournalReader, ModelIdentity, ReplayConfig
from src.monitoring.__main__ import main
from src.monitoring.journal import ReplayBusyError
from src.monitoring.replay import Replay, ReplayRequestError
from src.monitoring.source import ReplaySource
from src.splits import SplitConfig


@pytest.fixture
def identity_payload():
    """Explicit fixture identity; no artifact loader or production fallback is used."""
    return {
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


@pytest.fixture
def setup_replay(tmp_path, identity_payload):
    """Two validation origins with 48 targets each, on a numbered source scale."""
    split = SplitConfig(("NSW", "TAS"), 656, 512, 608, 512, 48, 48)
    frame = pd.DataFrame({"NSW": np.arange(608), "TAS": -np.arange(608) - 1})
    source = ReplaySource(frame, split, "numbered-fixture")
    config = ReplayConfig(
        512, 608, ("NSW",), ModelIdentity.from_api(identity_payload), retry_delay_seconds=0
    )
    calls = []

    def prediction(history):
        medians = [float(history[-1] + step) for step in range(1, 49)]
        return {
            **deepcopy(identity_payload),
            "value_scale": "source",
            "context_length": 512,
            "horizon": 48,
            "p10": [mid - 1 for mid in medians],
            "p50": medians,
            "p90": [mid + 1 for mid in medians],
        }

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={**prediction(body["history"]), "state": body["state"]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        yield SimpleNamespace(
            path=tmp_path / "run.sqlite",
            source=source,
            config=config,
            client=client,
            calls=calls,
            prediction=prediction,
            frame=frame,
        )


def test_context_and_48_step_joins_are_positional_and_resumable(setup_replay):
    case = setup_replay
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        counts = replay.run()
        assert (counts.forecast_steps, counts.matched_steps, counts.pending_steps) == (48, 0, 48)
        assert counts.clock_position == 512 and counts.next_origin == 560
        assert case.calls[0]["history"] == list(range(512))
        assert replay.release(512) == counts
        assert replay.release(513).matched_steps == 1
        with JournalReader(case.path) as reader:
            first = reader.matched_observations()[0]
            assert (first.origin, first.horizon, first.target_position) == (512, 1, 512)
            assert (first.p10, first.p50, first.p90, first.observed_value) == (511, 512, 513, 512)
            assert first.seasonal_lag48 == 464 and first.released_position == 513
            assert reader.released_observations()[0].position == 512
            run_id = reader.run_info().run_id
        assert replay.release(520).matched_steps == 8
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        counts = replay.run()
        assert len(case.calls) == 2
        assert case.calls[1]["history"] == list(range(48, 560))
        assert (counts.completed_forecasts, counts.matched_steps, counts.next_origin) == (
            2,
            48,
            None,
        )
        assert replay.release(608).complete
        assert replay.run().complete
        assert replay.release(608).matched_steps == 96
    with JournalReader(case.path) as reader:
        assert reader.run_info().run_id == run_id
        rows = reader.matched_observations()
        assert len({(row.forecast_id, row.horizon) for row in rows}) == 96
        assert all(row.target_position == row.origin + row.horizon - 1 for row in rows)
        assert all(row.observed_value == row.target_position for row in rows)
        assert len(reader.released_observations()) == 96
        assert len({row.forecast_id for row in rows}) == 2


def test_release_delay_is_a_strict_truth_gate(setup_replay):
    case = setup_replay
    config = replace(case.config, observation_delay=5)
    with Replay(case.path, case.source, config, client=case.client) as replay:
        replay.run()
        assert replay.release(517).matched_steps == 0
        assert replay.release(518).matched_steps == 1
    with JournalReader(case.path) as reader:
        assert reader.matched_observations()[0].released_position == 518
        assert len(reader.released_observations()) == 1
    with pytest.raises(ValueError, match="release"):
        case.source.observation("NSW", 512, 517, 5)


def test_origin_major_order_and_negative_source_values(setup_replay):
    case = setup_replay
    config = replace(case.config, states=("TAS", "NSW"))
    with Replay(case.path, case.source, config, client=case.client) as replay:
        counts = replay.run(max_forecasts=3)
        assert counts.completed_forecasts == 3 and counts.next_origin == 560
    assert [(body["state"], body["history"][-1]) for body in case.calls] == [
        ("TAS", -512),
        ("NSW", 511),
        ("TAS", -560),
    ]


def test_overlap_deduplicates_observations_and_uses_monotonic_match_pages(setup_replay):
    case = setup_replay
    split = replace(case.source.split, origin_stride=1)
    source = ReplaySource(case.frame, split, "overlap-fixture")
    config = replace(case.config, origin_stop=514, origin_stride=1)
    with Replay(case.path, source, config, client=case.client) as replay:
        replay.run(max_forecasts=2)
        with JournalReader(case.path) as reader:
            first = reader.matched_observations()
            assert len(first) == 1 and first[0].target_position == 512
        replay.release(514)
        with JournalReader(case.path) as reader:
            page = reader.matched_observations(after_id=first[-1].id)
            assert {(row.origin, row.horizon) for row in page} == {(512, 2), (513, 1)}
        assert replay.release(561).matched_steps == 96
    with JournalReader(case.path) as reader:
        observations = reader.released_observations()
        matches = reader.matched_observations()
        assert [row.position for row in observations] == list(range(512, 561))
        assert Counter(row.target_position for row in matches)[513] == 2
        assert len({(row.state, row.position) for row in observations}) == 49
        assert len(reader.forecast_steps(after_id=48, limit=100)) == 48


def test_reader_snapshot_is_coherent_during_writes(setup_replay):
    case = setup_replay
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        replay.run()
        with JournalReader(case.path) as reader:
            with reader.snapshot() as snapshot:
                assert snapshot is reader
                assert reader.counts().matched_steps == 0
                replay.release(520)
                assert reader.counts().matched_steps == 0
                assert reader.matched_observations() == []
                assert reader.released_observations() == []
            assert reader.counts().matched_steps == len(reader.matched_observations()) == 8
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                reader.connection.execute("DELETE FROM run")


@pytest.mark.parametrize("failure", [503, 429, "timeout"])
def test_transient_failure_retries_once_without_duplicate_records(setup_replay, failure):
    case = setup_replay
    attempts = []

    def handler(request):
        attempts.append(json.loads(request.content))
        if len(attempts) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("fixture timeout", request=request)
            return httpx.Response(failure)
        body = attempts[-1]
        return httpx.Response(
            200, json={**case.prediction(body["history"]), "state": body["state"]}
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        Replay(case.path, case.source, case.config, client=client) as replay,
    ):
        counts = replay.run()
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert (counts.completed_forecasts, counts.forecast_steps, counts.failed_attempts) == (
        1,
        48,
        1,
    )


def test_failed_http_calls_are_bounded_across_restarts(setup_replay):
    case = setup_replay
    config = replace(case.config, max_attempts=2)
    calls = []

    def handler(request):
        calls.append(request)
        assert request.extensions["timeout"]["read"] <= config.timeout_seconds
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        for _ in range(2):
            with Replay(case.path, case.source, config, client=client) as replay:
                with pytest.raises(RuntimeError, match="budget exhausted"):
                    replay.run()
                counts = replay.journal.counts()
                assert counts.failed_attempts == 2 and counts.forecast_steps == 0
                assert counts.next_origin == 512 and counts.clock_position == 512
    assert len(calls) == 2


def test_interrupted_inflight_attempt_consumes_budget_and_can_resume(setup_replay):
    case = setup_replay

    def interrupted(_request):
        raise KeyboardInterrupt()

    with (
        httpx.Client(transport=httpx.MockTransport(interrupted)) as client,
        Replay(case.path, case.source, case.config, client=client) as replay,
        pytest.raises(KeyboardInterrupt),
    ):
        replay.run()
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        counts = replay.run()
        assert counts.failed_attempts == 0 and counts.unfinished_attempts == 1
        assert counts.forecast_steps == 48


def test_active_request_is_unfinished_until_its_result_is_committed(setup_replay):
    case = setup_replay

    def handler(request):
        with JournalReader(case.path) as reader:
            counts = reader.counts()
            assert counts.failed_attempts == 0
            assert counts.unfinished_attempts == 1
            assert counts.completed_forecasts == 0
        body = json.loads(request.content)
        return httpx.Response(
            200, json={**case.prediction(body["history"]), "state": body["state"]}
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        Replay(case.path, case.source, case.config, client=client) as replay,
    ):
        counts = replay.run()
        assert counts.failed_attempts == counts.unfinished_attempts == 0
        assert counts.completed_forecasts == 1


@pytest.mark.parametrize("status", [301, 400, 401, 404, 422])
def test_nonretryable_http_stops_without_advancing(setup_replay, status):
    case = setup_replay
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        Replay(case.path, case.source, case.config, client=client) as replay,
    ):
        with pytest.raises(ReplayRequestError, match="Nonretryable"):
            replay.run()
        assert replay.journal.counts().next_origin == 512
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"state": "TAS"},
        {"model_revision": "local-sha256:" + "c" * 64},
        {"p50": [1.0] * 47},
        {"p90": [-999.0] * 48},
        {"p10": [float("nan")] * 48},
        {"context_length": 511},
        {"horizon": 49},
        {"value_scale": "MW"},
        {"settings": None},
        {"p50": [True] * 48},
    ],
)
def test_response_schema_and_identity_fail_closed(setup_replay, change):
    case = setup_replay
    response = {**case.prediction(list(range(512))), "state": "NSW", **change}
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=json.dumps(response).encode())
            )
        ) as client,
        Replay(case.path, case.source, case.config, client=client) as replay,
    ):
        with pytest.raises(ReplayRequestError):
            replay.run()
        counts = replay.journal.counts()
        assert counts.forecast_steps == 0 and counts.failed_attempts == 1
        assert counts.next_origin == 512


@pytest.mark.parametrize("change_identity", ["checkpoint", "seed", "training_run"])
def test_valid_but_different_model_or_settings_are_refused(setup_replay, change_identity):
    case = setup_replay
    response = {**case.prediction(list(range(512))), "state": "NSW"}
    if change_identity == "checkpoint":
        response["checkpoint"]["sha256"] = "c" * 64
        response["model_revision"] = "local-sha256:" + "c" * 64
    elif change_identity == "seed":
        response["settings"]["seed"] = 8
    else:
        response["checkpoint"]["training_run_id"] = "c" * 32
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
        ) as client,
        Replay(case.path, case.source, case.config, client=client) as replay,
        pytest.raises(ReplayRequestError, match="declared run"),
    ):
        replay.run()


@pytest.mark.parametrize("content", [b"not json", b"{}", b"[1,2]", b" " * 131073])
def test_malformed_or_oversized_response_is_bounded(setup_replay, content):
    case = setup_replay
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
        ) as client,
        Replay(case.path, case.source, case.config, client=client) as replay,
    ):
        with pytest.raises(ReplayRequestError):
            replay.run()
        assert replay.journal.counts().failed_attempts == 1


@pytest.mark.parametrize(
    "change",
    [
        {"observation_delay": 1},
        {"states": ("TAS",)},
        {"origin_stop": 560},
        {"endpoint": "http://other.invalid:8000"},
        {"timeout_seconds": 5},
    ],
)
def test_different_config_cannot_resume_an_existing_journal(setup_replay, change):
    case = setup_replay
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        replay.run()
    with pytest.raises(ValueError, match="mismatch"):
        Replay(case.path, case.source, replace(case.config, **change), client=case.client)
    assert len(case.calls) == 1


def test_changed_future_source_cannot_resume_and_never_enters_input(setup_replay):
    case = setup_replay
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        replay.run()
    frame = case.frame.copy()
    frame.loc[607, "NSW"] = 1000000
    changed = ReplaySource(frame, case.source.split, "numbered-fixture")
    assert changed.history("NSW", 512) == case.calls[0]["history"]
    with pytest.raises(ValueError, match="mismatch"):
        Replay(case.path, changed, case.config, client=case.client)


def test_atomic_forecast_commit_rolls_back_steps_and_progress(setup_replay):
    case = setup_replay
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        replay.journal.connection.execute("""CREATE TRIGGER interrupt_step BEFORE INSERT ON steps
            WHEN NEW.horizon=24 BEGIN SELECT RAISE(ABORT,'fixture interruption'); END""")
        with pytest.raises(sqlite3.IntegrityError):
            replay.run()
        counts = replay.journal.counts()
        assert counts.completed_forecasts == counts.forecast_steps == 0
        assert counts.next_origin == 512
        replay.journal.connection.execute("DROP TRIGGER interrupt_step")
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        assert replay.run().forecast_steps == 48
    assert len(case.calls) == 2 and case.calls[0] == case.calls[1]


def test_atomic_release_rolls_back_truth_joins_and_clock(setup_replay):
    case = setup_replay
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        replay.run()
        replay.journal.connection.execute("""CREATE TRIGGER interrupt_match BEFORE INSERT ON matches
            WHEN NEW.step_id=3 BEGIN SELECT RAISE(ABORT,'fixture interruption'); END""")
        with pytest.raises(sqlite3.IntegrityError):
            replay.release(516)
        assert replay.journal.counts().clock_position == 512
        assert replay.journal.released_observations() == []
        assert replay.journal.matched_observations() == []
        replay.journal.connection.execute("DROP TRIGGER interrupt_match")
        assert replay.release(516).matched_steps == 4


def test_single_writer_lock_does_not_block_readers_and_is_released(setup_replay):
    case = setup_replay
    with Replay(case.path, case.source, case.config, client=case.client):
        with pytest.raises(ReplayBusyError):
            Replay(case.path, case.source, case.config, client=case.client)
        with JournalReader(case.path) as reader:
            assert reader.counts().completed_forecasts == 0
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        assert replay.run().completed_forecasts == 1


def test_clock_cannot_skip_unissued_origins_or_go_backwards(setup_replay):
    case = setup_replay
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        with pytest.raises(ValueError, match="skip"):
            replay.release(513)
        replay.run()
        with pytest.raises(ValueError, match="skip"):
            replay.release(561)
        replay.release(520)
        with pytest.raises(ValueError, match="backwards"):
            replay.release(519)


def _write_validation(directory, case):
    params = directory / "params.yaml"
    cfg = case.source.split
    params.write_text(
        yaml.safe_dump(
            {
                "data": {"states": list(cfg.states), "common_length": cfg.common_length},
                "split": {"train_end": cfg.train_end, "validation_end": cfg.validation_end},
                "forecast": {
                    key: asdict(cfg)[key] for key in ("context_length", "horizon", "origin_stride")
                },
            }
        )
    )
    case.frame.iloc[:512].to_parquet(directory / "train.parquet")
    case.frame.iloc[512:].to_parquet(directory / "validation.parquet")
    return params


def test_validation_loader_hashes_files_and_never_opens_final_test(tmp_path, setup_replay):
    case = setup_replay
    params = _write_validation(tmp_path, case)
    source = ReplaySource.from_validation(tmp_path, params)
    assert source.history("NSW", 560) == list(range(48, 560))
    manifest = json.loads(source.reference)
    assert set(manifest["file_hashes"]) == {"params.yaml", "train.parquet", "validation.parquet"}
    assert not (tmp_path / "test.parquet").exists()
    params.write_text(params.read_text() + "\n# changed protocol file bytes\n")
    assert ReplaySource.from_validation(tmp_path, params).source_sha256 != source.source_sha256


@pytest.mark.parametrize(
    "bad_frame", ["positions", "columns", "nan", "strings", "boolean", "complex"]
)
def test_loader_refuses_invalid_positional_sources(setup_replay, bad_frame):
    case = setup_replay
    frame = case.frame.copy()
    if bad_frame == "positions":
        frame.index = pd.date_range("2000-01-01", periods=608, freq="30min")
    elif bad_frame == "columns":
        frame = frame[["TAS", "NSW"]]
    elif bad_frame == "strings":
        frame = frame.astype(str)
    elif bad_frame == "boolean":
        frame = frame.astype(bool)
    elif bad_frame == "complex":
        frame = frame.astype(complex)
    else:
        frame.loc[550, "NSW"] = float("nan")
    with pytest.raises(ValueError):
        ReplaySource(frame, case.source.split, "invalid")


def test_invalid_origins_and_run_budgets_never_send_http(setup_replay):
    case = setup_replay
    with pytest.raises(ValueError, match="protocol"):
        Replay(case.path, case.source, replace(case.config, origin_start=513), client=case.client)
    with pytest.raises(ValueError, match="protocol"):
        Replay(
            case.path, case.source, replace(case.config, origin_stop=10**100), client=case.client
        )
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        for kwargs in (
            {"max_forecasts": 0},
            {"max_forecasts": True},
            {"max_forecasts": 1001},
            {"max_seconds": float("inf")},
            {"max_seconds": 0},
        ):
            with pytest.raises(ValueError):
                replay.run(**kwargs)
    assert case.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observation_delay": -1},
        {"observation_delay": True},
        {"max_attempts": 11},
        {"timeout_seconds": float("nan")},
        {"retry_delay_seconds": 6},
        {"states": ("NSW", "NSW")},
        {"states": ("WA",)},
        {"endpoint": "http://name:secret@localhost"},
        {"endpoint": "file:///private"},
    ],
)
def test_invalid_configs_are_refused(setup_replay, kwargs):
    with pytest.raises(ValueError):
        replace(setup_replay.config, **kwargs)


def test_reader_refuses_unknown_schema_and_invalid_pages(tmp_path, setup_replay):
    empty = tmp_path / "unknown.sqlite"
    sqlite3.connect(empty).close()
    with pytest.raises(ValueError, match="schema"):
        JournalReader(empty)
    case = setup_replay
    with (
        Replay(case.path, case.source, case.config, client=case.client),
        JournalReader(case.path) as reader,
    ):
        for kwargs in ({"limit": 0}, {"limit": 10001}, {"after_id": -1}, {"after_id": True}):
            with pytest.raises(ValueError):
                reader.matched_observations(**kwargs)


def test_journal_aliases_share_the_single_writer_lock(tmp_path, setup_replay):
    case = setup_replay
    with Replay(case.path, case.source, case.config, client=case.client):
        alias = tmp_path / "alias.sqlite"
        alias.symlink_to(case.path)
        with pytest.raises(ReplayBusyError):
            Replay(alias, case.source, case.config, client=case.client)


def test_corrupt_manifest_hash_is_refused_on_read_and_resume(setup_replay):
    case = setup_replay
    with Replay(case.path, case.source, case.config, client=case.client) as replay:
        replay.journal.connection.execute("UPDATE run SET config_sha256=?", ("0" * 64,))
    with JournalReader(case.path) as reader, pytest.raises(ValueError, match="hashes"):
        reader.run_info()
    with pytest.raises(ValueError, match="hashes"):
        Replay(case.path, case.source, case.config, client=case.client)


def test_cli_and_real_http_use_declared_test_predictor(
    tmp_path, setup_replay, identity_payload, capsys
):
    case = setup_replay
    histories = []

    class DeclaredTestPredictor:
        """Only this test injects the numbered predictor into the real API lifecycle."""

        def predict(self, history):
            """Return an explicit fixture forecast and capture its exact input."""
            histories.append(history)
            return case.prediction(history)

    application = create_app(Settings(), predictor_factory=lambda _: DeclaredTestPredictor())
    server = uvicorn.Server(uvicorn.Config(application, log_level="error", lifespan="on"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        endpoint = f"http://127.0.0.1:{listener.getsockname()[1]}"
        thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            params = _write_validation(tmp_path, case)
            identity = tmp_path / "identity.json"
            identity.write_text(json.dumps(identity_payload))
            args = [
                "run",
                "--journal",
                str(case.path),
                "--source-dir",
                str(tmp_path),
                "--params",
                str(params),
                "--identity",
                str(identity),
                "--endpoint",
                endpoint,
                "--origin-start",
                "512",
                "--origin-stop",
                "608",
                "--states",
                "NSW",
            ]
            assert main(args) == 0
            assert json.loads(capsys.readouterr().out)["counts"]["completed_forecasts"] == 1
            assert main(args + ["--release-through", "608"]) == 0
            assert json.loads(capsys.readouterr().out)["counts"]["complete"]
            assert main(args) == 0
            capsys.readouterr()
            assert histories == [list(range(512)), list(range(48, 560))]
            assert main(["status", "--journal", str(case.path)]) == 0
            assert json.loads(capsys.readouterr().out)["counts"]["matched_steps"] == 96
            assert main(args + ["--observation-delay", "1"]) == 1
            assert json.loads(capsys.readouterr().out)["error"] == "ValueError"
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()
