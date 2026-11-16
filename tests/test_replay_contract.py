"""Shared replay schema and hand-calculated consumer fixture expectations."""

from dataclasses import FrozenInstanceError
import hashlib
import json

import httpx
import pytest

from src.monitoring import SCHEMA_VERSION, JournalReader
from src.monitoring.replay import Replay
from tests.replay_fixture import create_replay_fixture


def test_shared_fixture_keeps_forecast_quality_and_unique_drift_samples_distinct(tmp_path):
    fixture = create_replay_fixture(tmp_path / "fixture")
    assert SCHEMA_VERSION == 1
    assert not (fixture.directory / "test.parquet").exists()
    with JournalReader(fixture.journal) as reader, reader.snapshot():
        counts = reader.counts()
        assert (counts.completed_forecasts, counts.forecast_steps, counts.matched_steps) == (
            4,
            192,
            6,
        )
        assert (counts.clock_position, counts.pending_steps, counts.complete) == (516, 186, False)
        assert counts.failed_attempts == counts.unfinished_attempts == 0
        samples = reader.released_observations()
        assert [
            (row.state, row.position, row.value, row.released_position) for row in samples
        ] == [
            ("NSW", 512, 512, 515),
            ("TAS", 512, -513, 515),
            ("NSW", 513, 513, 516),
            ("TAS", 513, -514, 516),
        ]
        matches = reader.matched_observations()
        assert {(row.state, row.origin, row.horizon) for row in matches} == {
            (state, origin, horizon)
            for state in ("NSW", "TAS")
            for origin, horizon in ((512, 1), (512, 2), (513, 1))
        }
        assert all(row.p50 == row.observed_value for row in matches)
        assert all(abs(row.seasonal_lag48 - row.observed_value) == 48 for row in matches)
        assert all(row.p90 - row.p10 == 2 for row in matches)
        assert all(row.released_position <= counts.clock_position for row in matches)
        with pytest.raises(FrozenInstanceError):
            matches[0].observed_value = 0


def test_fixture_restart_retains_request_ids_and_late_join_cursors(tmp_path):
    fixture = create_replay_fixture(tmp_path / "fixture")

    def unexpected_request(_request):
        pytest.fail("A completed fixture must not send HTTP on restart")

    with (
        httpx.Client(transport=httpx.MockTransport(unexpected_request)) as client,
        Replay(fixture.journal, fixture.source, fixture.config, client=client) as replay,
        JournalReader(fixture.journal) as reader,
    ):
        with reader.snapshot():
            info = reader.run_info()
            before = reader.forecast_steps()
            cursor = reader.matched_observations()[-1].id
            assert replay.run(max_forecasts=4).completed_forecasts == 4
            assert replay.release(563).complete
            with reader.snapshot():
                assert reader.counts().matched_steps == 6
                assert reader.matched_observations(after_id=cursor) == []
        assert reader.run_info() == info
        assert reader.forecast_steps() == before
        matches = reader.matched_observations(after_id=cursor, limit=10000)
        assert len(matches) == 186
        assert len(reader.released_observations()) == 98
        assert reader.counts().matched_steps == 192
        for row in before:
            history = [
                float(position if row.state == "NSW" else -position - 1)
                for position in range(row.origin - 512, row.origin)
            ]
            payload = json.dumps(
                {"state": row.state, "history": history}, sort_keys=True, separators=(",", ":")
            )
            input_hash = hashlib.sha256(payload.encode()).hexdigest()
            identity = f"{info.run_id}:{row.state}:{row.origin}:{input_hash}"
            assert row.input_sha256 == input_hash
            assert row.forecast_id == hashlib.sha256(identity.encode()).hexdigest()
        assert all(row.observed_value == row.p50 for row in matches)
        assert all(row.released_position <= 563 for row in matches)


def test_fixture_generator_preserves_an_existing_directory(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    marker = occupied / "preserve.txt"
    marker.write_text("unrelated contents", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_replay_fixture(occupied)
    assert list(occupied.iterdir()) == [marker]
    assert marker.read_text(encoding="utf-8") == "unrelated contents"
