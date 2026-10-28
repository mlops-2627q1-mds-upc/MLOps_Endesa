"""Durable replay writes and public read-only, snapshot-consistent exports."""

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from src.api.schemas import ForecastResponse
from src.monitoring.source import ReplaySource
from src.monitoring.types import (
    ForecastStep,
    MatchedObservation,
    ModelIdentity,
    ReleasedObservation,
    ReplayConfig,
    ReplayCounts,
    RunInfo,
    sha256,
)

SCHEMA_VERSION = 1
SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE run (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1), run_id TEXT NOT NULL,
    config_json TEXT NOT NULL, config_sha256 TEXT NOT NULL, source_sha256 TEXT NOT NULL,
    source_reference TEXT NOT NULL, clock_position INTEGER NOT NULL, next_index INTEGER NOT NULL
);
CREATE TABLE attempts (
    id INTEGER PRIMARY KEY, request_index INTEGER NOT NULL, number INTEGER NOT NULL,
    outcome TEXT NOT NULL, UNIQUE(request_index, number)
);
CREATE TABLE forecasts (
    forecast_id TEXT PRIMARY KEY, state TEXT NOT NULL, origin INTEGER NOT NULL,
    input_sha256 TEXT NOT NULL, UNIQUE(state, origin)
);
CREATE TABLE steps (
    id INTEGER PRIMARY KEY, forecast_id TEXT NOT NULL REFERENCES forecasts(forecast_id),
    horizon INTEGER NOT NULL CHECK(horizon BETWEEN 1 AND 48), target_position INTEGER NOT NULL,
    p10 REAL NOT NULL, p50 REAL NOT NULL, p90 REAL NOT NULL, seasonal_lag48 REAL NOT NULL,
    UNIQUE(forecast_id, horizon)
);
CREATE TABLE observations (
    id INTEGER PRIMARY KEY, state TEXT NOT NULL, position INTEGER NOT NULL,
    value REAL NOT NULL, released_position INTEGER NOT NULL, UNIQUE(state, position)
);
CREATE TABLE matches (
    id INTEGER PRIMARY KEY, step_id INTEGER NOT NULL UNIQUE REFERENCES steps(id),
    observation_id INTEGER NOT NULL REFERENCES observations(id)
);
PRAGMA user_version=1;
COMMIT;
"""


class ReplayBusyError(RuntimeError):
    """Another process owns the journal writer lock."""


@contextmanager
def _writer_lock(path: Path):
    # Advisory OS locks survive no process and therefore need no stale-lock reset.
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt  # pylint: disable=import-outside-toplevel,import-error

            stream.write(b"\0")
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ReplayBusyError("Journal already has an active writer") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # pylint: disable=import-outside-toplevel

            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ReplayBusyError("Journal already has an active writer") from error
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _page(after_id: int, limit: int) -> None:
    if (
        type(after_id) is not int
        or after_id < 0
        or type(limit) is not int
        or not 1 <= limit <= 10000
    ):
        raise ValueError("after_id must be nonnegative and limit must be 1..10000")


class JournalReader:
    """Read an existing journal without creating it or exposing hidden observations."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(
            path.resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None
        )
        self.connection.row_factory = sqlite3.Row
        try:
            self._validate_schema()
            self.connection.execute("PRAGMA query_only=ON")
        except BaseException:
            self.connection.close()
            raise

    def _validate_schema(self):
        if self.connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("Unsupported replay journal schema")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self) -> None:
        """Close the SQLite reader."""
        self.connection.close()

    @contextmanager
    def snapshot(self):
        """Group counts and paginated exports in one committed SQLite snapshot."""
        nested = self.connection.in_transaction
        if not nested:
            self.connection.execute("BEGIN")
        try:
            yield self
        finally:
            if not nested:
                self.connection.rollback()

    def run_info(self) -> RunInfo:
        """Return the persisted run/source/config/model identities."""
        row = self.connection.execute("SELECT * FROM run WHERE singleton=1").fetchone()
        if row is None:
            raise ValueError("Journal has no initialized replay run")
        if (
            sha256(row["config_json"]) != row["config_sha256"]
            or sha256(row["source_reference"]) != row["source_sha256"]
        ):
            raise ValueError("Journal identity hashes do not match the stored manifests")
        return RunInfo(
            row["run_id"],
            row["config_sha256"],
            row["source_sha256"],
            row["source_reference"],
            row["config_json"],
            ModelIdentity(**json.loads(row["config_json"])["model_identity"]),
        )

    def counts(self) -> ReplayCounts:
        """Count committed records, recorded failures and calls with unknown outcomes."""
        with self.snapshot():
            config = json.loads(self.run_info().config_json)
            progress = self.connection.execute(
                "SELECT clock_position, next_index FROM run"
            ).fetchone()
            origins = range(config["origin_start"], config["origin_stop"], config["origin_stride"])
            expected = len(origins) * len(config["states"])
            completed = self.connection.execute("SELECT count(*) FROM forecasts").fetchone()[0]
            steps = self.connection.execute("SELECT count(*) FROM steps").fetchone()[0]
            matched = self.connection.execute("SELECT count(*) FROM matches").fetchone()[0]
            failed = self.connection.execute(
                "SELECT count(*) FROM attempts WHERE outcome NOT IN ('started', 'succeeded')"
            ).fetchone()[0]
            unfinished = self.connection.execute(
                "SELECT count(*) FROM attempts WHERE outcome = 'started'"
            ).fetchone()[0]
            next_index = progress["next_index"]
            next_origin = (
                origins[next_index // len(config["states"])] if next_index < expected else None
            )
            return ReplayCounts(
                expected,
                completed,
                steps,
                matched,
                steps - matched,
                failed,
                progress["clock_position"],
                next_origin,
                completed == expected and steps == matched,
                unfinished,
            )

    def _step_rows(self, after_id: int, limit: int, matched: bool):
        _page(after_id, limit)
        fields = "f.forecast_id, f.state, f.origin, f.input_sha256, s.horizon, s.target_position, s.p10, s.p50, s.p90, s.seasonal_lag48"
        if matched:
            query = f"""SELECT m.id, s.id AS forecast_step_id, {fields},
                o.value AS observed_value, o.released_position FROM matches m
                JOIN steps s ON s.id=m.step_id JOIN forecasts f ON f.forecast_id=s.forecast_id
                JOIN observations o ON o.id=m.observation_id WHERE m.id>? ORDER BY m.id LIMIT ?"""
        else:
            query = f"""SELECT s.id, {fields} FROM steps s
                JOIN forecasts f ON f.forecast_id=s.forecast_id WHERE s.id>? ORDER BY s.id LIMIT ?"""
        return self.connection.execute(query, (after_id, limit)).fetchall()

    def forecast_steps(self, *, after_id: int = 0, limit: int = 1000) -> list[ForecastStep]:
        """Export committed predictions, including pending steps but no hidden truth."""
        with self.snapshot():
            info = self.run_info()
            return [
                ForecastStep(
                    **dict(row),
                    model_identity=info.model_identity,
                    source_reference=info.source_reference,
                )
                for row in self._step_rows(after_id, limit, False)
            ]

    def matched_observations(
        self, *, after_id: int = 0, limit: int = 1000
    ) -> list[MatchedObservation]:
        """Export unique released joins, paginating by match ID rather than forecast ID."""
        with self.snapshot():
            info = self.run_info()
            return [
                MatchedObservation(
                    **dict(row),
                    model_identity=info.model_identity,
                    source_reference=info.source_reference,
                )
                for row in self._step_rows(after_id, limit, True)
            ]

    def released_observations(
        self, *, after_id: int = 0, limit: int = 1000
    ) -> list[ReleasedObservation]:
        """Export each released state/position once, independent of forecast overlap."""
        _page(after_id, limit)
        with self.snapshot():
            info = self.run_info()
            rows = self.connection.execute(
                "SELECT * FROM observations WHERE id>? ORDER BY id LIMIT ?", (after_id, limit)
            ).fetchall()
            return [
                ReleasedObservation(
                    **dict(row),
                    source_reference=info.source_reference,
                    config_sha256=info.config_sha256,
                )
                for row in rows
            ]


class ReplayJournal(JournalReader):
    """Single-writer journal; consumers should use JournalReader."""

    def __init__(self, path: Path, source: ReplaySource, config: ReplayConfig):
        self._resources = ExitStack()
        self.source, self.config = source, config
        source.validate_config(config)
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._resources.enter_context(_writer_lock(path.with_suffix(path.suffix + ".lock")))
            self.connection = sqlite3.connect(path, isolation_level=None)
            self._resources.callback(self.connection.close)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys=ON")
            # SQLite can leave an empty file after interrupted first initialization.
            # Recover only a pristine database, without altering foreign schemas.
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            application = self.connection.execute("PRAGMA application_id").fetchone()[0]
            populated = self.connection.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            if version == 0 and application == 0 and populated is None:
                self.connection.executescript(SCHEMA)
            self._validate_schema()
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self._bind_run()
        except BaseException:
            self._resources.close()
            raise

    def close(self) -> None:
        """Close the writer and release its OS lock."""
        self._resources.close()

    @contextmanager
    def transaction(self):
        """Commit all related records and progress together, or roll them all back."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def _bind_run(self):
        config_json = self.config.to_json()
        with self.transaction():
            row = self.connection.execute("SELECT * FROM run").fetchone()
            if row is None:
                self.connection.execute(
                    "INSERT INTO run VALUES (1,?,?,?,?,?,?,0)",
                    (
                        uuid4().hex,
                        config_json,
                        sha256(config_json),
                        self.source.source_sha256,
                        self.source.reference,
                        self.config.origin_start,
                    ),
                )
            elif (row["config_json"], row["source_sha256"], row["source_reference"]) != (
                config_json,
                self.source.source_sha256,
                self.source.reference,
            ):
                raise ValueError(
                    "Journal source/config/model mismatch; use the original run or a new journal"
                )
            self.run_info()

    def next_index(self) -> int:
        """Read the first uncommitted request index."""
        return self.connection.execute("SELECT next_index FROM run").fetchone()[0]

    def start_attempt(self, index: int) -> int:
        """Persist consumption of the finite retry budget before sending HTTP."""
        with self.transaction():
            count = self.connection.execute(
                "SELECT count(*) FROM attempts WHERE request_index=?", (index,)
            ).fetchone()[0]
            if count >= self.config.max_attempts:
                raise RuntimeError("Durable HTTP attempt budget exhausted for this forecast")
            cursor = self.connection.execute(
                "INSERT INTO attempts(request_index,number,outcome) VALUES (?,?,'started')",
                (index, count + 1),
            )
            return cursor.lastrowid

    def fail_attempt(self, attempt: int, reason: str) -> None:
        """Retain a bounded reason code without storing response bodies or raw errors."""
        with self.transaction():
            self.connection.execute("UPDATE attempts SET outcome=? WHERE id=?", (reason, attempt))

    def commit_forecast(
        self,
        index: int,
        attempt: int,
        input_hash: str,
        response: ForecastResponse,
        baseline: list[float],
    ) -> None:
        """Atomically commit all 48 steps, successful attempt and next request cursor."""
        state, origin = self.config.request_at(index)
        with self.transaction():
            if index != self.next_index():
                raise RuntimeError("Replay progress changed before forecast commit")
            forecast_id = sha256(f"{self.run_info().run_id}:{state}:{origin}:{input_hash}")
            self.connection.execute(
                "INSERT INTO forecasts VALUES (?,?,?,?)", (forecast_id, state, origin, input_hash)
            )
            self.connection.executemany(
                """INSERT INTO steps(forecast_id,horizon,target_position,
                p10,p50,p90,seasonal_lag48) VALUES (?,?,?,?,?,?,?)""",
                [
                    (
                        forecast_id,
                        step + 1,
                        origin + step,
                        response.p10[step],
                        response.p50[step],
                        response.p90[step],
                        baseline[step],
                    )
                    for step in range(48)
                ],
            )
            self.connection.execute(
                "UPDATE attempts SET outcome='succeeded' WHERE id=?", (attempt,)
            )
            self.connection.execute("UPDATE run SET next_index=?", (index + 1,))

    def advance_clock(self, position: int) -> None:
        """Release source observations and join only the steps visible at this clock."""
        if type(position) is not int:
            raise ValueError("Clock position must be an integer")
        with self.transaction():
            counts = self.counts()
            upper = (
                counts.next_origin
                if counts.next_origin is not None
                else self.config.origins[-1] + 48 + self.config.observation_delay
            )
            if not counts.clock_position <= position <= upper:
                raise ValueError("Clock cannot move backwards or skip an unissued origin")
            # The last clock's release boundary is a durable observation cursor too.
            start = max(
                self.config.origin_start, counts.clock_position - self.config.observation_delay
            )
            stop = min(self.config.origins[-1] + 48, position - self.config.observation_delay)
            self.connection.executemany(
                """INSERT INTO observations(state,position,value,released_position)
                VALUES (?,?,?,?)""",
                [
                    (
                        state,
                        target,
                        self.source.observation(
                            state, target, position, self.config.observation_delay
                        ),
                        target + 1 + self.config.observation_delay,
                    )
                    for target in range(start, stop)
                    for state in self.config.states
                ],
            )
            self.connection.execute("""INSERT INTO matches(step_id, observation_id)
                SELECT s.id, o.id FROM steps s JOIN forecasts f ON f.forecast_id=s.forecast_id
                JOIN observations o ON o.state=f.state AND o.position=s.target_position
                WHERE NOT EXISTS (SELECT 1 FROM matches m WHERE m.step_id=s.id)
                ORDER BY o.released_position, o.id, s.id""")
            self.connection.execute("UPDATE run SET clock_position=?", (position,))
