# Historical positional replay

The replay client sends validation histories to the [forecast HTTP API](api.md)
and records forecasts before releasing their observations. It runs outside the
endpoint. This is historical monitoring demonstration data, not live demand,
new final-test evaluation, or evidence of deployment on Virtech.

An origin `o` sends exactly the 512 source positions `[o-512, o)`. Step horizons
are **1 through 48**, with target position `o + horizon - 1`. The existing
[split protocol](../src/splits.py) supplies validation boundaries and stride.
Values remain in their source scale, including negative TAS values. The client
does not invent civil timestamps or assert MW units.

At replay clock `c`, model history may include positions strictly below `c`.
For monitoring, a target at `t` becomes available at **`t + 1 + delay`**.
The delay models label arrival to monitoring; it does not delay the historical
measurements available to the next forecast request. These are distinct channels.
For example, origin 512 uses positions 0..511; its first target, position 512,
is released at clock 518 with delay 5. Its last target, 559, releases at 565.

## Run a bounded validation range

Use the locked development environment (`uv sync --locked --dev`), including
the existing HTTPX dependency. Start the selected-model service as described in
the API guide after verifying its artifacts. Replay never loads a predictor or
falls back to a base model or a fixture. The integration test alone injects a
declared test predictor into the real API lifecycle.

Prepare an identity JSON file for the **expected** selected artifact and runtime
settings. Use the API response fields `model_id`, `model_revision`, `checkpoint`
(`sha256`, `training_run_id`) and `settings` (all executed settings documented in
the API guide). Review those against the intended release before replay; the
client will not adopt the identity of the first response automatically. Extra
response fields in this file are ignored. A checksum alone is insufficient:
training run, seed, sampling, CPU threads and the other executed settings must
also match.

After retrieving the DVC validation split, this command declares two origins
from the checked-in protocol, one state and a four-position monitoring delay:

```sh
uv run --dev python -m src.monitoring run \
  --source-dir data/processed --params params.yaml \
  --identity temporary/selected-api-identity.json \
  --journal reports/replay/validation.sqlite \
  --endpoint http://127.0.0.1:8000 \
  --origin-start 195696 --origin-stop 195792 --states NSW \
  --observation-delay 4 --max-forecasts 2 --max-seconds 60 \
  --release-through 195796

uv run --dev python -m src.monitoring status \
  --journal reports/replay/validation.sqlite
```

`origin-stop` is exclusive. Requests process origins chronologically, then states
in the supplied order. Repeating the command resumes the same run; a completed
run sends no further HTTP requests. Every run prints its identities and counts.
The example's expected final counts are 2 forecasts, 96 steps, 96 matches and
zero pending steps; these are positional expectations, not measured model scores.
The source loader opens only `train.parquet` and `validation.parquet`, verifies
their integer positions and ordered numeric columns, and hashes both files and
the exact parameter bytes. It never opens `test.parquet`.

`--max-forecasts` bounds successful requests per invocation (default 1, maximum
1000). `--max-seconds` defaults to 60 and cannot exceed 3600. The latter checks
between calls and response chunks; HTTPX separately bounds connect/read/write/
pool waits. It is not a hard process deadline. Response bodies are capped at
128 KiB. Each forecast has at most `--max-attempts` durable HTTP attempts
(default 3, maximum 10), each with `--timeout-seconds` (default 10, maximum 60).
The retry delay defaults to 0.1 seconds and is bounded at 5 seconds. Transport
failures and HTTP 408/425/429/500/502/503/504 consume attempts and may retry;
other statuses, malformed responses, crossed quantiles and identity/settings
mismatches stop the invocation. There are no redirects or environment proxies
in the default client. A stopped invocation exits nonzero and preserves progress.

Without `--release-through`, the clock stops at the last requested origin.
The next request advances it to its origin, releasing earlier labels there.
`--release-through` explicitly advances further, but cannot pass the next
unissued origin. After all requests it can reach `last_origin + 48 + delay` to
release the last targets. It cannot move backwards. The source range and delay
are immutable; use the same declaration on restart. Invocation limits and the
release-through position may change without creating a different run.

## Consume committed records

Consumers use public frozen dataclasses and `JournalReader` from `src.monitoring`.
They do not query SQLite tables. Every page is bounded to 1..10000 rows (default
1000). Hold one snapshot when counts and several pages must describe the same
committed state:

```python
from pathlib import Path
from src.monitoring import JournalReader

with JournalReader(Path("reports/replay/validation.sqlite")) as reader:
    with reader.snapshot():
        run = reader.run_info()
        counts = reader.counts()
        cursor = 0
        while rows := reader.matched_observations(after_id=cursor, limit=1000):
            for row in rows:
                print(row.state, row.origin, row.horizon,
                      row.p50, row.observed_value, row.seasonal_lag48)
            cursor = rows[-1].id
```

Close snapshots promptly; WAL allows the writer to commit while a reader retains
its snapshot. Outside an explicit snapshot, each method reads a fresh committed
state. Consumers should persist their own run ID and export cursor and refuse to
reuse that cursor for another run.

| Reader method | Record and meaning |
| --- | --- |
| `run_info()` | `RunInfo`: run ID, config/source SHA-256, canonical config/source manifest, `ModelIdentity`. |
| `counts()` | `ReplayCounts`: expected/completed forecasts, step/matched/pending counts, failed/unfinished attempts, clock, next origin and completion. |
| `forecast_steps(after_id=0, limit=1000)` | `ForecastStep`: stable step ID, forecast ID, state, origin, 1-based horizon, target position, p10/p50/p90, lag-48 baseline, model/source identity and input hash. Includes pending predictions without hidden labels. |
| `matched_observations(after_id=0, limit=1000)` | `MatchedObservation`: the same forecast fields plus forecast step ID, observed value and scheduled release position. Its `id` is a separate monotonic **match** ID. |
| `released_observations(after_id=0, limit=1000)` | `ReleasedObservation`: monotonic observation ID, state, position, value, scheduled release position, source reference and config hash. Unique per `(state, position)`, regardless of forecast overlap. |

The exported `seasonal_lag48` is computed with the existing
`evaluation.seasonal_forecast` from that request's identical history. Input hashes
cover the canonical `{state, history}` payload. Source references are canonical
JSON manifests containing the label, split, file hashes and numeric content hash.
`ModelIdentity` includes model ID/revision, checkpoint checksum, training run and
canonical `settings_json`. Release positions are positional integers, never wall
times. Deduplicated observations cover `[first_origin, last_origin + 48)` as the
clock permits. Separate match IDs prevent overlapping windows from causing late
joins to be skipped when consumers paginate incrementally.

`pending_steps = forecast_steps - matched_steps`. Failed HTTP requests and
unissued forecasts contribute no steps. Overlapping forecasts each receive their
own join for a target; drift instead uses the deduplicated observation reader.
`next_origin` becomes `None` when all requests finish, even if labels remain
pending. `complete` requires all expected forecasts and all their released joins.
`failed_attempts` counts recorded transport, HTTP, schema or identity failures.
`unfinished_attempts` counts calls started without a committed result: a request
may still be running, or an interruption may have left its remote outcome unknown.
Neither a running call nor an unknown outcome is reported as a confirmed failure.
Both consume the durable attempt budget until the run ends; an unknown outcome
stays visible even if a later attempt succeeds.

## Persistence, retries and retention

SQLite uses full synchronization, foreign keys and WAL. An empty database left by
interrupted first initialization is initialized on retry; foreign or unsupported
schemas are refused. An OS advisory lock
admits one writer per journal and is released automatically on process exit;
readers open the journal read-only. POSIX locking is tested locally; the Windows
locking branch requires platform verification. A forecast's 48 records, successful
attempt and request cursor commit in one transaction. Observations, joins and the
clock commit in another. The attempt is recorded **before** HTTP. A crash after a
remote success can repeat the HTTP call on restart, since the endpoint has no
idempotency key, but cannot duplicate local forecasts or joins. That interrupted
attempt still consumes the finite budget. Exhausting it stops the run; restarting
does not reset it. A new journal starts a separately identified run.

A journal binds the endpoint, state order, range, stride, delay, retry/timeout
policy, model/settings and exact source manifests. A changed declaration is
refused. It stores predictions, baselines, attempt reason codes and released
observations, not raw 512-value request histories, future labels or HTTP error
bodies. It has no automatic pruning or expiry: retain the entire journal as the
record of the declared bounded run. Archive it with SQLite's backup API or after
closing writers/readers so committed WAL contents are included. Keep journals and
sidecars under ignored `reports/replay/`; do not commit them or source datasets.

## Local evidence

```sh
uv run --dev pytest tests/test_replay.py tests/test_api.py tests/test_splits.py -q
uv run --dev ruff format --check
uv run --dev ruff check
uv run --dev pylint src tests
```

The replay tests use numbered positions to check input boundaries, every target
join, delayed and partial release, overlapping origins, retry exhaustion across
restarts, interrupted attempts, transactional rollback, schema/identity refusal,
source/config mismatch, snapshot consistency and a real local HTTP lifecycle with
a declared test predictor. They require no model weights or training. These are
source/interface checks. No frozen final-test data is evaluated.

On 28 October, a [bounded selected-model replay](../monitoring/replay/validation-smoke.json)
also passed on native macOS at commit `a021c3a`: the first 30 validation origins
`[195696, 197136)` across all five states produced 150 forecasts and 7,200 matched
steps, with delay 4 and zero final pending steps or failures. The first origin
was partially released before closing and reopening the writer; repeating the
completed invocation left counts unchanged. The API used one worker and two CPU
threads and was stopped afterward. The recorded 160.18 seconds covers startup
and the whole replay; it is not a request-latency benchmark or target-capacity
measurement. This subset does not establish full-validation interval calibration.

To reproduce that range with the CLI above, use `--origin-stop 197136`, all five
states, `--max-forecasts 150 --max-seconds 300 --release-through 197140` and a new
journal. Keep delay 4 and the recorded artifact/settings. Container execution,
Windows lock checks and Virtech end-to-end verification remain pending.
