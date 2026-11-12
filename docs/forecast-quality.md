# Forecast quality from delayed observations

The quality consumer scores selected-model forecasts and the stored lag48 seasonal
baseline on exactly the same released journal joins. It runs separately from the
forecast API and uses only the public [replay reader](replay.md) methods.

Run from the checkout with the journal's exact train/validation artifacts and
`params.yaml` available:

```sh
python -m src.forecast_quality \
  --journal reports/replay/run.sqlite \
  --data-dir data/processed --params params.yaml \
  --minimum-samples 30 --max-steps 100000 --page-size 1000 \
  > temporary/forecast-quality.json
```

The command writes deterministic JSON to stdout. A failure returns a nonzero exit
code and writes no partial report. Keep generated reports in an ignored directory;
check the exit code before using redirected output. No model inference occurs.

`ReplaySource.from_validation` verifies the exact train, validation and parameter
identities against the journal before any training scale is derived. MASE reuses
`src.evaluation.mase_scale` with lag336 and every training position `[0, train_end)`;
it does not use validation errors or the final test. A zero training scale is an
error. The local validation values establish source identity only: scored actuals
come exclusively from released journal joins.

Each declared state has an aggregate result and all 48 horizon results, including
groups with no forecasts. MAE is the mean absolute point error and MASE is that
state's MAE divided by its training scale. The baseline uses the journal's stored
`seasonal_lag48`, computed from each forecast's identical 512-position history.
Coverage is the fraction satisfying `p10 <= observed <= p90`; width is the mean
`p90 - p10` over those same matched rows. Values retain the source scale. Positions
refer to 30-minute steps, rather than recovered wall-clock timestamps.

The default minimum is 30 matched rows **per reported group**, declared before
scoring. It is an evidence-display threshold, not a claim of independent samples
or statistical power. `no_forecasts`, `pending_observations` and
`insufficient_samples` groups contain null metrics for both model roles, alongside
their actual counts. A group with enough released samples may still have pending
steps; those counts remain visible. Expected forecasts include unissued requests,
while pending steps count only committed forecasts awaiting observations.

The report records origin/target intervals, sample policy, model and executed
settings, run/config/source identities and training scales in metadata. Forecast
and matched-forecast counts accompany unique matched state/target counts.
Overlapping targets contribute separate forecast errors and remain correlated.
Empirical coverage describes this replay; nominal p10/p90 does not establish
calibrated 80% coverage, and no coverage SLO is defined.

Counts and all paginated rows share one committed read transaction. Duplicate
exports do not add errors twice; conflicting IDs, forecasts, identities, joins,
overlapping truth values or release positions fail the report. Counts, complete
48-step forecasts and the chronological request prefix are checked. Each row
stream is bounded by `max_steps` (at most 1,000,000); excess volume or a truncated
export fails instead of scoring a subset. Read transactions close before metric
aggregation.

The consumer uses schema 1 from replay PR #89, verified at
`067ff119530b0f8cf221778a6a2fc5ce8d5cdda9`. It reads `run_info`, `counts`,
`forecast_steps` and `matched_observations`, checks identity before and inside the
snapshot, and starts each paginated export at zero. It does not persist cursors;
an incremental consumer must bind every cursor to its `run_id`. The service
configuration follows PR #90 at `da39aa6acc4ebed5bdf6878c23f8f7ae4e2e88c9`.
These source dependencies still require human review and integration.

Focused behavioral checks:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --locked --dev pytest tests/test_forecast_quality.py \
  tests/test_forecast_quality_exporter.py tests/test_replay_contract.py -q
```

These fixtures establish arithmetic, training-only scaling, positional release,
source verification, duplicate/conflict handling and concurrent snapshot behavior.
The shared delayed-overlap fixture from `tests/replay_fixture.py` also exercises
the consumer through the public reader. Fixture arithmetic, retained selected-model
coverage, hosted CI, local stack rendering and target deployment are separate evidence.

## Separate exporter and monitoring stack

[The exporter](../monitoring/quality/exporter.py) reads one JSON report per scrape;
it never loads a model, opens datasets or queries a journal. Publish atomically
after a successful report command so readers see a complete file version:

```sh
mkdir -p temporary/quality
python -m src.forecast_quality \
  --journal reports/replay/run.sqlite --data-dir data/processed --params params.yaml \
  --minimum-samples 30 > temporary/quality/report.json.tmp && \
  mv temporary/quality/report.json.tmp temporary/quality/report.json
ENDESA_QUALITY_REPORT="$PWD/temporary/quality/report.json" \
  python -m uvicorn monitoring.quality.exporter:create_app --factory \
  --host 127.0.0.1 --port 9101 --workers 1
```

`GET /metrics` exposes `endesa_quality_report_present`, `report_valid`,
`report_fresh`, `report_available`, `report_age_seconds` and
`report_max_age_seconds` under the same `endesa_quality_` prefix. Availability is
1 only for a valid, fresh report. All measurement and sample-count series disappear
when the file is missing, invalid, oversized, future-dated or stale; no previous
values are cached. The exporter stays reachable with availability 0, so Prometheus
can distinguish report failure from exporter failure. JSON is limited to 1 MiB.
Duplicate JSON keys, noninteger schema/horizon fields and state/horizon counts
that describe different forecast populations also invalidate the whole report.

`ENDESA_QUALITY_MAX_AGE_SECONDS` defaults to 300 and accepts 1..86400. This is an
operational report-refresh limit, not a coverage or forecasting SLO. Report age
measures file publication, **not** the age of observations or replay progress.
A completed historical run remains a historical run after its report is refreshed.
Schedule refresh separately if continuous scrapes are wanted; no scheduler or
automatic replay is installed by this change.

Quality series use only `state` (the five known states), `horizon` (`all` or 1..48)
and, for error/interval metrics, `model_role` (`selected` or `seasonal_lag48`).
`endesa_quality_sample_status` uses 0 for no forecasts, 1 for pending observations,
2 for insufficient samples and 3 for sufficient samples. Null metrics are omitted;
a computed zero error is exported as zero. Model/settings/source/config identities
stay in the JSON metadata.

The [Compose override](../monitoring/quality/compose.yml) extends the
[service stack](monitoring-service.md). It uses the API's exact image selector, mounts
the exporter source and report directory read-only, and exposes port 9101 only
to the dedicated monitoring network. The ordinary Docker bridge permits outbound
traffic; host publications for the other services bind to `127.0.0.1`. This is
not egress isolation. No checkpoint is mounted into the quality service. It has a
read-only root, one CPU, a 256 MiB memory limit and a 16 MiB `/tmp` mount. The
source mount must come from the same reviewed checkout as the image;
it is an explicit local integration configuration, not an immutable deployment
artifact. No dependency or API instrumentation changes are required.
The combined dashboard provider replaces the base stack's `service.yml`, whose
mount point already exists under its read-only provisioning directory; it retains
both the service and quality dashboards.

```sh
export QUALITY_REPORT_DIR="$PWD/temporary/quality"
export COMPOSE_PROJECT_NAME="endesa-quality-$(date +%s)"
export GIT_SHA="$(git rev-parse HEAD)"
# Set FORECAST_CHECKPOINT_DIR and GRAFANA_ADMIN_PASSWORD for the local stack.
# Choose unused FORECAST_PORT, PROMETHEUS_PORT and GRAFANA_PORT before starting.
docker compose -f monitoring/service/compose.yml -f monitoring/quality/compose.yml config --quiet
docker compose -f monitoring/service/compose.yml -f monitoring/quality/compose.yml up --build -d
```

The override retains the API/Prometheus jobs and service alerts, adds a separate
`forecast-quality` scrape job, and provisions **Endesa / Endesa forecast quality**
(`/d/endesa-quality`) with datasource UID `endesa-prometheus`. State error/baseline
comparisons, coverage/width, counts and per-horizon evidence tables share scrape
and report-availability gates. Missing data stays missing. The state selector
controls the horizon tables; sample statuses explain omitted metric cells. The
report-declared minimum appears beside availability and report age. Availability
0 is red, 1 is green, and absent evidence is neutral. Time-series gaps stay open;
legends use the final sample, including null, so an earlier value does not imply
current availability. No numerical coverage/error threshold is configured.

After a controlled report rename or an age-limit breach, availability must become
0 and quality measurements must disappear; restore and atomically refresh the
report to recover. Stopping the exporter must set `up{job="forecast-quality"}` to
0 and remove current panel measurements. These conditions are evidence-display
checks, with no external notification receiver or quality alert threshold.

Stop only the selected project after inspection, retaining its data and unrelated
resources:

```sh
docker compose -f monitoring/service/compose.yml -f monitoring/quality/compose.yml stop --timeout 10
```

## Quality-state and fixture contract for #84

`up{job="forecast-quality"}` describes the scrape. Availability, age and all
measurement queries match their own `job,instance`; one healthy exporter cannot
expose another failed exporter's retained samples. Quality measurements and
counts additionally require `endesa_quality_report_available == 1`. A missing
target stays absent. The exporter health check proves HTTP reachability only.

| Report state with exporter reachable | Present | Valid | Fresh | Available | Measurement/count series |
| --- | ---: | ---: | ---: | ---: | --- |
| Valid and within age limit | 1 | 1 | 1 | 1 | Present; insufficient metrics omitted |
| Missing | 0 | 0 | 0 | 0 | Absent |
| Malformed, conflicting, oversized or future dated | 1 | 0 | 0 | 0 | Absent |
| Valid but stale | 1 | 1 | 0 | 0 | Absent |

Report status metrics use the `endesa_quality_report_` prefix. A failed scrape
sets `up` to 0; dashboard report and measurement queries become absent even if
Prometheus retains earlier samples. Restoring a valid report and successful
scrapes recovers measurements. File freshness is unrelated to observation age.

The deterministic fixture uses NSW and TAS, origins 512/513, stride1, delay2,
training MASE scales 336, p50 equal to truth, width2 and baseline absolute error48.
With `minimum_samples=2`, clock516 has 6 matched / 186 pending rows overall.
Each state has 3 matched / 93 pending, 2 matched forecasts and 2 unique targets;
horizons 1/2/3 have 2/1/0 matches and statuses sufficient/insufficient/pending.
Releasing at 563 produces 192 matches, 0 pending and 98 unique state/targets;
every state/horizon has 2 sufficient samples. Minimum30 would keep the final
horizon metrics suppressed. These are synthetic expectations, not model results.

The [Prometheus scenarios](../monitoring/quality/dashboard-tests.yml) exercise all
18 dashboard expressions for sufficient and insufficient samples, unavailable
reports, failed/missing targets and mixed healthy/failed exporters. Run them with
the pinned Prometheus image, without network access:

```sh
docker run --rm --network none --workdir /quality \
  --mount "type=bind,src=$PWD/monitoring/quality,dst=/quality,readonly" \
  --entrypoint /bin/promtool \
  prom/prometheus:v3.15.1@sha256:fa3302279710be2f21a8301ec17a8b4f2301fd829bd113839b3b6b02b6f191a2 \
  test rules dashboard-tests.yml
```

For actual #84 execution, retain the source/image/config identities, report hash
and run ID, successful/failed scrape times and actual Grafana screenshots at each
transition. Use an isolated owned project and fixture report for tooling checks;
selected-model and deployment evidence require their separately reviewed runtime
prerequisites. Full validation coverage and owner assessment remain pending;
no coverage target is adopted by this configuration.

## Bounded selected-model observation

The selected-model validation replay covers the first
30 declared origins `[195696,197136)`, stride48, across all five states, with a
four-position observation delay. Its exact source/model identities match the
consumer's local train/validation verification. All 150 forecasts have 7,200
released joins and zero pending steps: 1,440 rows per state, and 30 per state/horizon.
The predeclared display minimum was 30. The selected point and lag48 baseline use
the same joins.

| State | Selected MAE | Selected MASE | Lag48 MAE | Lag48 MASE | Empirical coverage | Mean width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NSW | 210.536579 | 0.503993 | 427.793071 | 1.024072 | 74.93% | 518.677085 |
| VIC | 294.913368 | 1.001068 | 474.927613 | 1.612117 | 71.25% | 653.745474 |
| QLD | 118.810194 | 0.552093 | 217.827770 | 1.012212 | 62.22% | 276.757540 |
| SA | 111.665256 | 0.855924 | 167.524781 | 1.284092 | 63.13% | 236.284354 |
| TAS | 17.619535 | 0.385934 | 29.319810 | 0.642213 | 61.25% | 40.347798 |

This is measured coverage of a bounded historical validation subset. It is below
the nominal 80% central interval in each state here; it does not establish full
validation calibration, a coverage target, general model superiority, or target
deployment performance. No threshold tuning, final-test scoring or additional
model inference was performed by the consumer. The
[versioned summary](../monitoring/quality/validation-subset.json) retains identities,
counts and exact aggregate results. The journal and full per-horizon JSON report
are retained outside Git.
