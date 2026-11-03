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

Focused behavioral checks:

```sh
python -m pytest tests/test_forecast_quality.py -q
```

These fixtures establish arithmetic, training-only scaling, positional release,
source verification, duplicate/conflict handling and concurrent snapshot behavior.
Actual selected-model replay coverage, monitoring-stack integration, hosted CI,
peer review and target deployment are separate evidence.

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

The [Compose override](../monitoring/quality/compose.yml) extends the existing
[private service stack](monitoring-service.md). It reuses its built image, mounts
the exporter source and report directory read-only, and exposes port 9101 only
inside the internal monitoring network. No checkpoint is mounted into this
service. The source mount must come from the same reviewed checkout as the image;
it is an explicit local integration configuration, not an immutable deployment
artifact. No dependency or API instrumentation changes are required.

```sh
export QUALITY_REPORT_DIR="$PWD/temporary/quality"
# Set the base stack's GIT_SHA, FORECAST_CHECKPOINT_DIR and local Grafana password.
docker compose -f monitoring/service/compose.yml -f monitoring/quality/compose.yml config --quiet
docker compose -f monitoring/service/compose.yml -f monitoring/quality/compose.yml up --build -d
```

The override retains the API/Prometheus jobs and service alerts, adds a separate
`forecast-quality` scrape job, and provisions **Endesa / Endesa forecast quality**
(`/d/endesa-quality`) with datasource UID `endesa-prometheus`. State error/baseline
comparisons, coverage/width, counts and per-horizon evidence tables share scrape
and report-availability gates. Missing data stays missing. The state selector
controls the horizon tables; sample statuses explain omitted metric cells.

After a controlled report rename or an age-limit breach, availability must become
0 and quality measurements must disappear; restore and atomically refresh the
report to recover. Stopping the exporter must set `up{job="forecast-quality"}` to
0 and remove current panel measurements. These conditions are evidence-display
checks, with no external notification receiver or quality alert threshold.

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
