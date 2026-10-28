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
