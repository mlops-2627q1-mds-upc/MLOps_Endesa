# Positional demand drift prototype

Issue [#82](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/82)
adds a training-only Alibi Detect comparison for NSW, VIC, QLD, SA and TAS.
The declared validation control raised **8 alerts in 30 state windows**. The
declared level and variability interventions took **56–112 nominal days** to
detect. This is an investigation signal requiring further calibration before
paging; neither the nominal alpha nor an input alert establishes forecast error.

## Reference, features and testing schedule

[The configuration](../monitoring/drift/config.json) versions the feature recipe,
states, split boundaries, reference size, window size and finite test budget.
Positions follow [the split protocol](problem-definition.md): one step is nominally
30 minutes, demand remains in the source scale, and no calendar labels are inferred.

The last 104 whole 336-position training cycles, `[160608, 195552)`, supply both
the frozen mean phase pattern and the reference features. Their shared fit is
training evidence, not an independent control. The incomplete training tail is
excluded. One disjoint cycle yields two source-scale features: the mean and
population standard deviation of demand minus the fixed phase pattern. This
removes a repeating positional pattern while retaining interpretable level and
residual variability. It does not model annual seasonality or make cycles iid.

Six disjoint eight-cycle windows cover `[195888, 212016)`. Each comparison needs
104 reference feature rows and eight complete observed cycle rows for a state.
The validation alignment prefix `[195696, 195888)` and tail `[212016, 213216)`
remain unused: 192 and 1,200 positions respectively. No overlapping 512-value
forecast contexts become detector samples.

Alibi's two-sided `KSDrift` compares each state's two features, with a frozen
reference and Bonferroni feature correction. The nominal 0.05 budget is divided
over five states and six scheduled windows, then two features: the feature
threshold is `0.05 / 60`. The schedule cannot extend itself or refresh its
reference. Serial dependence and nonstationarity remain, so these p-values do
not provide calibrated iid false-positive guarantees for demand. The six looks
form one campaign; re-running controls does not create independent evidence.

Results include state, `[start, end)` positions, source/config/reference/campaign
hashes, sample and missing counts, feature means, KS distances, p-values and the
threshold. An incomplete state returns `insufficient_data` without running KS.
Missing rows, columns and NaNs count as missing; infinity, booleans, nonnumeric
values and duplicate or unordered positions fail validation. Negative TAS values
are valid. Constant reference/current data are supported.

## Declared assessment and observed limitations

The deterministic seasonal fixture and DVC validation data each receive three
separate campaigns: unchanged control, a level offset of one training raw-demand
standard deviation, and doubled residual variation around the training phase
pattern. Both interventions begin after the first complete window. Settings
were declared before results, with no parameter search or final-test access.

| Data / state | Unchanged control alerts | Level delay | Variability delay |
| --- | --- | --- | --- |
| Synthetic seasonal fixture, every state | 0/6 | 2,688 steps / 56 days | 2,688 / 56 days |
| DVC validation, NSW | 2/6 | 2,688 / 56 days | 5,376 / 112 days |
| DVC validation, VIC | 1/6 | 2,688 / 56 days | 2,688 / 56 days |
| DVC validation, QLD | 2/6 | 2,688 / 56 days | 2,688 / 56 days |
| DVC validation, SA | 2/6 | 2,688 / 56 days | 5,376 / 112 days |
| DVC validation, TAS | 1/6 | 2,688 / 56 days | 2,688 / 56 days |

The fixture has a known unchanged generating process and produced 0/30 control
false alerts. The historical control's 8/30 alarms (26.7%) are a false-alert proxy,
because natural changes may already exist in validation. Every declared strong
intervention was eventually detected, with no pre-intervention alerts, but NSW
and SA variability changes were missed in the first affected window. The
persistent level intervention also went unflagged in later SA/VIC windows;
first detection does not establish sustained sensitivity. Smaller changes were
not assessed. Null delay plus `undetected_exposure_positions` records a failure
to detect when it occurs, rather than inventing a delay.

The [versioned summary](../monitoring/drift/assessment-summary.json) records both
data identities, the fixed interventions, control alarms and missed windows.
The retained real selected-model replay covers only 30 nominal days. Reading its
7,200 released observations through the final CLI produced `insufficient_data`
for every scheduled detector window, with 1,248/2,688 positions present per state
in the first aligned window. No KS comparison or new inference ran for that
short-range check; identical checkpoint reuse preserved the same result.

## Replay consumption and retry behavior

The adapter uses [#81's public reader](replay.md) and its snapshot context. It
pages only `released_observations`, which contains one row per `(state, position)`;
forecast steps and matched overlapping targets are never drift samples. It
checks the exact public replay source/config identities against the verified
training/validation artifacts, rejects malformed or duplicated rows, and gates
observations by the snapshot's positional release clock. A short replay can
therefore produce only `insufficient_data`, even if all its forecasts completed.

This branch integrates main `26f260e33766689bcdce4a6db6062a9e1e74475d`
and replay #89 head `067ff119530b0f8cf221778a6a2fc5ce8d5cdda9`. The
[shared replay fixture](../tests/replay_fixture.py) has four unique observations
at clock 516 and 98 at clock 563, despite 192 matched forecast steps at completion.
The drift consumer regression uses a test-only training reference while retaining
the full 2,688-position window minimum. Both snapshots remain insufficient for
every state; none invokes KS. This fixture checks the reader integration and
negative TAS values, rather than reproducing the production training reference.

The CLI saves completed tests and their input hashes atomically with the result.
An identical retry reuses those results. Changed tested values, a changed
campaign/reference/source, a corrupt checkpoint, or a backward clock fail closed.
An existing empty or null report is rejected and preserved; only a missing report
starts a new checkpoint.
Incomplete windows can be retried before later complete windows for that state;
an incomplete earlier slot remains reportable on subsequent polls. A file lock
prevents concurrent CLI writers to one checkpoint. Retain the same output path
for the declared replay campaign; deleting it is not an authorized budget reset.

The Python `assess_replay` checkpoint also saves `replay_snapshot`, containing
`run_id`, `clock_position` and `last_scanned_id`. Retain all three with the test
ledger. A retry rejects a different run or a backward clock/cursor even if no
window has completed. Each call rescans the bounded snapshot from observation ID
zero so it can verify tested input hashes; `last_scanned_id` records that scan and
is not an incremental input argument. Ledgers without snapshot metadata remain
readable and acquire it on the next successful call. Their original report is
needed for the CLI's backward-clock guard during that migration.

## Historical scenario interface for #84

`recorded_scenario` in [src/drift/scenario.py](../src/drift/scenario.py) and the
`scenario` CLI replay the retained validation assessment. They read only the
versioned summary, verify its frozen SHA-256, and expose at most six windows
across five states for `unchanged`, `level` or `variability`. They run no KS
comparisons and open no datasets, model, journal or service. The normal control
retains all 8/30 alarms; the shifts retain their 56-112 nominal-day first detection
delays and missed windows. Source, reference, configuration and original report
hashes identify the evidence being played back. Thresholds and original results
are unchanged.

The caller supplies `clock_position` and a nonnegative positional observation
delay. Window `[start, end)` becomes available at `end + delay`, following #81's
last-target + 1 + delay convention. Before that position, its status is `pending`.
This status describes playback availability; it is not a measured missing-sample
count. `recorded_summary` retains the complete prior assessment, while `windows`
and `timing.replay_first_signal_position` describe the requested playback clock.
`timing.replay_detection_delay_positions` includes the supplied label delay;
the recorded detector delay in `recorded_summary` excludes it. The unchanged
control reports its first alarm positions, with null detection delay because
there is no intervention onset.

For example, with a seven-position delay the first affected window is available
at 201271. NSW and SA variability shifts remain missed there; their first signal
is available at 203959, or 5,383 positions after onset. These are nominal replay
positions. `wall_clock_alert_latency_seconds` remains null, and
`alert_delivery_verified` remains false. Process execution time cannot establish
alert latency. The persistent interventions were never removed, so
`recovery_assessed` is false. Later `no_drift` windows under a declared shift are
marked `missed_declared_shift`; they cannot establish recovery or alert clearing.

Run each command with a separate output path; assessment playback refuses to
overwrite prior evidence:

```sh
monitoring/drift/.venv/bin/python -m src.drift scenario \
  --scenario unchanged --clock-position 212016 \
  --output monitoring/drift/output/scenario-control.json
monitoring/drift/.venv/bin/python -m src.drift scenario \
  --scenario variability --clock-position 201271 --observation-delay 7 \
  --output monitoring/drift/output/scenario-first-window.json
monitoring/drift/.venv/bin/python -m src.drift scenario \
  --scenario variability --clock-position 203959 --observation-delay 7 \
  --output monitoring/drift/output/scenario-first-signal.json
```

This interface supplies deterministic evidence for a response demonstration.
It does not satisfy #84's runtime alert-delivery, wall-clock timing or recovery
criteria. Those need a later isolated execution with recorded stimulus, receiver
and recovery timestamps, alongside an appropriate normal control. Keep the
historical calibration limits visible when interpreting that demonstration.

## Reproduction

Alibi Detect 0.13.0 requires NumPy below 2. The independent Python 3.11
[uv project and lock](../monitoring/drift/pyproject.toml) preserve the serving
environment. PyArrow is constrained below 26 because its version 26 wheel
rejects NumPy 1.26.4 at import. No Torch/TF extras are installed; Alibi's base
package still brings its upstream scientific/image/transformer dependencies.
The small dotenv/loguru/typer dependencies support existing project imports.

Run from the repository root:

```sh
uv sync --project monitoring/drift --locked --dev
monitoring/drift/.venv/bin/python -m pytest --noconftest tests/test_drift*.py -q
monitoring/drift/.venv/bin/python -m src.drift fixture \
  --output monitoring/drift/output/fixture-check.json
monitoring/drift/.venv/bin/python -m src.drift assess \
  --data-dir data/processed \
  --output monitoring/drift/output/validation-check.json
monitoring/drift/.venv/bin/python -m src.drift replay \
  --data-dir data/processed --journal /path/to/declared-replay.sqlite \
  --output monitoring/drift/output/replay-checkpoint.json
```

`--noconftest` excludes the unrelated project MLflow fixture from this small
environment. The dedicated [workflow](../.github/workflows/drift.yml) explicitly
imports the real backend and runs pure, real KS and actual journal integration
checks. Main CI tests pure logic and skips only the real Alibi cases when the
optional dependency is absent. Workflow configuration is not proof of a remote
successful run. On Windows, use `.venv/Scripts/python.exe` for these commands.

The data loader verifies the selected train/validation DVC output checksums,
raw source identity, split source and parameters; it never opens the test file.
This verifies local artifact provenance, not remote access or complete pipeline
reproduction. Assessments refuse to overwrite an existing output. Generated
references/journals/checkpoints stay under ignored `monitoring/drift/output/`.
The default time budget is 120 seconds, capped at 300, checked between pages and
comparisons; it is not a hard process deadline. Pagination is capped at 128
pages of 1,000 records. No command trains or runs model inference.

After an alert, first check source/version, release coverage and feature counts;
inspect the affected state's phase pattern, level and residual variation. Check
seasonal or data-quality explanations, then compare delayed forecast error and
the seasonal baseline on matching targets before considering model changes.
Keep the frozen reference while investigating; neither automatic retraining nor
operational paging is implemented.

Treat `insufficient_data` as a coverage problem before interpreting demand drift.
For a complete window, compare the two feature changes with the unchanged
control's alarm rate and retain missed shifts in the investigation record. Input
drift alone does not establish forecast degradation; compare released forecast
errors and the lag-48 baseline on identical targets through #83. A later
`no_drift` result cannot establish model recovery. Human assessment of the
false-alert burden, detection delay and calibration remains required before
operational notifications. Formal decision PR #97 remains separate; additional
implementation proposals stay in ignored issue-specific temporary text notes.

The [official KS documentation](https://docs.seldon.ai/alibi-detect/cd/methods/offline/ksdrift)
defines the featurewise test and correction API;
[the pinned release](https://pypi.org/project/alibi-detect/0.13.0/) supplies the
dependency specification. This project's aggregation, schedule and empirical
interpretation are local engineering choices pending teammate review.
