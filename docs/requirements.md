# Requirements: functional and non-functional

**Status:** The six open questions of the first draft were settled in the
[PR #30](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/30) discussion
(issue [#29](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/29)):
Pablo Pérez, Dídac Cayuela and Pau Adal agreed in writing, and Sindri Másson
(author) agrees. Items marked *Proposed* or *Open* are not yet agreed. Drafted by Sindri with AI
assistance (Claude Code) from the project documents, the team discussion and the
course guidance below.

## Method

Each requirement names its source and how it is verified. Non-functional
requirements also name their **ISO/IEC 25010** characteristic (with the AI-specific
additions of **ISO/IEC 25059** where relevant) and their **scope**: the data, the
model, or the whole system. The scope follows Habibullah, Gay and Horkoff (2023),
*Non-functional requirements for machine learning*, Requirements Engineering 28,
which found that practitioners rarely state requirements for data. Requirements are
written to be measurable; where a number is still unknown, the requirement says what
must be measured first. Characteristics that do not apply are listed with a reason.

Requirement numbers are stable: issues and reviews refer to them, so new
requirements are added at the end of each list.

Sources:

- **Agreed (#5):** the evaluation contract [agreed in issue #5](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/5#issuecomment-5972245779),
  recorded in the [problem definition](problem-definition.md#evaluation-contract-for-m2).
- **Agreed (#30):** settled in the PR #30 discussion described above.
- **Documented:** stated in an existing project document: the
  [dataset card](DATASET_CARD.md), [model card](MODEL_CARD.md), an EDN entry or
  [CONTRIBUTING](../CONTRIBUTING.md).
- **Course Mx:** required by a course milestone.
- **Proposed / Open:** needs a team decision or a measurement.

## Scope

The system builds, evaluates, serves, and monitors one ML component: given past
half-hourly demand for one of five Australian states, it forecasts the next 48
steps. The immediate users are the team and course reviewers. It is not intended
for operational grid scheduling ([model card](MODEL_CARD.md#intended-use-and-inputs)).

## Functional requirements

### Data

- **FR-1 Pinned data.** Retrieve the Monash archive at revision `58aafbe` and
  reject it if its SHA-256 does not match the recorded value.
  *Source:* Documented (dataset card); implementation planned in
  [#46](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/46).
  *Verify:* test with a corrupted file.
- **FR-2 Parsing.** Parse the TSF file into five state series, mapping source
  code `QUN` to `QLD`.
  *Source:* Documented (dataset card). *Verify:* unit test.
- **FR-3 Splits.** Cut each state to the first 230,736 positions, split
  train/validation/test at 195,696 and 213,216, and create a forecast origin
  every 48 steps.
  *Source:* Agreed (#5); implemented in [`src/splits.py`](../src/splits.py) and
  [`params.yaml`](../params.yaml). *Verify:* [`tests/test_splits.py`](../tests/test_splits.py).
- **FR-4 Data validation.** Validate the five raw series and the processed blocks
  as DVC stages (`configure_gx`, `validate-data`) after the split: nine Great
  Expectations suites (row counts, column order, float32, no nulls, value ranges
  from the training block only) and Python checks for infinite values, the 22
  documented Tasmanian negatives, complete block indexes and the forecast-origin
  grid. A failed check stops the DVC pipeline at `validate-data`.
  *Source:* Course M3; implemented in [#37](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/37)
  ([data validation](data-validation.md#what-is-checked), [EDN-008](edn/0008-data-validation-design.md)).
  *Verify:* `dvc repro`: 118 expectations pass on the real data; synthetic failure
  cases in `tests/test_validation_checks.py`.

### Modelling and evaluation

- **FR-5 Naive baselines.** Produce seasonal naive forecasts at lag 48 (one
  day) and lag 336 (one week).
  *Source:* Agreed (#5). *Verify:* unit test on a toy series.
- **FR-6 Zero-shot Chronos.** Forecast with Chronos-T5-small at revision
  `a971ba2` without project training: 512-value context, 48-step horizon,
  median of sampled trajectories, seed and sample count recorded.
  *Source:* Agreed (#5); Documented (model card). *Verify:* logged run.
- **FR-7 Fine-tuning.** Fine-tune Chronos on the training block only and select
  its configuration on validation data only.
  *Source:* Agreed (#5); carried out in [#13](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/13)
  (Experiment 004, [PR #53](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/53)).
  *Verify:* code review; training reads only windows ending before `train_end`.
- **FR-8 Evaluation.** Evaluate all methods on the same origins: MAE per state,
  equally weighted mean MASE (lag-336 scale from the training block), and the
  share of error in the worst 5% of origins. **Holiday errors are deferred:** the
  interval alignment of the source timestamps (00:00 or 00:30) is unresolved
  ([#6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6)), so they are
  not part of model selection or the test report. They may later be computed from
  stored predictions under both alignments as an exploratory analysis.
  *Source:* Agreed (#5); holiday deferral Agreed (#30); the matching amendment of
  the #5 protocol is still to be made. *Verify:* evaluation report.
- **FR-9 Run tracking.** Log each run with Git commit, data version,
  parameters, seed, metrics, and artifacts.
  *Source:* Documented (CONTRIBUTING); Course M2. *Verify:* MLflow run record.
- **FR-10 Model promotion.** Version the selected model with DVC and serve it only
  if it meets the decision rule fixed before the test block was opened (commit
  `903f7c4`): its test mean MASE must be lower than that of untuned Chronos, lag 48
  and lag 336, each by more than option 1's validation seed spread (0.0091).
  Otherwise serve untuned Chronos. **Outcome:** the rule is met (0.5571 against
  0.8175, 0.9885 and 0.9617, as reported in PR #53), so the fine-tuned checkpoint
  `models/chronos-t5-small-fine-tuned` is selected for later serving. The API is not deployed.
  *Source:* Agreed (#5) selection; promotion rule and decision Agreed (#30); Course M2/M5.
  *Verify:* DVC record of the checkpoint; the serving code loads it.

### Serving

- **FR-11 Forecast endpoint.** Accept a state and its 512 most recent values; return
  48 point forecasts (the median of the sampled trajectories) and their 10th and
  90th percentiles (an 80% interval), with the model version and the value scale.
  The service is stateless; the state is used for input validation, logging and
  monitoring, not by the model. A date-based replay for monitoring is a separate
  script, not an endpoint.
  *Source:* Agreed (#30); Course M4; inference in [PR #52](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/52),
  API in [#49](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/49).
  *Verify:* endpoint tests; forecasts match the evaluator for the same history.
- **FR-12 Input validation.** Reject invalid input (wrong length, non-numeric or
  non-finite values, unknown state) with a clear error.
  *Source:* Course M4. *Verify:* endpoint tests.
- **FR-13 Versioning and health.** Include the model version in each response;
  provide a health-check endpoint.
  *Source:* Proposed. *Verify:* endpoint tests.

### Monitoring

- **FR-14 Prediction logging.** Log requests and predictions, and compute forecast
  error once the observed values arrive.
  *Source:* Course M6. *Verify:* dashboard or report.
- **FR-15 Drift detection.** Detect drift in the input distribution relative to
  the training data.
  *Source:* Course M6. *Verify:* simulated drift test.

### Model testing

- **FR-16 Behavioural model tests.** Test the selected model's behaviour on synthetic
  series with limits declared before inference: minimum functionality (constant,
  zero and repeating demand), directional expectations and invariance.
  *Source:* Course M3; proposed in PRs
  [#57](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/57),
  [#58](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/58) and
  [#59](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/59).
  *Verify:* the opt-in real-model test runner; results recorded with their limits.

## Non-functional requirements

- **NFR-1 Reproducibility.** *ISO 25010: reliability. Scope: system.* On a fresh
  clone of a given commit, on Windows, macOS or Linux:
  - tracked files and DVC hashes are byte-identical (LF line endings in the
    repository and in every file the code writes);
  - deterministic results (splits, seasonal baselines, validation checks) reproduce
    exactly;
  - sampled Chronos forecasts reproduce exactly on the same device type and seed,
    and within a stated tolerance across devices.

  *Source:* Agreed (#5); Course M2; byte-identical files Agreed (#30); cross-device
  tolerance **Proposed**: about 0.002 in mean MASE, based on the observed 0.0015
  GPU/CPU difference of the control in Experiment 004.
  *Verify:* the lag-336 baseline reproduced to every digit on macOS and Windows
  (#28 review); the minimum functionality cases matched macOS on Windows to 4–6
  significant digits (#57 review).
- **NFR-2 Traceability.** *ISO 25010: security / accountability. Scope: system.*
  Every reported result links code commit, data version, checkpoint, configuration
  and run ID, and important runs start from a clean working tree (`git.dirty = false`).
  *Source:* Agreed (#5). *Verify:* MLflow tags and artifacts.
- **NFR-3 No leakage.** *ISO 25059: functional correctness. Scope: data and model.*
  Test targets are never used for training, preprocessing fits, or model selection;
  no forecast context includes values at or after its origin.
  *Source:* Agreed (#5). *Verify:* tests on context windows and training-only scales.
- **NFR-4 Portability.** *ISO 25010: flexibility / installability. Scope: system.*
  Setup works on Windows, macOS and Linux with uv and Python 3.11
  ([setup guide](setup.md)), and later in a container. On Windows, Smart App
  Control must be off or WSL used.
  *Source:* Documented; Course M5. *Verify:* a teammate's clean setup on each OS.
- **NFR-5 Code quality.** *ISO 25010: maintainability / testability. Scope: system.*
  On every PR, CI runs Ruff (lint and format), Pylint (score at least 9.0 and no
  error messages) and pytest. The data validation gate (NFR-14) runs in the DVC
  pipeline.
  *Source:* Course M3/M5; implemented in `.github/workflows/ci.yml`,
  `pyproject.toml` (#38) and `dvc.yaml` (#37). *Verify:* CI checks on each PR.
- **NFR-6 Energy.** *ISO 25010: performance efficiency / resource utilisation.
  Scope: model.* Training and inference energy are measured (CodeCarbon, with NVML
  power samples for the GPU) and reported with their measurement method and
  limitations ([energy measurement](evaluation.md#energy-measurement),
  [EDN-005](edn/0005-energy-measurement.md)).
  *Source:* Course M3. *Verify:* `energy.*` metrics on each run.
- **NFR-7 Latency.** *ISO 25010: performance efficiency / time behaviour. Scope:
  system.* A 48-step forecast for one state returns within *X* seconds at the 95th
  percentile on the serving machine. The only measurement so far is 0.54–0.58 s
  per call on an Apple M4 CPU with 20 samples (PR #52), which is not the serving
  machine.
  *Source:* Agreed (#30) to measure first. *Verify:* load test. *X*: **Open**,
  set from the serving-machine measurement with a margin.
- **NFR-8 Compute budget.** *ISO 25010: performance efficiency / resource
  utilisation. Scope: model.* The fine-tuning budget was set per run during the
  #13 pilot rather than upfront: a 1,000-step run takes about 680 s and 0.028 kWh
  on an RTX 4060 (8 GB), with stop rules of 1,800 s per training run and one hour
  per evaluation. The whole experiment measured 1.126 kWh and 196 g CO₂eq over 37
  runs, of which about 0.51 kWh was lost to a since-fixed GPU memory issue
  (Experiment 004, PR #53).
  *Source:* Agreed (#30). *Verify:* `energy.*` metrics and runtimes in MLflow.
- **NFR-9 Security.** *ISO 25010: security / confidentiality. Scope: system.* No
  credentials in the repository; each contributor uses their own DagsHub
  credentials in an ignored `.env`. Whether the API needs authentication is a team
  decision.
  *Source:* Documented (CONTRIBUTING); authentication **Open**.
- **NFR-10 Transparency.** *ISO 25059: transparency. Scope: system.* Outputs and
  documentation state the limitations: historical data, not for operational use,
  and values in the stored source scale. Any MW label keeps the qualification that
  MW is supported by the extraction package but not verified for this archive
  ([EDN-004](edn/0004-dataset-source-limits.md)).
  *Source:* Documented (cards, EDN-004); unit wording Agreed (#30).
- **NFR-11 Licensing.** Attribute the data (CC BY 4.0) and the base model
  (Apache 2.0). *Scope: system.*
  *Source:* Documented (cards).
- **NFR-12 Availability.** *ISO 25010: reliability / availability. Scope: system.*
  No uptime requirement; the service is a course demonstration.
  *Source:* Proposed.
- **NFR-13 Data integrity.** *ISO 25010: security / integrity. Scope: data.* The raw
  archive matches its recorded SHA-256 (FR-1), and every input to a run matches the
  hashes in `dvc.lock`: `dvc status` reports no changes on a fresh clone.
  *Source:* Documented (dataset card); Course M2. *Verify:* `dvc status`; the
  evaluator refuses changed DVC inputs.
- **NFR-14 Data completeness and consistency.** *ISO 25010: functional correctness.
  Scope: data.* The validation stage (FR-4) passes with 0 failed expectations before
  evaluation or training results are reported.
  *Source:* Course M3; implemented in #37. *Verify:* `reports/validation.json` shows
  `success: true`.
- **NFR-15 Accuracy.** *ISO 25059: functional correctness. Scope: model.* The selected
  model has a lower test mean MASE than untuned Chronos and both seasonal baselines,
  by more than the validation seed spread (FR-10). Met by the fine-tuned checkpoint:
  0.5571, which is 31.9% below untuned Chronos and 43.6% below lag 48.
  *Source:* Agreed (#30). *Verify:* the single test evaluation (PR #53).
- **NFR-16 Robustness.** *ISO 25059: robustness. Scope: model.* The selected model is
  better than each baseline in every state, not only on average. Met on test in all
  five states. Its error is more concentrated in the worst origins than the
  control's (worst-5% share 0.183 against 0.147), which is reported, not hidden.
  *Source:* Proposed. *Verify:* per-state test MASE.
- **NFR-17 Interval reporting.** *ISO 25059: transparency. Scope: model.* The 80%
  interval's empirical coverage is measured on validation before any coverage
  target is set. With 20 samples, the 10th and 90th percentiles are coarse.
  *Source:* Agreed (#30). *Verify:* coverage report. Target: **Open**.

### ISO characteristics considered and not applied

| Characteristic | Reason |
| --- | --- |
| Safety (ISO 25010) | Not used for operational decisions or grid control. |
| User controllability, intervenability (ISO 25059) | The service takes no automated action that an operator would need to stop. |
| Capacity and scalability | One user group and low request rates; NFR-7 covers the response time that matters. |
| Fairness and privacy | The data are aggregate demand per state, with no personal data or decisions about people. |
| Interaction capability (usability) | No user interface beyond the API; FR-12 covers clear error messages. |

## Remaining open items

1. **Latency target** (NFR-7): measure on the serving machine, then set *X*.
2. **Interval coverage target** (NFR-17): measure the 80% interval's coverage on validation first.
3. **Cross-device tolerance** (NFR-1): agree the proposed 0.002 mean-MASE tolerance or another value.
4. **Holiday errors** (FR-8): amend the #5 protocol to record the deferral; revisit once
   the timestamp alignment in #6 is resolved.
5. **API authentication** (NFR-9): decide whether the demonstration API needs it.
