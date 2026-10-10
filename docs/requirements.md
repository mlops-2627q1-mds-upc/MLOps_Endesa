# Requirements: functional and non-functional

**Status:** Draft for team review. Proposed by Sindri Másson; drafted with AI assistance 
from the existing [problem definition](problem-definition.md), [dataset card](DATASET_CARD.md), 
[model card](MODEL_CARD.md), and course milestones. Requirements marked *Proposed* and the open 
questions below have not been agreed by the team.

Each requirement names its source:

- **Agreed (#5):** the evaluation contract [agreed in issue #5](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/5#issuecomment-5972245779),
  recorded in the [problem definition](problem-definition.md#evaluation-contract-for-m2).
- **Documented:** stated in an existing project document: the
  [dataset card](DATASET_CARD.md), [model card](MODEL_CARD.md), or
  [CONTRIBUTING](../CONTRIBUTING.md).
- **Course Mx:** required by a course milestone.
- **Proposed:** a suggestion that needs a team decision.

## Scope

The system builds, evaluates, serves, and monitors one ML component: given past
half-hourly demand for one of five Australian states, it forecasts the next 48
steps. The immediate users are the team and course reviewers. It is not intended
for operational grid scheduling ([model card](MODEL_CARD.md#model-details-and-intended-use)).

## Functional requirements

### Data

- **FR-1 Pinned data.** Retrieve the Monash archive at revision `58aafbe` and
  reject it if its SHA-256 does not match the recorded value.
  *Source:* Documented (dataset card). *Verify:* test with a corrupted file; `dvc pull`.
- **FR-2 Parsing.** Parse the TSF file into five state series, mapping source
  code `QUN` to `QLD`.
  *Source:* Documented (dataset card). *Verify:* unit test.
- **FR-3 Splits.** Cut each state to the first 230,736 positions, split
  train/validation/test at 195,696 and 213,216, and create a forecast origin
  every 48 steps.
  *Source:* Agreed (#5); implemented in [`src/splits.py`](../src/splits.py) and
  [`params.yaml`](../params.yaml). *Verify:* [`tests/test_splits.py`](../tests/test_splits.py).
- **FR-4 Data validation.** Check lengths and finite values; preserve
  Tasmania's negative values without clipping.
  *Source:* Documented (dataset card); Course M3. *Verify:* data tests.

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
  *Source:* Agreed (#5). *Verify:* code review; logged parameters.
- **FR-8 Evaluation.** Evaluate all methods on the same origins: MAE per state,
  equally weighted mean MASE (lag-336 scale from the training block), holiday
  errors, and the share of error in the worst 5% of origins.
  *Source:* Agreed (#5). *Verify:* evaluation report.
- **FR-9 Run tracking.** Log each run with Git commit, data version,
  parameters, seed, metrics, and artifacts.
  *Source:* Documented (CONTRIBUTING); Course M2. *Verify:* MLflow run record.
- **FR-10 Model versioning.** Version the trained model and promote it only
  when it meets the decision rule.
  *Source:* Agreed (#5) decision rule; Course M2/M5. *Verify:* model registry or DVC record.

### Serving

- **FR-11 Forecast endpoint.** Accept a state and its 512 most recent values;
  return 48 forecasts with prediction intervals.
  *Source:* Proposed; Course M4. *Verify:* endpoint tests.
- **FR-12 Input validation.** Reject invalid input (wrong length, non-numeric
  values, unknown state) with a clear error.
  *Source:* Proposed; Course M4. *Verify:* endpoint tests.
- **FR-13 Versioning and health.** Include the model version in each response;
  provide a health-check endpoint.
  *Source:* Proposed. *Verify:* endpoint tests.

### Monitoring

- **FR-14 Prediction logging.** Log requests and predictions, and compute
  forecast error once the observed values arrive.
  *Source:* Course M6. *Verify:* dashboard or report.
- **FR-15 Drift detection.** Detect drift in the input distribution relative to
  the training data.
  *Source:* Course M6. *Verify:* simulated drift test.

## Non-functional requirements

- **NFR-1 Reproducibility.** A fresh clone at a given commit rebuilds the same
  splits, and a rerun with the same seed reproduces the reported metrics.
  *Source:* Agreed (#5); Course M2.
- **NFR-2 Traceability.** Every reported result links code commit, data
  version, checkpoint, configuration, and run ID.
  *Source:* Agreed (#5).
- **NFR-3 No leakage.** Test targets are never used for training, preprocessing
  fits, or model selection; no forecast context includes values at or after its
  origin.
  *Source:* Agreed (#5).
- **NFR-4 Portability.** Setup works on Windows and Linux with uv and Python
  3.11 ([setup guide](setup.md)), and later in a container.
  *Source:* Documented; Course M5.
- **NFR-5 Code quality.** `ruff` and `pytest` pass on every PR, enforced by CI
  once it exists.
  *Source:* Course M3/M5.
- **NFR-6 Energy.** Training and inference energy are measured with CodeCarbon,
  with measurement limitations stated.
  *Source:* Course M3.
- **NFR-7 Latency.** A 48-step forecast for one state returns within *X*
  seconds (95th percentile) on the target hardware.
  *Source:* Proposed; *X* to be set after a measurement.
- **NFR-8 Compute budget.** Fine-tuning fits the team's available hardware
  within *Y* GPU or CPU hours.
  *Source:* Proposed; *Y* not decided.
- **NFR-9 Security.** No credentials in the repository. Whether the API needs
  authentication is a team decision.
  *Source:* Documented (CONTRIBUTING); Proposed.
- **NFR-10 Transparency.** Outputs and documentation state the limitations:
  unverified unit, historical data, not for operational use.
  *Source:* Documented (cards).
- **NFR-11 Licensing.** Attribute the data (CC BY 4.0) and the base model
  (Apache 2.0).
  *Source:* Documented (cards).
- **NFR-12 Availability.** No uptime requirement; the service is a course
  demonstration.
  *Source:* Proposed.

## Open questions

1. **Holiday errors and timestamps.** FR-8 requires errors on public holidays,
   but observation timestamps and time zone are unverified
   ([dataset card](DATASET_CARD.md#considerations-for-using-the-data)). Holidays
   cannot be identified until the calendar is verified in
   [issue #6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6);
   otherwise this metric needs to be dropped or deferred.
2. **API input.** Should clients send the raw 512-value history (FR-11), or a
   state and date, with the system looking up the history?
3. **Prediction intervals.** Which quantiles (for example 10–90%), and is there
   a coverage target?
4. **Latency and compute targets.** NFR-7 and NFR-8 need a measurement on the
   team's hardware before targets are set.
5. **Target unit.** API responses need a documented unit, which is still
   unverified.
6. **Success criteria.** For the M1 practice "Problem selection and
   requirements engineering", the [course evidence register](rubric-evidence.md)
   still lists an agreed success criterion as needed. The decision rule in #5
   defines the comparison, but not when the fine-tuned model is good enough to
   deploy: must it beat both comparators, and by a minimum margin?
