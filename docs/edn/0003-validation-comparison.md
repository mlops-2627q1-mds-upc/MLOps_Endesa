# EDN-003: Bounded validation comparison before project fine-tuning

- **Date:** 2026-10-05.
- **Milestone:** M2 - Reproducibility.
- **Activity / Topic:** Real experiment records for #12.
- **Decision Participants:** Dídac Cayuela.
- **Status:** Sindri's assessment recorded; Windows corrections implemented; current-head peer review pending.

## Decision

Use the already agreed split, horizon and metrics from #5/#17 to compare lag-48,
lag-336 and the pinned untuned Chronos checkpoint on all validation origins.
Keep the final test block closed. Compute each MASE scale on training data only,
select the primary baseline on validation, and preserve predictions and run IDs.
Use local CPU inference with a fixed seed, 20 samples, batch size 16, four threads
and a one-hour limit. Keep model dependencies optional so ordinary CI stays small.

## Rationale

The split implementation is reviewed and merged, so actual comparable runs can
replace synthetic tracking evidence. Full validation covers the agreed origins;
a small sample would not satisfy that comparison. Local inference avoids paid
compute and a training run. A recorded budget prevents an open-ended experiment.
The source calendar is unverified, so holiday breakdowns remain explicitly
unavailable while positional error metrics and worst-origin shares can be measured.

## AI Involvement

Information seeking; Alternative assessment; Recommendation; Solution generation.
OpenAI Codex checked the merged split and upstream Chronos disclosure, selected
the initial inference settings, implemented the evaluator and tests, executed
the comparison, and verified the shared predictions, scores and version records.
On 2026-10-08, Codex implemented Sindri's Windows line-ending and target-hash
corrections, added regression checks and updated the reproduction documentation.

## Response to AI

Accepted to proceed with #12 after Dídac reported his approval of #17. This is
authorization for the task. The CPU/batch/budget choices are the agent's initial
implementation choices, not a separately recorded team agreement.
Dídac authorized addressing Sindri's Windows findings on 2026-10-08.

## Assessment of the AI Contribution

In his [2026-10-06 review](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/28#pullrequestreview-5426279134)
of `eff29e2`, Sindri found the generated code consistent with the agreed protocol,
with no leakage or indexing errors. He identified missing Windows line-ending
handling in Git checkout and target hashing.

Sindri reproduced lag-336 on Windows in shared run
[`a5102636b26e47f1be7110e4a5adbd7a`](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/a5102636b26e47f1be7110e4a5adbd7a),
matching the recorded errors and 87,600 predictions after forcing LF checkout.
He agreed with the scoped validation interpretation: untuned Chronos scored below
both seasonal baselines. He did not rerun Chronos and retained the single-seed,
validation-only and upstream pretraining-disclosure limitations.

The 2026-10-08 corrections enforce LF checkout and CSV hashing. Regression checks
simulate Windows settings; a native Windows recheck of this revision and
current-head peer approval remain pending.

## Other Evidence

- [Issue #12](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/12).
- [Reviewed split PR #27](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/27).
- [Evaluation commands and limitations](../evaluation.md).
- [Completed comparison and reproduction target](../experiments/001-baseline-chronos.md).
- Source: `src/evaluation.py`, `src/modeling/evaluate.py`; tests: `tests/test_evaluation.py`.
