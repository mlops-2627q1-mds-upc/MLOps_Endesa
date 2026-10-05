# EDN-003: Bounded validation comparison before project fine-tuning

- **Date:** 2026-10-05.
- **Milestone:** M2 - Reproducibility.
- **Activity / Topic:** Real experiment records for #12.
- **Decision Participants:** Dídac Cayuela.
- **Status:** Implementation authorized by Dídac; implementation review and result assessment pending.

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
the initial inference settings, and implemented the evaluator, tests and records.

## Response to AI

Accepted to proceed with #12 after Dídac reported his approval of #17. This is
authorization for the task. The CPU/batch/budget choices are the agent's initial
implementation choices, not a separately recorded team agreement.

## Assessment of the AI Contribution

Human assessment of the code, result interpretation and reproduction is pending.
No peer reproduction or forecast improvement is inferred from authorization.

## Other Evidence

- [Issue #12](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/12).
- [Reviewed split PR #27](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/27).
- [Evaluation commands and limitations](../evaluation.md).
- Source: `src/evaluation.py`, `src/modeling/evaluate.py`; tests: `tests/test_evaluation.py`.
