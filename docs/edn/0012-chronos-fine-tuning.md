# EDN-012: Bounded Chronos fine-tuning and its selection

- **Date:** 2026-10-12.
- **Milestone:** M2; planned in #13 as a model-building experiment.
- **Activity / Topic:** Fine-tuning experiment for #13.
- **Decision Participants:** Sindri Másson; Pablo Pérez Cano proposed the options and reviewed the validation results.
- **Status:** Validation results and test path reviewed by Pablo; test run completed and meets the decision rule. Implementation, recorded results and provenance repair were included in approved and merged [PR #69](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/69) on 2026-10-17.

## Decision

Fine-tune the pinned Chronos-T5-small checkpoint on windows from the training block
only, with the five options Pablo proposed in #13, each changing one setting of the
Chronos paper's fine-tuning schedule, and three seeds per option. Train and score every
option on the GPU, including an untrained control saved through the same checkpoint
path. Select on validation: option 1 with seed 8, its best run. Do not run the AdaLoRA
round, since LoRA was not competitive. Score the test block once, with a decision rule
fixed beforehand in Experiment 004.

## Rationale

A validation pass takes about 34 minutes on the CPU and 4.3 minutes on the GPU, so
scoring about 16 runs on the CPU would have taken 8 to 9 hours. Scoring the control on
the GPU as well keeps every option under the same conditions; its difference from the
untuned CPU result on the same machine (0.0015 in mean MASE) is reported. A batch of 32
does not fit the RTX 4060's 8 GB, so it is trained as two accumulated micro-batches of
16, which gives an effective batch size of 32.

Option 1 scored lowest on validation (mean MASE 0.570 over three seeds, against 0.874
for the control). Seed 8 is its best validation run. Both choices were fixed using
validation, before evaluating test targets.
The decision rule requires a test margin larger than option 1's observed validation
seed spread (0.0091). This is an engineering margin, not a confidence bound.

## AI Involvement

Information seeking; Recommendation; Solution generation.
Claude Code (Anthropic) implemented the training, the round and the evaluator's
checkpoint, device and test-block options, with their tests. It diagnosed the 8 GB
limit and the cached GPU memory that slowed the round's evaluations and drafted
Experiment 004 and this entry.
On 2026-10-17, Codex integrated the branches, repaired checkpoint lineage and DVC
discovery, and extended the existing behaviour runners to local checkpoints, with
regression tests. It executed the fixed selected-checkpoint checks, independently
recalculated their recorded metrics and prepared
[Experiment 008](../experiments/008-selected-checkpoint-behaviour.md), retaining
the interrupted resource adjustment. Dídac requested the submission fixes.

## Response to AI

Accepted with modifications. Claude Code first recommended scoring seed 7 on test,
because it was fixed before any result; Sindri rejected this, since seed 7 was as
arbitrary as any other, and selected seed 8 on validation.

## Assessment of the AI Contribution

Pablo read the 18 validation runs back from MLflow and confirmed the table, the clean
commits, the shared target hash, the link from each evaluation to its training run, and
that training reads only the training block. Pablo agreed with selecting option 1, with
skipping AdaLoRA, and with scoring seed 8 once on test, and approved the test path before
the test run. On test, option 1 with seed 8 scored a mean MASE of 0.5571, against 0.8175
for the control and 0.9885 for lag 48, which meets the decision rule.

The test results, provenance repair and selected-checkpoint checks were included
in [PR #69](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/69), approved by
Toni and merged on 2026-10-17. His
[review](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/69#pullrequestreview-5559556788)
reports 235 passing routine tests, clean Ruff, Pylint 9.63 and 83% coverage.
The recorded model results retain their original executions in Experiments 004
and 008.

## Other Evidence

- [Issue #13 and Pablo's proposal and review](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/13).
- [Experiment 004: results, costs, retained attempts and the test plan](../experiments/004-chronos-fine-tuning.md).
- [Selected-checkpoint behaviour: actual outcomes and retained interruption](../experiments/008-selected-checkpoint-behaviour.md).
- [Fine-tuning commands](../evaluation.md#fine-tuning).
- Source: `src/modeling/train.py`, `src/modeling/fine_tune_round.py`, `src/modeling/evaluate.py`, `src/evaluation.py`; tests: `tests/test_train.py`, `tests/test_fine_tune_round.py`, `tests/test_evaluation.py`.
