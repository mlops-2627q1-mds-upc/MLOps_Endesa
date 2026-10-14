# EDN-009: Paired scale and perturbation checks

- **Date:** 2026-10-14.
- **Milestone:** M3.
- **Decision Participants:** Dídac Cayuela.
- **Status:** Selected for implementation; numerical thresholds and findings await peer review.

## Decision

Dídac chose paired scale and bounded-perturbation checks for issue #55, using
synthetic and training/validation histories. Compare every forecast step, undo
positive scale changes, and record mean and maximum differences relative to
the history's mean absolute value. Declare the thresholds before inference and
retain every outcome, including counterexamples.

## Rationale

Independent sampling could make an unchanged history appear unstable. Reuse the
merged evaluator with one checkpoint load, batch size 1, and the same seed reset
before each history. This costs more calls than batching pairs, but matches the
random stream on both sides and keeps the comparison interpretable. Comparing
only forecast averages could hide an isolated large change; both mean and
maximum step differences are therefore checked.

The scale relationship has a mathematical justification under exact normalization;
quantization and float32 arithmetic still need empirical checks. Small-noise
stability is a robustness hypothesis. The [protocol](../forecast-invariance.md)
sets initial numerical and investigation thresholds before running the model.
They are not tuned on these outcomes or the final test set.

## AI Involvement

Alternative generation; Alternative assessment; Solution generation. OpenAI
Codex helped compare implementation approaches, write the runner and tests,
and prepare the reproducible evidence.

## Response to AI

Dídac used the proposals to make the design decision and requested implementation
and a reviewable PR in the task worktree.

## Assessment of the AI Contribution

The proposals supported Dídac's decision. Peer assessment of the implementation,
thresholds and model findings is pending.

## Other Evidence

- [Issue #55](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/55).
- [Case construction and comparisons](../../src/invariance.py).
- [Runner](../../src/modeling/invariance.py) and [tests](../../tests/test_invariance.py).
- [Experiment record](../experiments/004-forecast-invariance.md).
