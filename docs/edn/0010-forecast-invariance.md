# EDN-010: Paired scale and perturbation checks

- **Date:** 2026-10-14.
- **Milestone:** M3.
- **Decision Participants:** Dídac Cayuela (decision owner); Pau Adal (protocol and implementation review).
- **Status:** Base-checkpoint protocol and implementation approved in PR #59; local-checkpoint extension awaits current-version review.

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

The proposals supported Dídac's decision. In his
[2026-10-15 review](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/59#pullrequestreview-5534033689),
Pau reported 117 passing routine tests on Windows, with the 70 real-model cases
skipped, and accepted the design and thresholds subject to numbering corrections.
That review does not add an independent real-model run.

## Other Evidence

- [Issue #55](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/55).
- [Case construction and comparisons](../../src/invariance.py).
- [Runner](../../src/modeling/invariance.py) and [tests](../../tests/test_invariance.py).
- [Experiment record](../experiments/006-forecast-invariance.md).

On 2026-10-17, Pau Adal approved `c8acaa9` in [PR #59](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/59).
