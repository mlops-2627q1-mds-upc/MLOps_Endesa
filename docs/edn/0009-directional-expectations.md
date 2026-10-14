# EDN-009: Directional expectations for sustained demand changes

- **Date:** 2026-10-14.
- **Milestone:** M3 - Quality assurance.
- **Activity / Topic:** Controlled model expectations for #56.
- **Decision Participants:** Dídac Cayuela (task scope and implementation request).
- **Status:** Proposed; numerical hypotheses and teammate review pending.

## Decision

For #56, propose paired synthetic histories that add or subtract a constant level
over one, two or four complete daily cycles while retaining the daily pattern.
Compare the average of the 48-step median forecast with a minimum signed response
of 10% of the input offset. An independently forecast unchanged history must
match the reference within 0.01 synthetic units at every step. Declare these rules
before inference, use matched seeds and sampling, and retain all counterexamples.

## Rationale

Issue #56 calls for justified directional expectations without assuming that
Chronos always responds monotonically. A full-cycle level shift supports a
specific persistence hypothesis. An isolated spike or arbitrary trend would make
the expected response harder to justify. A horizon summary allows individual
steps to differ, while the effect margin prevents unchanged forecasts from
passing. The margin is proposed for review, not calibrated against observed
results. A single checkpoint load and a 180-second process limit bound the run.
The [experiment protocol](../experiments/005-directional-expectations.md) states
the formula, cases, unchanged control and sampling settings.

## AI Involvement

Alternative generation; Recommendation; Solution generation. Codex proposed the
controlled cases and comparison rules, implemented tests using the existing
evaluator, and prepared the bounded run and evidence record.

## Response to AI

Dídac requested implementation of #56 and professional, maintainable code under
the project instructions. The proposed numerical rules and findings remain for
human review; no acceptance or team agreement is inferred from that request.

## Assessment of the AI Contribution

Human assessment is pending. The experiment separates implementation checks from
real-checkpoint evidence so a teammate can assess the hypotheses and limitations.

## Other Evidence

- [Issue #56](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/56).
- [Experiment 005 and reproduction](../experiments/005-directional-expectations.md).
- [Directional runner](../../src/modeling/directional.py) and
  [tests](../../tests/test_directional.py).
