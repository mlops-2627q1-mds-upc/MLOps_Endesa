# EDN-011: Directional expectations for sustained demand changes

- **Date:** 2026-10-14.
- **Milestone:** M3 - Quality assurance.
- **Activity / Topic:** Controlled model expectations for #56.
- **Decision Participants:** Dídac Cayuela (task scope and implementation request); Pau Adal (protocol and implementation review).
- **Status:** Base-checkpoint protocol and implementation approved in PR #58; local-checkpoint extension awaits current-version review.

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
The [experiment protocol](../experiments/007-directional-expectations.md) states
the formula, cases, unchanged control and sampling settings.

## AI Involvement

Alternative generation; Recommendation; Solution generation. Codex proposed the
controlled cases and comparison rules, implemented tests using the existing
evaluator, executed the bounded checks and drafted the evidence record.

## Response to AI

Dídac requested implementation of #56 and professional, maintainable code under
the project instructions. Pau subsequently reviewed the proposed rules and
implementation and requested numbering corrections before approval.

## Assessment of the AI Contribution

In his [2026-10-15 review](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/58#pullrequestreview-5534363238),
Pau reported 107 passing routine tests on Windows, with 13 real-model cases
skipped, and found the paired cases, horizon-mean response and unchanged control
appropriate. He requested documentation renumbering before approval; the review
does not add an independent real-model run.

## Other Evidence

- [Issue #56](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/56).
- [Experiment 007 and reproduction](../experiments/007-directional-expectations.md).
- [Complete real-checkpoint report](../evidence/issue-56-directional-expectations.json).
- [Directional runner](../../src/modeling/directional.py) and
  [tests](../../tests/test_directional.py).

On 2026-10-17, Pau Adal approved `43d6851` in [PR #58](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/58).
