# EDN-009: Known continuations and declared model-test limits

- **Date:** 2026-10-14
- **Milestone:** M3 - Quality assurance.
- **Activity / Topic:** Minimum functionality tests for issue #54.
- **Decision Participants:** Dídac Cayuela (task owner); Sindri Másson and Pau Adal (protocol and implementation review).
- **Status:** Base-checkpoint protocol and implementation approved in PR #57; local-checkpoint extension awaits current-version review.

## Decision

Test constant demand, zero demand and a repeated 48-step sine pattern using the
agreed 512-value context and 48-step horizon. Use three demand levels and two
relative pattern amplitudes. Require both a mean and a maximum absolute-error
limit, with absolute limits for the zero case. Declare the numeric limits in
[the protocol](../minimum-functionality.md) before inference and preserve all
outcomes, including failures, when running the pinned real Chronos checkpoint.

## Rationale

A zero seasonal scale makes MASE undefined on these cases. Percentage errors
are undefined at zero. Mean absolute error gives an interpretable overall error,
while a maximum-error limit detects isolated bad steps. Pattern limits depend
on amplitude rather than the demand level: accepting a flat forecast merely
because the level is large would miss the behaviour being checked.

Exact equality would conflate token quantization with failure to continue a
series. The declared zero limits account for the checkpoint's unit-scale bin
spacing. Conversely, increasing limits after observing results would remove
the checks' value. Failed cases should remain visible and inform later model
assessment. The limits are proposed synthetic regression criteria; they do not
represent an agreed operational accuracy requirement.

Reuse the existing evaluator and keep inference separate from fast default
tests. A bounded process loads the checkpoint once, uses at most two CPU threads,
and records the full case set before assertions. This avoids a new inference
implementation and does not depend on the pending PR #52.

## AI Involvement

Recommendation; Solution generation. Codex proposed the metrics and limits and
implemented the cases, test harness, bounded reproduction and evidence records.

## Response to AI

Dídac requested implementation and a reviewable PR with maintainable code and
clear writing. Sindri and Pau subsequently approved the original protocol and
implementation in PR #57; this records their reviews rather than team consensus.

## Assessment of the AI Contribution

In the [2026-10-15 reviews](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/57),
Sindri independently reproduced all ten model cases on Windows and checked that
the pattern limits reject a flat forecast. Pau checked the protocol hash, metric
choice and zero-case limits. Sindri suggested clarifying that these are minimum
functionality checks, with the declared limits unchanged.

## Other Evidence

- [Issue #54](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/54).
- [Protocol and reproduction](../minimum-functionality.md).
- [Initial real-model results](../experiments/005-minimum-functionality.md) and [complete report](../evidence/issue-54-minimum-functionality.json).
- [Declared cases and bounded runner](../../src/modeling/minimum_functionality.py).
- [Real-model tests](../../tests/test_minimum_functionality_model.py) and [fast harness tests](../../tests/test_minimum_functionality.py).

On 2026-10-17, Pau Adal approved `32bdbe1` in [PR #57](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/57).
