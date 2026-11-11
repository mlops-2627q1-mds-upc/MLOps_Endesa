# EDN-013: Verified checkpoint loading

Date: 2026-11-11 (recorded).

Milestone: M4

Activity / Topic: Model loading (#48)

Decision Participants: Dídac Cayuela

Status: Draft for wording review before push.

## Decision

I chose to verify the selected checkpoint against its tracked provenance before loading
it. Missing or mismatched files stop startup. The pinned base model remains an explicit
comparison option.

## Rationale

A directory move should not change model selection, and an automatic fallback could hide
a broken deployment. I reused the checkpoint helper instead of adding another validation
path. Prediction keeps the evaluation settings, with serialized sampling and restored
CPU RNG state.

## AI Involvement

Solution generation; Other. Codex implemented the loading checks, sampling lock and
tests, and ran the selected-checkpoint smoke check.

## Response to AI

Other. I used implementation and validation support.

## Assessment of the AI Contribution

I used Codex to implement the choice consistently with evaluation. The relocation and
RNG checks support that contract; model accuracy and interval coverage need separate
evidence.

## Other Evidence

[PR #52](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/52).
