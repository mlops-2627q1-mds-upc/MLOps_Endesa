# EDN-019: Use SQLite for restartable historical replay

Date: 2026-11-11 (recorded).

Milestone: M6

Activity / Topic: Replay and delayed observations (#81)

Decision Participants: Dídac Cayuela

Status: Draft for wording review before push.

## Decision

I chose one SQLite journal per replay run, with immutable source/model identity,
transactional progress and a retry budget that survives restarts. Replay releases
observations to monitoring according to its positional clock.

## Rationale

CSV would need separate cursor, deduplication and recovery mechanisms; a remote event
service would add another deployment dependency. SQLite gives readers a consistent
snapshot while the writer commits. Unique observations prevent overlapping forecasts
from inflating drift samples.

## AI Involvement

Solution generation; Other. Codex developed the transaction and recovery details for the
SQLite approach I chose, then implemented and tested them. Agent review found and
corrected an interrupted-initialization defect.

## Response to AI

Other. I used implementation and recovery-test support.

## Assessment of the AI Contribution

I kept the transactional design because a lost HTTP response can repeat remote work but
must not duplicate local forecasts or reset the retry budget. The interruption tests
exposed a recovery case that the first implementation missed.

## Other Evidence

[PR #89](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/89). The retained
selected-model replay supports the downstream quality checks; it does not establish
monitoring deployment.
