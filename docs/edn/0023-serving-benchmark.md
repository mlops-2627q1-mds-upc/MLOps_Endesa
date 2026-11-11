# EDN-023: Measure serving with an isolated, supervised benchmark

Date: 2026-11-11 (recorded).

Milestone: M4

Activity / Topic: Serving performance (#76)

Decision Participants: Dídac Cayuela

Status: Recorded; peer review pending.

## Decision

I chose a separate container lifecycle for cold-start and warm-request measurements,
with fixed inputs/settings and a separate confirmation run. A successful supervisor
exit, verified cleanup and matching report hashes are required before using a result.

## Rationale

Timing an already running endpoint cannot measure startup. Isolation protects the main
API, while supervision prevents a complete-looking report from hiding a failed worker or
cleanup. Memory readings use the actual container scope.

## AI Involvement

Information seeking; Alternative generation; Alternative assessment; Recommendation;
Solution generation; Other. I used Codex 6.1 Sol Max to compare measurement options,
develop the protocol details and implement and test the approach I chose.

## Response to AI

Used as input for further analysis.

## Assessment of the AI Contribution

I chose a separate lifecycle to measure startup and protect the main API. Agent review
found two supervision defects that were corrected. I kept the measured comparison
separate from adopting a performance requirement.

## Other Evidence

[PR #93](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/93). All 94 requests
passed, but both isolated stops exited 137; graceful shutdown remains unverified.
