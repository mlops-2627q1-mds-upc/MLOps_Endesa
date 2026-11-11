# EDN-014: Keep replay and monitoring outside the API

Date: 2026-11-11 (recorded).

Milestone: M4

Activity / Topic: Component responsibilities (#74)

Decision Participants: Dídac Cayuela

Status: Draft for wording review before push.

## Decision

I separated stateless forecasting from historical replay and model monitoring. The API
owns model loading, validation and inference; an external journal owns replay progress
and delayed observations.

## Rationale

Replay needs persistent state, while each forecast request only needs its input history.
Running joins and drift checks inside requests would add dependencies and work to
inference. I kept one model process with two CPU threads to limit duplicated memory and
compute.

## AI Involvement

Information seeking; Alternative generation; Alternative assessment; Recommendation;
Solution generation. I used Codex to compare component arrangements and prepare the
architecture description for the boundaries I selected.

## Response to AI

Used as input for further analysis.

## Assessment of the AI Contribution

The comparison helped me assess the cost of putting replay and monitoring inside
requests. I chose to keep them separate; public access, extra admission controls and
rollback remain follow-up proposals.

## Other Evidence

[Issue #74](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/74), [API PR
#86](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/86) and [replay PR
#89](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/89).
