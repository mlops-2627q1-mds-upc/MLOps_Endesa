# EDN-017: Separate release validation from fast PR checks

Date: 2026-11-11 (recorded).

Milestone: M5

Activity / Topic: Release-validation strategy (#77)

Decision Participants: Dídac Cayuela

Status: Recorded; peer review pending.

## Decision

I chose a manual release command that retrieves and verifies the selected artifacts,
then runs the established data, minimum-functionality and directional checks. Fast PR
checks stay separate.

## Rationale

A fixture smoke does not verify artifact retrieval. Running every behavioural suite on
every PR would add compute and private-data access to routine changes. I kept fixed
thresholds, two CPU threads and a 900-second command budget, with retained reports on
failure.

## AI Involvement

Information seeking; Alternative generation; Alternative assessment; Recommendation;
Solution generation; Other. Codex implemented and tested the release protocol I chose. I
used Codex 6.1 Sol Max to assess whether the cold-cache result called for code changes.

## Response to AI

Used as input for further analysis.

## Assessment of the AI Contribution

The empty-cache run verified retrieval as well as the model checks. I retained the
implementation because it passed without a reproduced defect; the follow-up needed
better evidence rather than a rewrite.

## Other Evidence

[PR #87](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/87). Local validation
passed; the release-validation Actions run and runner authentication remain unverified.
