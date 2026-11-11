# EDN-016: Separate the serving image from model provisioning

Date: 2026-11-11 (recorded).

Milestone: M5

Activity / Topic: Container packaging (#50)

Decision Participants: Dídac Cayuela

Status: Recorded; peer review pending.

## Decision

I chose a non-root CPU image with locked serving dependencies and a read-only checkpoint
mount. Build inputs use an exact-file allowlist. Fixture predictors stay in the test
harness.

## Rationale

Baking weights into each image would duplicate artifacts and tie model provisioning to
application builds. Installing the full project would add training tools to serving. The
separate serving group limits that dependency set without reorganizing the development
environment.

## AI Involvement

Solution generation; Other. Codex implemented the image, dependency group, context rules
and fixture workflow, then checked the build context and CI results.

## Response to AI

Other. I used packaging and test support.

## Assessment of the AI Contribution

I kept the production/test separation because a fixture must not hide missing real
weights. The normal entry point fails without the selected artifact; the fixture runs
only through an explicit test command.

## Other Evidence

[PR #88](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/88). The
selected-model deployment has separate evidence in the #75 record.
