# EDN-006: Require CI against the current main branch

- **Date:** 2026-10-11
- **Milestone:** M5 - Model packaging
- **Decision Participants:** Dídac Cayuela (task owner; requested implementation).
- **Status:** Implemented in GitHub; documentation and teammate review pending.

## Decision

Require `CI / lint` and `CI / test` from GitHub Actions (app ID `15368`) on `main`.
Require PR branches to be up to date with `main` before merging. Extend the
existing branch protection, preserving peer approval, stale-approval dismissal,
conversation resolution, administrator enforcement, and history protection.

## Rationale

Successful CI runs did not make the jobs merge requirements. The jobs introduced
in [PR #21](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/21) are now
proven on `main`, after the repair in
[PR #25](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/25).
Binding their names to GitHub Actions prevents a status from another source from
satisfying the gate.

A loose gate would accept results from an older base branch. PR #21 passed its
PR checks but failed on the combined `main` tree after another change merged;
the [M5 evidence register](../rubric-evidence.md) links those runs. Requiring an
up-to-date branch addresses that gap at the cost of another CI run when `main`
advances. Extending the existing protection avoids maintaining overlapping
branch-protection and ruleset policies.

## AI Involvement

Information seeking; Alternative assessment; Recommendation; Solution generation.

Codex inspected the workflow, successful runs, and live GitHub settings; proposed
the strict update requirement; configured the gate; verified deliberate failures
and recovery on a temporary organization PR; and drafted the evidence and PR.

## Response to AI

Dídac requested completion of [#20](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/20)
and a review-ready PR. The implementation follows that authorization. His
assessment of the specific configuration and a teammate's review remain pending.

## Assessment of the AI Contribution

Pending human assessment. The recorded checks establish the remote configuration
and failure/recovery behavior; they do not establish team consensus or extend the
gate to training, model promotion, or deployment.

## Other Evidence

- [Verified configuration and failure/recovery results](../project-setup.md#required-ci-checks).
- [Temporary verification PR #42](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/42).
- [Working agreement](../../CONTRIBUTING.md#repository-enforcement) and
  [original workflow decision](0001-team-workflow.md).
- [GitHub's strict and loose status-check semantics](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging).
