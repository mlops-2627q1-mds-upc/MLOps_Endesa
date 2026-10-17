# EDN-006: Require CI against the current main branch

- **Date:** 2026-10-11
- **Milestone:** M5 - Model packaging
- **Decision Participants:** Dídac Cayuela (task owner for required CI checks); Pablo Pérez Cano (configuration and documentation review on 2026-10-16).
- **Status:** Required checks verified by Pablo; documentation approved and merged in PR #43 on 2026-10-17.

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

Dídac directed [#20](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/20)
to require the proven CI jobs, verify failure and recovery, and preserve the
existing peer-review rules. The task waited for the repaired `main` checks to
pass before enforcement. Codex proposed the strict up-to-date option and
implemented the gate; Pablo checked the resulting configuration and requested
documentation corrections.

## Assessment of the AI Contribution

In his [2026-10-16 review comment](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/43#issuecomment-6213376233),
Pablo described the failure/recovery verification as thorough and independently
confirmed the required check source, strict mode, peer approval and administrator
enforcement. He requested clearer gate scope and participant records before
approval. That review provides the recorded assessment of the configured gate.

## Other Evidence

- [Issue #20: enforcement scope and readiness check](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/20#issuecomment-6013452979).
- [Verified configuration and failure/recovery results](../project-setup.md#required-ci-checks).
- [Temporary verification PR #42](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/42).
- [Working agreement](../../CONTRIBUTING.md#repository-enforcement) and
  [original workflow decision](0001-team-workflow.md).
- [GitHub's strict and loose status-check semantics](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging).

On 2026-10-17, Pablo Pérez Cano approved `223b469` in [PR #43](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/43).
