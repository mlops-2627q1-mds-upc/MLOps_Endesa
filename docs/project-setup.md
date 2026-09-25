# GitHub setup status

Settings verified through GitHub's API on **2026-09-22**, using `didicayu`.
The [working agreement](../CONTRIBUTING.md) explains the workflow and
[reasons for each protection rule](../CONTRIBUTING.md#repository-enforcement).

## Repository settings

| Setting | Verified state |
| --- | --- |
| Main branch protection | Enabled for `main`; one approving peer review and resolved review conversations required. |
| Updated changes | New reviewable commits dismiss stale approvals. |
| Administrator enforcement | Enabled; no review bypass allowances configured. |
| History protection | Force pushes and deletion of `main` blocked. |
| Merge methods | Merge commits enabled; squash/rebase merging disabled; linear history not required. |
| Branch cleanup | Automatic deletion of merged branches enabled for this repository. |
| Required CI checks | None yet. Add working checks after CI jobs are introduced and run successfully. |
| Labels and milestones | All 12 agreed labels and milestones M1-M6 created. Milestone due dates are unset. |
| Existing work | Issues #1 and #2 and PR #4 assigned to `didicayu`, with M1 and the relevant type/area labels. |

Dídac, Pablo, and Sindri have repository Admin access. Write invitations have been
sent to Pau (`NIU1638529`) and Antoni (`toni646`); their access remains Read until
accepted. Existing teammate roles were preserved.

## Project

The [organization Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3)
is natively linked to the repository. It has Backlog, Ready, In progress, In review,
and Done; Priority P0-P3; Size S/M/L/XL; and weekly iterations from 2026-09-22
through the week of 2026-12-15.

| View | Configuration |
| --- | --- |
| Current week | Status board filtered by `iteration:@current`. |
| Backlog | Open tasks sorted by Priority. |
| Milestones | Table grouped by Milestone and sorted by Priority. |

All six default Project workflows were removed. Issue creation, card transitions,
and completion checks are manual; PR closing keywords still close linked issues
when merged. This avoids treating draft links or Not planned closures as Done.

[Documentation issue #1](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/1)
is **In review**, P1, M, and links to
[PR #4](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/4).
[Setup issue #2](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2) is
**In progress**, P1, S, with `blocked` for the remaining access dependencies.
Both are assigned to the current iteration, Week of 2026-09-22.

## Remaining access and review

The Project remains private. Repository Admin access did not grant permission to
change Project visibility or collaborators; a Project administrator must confirm
team/lecturer access and arrange editing access for the contributors. Pau and
Antoni also need to accept their repository invitations. Issue #2 tracks these
remaining steps.

Documentation is on `docs/1-team-working-agreement` in the organization repository.
PR #4 targets `main` and awaits peer approval and merge.
