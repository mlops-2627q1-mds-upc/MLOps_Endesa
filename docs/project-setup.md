# GitHub setup status

Repository settings and Project configuration checked through GitHub's API on
**2026-10-11**, using `didicayu`.
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
| Required CI checks | `CI / lint` and `CI / test` required from GitHub Actions (app ID 15368); branches must be up to date. Failure/recovery evidence and reviewed documentation are tracked in [#20](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/20). |
| Milestones | M1-M6 exist; milestone due dates are unset. |

Repository and Project access evidence is maintained in
[issue #2](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2).

## Project

The [organization Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3)
is natively linked to the repository. It has Backlog, Ready, In progress, In review,
and Done; Priority P0-P3; Size S/M/L/XL; and weekly iterations from 2026-09-22
through the week of 2026-12-15.

| View | Configuration |
| --- | --- |
| Current week | Status board; currently no saved filter. |
| Backlog | Open tasks sorted by Priority. |
| Milestones | Table grouped by Milestone and sorted by Priority. |

No Project workflows are configured. Issue creation, card transitions,
and completion checks are manual; PR closing keywords still close linked issues
when merged. This avoids treating draft links or Not planned closures as Done.

The Project is private. Repository permissions alone do not verify contributor
editing access or lecturer viewing access to it; confirmation remains tracked
in #2. The working agreement and EDN-001 were approved by Toni and merged in
[PR #4](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/4) on 2026-09-27.
