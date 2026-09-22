# GitHub setup status

Checked through GitHub's API on **2026-09-22**, using `didicayu`.
The [working agreement](../CONTRIBUTING.md) describes the intended workflow.

## Project

The organization [Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3)
has the five agreed statuses, Priority P0-P3, Size S/M/L/XL, and weekly iterations
from 2026-09-22 through the week of 2026-12-15.

| View | Configuration |
| --- | --- |
| Current week | Board with Status columns, filtered by `iteration:@current`. |
| Backlog | Open tasks sorted by Priority. |
| Milestones | Table grouped by Milestone and sorted by Priority. |

[Issue #1](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/1) is
**In review**, P1, M, in the first iteration, and links to
[PR #3](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/3).
[Issue #2](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2) is in
**Backlog**, P1, S, and holds the remaining setup checklist.

## Settings to complete

| Setting | Current state / next action |
| --- | --- |
| Repository access | `didicayu` has READ access. A maintainer needs to arrange team access and issue assignments; issue #1 names Dídac as owner in its body. |
| Repository link | The Project contains repository issues, but its native repository link is still unset. A maintainer must complete it. |
| Project visibility | Private. A Project administrator must confirm that teammates and the lecturer can view it. |
| Labels and milestones | Custom labels and repository milestones M1-M6 still need creation. |
| Main branch protection | Disabled; no protection rule or active ruleset applies to `main`. Require one peer approval, resolved conversations, and no force pushes or deletion. |
| Required CI checks | No CI jobs yet. Register required checks once the jobs exist and have run successfully. |
| Project automation | Six default workflows are enabled; their transitions still need checking against the working agreement, including Not planned closures. |

The access checks denied native repository linking, changing Project visibility,
and issue assignment. Issue #2 identifies the maintainer/administrator actions
needed to complete these settings.

The documentation branch is published from Dídac's fork because of READ access
to the organization repository. PR #3 targets the organization's `main` and
awaits peer review and merge.
