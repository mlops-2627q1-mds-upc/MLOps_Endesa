# MLOps_Endesa: GitHub Project Setup Record

**Verified on 2026-09-22 using GitHub's API as `didicayu`.** This record describes
actual configuration separately from the [working agreement](../CONTRIBUTING.md).

## Created and verified

- Organization: **mlops-2627q1-mds-upc**.
- Project: [MLOps_Endesa, number 3](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3).
- Target repository: [MLOps_Endesa](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa).
- Status options: Backlog, Ready, In progress, In review, Done.
- Priority options: P0, P1, P2, P3, with their agreed descriptions.
- Size options: S, M, L, XL, with their agreed effort ranges.
- Weekly Iteration field, beginning 2026-09-22 and covering the remaining course
  period through the week of 2026-12-15.
- Current week board, filtered by `iteration:@current` and grouped into Status
  columns; Backlog table sorted by ascending Priority; Milestones table grouped
  by the native Milestone field and sorted by ascending Priority.
- [Issue #1: Document the team working agreement and engineering decision](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/1),
  present as a Project item with Priority **P1**, Size **M**, and Iteration
  **Week of 2026-09-22**. Its card records the current review status.
- [Issue #2: Complete repository access and workflow enforcement](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2),
  present as **Backlog**, **P1**, **S**, awaiting access and accepted ownership.
- Project README with the workflow, repository URL, issue link, and setup status.

The documentation issue was opened after the initial local drafting. Its body
records that sequence, names Dídac as accountable owner, and distinguishes
validation, publication, peer review, and merging.

## Access limitations

GitHub reports `didicayu` has **READ** permission on the repository. The account
can create and edit this organization Project and create a repository issue, but
the following operations were denied by GitHub:

| Operation | Verified result |
| --- | --- |
| Native link between Project and repository | `LinkProjectV2ToRepository` denied; the Project's linked repository list remains empty. |
| Change Project visibility to public | "Viewer not authorized to change project visibility"; the Project remains private. |
| Assign issue #1 to `didicayu` | `ReplaceActorsForAssignable` denied; native Assignees remains empty. |

Adding the repository's issue to the Project succeeded. That association and the
repository URL in the Project README do not establish the native link shown in a
repository's Projects tab.

## Remaining setup

A repository maintainer must complete the native Project link and issue
assignment, or arrange the necessary access for Dídac. A Project administrator
must verify team/lecturer access and choose the appropriate visibility.

Repository milestones M1-M6 and the agreed custom labels have not been created.
The native Milestone field is present but empty; issue #1 records M1 in its body.
Both table views now have their agreed sorting/grouping. Six default Project
workflows report enabled: Auto-add sub-issues to project, Auto-close issue,
Item added to project, Item closed, Pull request linked to issue, and Pull request
merged. Their trigger/action details and behavior have not been verified. Inspect
them before relying on automatic transitions; in particular, closing an issue as
Not planned must not count as completed work.

Issue #2 holds the acceptance checklist for the outstanding configuration. Branch
protection is not verified; CI checks must be implemented and successfully run
before registering them as required checks. Project setup does not establish
that either is enforced.

## Documentation publication and review

Dídac authorized pushing the documentation and opening a PR. With READ access to
the organization repository, the source branch is `docs/team-working-agreement`
in [Dídac's fork](https://github.com/didicayu/MLOps_Endesa). The PR targets the
organization repository's `main`; publication and review evidence are tracked in
issue #1. Actual peer approval and merge are required before that task is Done.
The separate access/configuration task remains open until its checks are met.
