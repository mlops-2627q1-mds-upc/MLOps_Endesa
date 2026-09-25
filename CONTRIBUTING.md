# Working agreement

All five team members agreed to this workflow for planning, branches, and reviews.
The reasoning is recorded in [EDN-001](docs/edn/0001-team-workflow.md); current
GitHub settings and remaining setup are in the [setup record](docs/project-setup.md).

## Planning

Track work in the [MLOps_Endesa Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3).
Each task has one issue and one accountable owner. Link its implementation PR to
the issue instead of adding a second task card. Other contributors can help;
record their contributions in the PR or commits.

Assign each issue to the relevant course milestone:

| Milestone | Scope |
| --- | --- |
| M1 - Inception | Requirements, model/dataset cards, coordination |
| M2 - Reproducibility | Project structure, code/data versioning, experiment tracking |
| M3 - Quality assurance | Energy analysis, static analysis, code/data/model tests |
| M4 - Model deployment | Architecture, serving environment, API and API tests |
| M5 - Model packaging | Containers and CI/CD |
| M6 - Monitoring | Resource and model monitoring |

### Board states

| Status | When to use it |
| --- | --- |
| Backlog | Task recorded; scope or priority may still need discussion. |
| Ready | Outcome, acceptance criteria, owner, milestone, priority, and size are clear. Blocking dependencies are resolved. |
| In progress | The owner is working on the task. |
| In review | Results and checks are available for a teammate to review. A draft PR alone is not enough. |
| Done | Acceptance criteria met, review complete, evidence linked, and repository changes merged. |

For blocked work, keep the current status and add `blocked`, a dependency link,
and the next action. Close abandoned or duplicate work as **Not planned** and
exclude it from completed-work reporting.

Owners add issues to the Project and update their cards manually. Project
automations are disabled: linking a draft PR or closing an abandoned issue must
not mark work as reviewed or Done. Move an issue to Done after checking its
acceptance criteria and merge status.

### Fields and labels

Use Project fields for Status, Priority, Size, and weekly Iteration, and native
Assignees and Milestone fields. Keep priority, size, and status out of labels.

| Priority | Meaning |
| --- | --- |
| P0 | Urgent blocker threatening progress or a delivery. |
| P1 | Required for the milestone outcome. |
| P2 | Useful extension once required work is covered. |
| P3 | Optional idea. |

| Size | Estimated active human effort |
| --- | --- |
| S | 1-2 hours |
| M | 3-5 hours |
| L | 6-10 hours |
| XL | More than 10 hours or too uncertain; split or investigate before Ready. |

Revise estimates as the team gains experience. Record training time and compute
constraints separately; estimates are for planning, not individual assessment.

Choose one primary type and the relevant areas:

- Type: `type:feature`, `type:bug`, `type:experiment`, `type:docs`, `type:chore`.
- Area: `area:data`, `area:model`, `area:api`, `area:infra`, `area:monitoring`.
- Flags: `blocked`, `needs-decision`.

Start from the [task template](.github/ISSUE_TEMPLATE/task.md) or
[experiment template](.github/ISSUE_TEMPLATE/experiment.md). Use **Current week**
for the status board, **Backlog** for priority order, and **Milestones** for work
grouped by course milestone.

## GitHub Flow and review

1. Pick a Ready issue and create a branch from the latest `main` in the
   organization repository.
2. Name it `<type>/<issue>-<description>`, such as `feat/12-temporal-splits` or
   `docs/23-dataset-card`. Commit small, coherent changes.
3. Open a PR using the [template](.github/pull_request_template.md). A draft PR
   is useful for early discussion; mark it ready when the work can be reviewed.
4. Link checks and results, request a teammate's review, and address comments.
5. After one peer approval and passing applicable checks, merge with a merge
   commit. Resolve review conversations before merging.
6. Delete the merged branch and update the issue and evidence links.

`main` is the only long-lived branch. Team branches and PRs stay in the
organization repository. If you cannot push a task branch, ask a repository
administrator to check your access before starting it.

Use your own Git identity and credit shared work accurately. Merge commits retain
the individual commits; reviews, experiments, documentation, and support also
matter when assessing contributions.

### Commit messages

Write task commit subjects as `<type>: <imperative summary> (#<issue>)`.
Use the task issue number, not the PR number. Keep the subject to 72 characters
or fewer, start the summary with a verb such as `add`, `fix`, or `record`, and
omit a final period. Describe the change, not just the file:
`docs: record demand units and source version (#5)` is clearer than
`docs: update README`.

Choose the type by the commit's main purpose:

- `feat`: new behavior; `fix`: corrected behavior.
- `test`: tests only; `docs`: documentation only.
- `refactor`: code restructuring without a behavior change.
- `chore`: repository setup, dependencies, tooling, or DVC tracking metadata.
- `experiment`: versioned experiment configuration or recorded findings.

Keep related code and tests together when they form one coherent change; use
`feat` or `fix` for that commit. If the reason or constraint is not clear from
the subject, add a blank line and a short body explaining it. Put detailed run
results and reproduction evidence in the PR or experiment record. Reserve
`Closes #<issue>` for the PR description when merging completes the task, as
described below; `(#<issue>)` in a commit subject only links the work.

### Repository enforcement

The following settings apply to `main`. The [setup record](docs/project-setup.md)
records their verification and any outstanding setup.

| Setting | Reason |
| --- | --- |
| PR with at least one peer approval | A second contributor checks the change before it reaches the shared branch. |
| Dismiss approvals when new reviewable commits are pushed | Approval must cover the current change, not an earlier version. |
| Resolve review conversations before merge | Review findings must be addressed or explicitly resolved. |
| Apply protection to administrators too | The same review process applies to every contributor. |
| Block force pushes and deletion of `main` | Preserve shared history, contribution records, and reproducible versions. |
| Allow merge commits; disable squash and rebase merging | Retain the individual commits and the PR boundary. Linear history is not required because it would conflict with merge commits. |
| Automatically delete merged branches in this repository | Keep the branch list focused on active work. |

Required CI checks are not configured yet. Add them after the jobs exist and have
run successfully, so the rule refers to working checks. Peer review and local
validation apply now; they do not replace the planned code, data, and model CI.

Use `Closes #<issue>` when merging completes the task; otherwise use
`Related to #<issue>`. Keep deployment tasks open until the deployment is verified.
An experiment can be complete with a negative result if its method and conclusion
have been reviewed.

## Reproducibility and evidence

Link the issue and PR to the relevant code, data, configuration, run, and model
version so a teammate can reproduce the result.

- **Git:** source, configuration, dependency locks, tests, and DVC metadata.
  Keep credentials, environments, raw data, and model weights out of Git.
- **DVC:** shared data and model artifacts. Push artifacts before merging new
  references and have a teammate check that they can retrieve them.
- **MLflow:** runs with Git commit/dirty state, data and base-model versions,
  split configuration, seed, environment, metrics, and artifacts.

Use run IDs and configurations to distinguish experiments. Create a branch when
code or configuration changes. Run fast checks on PRs once CI is available;
expensive training and evaluation run separately. Model promotion needs evaluation
results.

Update the [course evidence register](docs/rubric-evidence.md) with the version,
check, result, and reviewer. Record consequential choices in the
[EDN](docs/edn/README.md), including the role and assessment of AI where applicable.

## Weekly coordination and deliveries

Hold a 20-30 minute planning meeting each week. Review feedback, dependencies,
availability, and the next tasks. Rotate the coordinator and reviewers, aim for
one active task per person, and review PRs within one working day when possible.
Communicate delays and help with blockers before starting more work.

Add a short Project update with participants, progress links, feedback, and next
actions. Build report material throughout the project and follow the course's
consensus/CATME process for the contribution assessment at each delivery.

The supplied lab schedule gives **19 October 2026, 23:55** for the initial report
and **14 December 2026, 23:55** for the final report, with presentations the next
day. Limits are 15 and 30 pages, excluding annexes. Check Atenea for changes.
Agree internal freezes before submission and tag the submitted version with
replication instructions.

## References

- Course laboratory slides, 2026-27: slides 18-19, 22-24, 34, and 38-41.
- [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow).
- [Projects practices](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/best-practices-for-projects).
- [Branch protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
