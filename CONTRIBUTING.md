# MLOps_Endesa: Team Working Agreement

This agreement defines how the five team members coordinate, review, and document
work on the electricity-demand forecasting component. Its purpose is to make
responsibilities clear, integrate changes safely, and leave a reproducible record
of the engineering work.

The initial decision is recorded in [EDN-001](docs/edn/0001-team-workflow.md).
The organization Project and documentation issue have been created. Repository
linking and access setup remain incomplete; branch protection and CI are separate
implementation steps. See the [verified setup record](docs/project-setup.md).

## Planning and responsibilities

Use the [MLOps_Endesa Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3)
owned by **mlops-2627q1-mds-upc** for this repository. Issues represent work items;
branches and pull requests are linked to their issues. Avoid adding the issue and
its implementation PR as separate task cards.

[Issue #1](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/1) tracks this
documentation; its Project card carries the current status, Priority P1, and
Size M. [Issue #2](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2)
tracks remaining repository access and workflow enforcement. The Project is
private; its native repository link and team/lecturer access need a maintainer
or Project administrator, as detailed in the setup record.

Each issue has one accountable owner. Other contributors can collaborate and
should be credited in the issue, PR, or commits. Reviewers should rotate so that
knowledge is shared across the team.

Use the six course milestones:

| Milestone | Scope |
| --- | --- |
| M1 - Inception | Requirements, model/dataset cards, coordination |
| M2 - Reproducibility | Project structure, code/data versioning, experiment tracking |
| M3 - Quality assurance | Energy analysis, static analysis, code/data/model tests |
| M4 - Model deployment | Architecture, serving environment, API and API tests |
| M5 - Model packaging | Containers and CI/CD |
| M6 - Monitoring | Resource and model monitoring |

### Board states

| Status | Entry condition |
| --- | --- |
| Backlog | A candidate task has been recorded. Scope or priority may still need discussion. |
| Ready | Outcome, acceptance criteria, dependencies, milestone, priority, and size are clear; blocking dependencies are resolved. |
| In progress | An owner is actively working on the task. |
| In review | The result and evidence are available for a teammate to assess. A draft PR alone does not meet this condition. |
| Done | Acceptance criteria are met, review is complete, and supporting evidence is linked. Repository changes are merged. |

A blocked task retains its current status and receives the `blocked` label, a
dependency link, and a comment describing the next action and who can take it.
If work is abandoned or duplicated, close the issue as **Not planned** with an
explanation and exclude it from completed-work reporting.

### Fields and labels

Store Status, Priority, and Size as single-select Project fields. Use the native
Assignees and Milestone fields, plus a weekly Iteration field for planned work.
Do not duplicate priority, size, or status in labels.

| Priority | Meaning |
| --- | --- |
| P0 | Urgent blocker threatening team progress or a delivery; use sparingly. |
| P1 | Required for the agreed milestone outcome. |
| P2 | A valuable extension that strengthens quality or the demonstration. |
| P3 | An optional idea to revisit when essential work is secure. |

| Size | Initial estimate of active human effort |
| --- | --- |
| S | 1-2 hours |
| M | 3-5 hours |
| L | 6-10 hours |
| XL | More than 10 hours or too uncertain to estimate; split or investigate before Ready. |

Estimates support planning and are recalibrated from experience. They are not
individual productivity targets. Record machine runtime and compute constraints
separately when relevant.

Use a small set of repository labels:

- Type: `type:feature`, `type:bug`, `type:experiment`, `type:docs`, `type:chore`.
- Area: `area:data`, `area:model`, `area:api`, `area:infra`, `area:monitoring`.
- Flags: `blocked`, `needs-decision`.

Choose one primary type and the relevant areas. Use the
[task template](.github/ISSUE_TEMPLATE/task.md) or the
[experiment template](.github/ISSUE_TEMPLATE/experiment.md).

Maintain three views of the same Project: **Current week** (board), **Backlog**
(table ordered by priority), and **Milestones** (table grouped by milestone).

## GitHub Flow and review

1. Select a Ready issue, assign its owner, and update the local `main` branch.
2. Create a short-lived branch from `main`. Use `<type>/<issue>-<description>`,
   for example `feat/12-temporal-splits`, `fix/19-missing-intervals`, or
   `docs/23-dataset-card`.
3. Commit coherent changes with descriptive messages. Open a draft PR when early
   feedback is useful. Keep unrelated changes in separate PRs.
4. Complete the [PR template](.github/pull_request_template.md), link evidence,
   and request review from another team member.
5. Address review comments, pass the applicable checks, and merge into `main`.
6. Delete the merged branch and update the issue and evidence links.

`main` is the only long-lived development branch. For the initial repository
setup, a descriptive branch without an issue number is acceptable; add the issue
link when the issue tracker is configured. Do not fabricate historical issues,
reviews, or approvals to reconstruct an earlier workflow.

If a contributor has read access only, push the task branch to their own fork
and open a PR against this repository's `main`. Keep the issue and Project card
in the organization repository. This is a contribution path while maintainers
resolve team access; it does not confer permission to change repository settings.

Use **merge commits** by default to preserve meaningful commits and authorship.
Each contributor uses their own Git identity associated with their GitHub account.
Credit shared work accurately. Contribution assessment also considers reviews,
experiments, documentation, and support to teammates; commit or line counts alone
do not describe contribution quality.

Configure protection for `main` to require one approving peer review, successful
applicable CI checks, and resolved review conversations, and to block force pushes
and deletion. Register required checks after the jobs exist and have run
successfully. Until enforcement is configured, apply the review policy manually.

Use `Closes #<issue>` in the PR description only when merging completes the issue.
Use `Related to #<issue>` for partial progress. A deployment task remains open
until the deployment itself has been verified. A documentation task is reviewed
for accuracy and usable links; an experiment is reviewed for method, results, and
interpretation. A negative experimental result can satisfy its acceptance criteria.

## Reproducibility and evidence

Maintain the connection between an issue, its PR/code commit, the DVC data version,
the MLflow run, the model artifact, and the release that uses it, as applicable.

- Git stores source, configuration, dependency locks, DVC metadata, tests, and
  concise documentation. Keep credentials, local environments, raw datasets, and
  model weights out of Git.
- DVC versions shared datasets and model artifacts. Before merging new artifact
  references, push the artifacts to the shared remote and have a teammate verify
  that they can retrieve them.
- MLflow records experiments. Important runs identify the Git commit and any
  uncommitted changes, data version, base-model revision, split configuration,
  seed, environment, metrics, and artifact references.
- Use configuration and run IDs to distinguish experiments; create a branch when
  the experiment requires a code or configuration change.
- Run fast checks on PRs as CI becomes available. Trigger expensive training and
  evaluation separately. Promoting a model requires its evaluation evidence.
- Update [rubric evidence](docs/rubric-evidence.md) when work provides evidence
  for a course practice. State what was checked, by whom, and on which version.

Record selected consequential decisions in the
[Engineering Decision Notebook](docs/edn/README.md). When AI contributes, record
its role, the human response, and the assessment of its contribution. An AI
suggestion or a completed configuration file is not evidence that a system works.

## Weekly coordination and deliveries

- Hold a 20-30 minute planning meeting to review lecturer feedback, dependencies,
  availability, and the next week's work. Rotate the coordinator.
- Aim for one active implementation task per person. Finish reviews and help with
  blockers before starting additional tasks.
- Aim to review within one working day when availability permits; communicate
  delays explicitly.
- Demonstrate progress each week. Add a short Project status update with the date,
  participants, evidence links, feedback, and next actions. Record decisions and
  tasks in their linked EDN entries and issues.
- Update the relevant report material throughout each milestone. Each delivery
  includes an evidence-based team contribution assessment following the course's
  consensus/CATME instructions.

The supplied lab schedule lists the initial report deadline as **19 October 2026,
23:55** and the final report deadline as **14 December 2026, 23:55**, with the
presentations on the following days. The report limits are 15 and 30 pages,
respectively, excluding annexes. Atenea remains authoritative for any changes.
Agree internal freezes several days before submission and identify the exact
submitted repository version with a release tag and replication instructions.

## References

- Course laboratory slides, *Machine Learning Systems in Production (MLOps)*,
  2026-27: slides 18-19, 22-24, 34, and 38-41.
- [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow).
- [GitHub Projects practices](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/best-practices-for-projects).
- [Branch protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
