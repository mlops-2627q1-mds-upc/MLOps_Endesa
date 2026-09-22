# EDN-001: Team workflow, review, and evidence policy

- **Date:** 2026-09-22.
- **Milestone:** M1 - Project Inception; also relevant to M2 and M5.
- **Activity / Topic:** Project coordination, code review, and traceability.
- **Decision Participants:** Dídac Cayuela (`didicayu`). Codex assisted the
  discussion and drafted the documentation. Participation or approval by the
  other four team members has not been recorded.
- **Status:** Accepted by Dídac for initial adoption and documentation; wider
  team review is pending. See the implementation update below for remote setup.

## Decision

Adopt the [team working agreement](../../CONTRIBUTING.md): one GitHub Project,
issue-based planning across M1-M6, five workflow states, and separate Priority and
Size fields. Implement changes through short-lived branches and peer-reviewed PRs
to `main`, using merge commits to preserve meaningful contribution history.
Connect completed work to reproducibility evidence and selected EDN entries.

GitHub Flow is already prescribed by the course. The choices recorded here concern
how the team organizes its board, reviews changes, preserves authorship, and
connects work to evidence.

## Rationale

A shared issue tracker keeps ownership, dependencies, and review discussions near
the code. Single-select fields avoid contradictory priority/size labels. One peer
approval and small tasks provide review without making every change depend on the
whole team. Weekly planning and rotating reviewers support shared knowledge.

Alternatives considered in the AI-assisted proposal were duplicating priority and
size in labels, tracking both issues and PRs as task cards, squashing contribution
history, and introducing more board automation immediately. The selected policy
reduces duplicate tracking, retains original authorship, and starts with explicit
manual transitions. The trade-offs are a less compact Git history and some manual
board maintenance; the team can revisit them after using the workflow.

The course assesses coordination, versioning, reproducibility, and individual
contributions (lab slides 22, 24, and 40). Evidence links make these practices
inspectable; task counts and configured tools alone do not establish their quality.

## AI Involvement

Information seeking; Alternative generation; Alternative assessment;
Recommendation; Solution generation.

Codex read the supplied course materials and demo repository, compared workflow
choices, and proposed the agreement and supporting templates.

## Response to AI

**Accepted.** Dídac instructed Codex to create the files documenting the proposed
working method and to add agent instructions if useful.

## Assessment of the AI Contribution

The proposal connected a concrete workflow to the course requirements and supplied
definitions for task readiness, completion, review, and evidence. The observable
human response was authorization to document and adopt it initially. No separate
written assessment from Dídac or review by the full team was provided in this
interaction. Its practical effectiveness remains to be assessed through use.

## AI Interaction Evidence

Selected excerpts from the interaction on 2026-09-22:

- Initial user proposal: "maybe we can use github flow with github projects to
  keep track of issues and branches. having a backlog, ready, in progres, in
  review, done."
- Assistant recommendation: "I recommend **GitHub Flow + one GitHub Project +
  the six course milestones**, with each completed task linked to evidence that
  helps you write and defend the report."
- User response: "Go ahead and create the necessary files to leave evidence of
  the decision made on how to work within the group. keep in mind this will be
  read by the teacher. Create the agents.md if necessary too"

These excerpts document the proposal and authorization; they are not minutes of a
meeting involving the full team.

## Other Evidence

- [Working agreement](../../CONTRIBUTING.md).
- [Task template](../../.github/ISSUE_TEMPLATE/task.md),
  [experiment template](../../.github/ISSUE_TEMPLATE/experiment.md), and
  [PR template](../../.github/pull_request_template.md).
- [Rubric evidence register](../rubric-evidence.md) and
  [agent instructions](../../AGENTS.md).
- Course lab slides, 2026-27, slides 22, 24, 30, and 38-41; and the
  [EDN instructions](README.md#source).

At initial drafting, the evidence consisted of this discussion and the local
documentation. No issue/PR identifiers, peer approval, Project URL, successful CI
run, or configured branch protection had been recorded.

## Implementation update - 2026-09-22

Dídac subsequently instructed Codex to create the organization Project and a task
for the documentation work. The
[Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3) was created with
the five agreed states, Priority, Size, weekly Iteration, and three views.
[Issue #1](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/1) was created
after the local draft and added as In progress, P1, and M. This sequence is
explicit in the issue; its creation is not backdated.

GitHub denied native repository linking, changing the Project's visibility, and
assigning the issue to Dídac. The issue body records the accountable owner, while
the native assignee remains unset. The [setup record](../project-setup.md) lists
the verified configuration and outstanding access requirements. A separate
[setup task #2](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2)
tracks the remaining settings. Backlog priority sorting and Milestones grouping
were subsequently configured and verified through the API.

Dídac also authorized pushing the documentation and opening a PR. Read-only
repository access requires publishing the branch through his own fork. Issue #1
tracks publication and review evidence. No peer approval or merge has been
recorded, and the task has not been marked Done.
