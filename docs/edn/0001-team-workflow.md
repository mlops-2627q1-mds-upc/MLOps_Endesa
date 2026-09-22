# EDN-001: Task planning and review

- **Date:** 2026-09-22
- **Milestone:** M1 - Project Inception; related to M2 and M5
- **Decision Participants:** Dídac Cayuela (`didicayu`)
- **Status:** Adopted for initial setup; team review in [PR #3](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/3)

## Decision

Use one GitHub Project with Backlog, Ready, In progress, In review, and Done.
Track tasks as issues, with separate Priority and Size fields. Each task has an
owner and links to its branch, PR, and supporting results. Require one teammate's
review before merging, and use merge commits to retain the individual commits.

Dídac proposed GitHub Flow, the Project board, its five states, priorities, and
sizes as the starting point. GitHub Flow is also prescribed by the course; this
entry records the planning and review choices around it.

## Rationale

- **One issue per task:** the issue holds scope and progress; the PR holds the
  change and review. Separate cards for both would duplicate progress tracking.
- **Priority and Size as fields:** each task has one value for each. Labels remain
  available for type, area, and blockers without competing priority labels.
- **One peer approval:** gives each change a second reader without requiring all
  five members to approve it. Reviewers rotate to share knowledge of the work.
- **Merge commits:** keep the sequence and authorship of meaningful changes
  visible for review and contribution assessment. The cost is a longer history
  than squash merging.

The board needs regular maintenance. Weekly planning and one active task per
person should keep that manageable; the team can revise these limits after use.
Completion depends on reviewed results and merged changes, rather than a card
movement alone.

## AI Involvement

Information seeking; Alternative generation; Alternative assessment;
Recommendation; Solution generation.

Codex checked the course materials and demo repository, helped compare planning
and review options, and drafted the agreement and templates from Dídac's initial
proposal. Dídac chose to adopt the resulting workflow.

## Response to AI

**Accepted** for the initial working agreement. Dídac subsequently requested
shorter documentation and clearer attribution of the original proposal.

## Assessment of the AI Contribution

The draft turned the initial workflow into concrete field definitions, review
rules, and templates. Dídac's feedback identified excessive process detail in the
EDN and asked for more focus on the engineering choices. This revision addresses
that feedback; the workflow's practical value remains to be assessed through use.

## Other Evidence

- [Working agreement](../../CONTRIBUTING.md) and [Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3).
- [Documentation task #1](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/1) and [PR #3](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/3).
- [Setup status](../project-setup.md), including the outstanding settings in [issue #2](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2).
- [Course EDN instructions](README.md#source).
