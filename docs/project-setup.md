# GitHub setup status

Branch protection and merge settings verified through GitHub's API on
**2026-10-11**, using `didicayu`. The Project and access notes below retain the
initial **2026-09-22** snapshot; [issue #2](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2)
tracks that separate setup work.
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
| Required CI checks | `CI / lint` and `CI / test`, both restricted to GitHub Actions (app ID `15368`). |
| Base branch freshness | Strict checks enabled: the PR branch must be up to date with `main`. |
| Initial labels and milestones | All 12 agreed labels and milestones M1-M6 created on 2026-09-22, without milestone due dates. |

## Required CI checks

Configured on 2026-10-11 for
[issue #20](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/20), after
confirming both exact check names and their GitHub Actions source in the
[successful main run](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/38158426444)
at `1ef5bc9ce00a5f61907f2b2127a52ba42950532c`. That run passed Ruff and 43 tests.
The [workflow](../.github/workflows/ci.yml) runs on pull requests to `main` and
pushes to `main`; neither trigger has path filters, and neither job has an `if`
condition. Dependencies are installed with `uv sync --locked --dev`.

The API reports `strict: true` and these required checks:

```json
{
  "checks": [
    { "context": "CI / lint", "app_id": 15368 },
    { "context": "CI / test", "app_id": 15368 }
  ],
  "strict": true
}
```

Comparing the branch-protection responses before and after configuration showed
that every other protection field was unchanged. The repository still allows
merge commits only and automatically deletes merged branches. The choice of a
strict gate and its trade-off are recorded in [EDN-006](edn/0006-required-ci-checks.md).

### Failure and recovery verification

[PR #42](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/42) used a
temporary test file to fail each required job independently. Its base stayed at
`1ef5bc9ce00a5f61907f2b2127a52ba42950532c` throughout the verification.

| Phase / PR head | Run | Required checks | Observed GitHub result |
| --- | --- | --- | --- |
| Unused import; `fbe12c6` | [Lint failure](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/38159658820) | Lint failed with Ruff `F401`; tests passed. | Both jobs had `isRequired: true`; `mergeStateStatus: BLOCKED`; `gh pr checks --required` exited 1. |
| Failing assertion; `21acfa2` | [Test failure](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/38159712941) | Lint passed; tests reported 1 intentional failure and 43 passes. | Both jobs had `isRequired: true`; `mergeStateStatus: BLOCKED`; `gh pr checks --required` exited 1. |
| Probe removed; `2d368a5` | [Recovery](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/38159790049) | Lint and all 43 tests passed. | Both jobs remained required; `gh pr checks --required` exited 0. Peer approval remained required. |

The [API observations](evidence/issue-20-ci-gate.json) record each head, required
check source, result, and observation time. The PR also reported `REVIEW_REQUIRED`
throughout: after CI recovered, it remained `BLOCKED` by the peer-review rule.
No merge was attempted, and the temporary PR was closed without merging.

To inspect the live rule and recorded PR checks with repository access:

```sh
gh api repos/mlops-2627q1-mds-upc/MLOps_Endesa/branches/main/protection
gh pr checks 42 --repo mlops-2627q1-mds-upc/MLOps_Endesa --required
gh run view 38159790049 --repo mlops-2627q1-mds-upc/MLOps_Endesa
```

The gate covers Ruff formatting/lint and the current pytest suite on Ubuntu
24.04 with Python 3.11. It does not retrieve shared DVC artifacts, run full model
training, establish native Windows portability, promote models, or deploy a
service. Those tasks need their own evidence.
[GitHub also accepts skipped or neutral required check conclusions](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging);
this verification observed actual successful
job execution, without skipped tests. Changes to CI must preserve that execution.

The remote rule is effective. Issue #20 stays open until the documentation PR
receives a teammate's review and is merged.

## Initial Project snapshot (2026-09-22)

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

At initial setup, [documentation issue #1](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/1)
was **In review**, P1, M, and linked to
[PR #4](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/4).
[Setup issue #2](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2) was
**In progress**, P1, S, with `blocked` for the remaining access dependencies.
Both were assigned to Week of 2026-09-22.

## Initial access and review snapshot (2026-09-22)

The Project was private. Repository Admin access did not grant permission to
change Project visibility or collaborators; a Project administrator was needed to
confirm team/lecturer access and arrange editing access for the contributors. Dídac,
Pablo, and Sindri had repository Admin access. Invitations had been sent to Pau
(`NIU1638529`) and Antoni (`toni646`); their access was Read until acceptance.
Issue #2 tracks the follow-up; these historical notes are not a current access audit.

The original documentation subsequently merged in
[PR #4](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/4).
