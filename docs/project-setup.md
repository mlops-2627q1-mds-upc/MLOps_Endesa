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
| Required CI checks | `CI / lint` and `CI / test`, both restricted to GitHub Actions (app ID `15368`). |
| Base branch freshness | Strict checks enabled: the PR branch must be up to date with `main`. |
| Milestones | M1-M6 exist; milestone due dates are unset. |

Repository and Project access evidence is maintained in
[issue #2](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2).

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

As of 2026-10-16, `make lint` runs Ruff formatting/lint and Pylint with
`fail-under = 9.0` and `fail-on = ["E"]`, added in
[PR #38](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/38). The test job
runs the default pytest suite on Ubuntu 24.04 with Python 3.11. It does not pull
DVC artifacts, so the nine data-backed checks in `tests/test_data.py`, added in
[PR #37](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/37), are skipped
when those files are absent. Synthetic validation tests still run in CI; a green
gate does not establish successful validation of the shared dataset.

The 2026-10-11 failure/recovery verification above executed all 43 tests without
skips; that is historical evidence for the gate at that revision.
[GitHub also accepts skipped or neutral required check conclusions](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging),
so executed tests and intentional skips must be reported separately. Shared-data
validation, full training, native Windows verification, model promotion and
deployment require their own evidence.

The remote rule is effective. Issue #20 stays open until the documentation PR
receives a teammate's review and is merged.

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
