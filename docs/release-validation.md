# Bounded release validation

`python -m src.release_validation` checks the selected checkpoint and tracked data
before a release. It runs separately from `CI / lint` and `CI / test` and does not
train, score final-test forecasts, publish a model, or deploy a service.

The acceptance rule is written to a new report before execution: every artifact
identity must match, data validation must pass, and all existing
[minimum-functionality](minimum-functionality.md) and
[directional](experiments/007-directional-expectations.md) cases must pass within
the budgets below. These are 10 synthetic minimum-functionality cases, 12
directional cases and the unchanged control. Their existing cases, thresholds,
seeds and sampling settings are reused. This command does not rerun invariance or
change the completed final-test evaluation in [Experiment 004](experiments/004-chronos-fine-tuning.md).

| Step | Hard wall-clock cap |
| --- | ---: |
| Optional DVC retrieval | 180 seconds |
| Artifact verification | 60 seconds |
| Configure private GX context | 30 seconds |
| Data validation | 120 seconds |
| Minimum functionality | 330 seconds (existing runner: 300 seconds) |
| Directional checks | 210 seconds (existing runner: 180 seconds) |
| Entire command | 900 seconds |

Every step receives the smaller of its cap and the remaining total budget. The
outer model caps include interpreter startup and report writing. Execution is
sequential, CPU only, with two CPU threads and offline model access. Hard timeouts
kill the complete process group, including nested workers; this command supports
Linux and macOS. Missing artifacts, failed checks and timeouts return nonzero.

## Run locally

Install the locked environment once, then run in a clean task checkout:

```sh
uv sync --locked --dev --extra forecast
uv run --no-sync python -m src.release_validation --pull
```

`--pull` retrieves only `data/raw.dvc`, the `split` stage's four processed Parquet
outputs in `dvc.lock`, and `models/chronos-t5-small-fine-tuned.dvc` from `origin`.
It verifies that the split stage declares exactly those four outputs. It does not
run `dvc repro`. For authenticated HTTP access, supply your authorized credentials
as `DVC_HTTP_USER` and `DVC_HTTP_PASSWORD` through your environment's secret
mechanism. Both are optional for public remotes and required together otherwise.
Credentials are passed to DVC in memory, never written to remote configuration or
included in the report, and removed from the child environment for later checks.

To establish retrieval independently of an existing model cache, start with an
absent selected-model directory and an empty private DVC cache. Record those
conditions before running `--pull`. Use an ignored `.dvc/config.local` to select
the private `cache.dir`; preserve any existing configuration and verified weights.
A successful run using a populated cache alone does not establish remote access.

Omit `--pull` when artifacts are already present. Existing artifacts can also be
read from another location without copying weights:

```sh
uv run --no-sync python -m src.release_validation \
  --data-dir /absolute/path/to/verified/data \
  --checkpoint /absolute/path/to/selected/checkpoint
```

Verification checks the selected weights against the repository's required
provenance using `describe_checkpoint`, including the originating training run.
The full model directory (including configuration files), raw directory and four
processed files must match their DVC hashes. Directory members cannot be
symlinks; extra files also change the hash and fail verification. Relocated
artifacts must match the same references. `--pull` accepts only repository-local
paths.

The existing GX configuration and data-validation commands run from a small
private source copy with a link to the verified data. They write their context,
Data Docs and report inside the current run directory. Original tracked GX files,
`reports/validation.json`, data and model files are not modified. Data checks
inspect final-test structure and completeness; model checks use synthetic inputs.

Each invocation creates a unique directory under `temporary/release-validation/`;
`--output-dir` selects a different parent. It retains `report.json`, step logs,
verified identities, original source/configuration hashes, the exact two-thread
parameter copy and the runners' detailed reports. The summary records the Git
commit and dirty state, selected weights and training run, DVC identities, commands,
budgets, exit codes, timings and any GitHub run/attempt identifiers. A failed step
stops later checks. Earlier evidence is not overwritten.

For a controlled failure that requires no data or inference:

```sh
uv run --no-sync python -m src.release_validation \
  --checkpoint /nonexistent/release-validation-checkpoint
```

Expect exit 1, a failed verification step and a retained report/log. A smaller
`--max-seconds` (1–900) can test the total budget; exceeding it exits 124. Do not
increase thresholds after observing a failed candidate.

## Manual GitHub workflow

[release-validation.yml](../.github/workflows/release-validation.yml) has only a
`workflow_dispatch` trigger. Once the reviewed workflow is available on the
default branch, dispatch it for the intended source ref. It installs the locked
forecast environment and calls the same command with `--pull`. Its job cap is
30 minutes, including setup and evidence upload. It has read-only repository
permissions, serializes runs and does not cancel a running validation.

The workflow references repository secrets `DVC_HTTP_USER` and
`DVC_HTTP_PASSWORD`. Both names were verified present on 2 November 2026; only
the GitHub secret names were inspected. Their presence does not establish
authentication or retrieval from Actions. On that date, this workflow was absent from `main`
and GitHub's workflow registry, so no remote release-validation run was attempted.
The remaining prerequisite is availability through the normal reviewed merge,
including the upstream inference dependency in PR #52. That PR has Pau's approval
on its current head but remains unmerged; PR #87 still requires human review.

After the reviewed workflow reaches `main`, record its intended source SHA and
dispatch it with the authorized GitHub account:

```sh
gh api repos/mlops-2627q1-mds-upc/MLOps_Endesa/commits/main --jq .sha
gh workflow run release-validation.yml \
  --repo mlops-2627q1-mds-upc/MLOps_Endesa --ref main
gh run list --repo mlops-2627q1-mds-upc/MLOps_Endesa \
  --workflow release-validation.yml --event workflow_dispatch --branch main \
  --json databaseId,headSha,status,conclusion,url
```

Select the resulting run by its source SHA and invocation time. Use its actual
run ID and attempt in the following commands; `RUN_ID` and `ATTEMPT` are
placeholders:

```sh
gh run watch RUN_ID --repo mlops-2627q1-mds-upc/MLOps_Endesa --exit-status
gh run view RUN_ID --repo mlops-2627q1-mds-upc/MLOps_Endesa \
  --json headSha,attempt,status,conclusion,url
gh run download RUN_ID --repo mlops-2627q1-mds-upc/MLOps_Endesa \
  --name release-validation-RUN_ID-ATTEMPT \
  --dir temporary/release-actions-RUN_ID-ATTEMPT
```

Confirm that `report.json` passed every step including `retrieve`. Check that
`source.github` records the actual run's `GITHUB_SHA`, `GITHUB_RUN_ID` and
`GITHUB_RUN_ATTEMPT`, and that the source, data and selected-model identities
match the intended candidate. Retain failed-run evidence too. PR CI and local retrieval are separate
evidence; neither substitutes for this authenticated Actions execution. GitHub's
[manual workflow instructions](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
require the workflow to exist on the default branch even when using `--ref`.

An `always()` step uses [upload-artifact](https://github.com/actions/upload-artifact)
to retain JSON reports, logs and executed parameters for 14 days, including failed
runs. It excludes data, weights, source copies and credential files. A setup
failure before the command starts may have only the Actions log. Existing PR
workflow triggers, check names and credentials remain unchanged.

## Local evidence on 2 November 2026

Combined run `8dfb4336d29947c19c4177f8e8dd2edc` passed on clean source commit
`66266e01f92dcf01e17901ad72bb6d99a4fd0fb1` in 56.76 seconds. Authenticated
`--pull` used an initially empty private cache and an absent local model directory,
retrieving all 15 DVC objects (198,693,558 bytes, including directory manifests)
in 15.96 seconds. Verification matched the raw and selected-model directory hashes
and all four processed-file hashes. This closes the previous local combined-pull
gap; it is not an Actions run.

The model directory identity was `0377ee21f89c25da36b95db59841c3d4.dir`, weights
SHA-256 `746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9`,
and training run `639eba4cc7e74de08ffab64beee715db`. All 118 data expectations,
10 minimum-functionality cases, 12 directional cases and unchanged control passed.
No training or final-test forecast evaluation ran. The shared compute lock covered
the entire subprocess lifetime, and the existing Python 3.11 environment was
reused with two CPU threads. Existing data hashes and the shared model's weights
hash were unchanged. Temporary downloaded weights and the private cache were
removed after retaining their identity receipts and reports.

Fresh controlled runs retained a missing-checkpoint failure (exit 1) and a
one-second-budget timeout (exit 124); neither reached model inference. The
27 focused release tests passed. Reports, logs, retrieval-isolation receipts and
commands are retained under this worktree's ignored
`temporary/release-validation-2026-11-02/`; earlier run records remain intact.
