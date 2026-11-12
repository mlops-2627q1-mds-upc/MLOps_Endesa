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
commit, tree and dirty state, selected weights and training run, DVC identities, commands,
budgets, exit codes, timings and any GitHub run/attempt identifiers. A failed step
stops later checks. Source/configuration hashes are checked again before success;
changes during execution fail validation. Earlier evidence is not overwritten.

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
authentication or retrieval from Actions. On 12 November 2026, this workflow was
still absent from `main` and GitHub's workflow registry, so it was not dispatched.
Inference dependency [PR #52](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/52)
merged as `ae9a38c18978251d2389f82367108ff097c3c3f1`; `main` also contains the API
from PR #86. PR #87 includes `main` at `26f260e33766689bcdce4a6db6062a9e1e74475d`
and still needs Pau's peer review and a reviewed merge before remote execution.

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
to retain JSON reports, handoff checksums, logs and executed parameters for 14 days, including failed
runs. It excludes data, weights, source copies and credential files. A setup
failure before the command starts may have only the Actions log. Existing PR
workflow triggers, check names and credentials remain unchanged.

## Publication handoff for #78

Every completed command also writes `handoff.json` and `handoff.sha256` beside
`report.json`, including on failure. The handoff is a public summary: it contains
canonical repository paths and hashes, without absolute machine paths, command
arguments, logs, credentials or raw data. Keep detailed reports in the retained
validation artifact; their hashes let a publication manifest identify the evidence.

| Field in handoff schema 1 | Meaning |
| --- | --- |
| `kind`, `schema_version`, `run_id` | `bounded-release-validation`, `1`, and the unique command run ID |
| `source.git_commit`, `git_tree`, `git_dirty` | Full commit/tree SHAs and checkout state captured before the checks |
| `source.sha256` | Original source/configuration file hashes, including `params.yaml`, `pyproject.toml`, `uv.lock`, DVC references, checkpoint provenance and the release workflow |
| `source.github` | `GITHUB_REPOSITORY`, `GITHUB_RUN_ID`, `GITHUB_RUN_ATTEMPT`, `GITHUB_SHA`, `GITHUB_WORKFLOW_REF`; values are null outside Actions |
| `checkpoint` | Canonical `path`, weights `sha256`, originating `training_run_id`, complete directory `md5`, and `provenance_sha256` |
| `data` | Raw-directory and four processed-file DVC MD5 identities, keyed by canonical path |
| `configuration` | `params_sha256`, `pyproject_sha256`, `uv_lock_sha256`, and `executed_params_sha256` for the CPU/two-thread parameter copy |
| `outcome` | `status`, `exit_code`, acceptance rule, UTC start/end, elapsed/total budget, and each step's name/status/exit code/elapsed/budget |
| `reports_sha256` | SHA-256 for each retained report and executed parameter file, keyed relative to the run directory |

`reports_sha256` covers `report.json`, `identities.json`, `params-cpu2.yaml`,
`minimum-functionality.json`, `directional/*/{plan,report}.json` and
`data-check/reports/*.json` when present. `handoff.sha256` hashes the exact bytes
of `handoff.json`; the handoff has no self-referential digest. Failure summaries
can omit identities or report files for checks that did not finish. They cannot
authorize publication.

The publication workflow in #78 can consume this interface without inference:

1. Select an authorized run of `.github/workflows/release-validation.yml` in this
   repository through the GitHub API. Require a successful `workflow_dispatch`
   run whose `headSha` equals the full image source SHA. Pin its run ID, attempt
   and unexpired `release-validation-RUN_ID-ATTEMPT` artifact; retain its artifact
   ID/digest. Never select a run only because it is the latest successful one.
2. Download that artifact, then run the following check from the same source
   checkout. Replace the run/attempt/directory placeholders and `IMAGE_SOURCE_SHA`.

   ```sh
   uv run --no-sync python -m src.release_validation \
     --check-handoff temporary/release-actions-RUN_ID-ATTEMPT/RUN_UUID/handoff.json \
     --source-commit IMAGE_SOURCE_SHA
   ```

   Exit 0 requires matching report hashes, a clean matching source commit, a
   successful retrieval followed by all five checks, complete result files, and
   elapsed times within every step/total budget. It also rejects a nonmatching
   Actions SHA. Missing files, failure, timeout, dirty source, tampering and a
   cache-only run return nonzero. This command performs no model loading or
   network access. It checks evidence consistency; the GitHub API selection in
   step 1 establishes where that evidence came from.
3. Match every `source.github` value to the selected run/attempt, compare
   `source.git_tree` and `source.sha256` with the image build checkout, and compare
   checkpoint/data identities with its tracked provenance and DVC references.
   A successful check for another commit cannot authorize this image, including
   a merge commit with an otherwise similar tree. Rerun validation for the exact
   release source when that source changes.
4. After the publication prerequisites are reviewed, record the immutable image
   digest with the source SHA, checkpoint/training-run identity, data/configuration
   identities, validation run/attempt and artifact ID/digest, the handoff SHA-256,
   and `reports_sha256`. Keep publication and independent pull/startup evidence
   separate from validation success.

No image is published by this command or its workflow. The pending decision record
is [EDN-017 in PR #97](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/blob/e85b168c5bc282463cd6a6f8edf34f3c073ab6f5/docs/edn/0017-release-validation.md);
this change does not duplicate it or supply peer approval.

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

That run remains evidence for `66266e01f92dcf01e17901ad72bb6d99a4fd0fb1`.
Its original `report.json` SHA-256 is
`fe08bf158e35a66d03b9f4d2d0f53783ab7cd3162a67a0d6c10cd2b46d7d149f`.
The integrated branch has a different dependency lock and adds the handoff
interface. No real-model run was repeated for this refresh, and the historical
report was not rewritten or given a retrospective publication handoff. Required
PR CI and fixture tests cover implementation separately; authenticated Actions
validation on the intended release source remains a later execution step.
