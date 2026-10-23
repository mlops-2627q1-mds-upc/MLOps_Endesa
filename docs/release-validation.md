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

The workflow references optional repository secrets `DVC_HTTP_USER` and
`DVC_HTTP_PASSWORD`. Their availability and remote access must be verified before
claiming that retrieval works in Actions. A local authenticated pull does not
establish runner access. The workflow has not been remotely executed as part of
this implementation.

An `always()` step uses [upload-artifact](https://github.com/actions/upload-artifact)
to retain JSON reports, logs and executed parameters for 14 days, including failed
runs. It excludes data, weights, source copies and credential files. A setup
failure before the command starts may have only the Actions log. Existing PR
workflow triggers, check names and credentials remain unchanged.
