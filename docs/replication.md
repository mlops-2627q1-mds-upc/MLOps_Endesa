# Replication and release evidence

The [final delivery checklist](final-delivery.md) pins the integrated
implementation candidate and distinguishes pending PRs from merged `main`.
The [editable report/presentation sources](delivery/README.md) remain DRAFT while
human review and runtime gates are open. Use the fast route below for source
checks; the later release route is a separately authorized execution.

Use the exact candidate or reviewed commit linked from the report cover. `main` and task
branches can move; record `git rev-parse HEAD` before running anything. The
submission tag is created only after review, merge and explicit authorization. The report and slides
must identify the same version.

The [17 October verification record](evidence/submission-replication-2026-10-17.json)
identifies the exact executed commits, successful repeated DVC reproduction,
retrieved checkpoint identity and deterministic baseline results.
[Coverage snapshots](evidence/coverage-2026-10-17.md) distinguish the component
baselines from the combined checks.

## Environment and fast checks

Clone the organization repository, check out the submitted commit, and follow
[setup](setup.md#setup-from-a-clean-checkout) for Python 3.11 and uv. Then run:

```sh
uv sync --locked --dev
make lint
make coverage
```

`make coverage` runs the default pytest suite and measures statements and branches
across `src/`. CI publishes its summary. No minimum coverage threshold has been
set. Default CI needs no credentials, data download or model weights; real-data
checks skip when their DVC inputs are absent, and real-model checks are opt-in.
The coverage percentage changes with the measured source and available tests;
do not copy a percentage from a different commit or environment.

## Retrieve and validate data and model artifacts

DVC and MLflow authentication are separate. Use your own authorized DagsHub
account; [the evaluation guide](evaluation.md#reproduction) provides an in-memory
DVC configuration example. Keep credentials out of Git and commands saved in
shell history. Once authentication is available:

```sh
uv run --locked dvc pull -r origin
uv run --locked dvc status
uv run --locked dvc repro
```

The unrestricted pull includes raw data, processed blocks and the selected
checkpoint (approximately 176 MiB), not just raw data. To retrieve only data:

```sh
uv run dvc pull data/raw.dvc data/processed/train.parquet data/processed/validation.parquet data/processed/test.parquet data/processed/origins.parquet -r origin
```

The successful validation report has `success: true`, zero failures and **118
aggregate checks: 74 GX expectations and 44 Python checks**. The Python checks
are grouped in four categories; they are already included in 118. Reproduction
must preserve the raw and processed hashes in `data/raw.dvc` and `dvc.lock`.
The validation pass checks held-out data integrity; it does not fit a model or
score a new forecast on the consumed test block.

To inspect the selected checkpoint's weights-bound training identity:

```sh
uv run python -c "from pathlib import Path; from src.modeling.checkpoints import describe_checkpoint; print(describe_checkpoint(Path('models/chronos-t5-small-fine-tuned')))"
```

Expected weights SHA-256:
`746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9`.
Expected training run: `639eba4cc7e74de08ffab64beee715db`.
The versioned sidecar supports the original published artifact without changing
its DVC hash. A metadata/weights mismatch must fail; an unrelated directory name
is never used to infer the training run.

## Reproduce deterministic validation baselines locally

Use a local MLflow store for this check so it does not create shared experiments:

```sh
uv run python -c "from pathlib import Path; Path('temporary/replication').mkdir(parents=True, exist_ok=True)"
uv run python -m src.modeling.evaluate --method seasonal_lag48 --tracking-uri sqlite:///temporary/replication/mlflow.db
uv run python -m src.modeling.evaluate --method seasonal_lag336 --tracking-uri sqlite:///temporary/replication/mlflow.db
```

Each method should produce 87,600 predictions on 1,825 validation origins:

| Method | Validation mean MASE |
| --- | ---: |
| Lag 48 | 1.042173206687521 |
| Lag 336 | 1.201974712215821 |

The output directory and run IDs are new. Compare predictions and metrics,
not generated run names or timestamps. These commands retain the default
validation block. Do not add `--block test` to a routine replication check.

## Optional selected-model behaviour checks

Install the locked forecast extra and run the existing declared behaviour cases:

```sh
uv sync --locked --dev --extra forecast
uv run --extra forecast python -m src.modeling.minimum_functionality --checkpoint models/chronos-t5-small-fine-tuned --max-seconds 300
uv run --extra forecast python -m src.modeling.directional --checkpoint models/chronos-t5-small-fine-tuned --max-seconds 180
uv run --extra forecast python -m src.modeling.invariance --checkpoint models/chronos-t5-small-fine-tuned --max-seconds 300
```

Run these sequentially. Minimum functionality and directional cases use synthetic
histories; invariance also uses training/validation windows. The reports record
the checkpoint digest, settings, declared limits and every outcome. A nonzero exit
can report a failed behaviour hypothesis, not just a setup problem. Preserve
that evidence; do not change limits to obtain passing results. The base-model
results remain separate. [Experiment 008](experiments/008-selected-checkpoint-behaviour.md)
records the selected-checkpoint outcomes and its two-thread configuration copy.
Use its process-group wrapper for the cooperative invariance deadline: 300 seconds
inside the runner and a 330-second external hard limit.

Historical training and held-out model results are in
[Experiment 004](experiments/004-chronos-fine-tuning.md). This replication route
does not retrain a model, rerun the consumed demand-test comparison, deploy an
API, or write to the shared experiment store.

## Local monitoring and release route

The [scenario guide](monitoring-scenarios.md#local-execution) provides bounded
in-process controls that load no checkpoint. Use a fresh output directory for
each run. Fresh Alibi uses the separate locked drift environment; Docker scenarios
require an inspected compatible cached image and unique owned target identity.
Recorded compact local results are in
[`monitoring/scenarios/local-evidence-2026-11-12.json`](../monitoring/scenarios/local-evidence-2026-11-12.json).
The selected-model, registry and target checks are separate:

1. Obtain human peer review and merge the applicable dependency PRs, then pin the
   exact clean integrated source. A changed commit needs matching release evidence.
2. With separate execution authorization, run
   [bounded release validation](release-validation.md#manual-github-workflow)
   on that source and retain the authenticated run ID, attempt, artifact ID/digest
   and complete successful handoff. A historical local pull or fast PR CI does
   not satisfy this gate.
3. Follow [private image publication](image-publication.md), retaining the
   validated source, private metadata observations, immutable image digest and
   trusted manifest. First publication is explicit and off by default. A 404 is
   unobserved access/existence evidence.
4. Verify independent reader access and native Linux amd64 selected-model startup
   from that digest. Check `/health`, `/ready`, `/forecast` and invalid input;
   retain actual 200/200/200/422 outcomes, selected weights/run and settings.
5. Halt any UPC/VPN-dependent step and wait for the user. After separately
   authorized access and target binding, use the exact
   [deployment/recovery runbook](deployment-release.md) with a published baseline,
   persistent lock and expected-current fence; capture restored forecasts and
   monitoring timestamps. Recorded status alone does not establish target health.

Do not execute target steps merely to test access. This documentation task
does not initiate VPN/SSH/tunnels, publish an image, submit artifacts, train a
model or repeat final-test tuning. The [delivery checklist](final-delivery.md)
records which clean setup/retrieval checks actually ran for this draft.
