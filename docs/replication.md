# Replicating the first delivery

Use the exact reviewed commit linked from the report cover. `main` and task
branches can move; record `git rev-parse HEAD` before running anything. The
submission tag is created only after review and merge. The report and slides
must identify the same version.

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

## Retrieve and validate artifacts

DVC and MLflow authentication are separate. Use your own authorized DagsHub
account; [the evaluation guide](evaluation.md#reproduction) provides an in-memory
DVC configuration example. Keep credentials out of Git and commands saved in
shell history. Once authentication is available:

```sh
uv run dvc pull -r origin
uv run dvc status
uv run dvc repro
make coverage
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

Install the locked forecast extra and run the existing declared synthetic cases:

```sh
uv sync --locked --dev --extra forecast
uv run --extra forecast python -m src.modeling.minimum_functionality --checkpoint models/chronos-t5-small-fine-tuned --max-seconds 300
uv run --extra forecast python -m src.modeling.directional --checkpoint models/chronos-t5-small-fine-tuned --max-seconds 180
uv run --extra forecast python -m src.modeling.invariance --checkpoint models/chronos-t5-small-fine-tuned --max-seconds 300
```

Run these sequentially. They use synthetic histories and record the actual
checkpoint digest, settings, declared limits and every outcome. A nonzero exit
can report a failed behaviour hypothesis, not just a setup problem. Preserve
that evidence; do not change limits to obtain passing results. The base-model
results remain separate. Supervise the invariance command with an external
300-second timeout because its internal deadline is cooperative.

Historical training and held-out model results are in
[Experiment 004](experiments/004-chronos-fine-tuning.md). This replication route
does not retrain a model, rerun the consumed demand-test comparison, deploy an
API, or write to the shared experiment store.
