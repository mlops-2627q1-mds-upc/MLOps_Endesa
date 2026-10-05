# MLflow experiment tracking

Tracking setup for [#12](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/12).
The shared server is the existing DagsHub repository used by DVC. This adapts the
[course MLflow demo](https://github.com/mlops-2627q1-mds-upc/MLOps-2627q1-demos/blob/main/docs/mlflow-demo.md)
with explicit forecasting metadata and local tests.

## Shared setup

1. From the repository root, run `uv sync --locked --dev`.
2. Copy `.env.template` to `.env` and fill in your own DagsHub username and access
   token. `MLFLOW_TRACKING_PASSWORD` holds the token. The tracking URI is
   `https://dagshub.com/pauadal03/MLOps_Endesa.mlflow`; `pauadal03` owns the
   repository, but each contributor authenticates as themselves and needs write
   access to it. Keep `.env` local; the template contains no credentials.
3. Run the smoke command:

   ```sh
   uv run python -m src.modeling.tracking_smoke
   ```

The command creates a four-row synthetic run in `MLOps_Endesa-smoke`, reads its
parameters and metrics back, and downloads the predictions CSV to verify its
contents. It prints the run ID and `verified: true` only after those checks pass.
It performs no data download, model loading, or training. Its `smoke.*` metrics
are fixture checks, not electricity-demand forecast results.

Open [DagsHub Experiments](https://dagshub.com/pauadal03/MLOps_Endesa/experiments)
and select the run ID. Each invocation creates a new run and preserves previous
records. The synthetic fixture is deterministic; a teammate can repeat this
command and check the same fixture values under their own account.

The project loads `.env` from the repository root. Variables already supplied
by the shell or notebook take precedence. For a notebook, load the project
configuration before calling the tracking helper. Do not put credentials in
notebook cells or in experiment settings.

## Local check

An explicit local store works without DagsHub credentials:

```sh
uv run python -m src.modeling.tracking_smoke --tracking-uri sqlite:///mlflow.db
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the URL printed by the UI command. `mlflow.db`, its journal files, and local
MLflow artifact directories are ignored by Git. A local run demonstrates local
logging only; the shared smoke command verifies DagsHub access separately.
With no tracking URI, the helper reports a configuration error instead of
silently using a local store.

## Logging real evaluations

Use `src.tracking.tracked_run` around the evaluator's execution. Supply a run
name, the exact experiment configuration, the data version actually loaded, and
the model revision actually used. Call `run.log_metrics({...})` with a batch of
per-state/aggregate errors or timed measurements, and `run.log_artifact(path)`
for predictions, plots, or reports. The returned `run.client` and `run.run_id`
also support the normal MLflow client API.

| Recorded information | Source |
| --- | --- |
| Git commit and dirty flag | Current checkout, including untracked source changes |
| Data version | Caller-supplied DVC reference; for the current five-series artifact, read `data/raw.dvc` |
| Model revision | Caller-supplied checkpoint revision, or an explicit algorithm version for a seasonal baseline |
| Configuration and searchable parameters | Caller-supplied JSON settings: method, seed, split boundaries, origins, horizon, context, sample count, device, and dtype as applicable |
| Environment | Python version, OS/architecture, installed package names/versions, lockfile and its SHA-256 |
| Metrics and output files | Caller-supplied results; metric writes are synchronous and batched |
| Execution status | `FINISHED` on success, `FAILED` when setup/logging or execution raises after run creation |

The helper uses an explicit client and run ID so it does not replace a notebook's
active MLflow run or global tracking URI. It records configuration and environment
once per run. Nonfinite metrics are rejected before writing the batch. An error
still reaches the caller after the run is marked failed. `run_body_seconds`
measures the context body, including its logging calls; measure model inference
separately when reporting latency.

Artifacts `config.json`, `provenance.json`, `environment.json`, and `uv.lock`
describe the recorded run. Environment variables, `.env`, Git diffs, and raw
datasets are not automatically uploaded. Supply only experiment settings in
`config` and deliberately selected output files as artifacts.

Before a real evaluation, retrieve the intended DVC artifact and check
`uv run dvc status`. The helper records the supplied version reference; it does
not validate the contents of locally modified data or pin/load a model for you.
Use the pinned model revision in [the model card](MODEL_CARD.md).

The split and forecast-origin implementation in
[#17](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/17) is merged.
Use the [validation evaluator](evaluation.md) to record real baseline/Chronos comparisons.
Record the selected evaluation settings and compute budget before running it.
Use identical validation targets for each method and keep final test targets out
of model selection. Keep source-unit/calendar and upstream pretraining limits
with the interpretation. [#5](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/5)
can remain open while the cards evolve.

## Validation

```sh
make lint
make test
```

Tracking tests use temporary SQLite stores, local artifacts, and a small Git
checkout. They exercise readback, configuration/version records, metric history,
artifact contents, dirty changes, failed executions, nonfinite values, missing
configuration, notebook state, and the smoke command. They require neither
remote credentials nor raw data/model weights. Shared run IDs and verification
results are recorded in the implementation PR and #12.

On 2026-10-02, shared run
[`eb71bd9c444c4af39fb4857bec087ac1`](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/0/runs/eb71bd9c444c4af39fb4857bec087ac1)
finished in `MLOps_Endesa-smoke` with MLflow 3.16.1 at clean Git commit
`501fd85f53657aa5d0b012f8ced8eecf86d637bf`. Parameter/metric readback and all five
artifact downloads were verified: predictions, configuration, provenance,
environment, and lockfile. The fixture produced `smoke.rows = 4` and
`smoke.mae = 0.75`; these are synthetic verification values. This checks shared
tracking access, not model performance or a second teammate's reproduction.

See [EDN-002](edn/0002-mlflow-tracking.md) for the logging choice and AI involvement.
Setup alone does not complete #12: comparable real runs and teammate reproduction
of a recorded result are still required.

References: [MLflow client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)
and [DagsHub MLflow integration](https://dagshub.com/docs/integration_guide/mlflow_tracking/).
