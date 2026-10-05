# Baseline and Chronos validation comparison

The evaluator for [#12](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/12)
uses the [agreed protocol](problem-definition.md) and the split implementation
merged in [PR #27](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/27).
It reads the DVC training and validation blocks. It never opens final test targets.

## Reproduction

1. Install `uv sync --locked --dev --extra forecast`. The optional `forecast`
   dependencies are needed for Chronos; baseline execution and CI use the base environment.
   Linux resolves PyTorch from its CPU index to avoid installing CUDA packages.
2. Restore the data with `uv run dvc pull -r origin` using your own DagsHub credentials.
   Check `uv run dvc status`; the evaluator rejects missing or modified DVC inputs.
3. Configure your own MLflow credentials as described in [the setup guide](mlflow.md).
4. Run `uv run --extra forecast python -m src.modeling.evaluate`. Each invocation
   creates new runs and a new ignored directory under `reports/evaluation/`.

DVC authenticates separately from MLflow. If the CLI is not already configured,
the same local `.env` credentials can be passed to DVC in memory:

```sh
uv run python - <<'PY'
import os
from dvc.repo import Repo
from src.config import PROJ_ROOT

remote = {
    "url": "https://dagshub.com/pauadal03/MLOps_Endesa.dvc",
    "auth": "basic",
    "user": os.environ["MLFLOW_TRACKING_USERNAME"],
    "password": os.environ["MLFLOW_TRACKING_PASSWORD"],
}
with Repo(str(PROJ_ROOT), config={"remote": {"origin": remote}}) as repo:
    repo.pull(remote="origin")
PY
```

This uses the constructor's configuration override; changing `repo.config`
after initialization can leave the remote filesystem using its earlier settings.
The command does not save credentials to DVC configuration or shell history.

For a fast reproduction of one real baseline, run:

```sh
uv run python -m src.modeling.evaluate --method seasonal_lag336
```

Use `--tracking-uri sqlite:///mlflow.db` for an explicitly local run. Shared
results must use the configured DagsHub server. The supported methods are `all`,
`seasonal_lag48`, `seasonal_lag336`, and `chronos`.

## Method and recorded evidence

Every method predicts the same 48-step targets at all 365 validation origins for
each of the five states: 1,825 forecasts and 87,600 point predictions per method.
Contexts contain the previous 512 observations and may include earlier observed
validation values. The lag-48 and lag-336 baselines repeat their last known cycle.
Chronos uses the pinned revision in `params.yaml`, 20 sampled trajectories and
their median as the point forecast. No project training or fitted preprocessing occurs.

Each state's MAE is reported in the source scale. Its MASE denominator is the
mean lag-336 absolute difference over training observations only, regardless of
which baseline wins. Mean MASE gives each state equal weight. The worst-origin
error share uses the largest `ceil(0.05 * origins)` origin errors: 19 of 365 here.
The primary baseline is selected by validation mean MASE, with lag 48 winning a tie.
This selection does not inspect the test block or select a Chronos configuration.

MLflow records the full protocol, model/data/code versions, seed, sample count,
device, float32 dtype, batch size, CPU thread count and dependency environment.
All runs in a comparison share a group ID and a hash of their state/origin/target
table. Output artifacts include `predictions.parquet`, `metrics.json`, training
MASE scales, parameters, DVC references, and the tracking helper's environment
and version files. Local `comparison.json` links the run IDs and selected baseline.

`model_load_seconds` includes checkpoint retrieval/loading. `inference_seconds`
measures the forecast loop, including batch preparation, result transfer, median
calculation and progress logging. It excludes MLflow upload time; the tracking
helper's `run_body_seconds` includes uploads. These are wall-clock measurements,
not isolated kernel latency or energy measurements. The initial local budget is
one hour, batch size 16 and four CPU threads, recorded in `params.yaml`.
The deadline is checked between batches and before recording a complete result;
an over-budget execution fails rather than reporting a partial comparison.

## Interpretation and limitations

The source's physical unit, time zone and individual civil timestamps remain
unverified in [#6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6).
Predictions therefore use positional time and the source scale. Holiday and
day-after-holiday error breakdowns from the protocol remain unavailable until a
verified calendar mapping exists; no holiday labels are inferred from positions.
Worst-origin error shares are available independently of that mapping.

The [Chronos paper, dataset table](https://arxiv.org/html/2403.07815v2#A2.T2)
lists Australian Electricity under zero-shot evaluation datasets, separate from
its training datasets. The [pinned model card](https://huggingface.co/amazon/chronos-t5-small/tree/a971ba21945c4f1796b17a91fe69214b5f4ad472)
points to that paper for training disclosure. This is upstream disclosure, not
an independent audit of the checkpoint's pretraining records. We describe it as
untuned in this project and retain that limitation when interpreting results.
Our validation period and metrics also differ from the paper's benchmark.

A fixed seed makes reruns comparable under the recorded software, device, batch
size and origin order. It does not establish bitwise equality across hardware.
A teammate must reproduce at least one real result and review the interpretation
before #12 is complete. Fine-tuning is tracked in #13 and final model-card evidence in #7.

The Chronos adapter follows the [upstream usage example](https://huggingface.co/amazon/chronos-t5-small/tree/a971ba21945c4f1796b17a91fe69214b5f4ad472),
with an explicit model revision and shared forecast targets.
