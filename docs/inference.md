# Reusable Chronos inference

Issue [#48](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/48) adds
`ChronosForecaster` in `src/modeling/predict.py`. Create one instance at application
startup and reuse its `predict(history)` method. The checkpoint is loaded once
per instance. The initial implementation uses CPU, matching `params.yaml`.

The input is one array of 512 numeric, finite observations in chronological
order, oldest first. Values remain in the stored source scale, including valid
negative TAS values. Strings, booleans and values outside float32's finite range
are rejected. State validation belongs to the API in
[#49](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/49).

Each result contains 48 values in `p10`, `p50` and `p90`, the model ID/revision,
context length, horizon and prediction settings. Forecast positions follow the
last supplied observation; no civil timestamps or unit conversions are inferred.
The quantiles follow the
[proposal in PR #30](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/30#issuecomment-6111711492),
which remains under team discussion. Interval coverage is not measured here.

## Usage

Install the locked environment with `uv sync --locked --dev --extra forecast`.
The loader follows the
[upstream Chronos example](https://github.com/amazon-science/chronos-forecasting/tree/v2.3.2),
using the exact checkpoint revision from `params.yaml` and `token=False`.

```python
from src.modeling.predict import ChronosForecaster, ForecastConfig

forecaster = ChronosForecaster(ForecastConfig.from_params())
result = forecaster.predict(history)  # history: 512 observations
```

For the command line, put that numeric array in `history.json`:

```sh
uv run --extra forecast python -m src.modeling.predict history.json \
  --output-path reports/inference/forecast.json
```

Omit `--output-path` to print JSON. Add `--local-files-only` to require the pinned
checkpoint to be in the local Hugging Face cache. Missing weights and prediction
failures raise `RuntimeError`; malformed histories raise `ValueError`. The CLI
reports an error and exits with status 1. Local outputs under `reports/inference/`
are ignored by Git.

The model uses 20 sampled trajectories and the evaluation's temperature 1,
top-k 50 and top-p 1. `p50` uses the same sample-median helper as evaluation;
`p10` and `p90` use linear interpolation. Calls on one instance are serialized.
Each call starts at the configured seed and restores the caller's CPU RNG state.
This makes repeated inputs comparable in the recorded environment; it does not
promise identical predictions across hardware or dependency versions.

The module performs no training or evaluation and does not access DVC or MLflow.
Fine-tuned artifact loading and final model selection will be checked when
[#13](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/13) produces a
reviewed checkpoint. API endpoints and hosted deployment remain separate work.
