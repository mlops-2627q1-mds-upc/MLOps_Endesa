# Reusable Chronos inference

`ChronosForecaster` in `src/modeling/predict.py` loads the selected fine-tuned
checkpoint once per instance. Create one at application startup and reuse
`predict(history)`. The default is `models/chronos-t5-small-fine-tuned`, selected
under [FR-10](requirements.md#modelling-and-evaluation). There is no fallback to the base
model if this artifact is missing, invalid or cannot be loaded.

The input is exactly 512 numeric, finite observations in chronological order,
oldest first. Values stay in the stored source scale, including valid negative
TAS values. Strings, booleans and values outside float32's finite range are
rejected. Each result contains 48 values in `p10`, `p50` and `p90`. Forecast
positions follow the last observation; no timestamps or unit conversions are
inferred. State validation belongs to the API in
[#49](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/49).

## Selected checkpoint and identity

Retrieve the [DVC artifact](../models/chronos-t5-small-fine-tuned.dvc) using your
own authorized DVC access, then install the locked forecasting environment:

```sh
uv sync --locked --dev --extra forecast
uv run dvc pull models/chronos-t5-small-fine-tuned.dvc
```

The loader uses `describe_checkpoint` from `src/modeling/checkpoints.py` to verify
the weights against the tracked
[provenance sidecar](../models/chronos-t5-small-fine-tuned.provenance.json).
Missing or malformed provenance, changed weights and conflicting training run
IDs fail before model loading. Local loading is always offline. The checkpoint
must be a full Chronos directory, including its model configuration and weights;
keep it unchanged throughout the serving process lifetime.

The response reports these identity fields without local filesystem paths:

- `model_id`: `chronos-t5-small-fine-tuned`.
- `model_revision`: `local-sha256:<weights hash>`.
- `checkpoint`: `sha256` and `training_run_id`, checked against the sidecar.
- `value_scale`: `source`.

For the currently selected artifact the weight hash is
`746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9`
and training run is `639eba4cc7e74de08ffab64beee715db`. The response also includes
`context_length`, `horizon` and the executed prediction `settings`.

## Usage

```python
from src.modeling.predict import ChronosForecaster

forecaster = ChronosForecaster()
result = forecaster.predict(history)  # history: 512 observations
```

For the command line, save that numeric array in `reports/inference/history.json`:

```sh
uv run --extra forecast python -m src.modeling.predict reports/inference/history.json \
  --output-path reports/inference/forecast.json
```

Omit `--output-path` to print JSON. `--checkpoint /path/to/checkpoint` can relocate
the selected artifact; it must still match the repository's selected sidecar.
The Python equivalent is `ForecastConfig.from_params(checkpoint=Path(...))`.
This option does not select arbitrary alternative weights.

For an explicit base-model comparison, use `--base-model --local-files-only`, or
`ChronosForecaster(ForecastConfig.from_params(checkpoint=None), local_files_only=True)`.
This loads the exact `evaluation.model_id` and `evaluation.model_revision` from
`params.yaml`, reports those Hub values, and returns `checkpoint: null`. The
`--local-files-only` flag requires an existing Hub cache; omitting it permits a
Hub download in explicit base-model mode. `--base-model` and `--checkpoint`
cannot be combined. The base model is not the selected FR-10 serving artifact.

Missing weights, identity failures, loading failures and malformed model output
raise `RuntimeError`; invalid histories or configurations raise `ValueError`.
The CLI prints an error and exits with status 1. Local outputs under
`reports/inference/` are ignored by Git.

## Sampling and concurrency

The module uses the evaluation's seed and sample count (currently 7 and 20),
CPU float32, temperature 1, top-k 50 and top-p 1. Serving defaults to two CPU
threads; `--cpu-threads` or `ForecastConfig.from_params(cpu_threads=...)` sets
that budget explicitly without changing the evaluation configuration.
`p50` uses evaluation's sample-median helper; `p10` and `p90` use linear
interpolation. Non-numeric, nonfinite, wrongly shaped or unordered outputs fail.

Model loading and predictions are serialized across forecaster instances in one
process because Torch's CPU RNG and thread setting are shared. Each operation
restores the previous CPU RNG state, including on failure; each prediction uses
the configured seed. This protects calls through this module. Other code using
Torch's global generator concurrently must coordinate with the serving process.
Repeated inputs are comparable within the same environment; predictions need
not be identical across hardware or dependency versions.

The module does not train, evaluate accuracy, publish artifacts, or contact
MLflow. Interval coverage and hosted deployment remain separate checks. The
[2026-10-11 smoke record](experiments/003-inference-smoke.md) describes only the
historical base-model implementation; it is not evidence for the selected
fine-tuned artifact.
