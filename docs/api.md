# Local forecasting API

The stateless FastAPI service accepts a state and 512 demand observations and
returns 48 positional p10, p50 and p90 forecasts. It reuses the
[selected-checkpoint inference module](inference.md). The state labels the
request; it is not an input feature or a selector for separate models.

## Startup

From the repository root, install the locked dependencies and retrieve the
selected artifact using your authorized DVC access:

```sh
uv sync --locked --dev --extra forecast
uv run dvc pull models/chronos-t5-small-fine-tuned.dvc
uv run --extra forecast uvicorn src.api.app:create_app --factory \
  --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000/docs` for the generated OpenAPI interface. The
startup command binds only to the local machine. Containers and hosted
deployment have separate configuration and verification requirements.

`create_app` captures these environment settings once. The lifespan reads the
forecast parameters and creates one `ChronosForecaster`, reused for requests.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FORECAST_PARAMS_PATH` | Repository `params.yaml` | Forecast and sampling settings; the contract remains 512 inputs and 48 outputs. |
| `FORECAST_CHECKPOINT` | Repository `models/chronos-t5-small-fine-tuned` | Relocate the selected artifact; its weights must match the tracked provenance sidecar. |
| `FORECAST_CPU_THREADS` | `2` | Positive integer for Torch CPU threads. |

Relative overrides resolve from the working directory. Keep the artifact and
parameters unchanged during serving. Loading is local only: absent weights,
checksum mismatch, invalid parameters or a loading failure abort startup and
Uvicorn exits unsuccessfully. There is no base-model or test-predictor fallback.
Use one worker to keep one model instance and one CPU budget; additional workers
would each load their own model.

## Requests and responses

`POST /forecast` takes a JSON object with only `state` and `history`:

- `state`: exactly `NSW`, `VIC`, `QLD`, `SA` or `TAS`.
- `history`: exactly 512 finite JSON numbers, oldest first. Integers and decimal
  numbers are accepted; booleans, numeric strings, nulls, nested arrays and
  values outside float32's finite range are rejected. Valid negatives are
  preserved, including TAS observations.

Generate complete example request JSON and save a complete response:

```sh
mkdir -p reports/inference
python - <<'PY'
import json
import math
from pathlib import Path

history = [15 + 20 * math.sin(2 * math.pi * i / 48) for i in range(512)]
Path("reports/inference/api-request.json").write_text(
    json.dumps({"state": "TAS", "history": history}) + "\n"
)
PY
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/forecast \
  -H 'Content-Type: application/json' \
  --data-binary @reports/inference/api-request.json \
  -o reports/inference/api-response.json
python -m json.tool reports/inference/api-response.json
```

Each successful response contains these fields:

| Field | Value |
| --- | --- |
| `state` | The request's state. |
| `p10`, `p50`, `p90` | 48 finite numbers each, with p10 <= p50 <= p90 at each position. |
| `model_id` | `chronos-t5-small-fine-tuned`. |
| `model_revision` | `local-sha256:<actual weights SHA-256>`. |
| `checkpoint` | The actual weight `sha256` and verified `training_run_id`. |
| `value_scale` | `source`. |
| `context_length`, `horizon` | `512`, `48`. |
| `settings` | Executed sample count, seed, device, CPU threads, dtype, sampling parameters, quantile method and point statistic. |

The API validates the predictor's output and returns only these public fields.
It does not substitute a configured name for the loaded artifact's identity.
Forecast positions 1 through 48 follow the last history observation; no
timestamps, clock alignment or unit conversion are inferred. Source scale does
not claim independently verified MW units. See the
[dataset card](DATASET_CARD.md) for those limits.

## Health and failures

| Request or condition | HTTP result |
| --- | --- |
| `GET /health` | `200 {"status":"ok"}`: process health only; no inference. |
| `GET /ready` | `200 {"status":"ready"}` when the lifespan has loaded a predictor. |
| No loaded predictor | `/ready` and a valid `/forecast` return `503 {"detail":"Model unavailable"}`. |
| Invalid request | `422` with `detail` entries containing `loc`, `msg` and `type`. |
| Prediction raises an error | `503 {"detail":"Forecast unavailable"}`. |
| Invalid prediction shape, values or identity | `500 {"detail":"Invalid forecast output"}`. |

For example, a numeric string in the first history position produces:

```json
{"detail":[{"loc":["body","history",0],"msg":"Input should be a valid number","type":"float_type"}]}
```

Validation responses omit submitted values and exception context; inference
responses omit internal paths, credentials and exception messages. An inference
error does not unload or recreate the model. Readiness reports a loaded model,
not a guarantee that every input will infer successfully. Startup must complete
before Uvicorn serves requests; unavailable lifecycle states are also exercised
directly in tests. Shutdown clears the application's predictor reference.

The synchronous forecast handler runs in FastAPI's thread pool; health and
readiness use asynchronous handlers. The inference module serializes its Torch
operations to protect process-global sampling state. The API has no history
store, training endpoint, retry loop or automatic reload.

## Verification

```sh
uv run pytest tests/test_api.py -q
make lint
make coverage
```

Tests inject a predictor factory through `create_app(settings, predictor_factory)`
and enter the same lifespan used by Uvicorn. They require no model weights or
network. They cover validation, output identity and quantile checks, lifecycle
reuse, fail-fast startup, controlled failures and probes during active inference.
The test factory is an explicit Python argument; deployment settings cannot
enable it.

Implementation references: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
[synchronous path operations](https://fastapi.tiangolo.com/async/#path-operation-functions),
and [Pydantic numeric validation](https://docs.pydantic.dev/latest/api/standard_library_types/#floats).
