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

A complete response from the smoke test below, with quantiles rounded to three
decimal places for readability:

```json
{
  "state": "TAS",
  "p10": [
    -2.478, -3.593, -4.348, -4.955, -4.955, -4.832, -4.336, -3.593,
    -2.366, -0.867, 0.756, 2.614, 4.658, 6.876, 9.366, 11.98,
    14.606, 17.232, 19.735, 21.99, 24.443, 26.301, 27.515, 28.345,
    29.856, 30.686, 31.516, 32.123, 31.826, 30.872, 30.067, 29.435,
    28.258, 27.317, 26.734, 25.396, 24.133, 21.94, 19.574, 16.861,
    14.371, 11.967, 9.762, 7.284, 4.931, 2.837, 0.731, -0.88
  ],
  "p50": [
    -2.354, -3.469, -4.336, -4.832, -4.955, -4.832, -4.336, -3.469,
    -2.354, -0.867, 0.867, 2.849, 4.955, 7.309, 9.787, 12.389,
    14.99, 17.592, 20.069, 22.671, 24.901, 26.759, 28.122, 29.732,
    31.281, 32.458, 33.325, 34.316, 34.75, 34.812, 34.316, 33.635,
    32.644, 31.9, 30.724, 29.113, 27.131, 25.025, 22.547, 20.069,
    17.468, 14.866, 12.389, 9.787, 7.309, 4.955, 2.849, 0.867
  ],
  "p90": [
    -2.218, -3.469, -4.336, -4.832, -4.955, -4.819, -4.212, -3.456,
    -2.23, -0.867, 0.991, 2.849, 4.955, 7.309, 9.787, 12.512,
    15.003, 17.716, 20.206, 22.671, 25.025, 26.908, 28.865, 30.513,
    32.0, 32.966, 33.845, 34.812, 34.936, 35.06, 34.688, 34.068,
    33.449, 32.334, 30.86, 29.299, 27.564, 25.595, 22.944, 20.255,
    17.778, 15.188, 12.686, 10.444, 9.552, 7.185, 4.732, 2.18
  ],
  "model_id": "chronos-t5-small-fine-tuned",
  "model_revision": "local-sha256:746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9",
  "checkpoint": {
    "sha256": "746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9",
    "training_run_id": "639eba4cc7e74de08ffab64beee715db"
  },
  "value_scale": "source",
  "context_length": 512,
  "horizon": 48,
  "settings": {
    "num_samples": 20,
    "seed": 7,
    "device": "cpu",
    "cpu_threads": 2,
    "dtype": "float32",
    "sampling": {
      "temperature": 1.0,
      "top_k": 50,
      "top_p": 1.0
    },
    "quantile_method": "linear",
    "point_statistic": "median"
  }
}
```

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

### Selected-model HTTP smoke, 2026-10-23

Clean source `a80daf7e0728ee3d68b3d0afd06bca61b28831d1` was tested with one
Uvicorn process, offline model loading, two CPU threads and two requests using
the synthetic TAS history above. `/health` and `/ready` returned 200; both
forecasts returned identical finite, ordered 48-element arrays and the full
executed settings. The server was stopped after the check.

The actual weights SHA-256 was
`746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9`,
verified against the selected provenance; training run
`639eba4cc7e74de08ffab64beee715db`. Settings were 20 samples, seed 7,
CPU float32, two threads, temperature 1, top-k 50, top-p 1, linear quantiles
and median p50.

On an Apple M4 with macOS 26.2 and Python 3.11.15, startup to process health was
3.487 s; the first request took 0.646 s and the second, warm request 0.633 s.
Versions were FastAPI 0.142.2, Uvicorn 0.54.0, Pydantic 2.13.5, Torch 2.14.1,
Chronos Forecasting 2.3.2 and Transformers 5.18.0. These two observations check
the local interface; they do not establish a latency target, load capacity,
forecast accuracy or deployment readiness.

To repeat the bounded requests after starting the server and generating the
example JSON above:

```sh
for attempt in 1 2; do
  curl --fail-with-body --silent --show-error http://127.0.0.1:8000/forecast \
    -H 'Content-Type: application/json' \
    --data-binary @reports/inference/api-request.json \
    -o "reports/inference/api-response-$attempt.json" \
    -w 'request_seconds=%{time_total}\n'
done
cmp reports/inference/api-response-1.json reports/inference/api-response-2.json
```

The smoke used `FORECAST_CHECKPOINT` to read the existing selected artifact from
another local checkout, without copying or changing it. A separate Uvicorn check
with missing selected weights exited with status 3 before serving requests.
Local test results were 85 API cases passed; full coverage ran 391 tests with
102 skipped, 86% total coverage and 100% API coverage. Lint passed. Required CI
checks and human peer review remain separate gates.

Implementation references: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
[synchronous path operations](https://fastapi.tiangolo.com/async/#path-operation-functions),
and [Pydantic numeric validation](https://docs.pydantic.dev/latest/api/standard_library_types/#floats).
