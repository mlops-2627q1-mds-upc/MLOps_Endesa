# Inference smoke check

On 2026-10-11, the pinned untuned Chronos checkpoint passed a local CPU smoke
check for [#48](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/48).
The input was a synthetic 512-value sequence with one negative value. This
checks the inference interface; it provides no accuracy or interval-coverage result.

Source commit: `3fa698f01f1d4883e9a5ff0cb44d485b06163bf8`. The
[recorded settings, hashes and results](../evidence/issue-48-inference-smoke.json)
identify the source, input and model revision. The final documentation commit
keeps the prediction module, parameters and dependency lock unchanged.

| Check | Observed result |
| --- | --- |
| Model | `amazon/chronos-t5-small`, revision `a971ba21945c4f1796b17a91fe69214b5f4ad472` |
| Environment | Apple M4, macOS 26.2 arm64, Python 3.11.15; locked forecasting extra |
| Loading | Cached checkpoint only, with `token=False` and `local_files_only=True`; construction took 2.189 s |
| Reuse | Two predictions used the same loaded pipeline |
| Output | Three finite, ordered quantile arrays of 48 values each |
| Repeatability | Repeated input produced identical results; caller CPU RNG state was preserved |
| Prediction durations | 0.578 s and 0.536 s |
| CLI | A separate CLI invocation returned the same JSON result |
| Fast checks | Local lint and all 82 tests passed, including 28 inference cases |

The prediction durations measure `predict(history)`, including validation and
quantile calculation. They exclude loading and JSON encoding. These are two
local calls, not an API latency benchmark or a service-level target. Peer review
and the #49 API implementation remain pending.

## Reproduction

From the checkout root, install `uv sync --locked --dev --extra forecast` and
generate the same input:

```sh
uv run --extra forecast python - <<'PY'
import json
from pathlib import Path
import numpy as np

positions = np.arange(512, dtype=np.float32)
history = (5000 + 300 * np.sin(2 * np.pi * positions / 48)).astype(np.float32)
history[0] = -3
directory = Path('reports/inference/issue-48')
directory.mkdir(parents=True, exist_ok=True)
(directory / 'history.json').write_text(json.dumps(history.tolist(), allow_nan=False) + '\n')
PY
```

Run the CLI against the cached checkpoint:

```sh
HF_HUB_OFFLINE=1 uv run --extra forecast python -m src.modeling.predict \
  reports/inference/issue-48/history.json --local-files-only \
  --output-path reports/inference/issue-48/forecast.json
```

If the pinned checkpoint is not cached, load it once using the
[normal invocation](../inference.md#usage), then repeat the offline check.
To check reuse and RNG state with the same input:

```python
import json
from pathlib import Path
import torch
from src.modeling.predict import ChronosForecaster

history = json.loads(Path('reports/inference/issue-48/history.json').read_text())
forecaster = ChronosForecaster(local_files_only=True)
state = torch.get_rng_state().clone()
first = forecaster.predict(history)
assert torch.equal(state, torch.get_rng_state())
assert first == forecaster.predict(history)
assert torch.equal(state, torch.get_rng_state())
```
