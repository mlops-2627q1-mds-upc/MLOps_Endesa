# Experiment 008: Behaviour of the selected checkpoint

On 17 October 2026, the selected fine-tuned checkpoint passed the existing
minimum-functionality, directional and invariance checks. Cases, ordering, seeds
and thresholds were retained from the pinned-base protocols in
[Experiment 005](005-minimum-functionality.md),
[Experiment 006](006-forecast-invariance.md) and
[Experiment 007](007-directional-expectations.md). These results concern the
selected artifact; peer review of the extension and outcomes is pending.

## Artifact and execution

| Item | Recorded value |
| --- | --- |
| Clean source commit | `81c5b82893ba1bb5c44409b48fe85908919a3931` |
| Local checkpoint | `models/chronos-t5-small-fine-tuned` |
| Weights SHA-256 | `746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9` |
| Training run | `639eba4cc7e74de08ffab64beee715db` |
| Inference | CPU float32; two CPU threads; batch size 1; 20 trajectories; seed 7 |
| Sampling | Temperature 1, top-k 50, top-p 1; interpolated sample median |
| Model access | Restored local weights; Hugging Face and Transformers offline flags enabled |

Training seed 8 identifies the selected artifact; seed 7 governs these checks.
Minimum functionality sets that seed once for its ordered cases. Directional
and invariance reset it before each history to match random draws. Reports
identify the executed local path and weights hash separately from the unused
Hub defaults in `settings`. The macOS arm64 environment records Chronos 2.3.2,
PyTorch 2.14.1, Transformers 5.18.0 and NumPy 2.4.6.

## Results and scope

| Suite | Outcomes | Reported runtime | Declared cap |
| --- | --- | ---: | ---: |
| [Minimum functionality](../evidence/selected-checkpoint-2026-10-17/minimum-functionality.json) | 10/10 cases passed | 11.732 s | 300 s |
| [Directional](../evidence/selected-checkpoint-2026-10-17/directional.json) | 12/12 changes and unchanged control passed | 12.720 s | 180 s |
| [Invariance](../evidence/selected-checkpoint-2026-10-17/invariance.json) | 70/70 pairs passed; zero counterexamples | 95.380 s | 300 s |

Minimum functionality checks three constant levels, zero demand and six
repeating patterns against known synthetic continuations. Directional checks
use a positive synthetic pattern with sustained increases and decreases. The
unchanged control's maximum absolute forecast difference was 0.0.

Invariance uses **14 histories**: four synthetic patterns (constant, seasonal,
trending and signed), plus ten stored windows: one training and one validation
window for each of NSW, VIC, QLD, SA and TAS. Five transformations per history
produce 70 pairs and 140 forecasts. Each stored window contains 512 observations
ending before its recorded origin. Only `train.parquet` and `validation.parquet`
are read; no demand-test targets are scored. Real windows are represented by
positions, aggregates and hashes; original observations are not exported.
Synthetic inputs and model forecasts are retained.

All 14 histories passed each transformation. The largest mean / maximum relative
forecast differences were 0 / 0 for scales 0.5 and 2; approximately
`1.19e-7` / `2.65e-7` for scale 10; 0.0111 / 0.0513 for 0.1% noise; and
0.0333 / 0.1066 for 1% noise. Scale limits remain `1e-5` / `1e-4`, and noise
limits remain 0.05 / 0.15.

These bounded cases do not establish general robustness, operational monotonicity,
interval coverage or demand forecast accuracy. No new training or energy measurement occurred. Durations are single-run observations,
not latency benchmarks. Historical performance and energy figures remain in
[Experiment 004](004-chronos-fine-tuning.md).

## Resource adjustment and retained interruption

The first invariance attempt read `cpu_threads: 4` from tracked `params.yaml`.
PyTorch's explicit setting overrode the environment's two-thread limit. Its
process group was stopped with SIGTERM after **54.624 seconds**, with 88 of
140 origin calls logged and no scored pairs returned. Its
[prepared report](../evidence/selected-checkpoint-2026-10-17/invariance-interrupted-prepared.json)
and [interruption record](../evidence/selected-checkpoint-2026-10-17/invariance-interruption.json)
are retained. This incomplete attempt is neither a passing run nor a model failure.

The completed attempt used an ignored configuration copy changing only
`evaluation.cpu_threads` from 4 to 2. Its
[resource manifest](../evidence/selected-checkpoint-2026-10-17/invariance-resource-manifest.json)
and exact [configuration copy](../evidence/selected-checkpoint-2026-10-17/params-cpu2.yaml)
record the difference. The original file's SHA-256 is
`a470aba4cedc7940052139315e4e03e5db5b26899d57d508ecfc693c6d79343c`;
the copy's is
`40deb953aa41e46927c9d1f19dbe65679992e5ce4148c4f86107c703d0c9bcf3`.
Protocols and thresholds did not change; execution settings differ from the
original four-thread invariance run.

Suites ran sequentially. Minimum functionality and directional have internal
process timeouts. Invariance has a cooperative 300-second deadline and was also
supervised with a **330-second process-group hard limit**. No completed invocation
reached its cap. Supervisor wall times were 12.764, 13.716 and 96.784 seconds;
the table uses the runners' reported durations. Source remained clean at the
recorded commit and all model processes ended, as the
[final state](../evidence/selected-checkpoint-2026-10-17/final-state.json) records.

## Reproduction

Use the recorded source commit in a clean checkout. Install the locked optional
environment and restore only the required artifacts with your own DVC access:

```sh
uv sync --locked --dev --extra forecast
uv run --locked --extra forecast dvc pull -r origin \
  models/chronos-t5-small-fine-tuned.dvc \
  data/processed/train.parquet data/processed/validation.parquet
```

Create the ignored resource copy and print both configuration hashes:

```sh
uv run --locked --extra forecast python - <<'PY'
from copy import deepcopy
import hashlib
from pathlib import Path
import yaml

original_bytes = Path("params.yaml").read_bytes()
original = yaml.safe_load(original_bytes)
assert original["evaluation"]["cpu_threads"] == 4
settings = deepcopy(original)
settings["evaluation"]["cpu_threads"] = 2
path = Path("temporary/selected-checkpoint-behaviour/params-cpu2.yaml")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
print("Original:", hashlib.sha256(original_bytes).hexdigest())
print("CPU2 copy:", hashlib.sha256(path.read_bytes()).hexdigest())
PY
```

Run the first two suites sequentially, with local offline weights:

```sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run --locked --extra forecast python -m src.modeling.minimum_functionality \
  --checkpoint models/chronos-t5-small-fine-tuned --max-seconds 300
uv run --locked --extra forecast python -m src.modeling.directional \
  --checkpoint models/chronos-t5-small-fine-tuned --max-seconds 180
```

Supervise full invariance with the same hard limit:

```sh
uv run --locked --extra forecast python - <<'PY'
import os
import signal
import subprocess
import sys

command = [
    sys.executable, "-m", "src.modeling.invariance",
    "--checkpoint", "models/chronos-t5-small-fine-tuned",
    "--params-path", "temporary/selected-checkpoint-behaviour/params-cpu2.yaml",
    "--max-seconds", "300",
]
process = subprocess.Popen(command, start_new_session=True)
try:
    status = process.wait(timeout=330)
except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    status = 124
raise SystemExit(status)
PY
```

Each runner creates new ignored reports. Retain failures, incomplete attempts and
settings when reproducing, without revising limits after outcomes are observed.
The process-group example uses the POSIX interface measured here; ordinary
mocked checks remain available through the portable test suite.

## Evidence verification

Successful JSON reports were copied byte-for-byte from the runtime outputs:

| Report | SHA-256 |
| --- | --- |
| Minimum functionality | `0bd050efc49f03525a58c33b29bb7f73d56255a34b030ae1159d1e50a6323ad1` |
| Directional | `2303a3bfa2f832987a7518b0f2c61fcca49b7da076e821f5d7fa2e9e513d49bc` |
| Invariance | `00223617deb2da1048403b066a7cbc7b948256f1451ae994bad95ff81c56328d` |

[Artifact checksums](../evidence/selected-checkpoint-2026-10-17/artifact-checksums.json)
cover every retained file. The [execution manifest](../evidence/selected-checkpoint-2026-10-17/manifest.json)
records source, weights, offline flags and thread limits. Individual
[minimum-functionality](../evidence/selected-checkpoint-2026-10-17/minimum-functionality-execution.json),
[directional](../evidence/selected-checkpoint-2026-10-17/directional-execution.json)
and [invariance](../evidence/selected-checkpoint-2026-10-17/invariance-execution.json)
execution records retain commands, exit codes and supervisor timings.
Recorded errors, directional changes, control difference and all 70 paired
metrics/pass decisions were independently recalculated from the retained arrays.
Protocols and real-window hashes were checked against the original base evidence.
