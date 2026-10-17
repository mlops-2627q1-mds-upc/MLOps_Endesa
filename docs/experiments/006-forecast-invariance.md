# Experiment 006: Paired forecast behaviour

The [protocol](../forecast-invariance.md) and
[EDN-010](../edn/0010-forecast-invariance.md) define the cases, relationships,
sampling controls and thresholds for
[issue #55](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/55).

On 2026-10-14, all 70 paired checks passed their declared thresholds. Each pair
compared the complete 48-step median forecasts. The
[complete report](../evidence/issue-55-forecast-invariance.json) retains the
140 forecasts, input-window hashes, versions, settings and every case outcome.
No counterexamples were observed in this run. Peer review is pending.

## Observed changes

Differences below are relative to `S = mean(abs(original history))`, after
dividing a scaled forecast by its input factor. Each column reports the largest
value across the 14 histories for that transformation.

| Transformation | Pairs within thresholds | Largest mean difference / S | Largest step difference / S |
| --- | ---: | ---: | ---: |
| Scale 0.5 | 14 / 14 | 0 | 0 |
| Scale 2 | 14 / 14 | 0 | 0 |
| Scale 10 | 14 / 14 | 1.17221e-7 | 2.64929e-7 |
| Noise bounded by 0.1% of S | 14 / 14 | 0.0122407 | 0.0403452 |
| Noise bounded by 1% of S | 14 / 14 | 0.0314933 | 0.0692990 |

The declared scale limits were `1e-5` mean and `1e-4` maximum; noise limits
were 5% mean and 15% maximum. The largest noise response was the synthetic
trending history at 1% input noise: a 3.15% mean and 6.93% maximum forecast
change. At 0.1% noise, SA validation reached 1.22% mean and 4.03% maximum.
Small input changes can therefore be amplified even when these initial
investigation thresholds pass. These results do not establish general
robustness or forecast accuracy.

## Versions and conditions

- Executed source and predeclared protocol: clean
  [`9469a13348be793a0a98c0113840d186684d3a40`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/commit/9469a13348be793a0a98c0113840d186684d3a40).
- Checkpoint: `amazon/chronos-t5-small` at
  `a971ba21945c4f1796b17a91fe69214b5f4ad472`, loaded once from the local cache.
- Sampling: seed 7 reset before each history, 20 trajectories, median point
  forecast, batch size 1, float32, temperature 1, top-k 50, top-p 1.
- Environment: macOS 26.2 arm64, CPU with four PyTorch threads, Python 3.11.15,
  Chronos 2.3.2, PyTorch 2.14.1, Transformers 5.18.0 and NumPy 2.4.6.
- Training input MD5: `76fc8921b85ad9d35e98b9b2451c1c09`; validation input MD5:
  `7d3a853328e8e21438ef90a4a17fcf03`. Both match the existing `dvc.lock` outputs.
- All five states used training positions `[195136, 195648)` and validation
  positions `[203920, 204432)`, selected before inference. Four synthetic patterns
  complete the 14-history set. No final test targets were opened.

The two input blocks were reconstructed locally from the
[pinned source archive](../DATASET_CARD.md#dataset-structure-and-provenance), verified
against SHA-256
`b1439c28a631766bd05ee327f1f430a887b9f53b38a8cd5c0769ded1fd4aaf5a`.
Their output bytes match the DVC versions above. Fresh retrieval from the DVC
remote was not verified in this execution.

The report's SHA-256 is
`70869a87b00a939f02cb198f342b156edb0b974963f3ee3da2026f454cdbace1`.
It was copied without modification from the runner's output. It contains full
forecasts and input identifiers; raw observations and checkpoint weights remain
outside Git.

## Runtime and reproduction

The runner took 121.502 seconds, including optional imports and preparation.
Its measured checkpoint load was 0.071 seconds and forecast loop 117.000 seconds.
The full pytest subprocess took 123.823 seconds under an external hard timeout
of 300 seconds. The internal cooperative deadline was also 300 seconds.

After restoring the locked environment and inputs as described in the
[reproduction guide](../forecast-invariance.md#reproduction), run:

```sh
uv run --extra forecast pytest tests/test_invariance.py \
  -k real_model --run-forecast-invariance -q
```

The observed result was `70 passed, 37 deselected in 121.58s`, exit status 0.
`HF_HUB_OFFLINE=1` was set for this cached-checkpoint execution. The equivalent
CLI is `uv run --extra forecast python -m src.modeling.invariance --max-seconds 300`.
For a hard timeout around the pytest command, use Python's subprocess timeout:

```sh
uv run --extra forecast python -c "import subprocess, sys; subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_invariance.py', '-k', 'real_model', '--run-forecast-invariance', '-q'], timeout=300, check=True)"
```

Routine validation also passed: `make test` reported 108 passed and 79 skipped.
The skips were 70 separately executed model checks and nine existing GX checks
whose raw DVC inputs were absent. Ruff formatting, Ruff lint and Pylint passed.
The synthetic seasonal reference tests check the harness; the real checkpoint
run above provides the model-behaviour evidence.

This is one checkpoint, seed, noise direction and window per state/block.
It neither measures quantile robustness nor predicts behaviour after fine-tuning.
Use the same declared protocol with an accepted checkpoint from #13 to compare
outcomes. A teammate still needs to review the protocol, thresholds and findings
before this evidence is treated as reviewed M3 work.
