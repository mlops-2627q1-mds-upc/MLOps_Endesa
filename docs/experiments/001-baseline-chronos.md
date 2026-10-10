# Experiment 001: Seasonal baselines and untuned Chronos

On 2026-10-05, all three methods completed the agreed validation comparison for
[#12](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/12).
The implementation and findings await peer reproduction and review in
[PR #28](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/28).

## Results

Lower mean MASE is better. Each method used the same 1,825 origins and 87,600
targets: 365 origins per state, with 512 past values and a 48-step horizon.
MASE uses each state's lag-336 training denominator and weights the five states
equally. Method links open the corresponding shared MLflow run.

| Method | Mean MASE | Forecast loop, seconds | Mean worst-19-origin error share |
| --- | ---: | ---: | ---: |
| [Seasonal lag 48](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/a0438b7d99324131a39c776bd1b19242) | 1.042173 | 0.001298 | 16.12% |
| [Seasonal lag 336](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/315c55ea0f7d44efbf9e3e5e5c78d33c) | 1.201975 | 0.002806 | 20.61% |
| [Untuned Chronos](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/d9dadcacba614f3aa5ed3021b1ee513a) | 0.872259 | 920.356929 | 16.83% |

Lag 48 is the primary baseline selected on validation. Untuned Chronos has
16.30% lower validation mean MASE than that baseline and lower MAE in all five
states. This result does not compare against a fine-tuned model or measure
held-out test performance.

Each table cell below contains **MAE in the source scale / MASE**. The physical
demand unit is still unverified.

| State | Lag 48: MAE / MASE | Lag 336: MAE / MASE | Chronos: MAE / MASE |
| --- | ---: | ---: | ---: |
| NSW | 402.308 / 0.963066 | 441.557 / 1.057022 | 350.736 / 0.839610 |
| VIC | 386.138 / 1.310725 | 411.843 / 1.397979 | 317.544 / 1.077888 |
| QLD | 238.281 / 1.107253 | 234.427 / 1.089345 | 185.559 / 0.862262 |
| SA | 126.470 / 0.969406 | 176.692 / 1.354357 | 112.098 / 0.859238 |
| TAS | 39.282 / 0.860417 | 50.730 / 1.111171 | 32.976 / 0.722295 |

The worst-origin statistic sums errors from the largest 19 of 365 origin errors
and divides by all origin errors, separately for each state. The aggregate is
their equal-weight mean. Its slightly larger value for Chronos does not imply
larger absolute tail errors: it measures concentration relative to total error.
Per-state shares are preserved in each run's `metrics.json`.

## Versions and execution

- Source: clean commit [`3bcc772c7d0b18d01febb29d220bc0884c729faa`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/commit/3bcc772c7d0b18d01febb29d220bc0884c729faa).
- Data: raw DVC MD5 `ab1300b4a36e9e1df9fabfeec15b0f90.dir`; `dvc.lock`
  SHA-256 `e7600130ee65dda143bc559a60efa7073ea5187b9689f3efb48fd795e2e2d44d`.
- Group: `d72b27c0c1434bd2ac7b753a62ac989e`; shared target-table SHA-256
  `f1aa58b74bba7b191bc56ad0615a3585b4bb3e961c823229f90e40f484d13197`.
- Checkpoint: `amazon/chronos-t5-small` at
  `a971ba21945c4f1796b17a91fe69214b5f4ad472`, without project training.
- Inference: seed 7, 20 trajectories, interpolated sample median, float32,
  batch size 16, CPU with four PyTorch threads; one-hour configured budget.
- Host: Apple M4, 32 GiB memory, macOS 26.2 arm64, ten logical CPUs.
  Python 3.11.15, Chronos 2.3.2, PyTorch 2.14.1,
  Transformers 5.18.0, MLflow 3.16.1;
  exact packages are in each run's environment and lockfile.

Chronos's forecast loop took 920.357 seconds, or 0.504 seconds per origin on
average. This includes batch preparation, sampling, median calculation and
progress output. Its checkpoint loading took 0.038 seconds from an existing
local cache; that is not a cold-start or download benchmark. Its run body took
929.684 seconds, including optional imports and artifact logging. Baseline loop
times measure vectorized forecast generation, excluding metrics and uploads.
These single wall-clock observations do not measure energy or serving latency.

## Verification and reproduction

All three shared runs were read back as `FINISHED` at the clean source commit.
All ten artifacts per run were downloaded: configuration, provenance, environment,
lockfile, predictions, metrics, training scales, parameters, raw DVC reference,
and split lockfile. Their targets, source/data references and lockfiles matched.
MAE, MASE and worst-origin shares were recalculated from the downloaded predictions
and training-only scales. Both seasonal forecasts were also reconstructed
independently from the observed history.

A fresh DVC cache retrieved eleven objects from the shared remote under Dídac's
own account and matched the versioned inputs. Local lint and all 40 tests passed;
[CI on the executed source](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/37356431172)
passed lint and tests.

For a teammate's fast reproduction, follow [the setup guide](../evaluation.md)
with their own credentials, then run:

```sh
uv run python -m src.modeling.evaluate --method seasonal_lag336
```

Expect 87,600 predictions, mean MASE `1.201974712215821`, the data reference and
target hash above, and a new shared run ID. Record that ID and the interpretation
review in PR #28 or #12.
Full Chronos reruns use the guide's optional `forecast` environment; identical
random seeds do not guarantee bitwise equality across hardware.

On 2026-10-06, [Sindri reproduced lag-336 on Windows](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/28#pullrequestreview-5426279134)
at `eff29e2` in shared run
[`a5102636b26e47f1be7110e4a5adbd7a`](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/a5102636b26e47f1be7110e4a5adbd7a).
He reported identical mean MASE, per-state errors, worst-origin shares and row
count. Matching the DVC data version required forcing LF checkout; the target hash
still differed because CSV serialization used Windows CRLF. The 2026-10-08
follow-up enforces LF checkout and target serialization, preserving the original
LF target hash above. Sindri did not rerun Chronos; native Windows verification
of the corrections remains pending.

## Limits and retained attempts

Final test targets were not loaded for prediction or scoring. Holiday and day-after-holiday
breakdowns await verified civil timestamps in
[#6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6).
The [upstream pretraining disclosure](../evaluation.md#interpretation-and-limitations)
lists Australian Electricity as a zero-shot evaluation dataset; this remains
disclosure rather than an independent audit of pretraining exposure. These
validation scores are not the paper's benchmark.

An earlier [Chronos attempt](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/09a8daa05b1a4391b1cfd66b8d8c90a9)
was interrupted by the agent after buffered progress was mistaken for a slow run.
It is retained as `FAILED` and excluded from these results. The comparison was
restarted with visible progress and unchanged forecast settings.
