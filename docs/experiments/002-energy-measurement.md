# Experiment 002: Energy of the validation comparison

On 2026-10-11, Sindri ran the [Experiment 001](001-baseline-chronos.md)
comparison again with energy measurement for
[#39](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/39), on the
machine planned for fine-tuning in
[#13](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/13).
The [energy measurement section](../evaluation.md#energy-measurement) describes
the method and its limitations.

## Results

Each method forecast the same 1,825 origins as in Experiment 001. Energy covers
only the forecast step. Method links open the corresponding shared MLflow run.

| Method | Mean MASE | Forecast loop, seconds | Energy, kWh | Emissions, kg CO₂eq |
| --- | ---: | ---: | ---: | ---: |
| [Seasonal lag 48](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/84f2d330225345ab96c7cd7140ae4a29) | 1.042173 | 0.000823 | 9.54 × 10⁻⁸ | 1.66 × 10⁻⁸ |
| [Seasonal lag 336](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/e3d4bd4368d04102aed5e96dd52cc2f0) | 1.201975 | 0.000833 | 6.64 × 10⁻⁸ | 1.16 × 10⁻⁸ |
| [Untuned Chronos](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/253c99e888444d36b255390a42a2ed37) | 0.872183 | 2041.635 | 0.0576 | 0.0100 |

Chronos's energy by component:

| Component | Energy, kWh | Share | Mean power | Method |
| --- | ---: | ---: | ---: | --- |
| CPU | 0.0233 | 40.5% | 41.2 W | `Windows EMI` |
| GPU | 0.0286 | 49.7% | 50.5 W | `nvml power samples (board power)` |
| RAM | 0.0057 | 9.8% | 10.0 W | `RAM power estimation model` |

Untuned Chronos has 16.31% lower validation mean MASE than lag 48, and used
0.0576 kWh, about 114 J per 48-step forecast. Chronos runs on the CPU; its GPU
energy is the board power the GPU reports while not used by the model, at the
idle level of about 50 W described in the method section. The seasonal forecasts
take under a millisecond, while CodeCarbon's measured durations were 0.024 s and
0.010 s; their energy is at the resolution of the measurement and does not
distinguish the two baselines.

## Versions and execution

- Source: clean commit [`68562d943b59b20889495245d31d42c94dc4cce4`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/commit/68562d943b59b20889495245d31d42c94dc4cce4).
- Data, target table and checkpoint: as in Experiment 001; group
  `355bd0480f0a48178ea65230dfd2f657`.
- Inference: seed 7, 20 trajectories, batch size 16, CPU with four PyTorch threads.
- Host: HP Victus 15L, AMD Ryzen 7 5700G, HP OEM NVIDIA GeForce RTX 4060,
  31 GB memory, Windows 11 (build 26200). Python 3.11.15, Chronos 2.3.2,
  PyTorch 2.14.1 (CPU build), Transformers 5.18.0, MLflow 3.16.1, CodeCarbon 3.3.1.

Chronos's forecast loop took 2041.635 seconds, or 1.119 seconds per origin. Its
checkpoint loading took 0.350 seconds from a local cache, and its run body
2067.346 seconds.

## Verification

All three runs were read back as `FINISHED` at the clean commit, with CPU, GPU
and RAM methods recorded. Each run's CPU and RAM methods appear in its own
`energy/codecarbon.log`. For Chronos, the CodeCarbon artifacts report
0.029010 kWh without the GPU; adding the sampled `energy.gpu_kwh` of 0.028626 kWh
gives the recorded `energy.kwh` of 0.057636 kWh.

Both seasonal baselines reproduced Experiment 001's mean MASE exactly, and the
target-table hash matched `f1aa58b74bba7b191bc56ad0615a3585b4bb3e961c823229f90e40f484d13197`
on native Windows. Chronos's mean MASE differs from Experiment 001 (0.872259 on
an Apple M4); identical seeds do not guarantee equal results across hardware.

## Retained attempt

An earlier [run group](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/392b49ac5a4e47b38968d494ee86acd8)
at `dca459f` (group `5bebf511ec9a40e2b33d3886a8766221`) completed, but CodeCarbon
logged its tracking methods only for the first run of the process, so the lag-336
and Chronos runs lack `energy.cpu_method` and `energy.ram_method`. The tracker now
sets up its hardware for every run. That group is retained and excluded from these
results.
