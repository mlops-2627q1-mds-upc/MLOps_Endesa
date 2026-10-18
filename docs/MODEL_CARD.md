# Chronos demand forecaster: project model card

This extends the [upstream Chronos-T5-small card](https://huggingface.co/amazon/chronos-t5-small/tree/a971ba21945c4f1796b17a91fe69214b5f4ad472)
with the Endesa experiment and its limits. The fine-tuning implementation,
checkpoint-lineage repair and first-delivery integration were approved by Toni
and merged in [PR #69](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/69)
on 17 October 2026. Recorded experiment results are separate from deployment evidence.

## Intended use and inputs

The component forecasts 48 half-hourly positions from the previous 512 demand
values for one of NSW, VIC, QLD, SA or TAS. It is a course comparison on historical
series, with no external features. Values remain in the stored source scale;
[the dataset card](DATASET_CARD.md#source-meaning-units-and-clock-limits)
explains the evidence for MW and the unresolved civil-clock mapping.
The component is not validated for operational electricity scheduling.

## Checkpoint and training provenance

| Item | Project record |
| --- | --- |
| Base | `amazon/chronos-t5-small`, revision `a971ba21945c4f1796b17a91fe69214b5f4ad472`; T5, approximately 46 million parameters |
| Selected artifact | `models/chronos-t5-small-fine-tuned`, versioned by [DVC metadata](../models/chronos-t5-small-fine-tuned.dvc) |
| DVC directory MD5 | `0377ee21f89c25da36b95db59841c3d4.dir` |
| Weights SHA-256 | `746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9` |
| Training run | [`639eba4cc7e74de08ffab64beee715db`](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/639eba4cc7e74de08ffab64beee715db) |
| Training source | Clean `ccddc8c8412ef1f07ae9a8b5161ba6d310e0ee45` |
| Selected configuration | Option 1, seed 8; 1,000 steps; AdamW, learning rate 0.001 decaying linearly to zero; float32; batch 32 as two accumulated micro-batches of 16 |
| Training inputs | Five training blocks only; windows of 512 context and 64 label steps ending before position 195,696 |
| Artifact lineage | New checkpoints carry weight-bound training metadata. The published artifact uses a [versioned sidecar](../models/chronos-t5-small-fine-tuned.provenance.json), preserving its existing DVC version. Unknown lineage is recorded as unknown. |

[Experiment 004](experiments/004-chronos-fine-tuning.md) records all options,
seeds, failed attempts, environment and run links. The reference configuration
and seed were selected on validation. The 0.0091 decision margin was fixed in
`903f7c4` before the test block was opened; it is the validation seed spread,
not a confidence interval or a significance test.

## Evaluation and recorded results

Each method forecasts the same 365 origins per state in each held-out block:
1,825 origins and 87,600 targets overall. Origins are 48 positions apart.
Contexts end before their origin; the lag-336 MASE scale is estimated on training
only. Results use the median of 20 sampled trajectories, with evaluation seed 7.
The primary baseline, lag 48, was chosen on validation.

| Method | Validation mean MASE | Test mean MASE |
| --- | ---: | ---: |
| Selected fine-tuned model, option 1 / seed 8 | 0.5649 | 0.5571 |
| Untuned Chronos control, GPU | 0.8737 | 0.8175 |
| Seasonal naive, lag 48 | 1.0422 | 0.9885 |
| Seasonal naive, lag 336 | 1.2020 | 0.9617 |

These are rounded results from Experiment 004, not a new execution during
submission integration. Test evaluation used clean
`0874b45142706df2838d42557af516733ecb90e0` on 12 October 2026; the selected-model
[evaluation run](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/afb1ac0b57644101a3331b903588a4a2)
identifies the weights and target table. The selected model's test MASE is
31.9% below the control and 43.6% below lag 48, exceeding the predeclared
margin against all three comparators. It is lower in every state. This is
one selected checkpoint on one test period; the test block has now been used.

The original [CPU validation comparison](experiments/001-baseline-chronos.md)
reports untuned MASE 0.872259 under a different execution device. Do not mix
that CPU result with the GPU control when interpreting the fine-tuning table.

## Behaviour and remaining failures

The pinned base checkpoint passed the declared [minimum functionality](experiments/005-minimum-functionality.md),
[invariance](experiments/006-forecast-invariance.md), and
[directional](experiments/007-directional-expectations.md) cases: 10, 70 and
12 paired cases plus an unchanged control respectively. These results concern
the base checkpoint.

The selected local artifact was checked separately on 17 October 2026:
[Experiment 008](experiments/008-selected-checkpoint-behaviour.md) records
10/10 minimum-functionality cases, 70/70 invariance pairs and 12/12 directional
cases plus the unchanged control passing the same declared limits. Invariance
used two CPU threads through an ignored configuration copy, after retaining an
interrupted four-thread attempt. Reports identify the actual weights hash.
These bounded checks do not establish behaviour on arbitrary demand or add
demand-test accuracy or energy measurements.

On test, the selected model's mean worst-5%-of-origins error share is 0.183,
against 0.147 for the control. Its remaining errors are more concentrated in
the worst origins. Interval coverage, a civil-calendar holiday breakdown and
operational robustness are not established by these point-error results.

## Resources, inference and deployment

The selected training run measured about 678 seconds and 0.0276 kWh on an
RTX 4060 with 8 GB. The experiment's 37 measured runs total 1.126 kWh;
one timed-out evaluation has no energy record. About 0.51 kWh of the measured
total was avoidable cost from a GPU-memory issue, documented in Experiment 004.
CodeCarbon/NVML figures are estimates, not wattmeter measurements.

Install the locked `forecast` extra to load the checkpoint. Its DVC directory
is approximately 176 MiB. [Replication instructions](replication.md) separate
fast checks, data validation, optional synthetic inference and historical results.
The checkpoint is selected for later serving; no deployed forecasting API or
serving latency/energy claim is made for this first delivery.

## Limitations and licensing

- The dataset is historical and covers five aggregate state series. Clock and
  interval alignment remain unresolved; evaluation uses positional indices.
- The upstream paper lists Australian Electricity as a zero-shot evaluation
  dataset. That disclosure is not an independent audit of pretraining exposure.
- Hardware changes can change sampled forecasts even with an identical seed.
  The observed validation seed spread is not a general uncertainty bound.
- Twenty trajectories give coarse p10/p90 estimates; an 80% interval's actual
  coverage remains to be measured before a coverage target is claimed.
- The base model is [Apache 2.0](https://huggingface.co/amazon/chronos-t5-small/tree/a971ba21945c4f1796b17a91fe69214b5f4ad472).
  The dataset is CC BY 4.0; retain both attributions when reusing artifacts.

See [requirements](requirements.md), [EDN-012](edn/0012-chronos-fine-tuning.md)
and the [Chronos paper](https://arxiv.org/abs/2403.07815) for the decision record
and upstream context.
