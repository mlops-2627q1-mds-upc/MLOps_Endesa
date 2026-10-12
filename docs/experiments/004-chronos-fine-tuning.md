# Experiment 004: Bounded Chronos fine-tuning

On 2026-10-12, Sindri fine-tuned the pinned Chronos checkpoint for
[#13](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/13), following the
options proposed by Pablo in that issue. Training used the training block only, and
every option was scored on the full validation block with the evaluator from
[Experiment 001](001-baseline-chronos.md). The test plan and decision rule were fixed
[below](#test-plan-and-decision-rule) before the test block was opened; the selected
model was then scored once on test, and [meets the rule](#results-on-test).

## Results on validation

Lower mean MASE is better. Every run scored the same 1,825 origins as Experiment 001,
with 20 samples and their median, on the GPU. Links open the evaluation runs.

| Option | Seed 7 | Seed 8 | Seed 9 | Mean |
| --- | ---: | ---: | ---: | ---: |
| [Control](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/8b35be0c36824a4592a3a8d23c4a296c): pinned checkpoint, untrained | 0.8737 | | | 0.8737 |
| 1 Reference: full fine-tuning, lr 0.001 to 0 over 1,000 steps | [0.5720](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/5670e714d1c94e23b642a469f5d4ec1b) | [0.5649](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/94c5bee1bec54b0a9d65d0df3d98abb6) | [0.5740](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/5dc947a9919444f4bf5e77fa43d8a368) | 0.5703 |
| 2 Lower learning rate: lr 0.0001 | [0.6195](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/b68ec76f6e534252b60c05d3e233848d) | [0.6193](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/5c5fb65d8b6749788e1162276fa8f017) | [0.6213](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/5aeff343d88344158441c70c311fb087) | 0.6200 |
| 3 Fewer steps: 300 | [0.6159](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/2330ace6700f40da83c3a46414fcd405) | [0.6033](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/becacf55226f4ecbb0dd10f006fe0e74) | [0.6128](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/baf3eb3300c54023bccbdbd07a7ffc50) | 0.6107 |
| 4 Recent data: windows from the last three training years | [0.5740](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/f71ccd9d2a4a420f961106c7b34c8bbc) | [0.5774](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/f59ba9d2548c47feaee5a046b6032a18) | [0.5849](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/1efd64f84bb2455b92906ec1f22e23cb) | 0.5788 |
| 5 LoRA: rank 8 on the attention projections, base frozen | [0.6325](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/298769d0962a4b54bf242a6812da12b1) | [0.6383](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/d3bc7314fcc14d1ab0e6006610084c65) | [0.6326](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/6a24d48ceecc43769971214abf82f65b) | 0.6345 |

For reference, lag 48 scores 1.0422 and lag 336 1.2020 on validation (Experiment 002).

Option 1 has the lowest mean validation MASE, 34.7% below the control and 45.3% below
lag 48. Each of its seeds scores below every seed of options 2, 3 and 5. Options 1 and 4
are close: their seed ranges meet at 0.5740. The range of option 1 over its three seeds,
0.0091, is the seed-to-seed variation used in the decision rule. LoRA improves on the
control but is the weakest option, so the AdaLoRA round proposed for a competitive LoRA
was not run. LoRA used option 1's learning rate; other learning rates were not tested.

Option 1 with seed 8 is selected on validation. Its training run is
[`639eba4cc7e74de08ffab64beee715db`](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/639eba4cc7e74de08ffab64beee715db).
The checkpoint is versioned with DVC as `models/chronos-t5-small-fine-tuned` (176 MB);
its `model.safetensors` has the SHA-256 recorded by its evaluation runs,
`746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9`. Retrieve it with
`uv run dvc pull models/chronos-t5-small-fine-tuned.dvc -r origin`, or with the in-memory
credentials snippet in [the evaluation guide](../evaluation.md#reproduction).

### Training length

Option 1 with seed 8 saved checkpoints every 250 steps. Scored on validation:

| Training step | 0 (control) | [250](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/7616e02f02354490a5740478d23f0b74) | [500](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/d084376c36f848a18be164378e5840c3) | [750](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/4dbe592df9a64910b24e867c6b6e3294) | 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean MASE | 0.8737 | 0.6498 | 0.6031 | 0.5781 | 0.5649 |

Validation MASE was still falling at 1,000 steps, by less in each interval. Intermediate
checkpoints are taken before the learning rate has decayed, so they do not stand in
for shorter schedules: step 250 scores 0.6498, while option 3 trained for 300 steps with
a full decay averages 0.6107. This diagnostic does not change the selection.

## Cost

| Run | Time | Energy, kWh | Notes |
| --- | ---: | ---: | --- |
| Training, 1,000 steps (options 1, 2, 4) | 677–680 s | 0.0276–0.0285 | 46.2 million trainable parameters |
| Training, 300 steps (option 3) | 203–204 s | 0.0083 | |
| Training, LoRA (option 5) | 632–633 s | 0.0257–0.0265 | 589,824 trainable parameters (1.26%); adapter 2.3 MB |
| Validation pass on GPU | 258–260 s | 0.0103–0.0104 | |

A training step takes 0.68 s. GPU energy comes from NVML power samples, as the
[energy measurement section](../evaluation.md#energy-measurement) describes. In a
1,000-step run, the GPU drew about 114 W on average, which over 679 s is 0.0215 of the
run's 0.0285 kWh; CPU and RAM account for the rest. For comparison with the GPU
validation pass above, a validation pass of the untuned model on the CPU took 2,042 s
and 0.0576 kWh (Experiment 002).

The table gives the cost of a normal run. The energy actually measured for this
experiment, including the failed and slow runs described under
[retained attempts](#retained-attempts), is:

| Runs | Count | Energy, kWh | Emissions, g CO₂eq |
| --- | ---: | ---: | ---: |
| Training, including the first pilot attempt | 17 | 0.411 | 71.5 |
| Evaluations at normal speed | 12 | 0.125 | 21.7 |
| Slow evaluations of options 1 to 3 | 8 | 0.591 | 102.8 |
| **Total measured** | 37 | **1.126** | **196.0** |

At normal speed, the eight slow evaluations would have used about 0.083 kWh, so the
GPU memory issue cost about 0.51 kWh, 45% of the total. The evaluation that hit its time
limit ran for 3,611 s and recorded no energy, because it failed before logging it.

## Versions and execution

- Source: clean commits [`8b3da48`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/commit/8b3da48968db525e7d8eae3985d25f48e59a50e3) (pilot),
  [`ccddc8c`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/commit/ccddc8c8412ef1f07ae9a8b5161ba6d310e0ee45) (options 1–3),
  [`1b78dfe`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/commit/1b78dfef3e9b28477564fc35542bc7d913cecc95) (option 4 and the option 3 rescore),
  [`f3c5b8e`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/commit/f3c5b8ea47ad20ed0cf29235a50ab03828653a86) (option 5) and
  [`e2cf097`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/commit/e2cf0977183738a5a4d39e4c51daf13e3456bdd4) (training-length diagnostic).
  Between `ccddc8c` and `f3c5b8e`, only LoRA and the GPU memory release changed; the
  full fine-tuning path is the same.
- Data and target table: as in Experiment 001; every evaluation shares the target-table
  hash `f1aa58b74bba7b191bc56ad0615a3585b4bb3e961c823229f90e40f484d13197`.
- Base checkpoint: `amazon/chronos-t5-small` at `a971ba21945c4f1796b17a91fe69214b5f4ad472`.
- Training: windows of 512 context and 64 label steps from the training block, through
  the checkpoint's tokenizer; batch 32 as two accumulated micro-batches of 16; AdamW
  without weight decay; gradient norm clipped at 1.0; float32. Options are defined in
  `params.yaml` and each changes one setting of option 1.
- Evaluation: seed 7, 20 samples and their median, batch 16, as in Experiment 001, on
  the GPU.
- Host: as in Experiment 002, with the NVIDIA driver 617.42. PyTorch 2.14.1 (CUDA 13.0
  build), Chronos 2.3.2, Transformers 5.18.0, peft 0.21.2, MLflow 3.16.1, CodeCarbon 3.3.1.

## Verification

Pablo read all 18 validation runs back from MLflow and confirmed the table, the clean
commits, the shared target hash, the link from each evaluation to its training run, and
that training reads only `train.parquet` with windows ending before `train_end`. Both
are covered by tests.

The control scored on the GPU differs from the untuned CPU result on the same machine
(0.8722, Experiment 002) by 0.0015 in mean MASE; per state the difference reaches 0.029.
The weights are identical, so this is the sampling of the 20 trajectories, which uses
different random numbers on the GPU. Option 1 with seed 7 gave 0.5718 in the pilot and
0.5720 in the round, which trained it again.

After the test path was added, the evaluator reproduced lag 48, lag 336 and option 1
with seed 8 on validation to every recorded digit, without opening the test block.

## Retained attempts

- The first pilot runs [`abe0719b`](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/abe0719b60c1434192912dcf6667f9b7)
  (control) and [`78934fe4`](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/78934fe4c4ec4e4fb692bcf7741141a4)
  (training) completed and logged their data, but MLflow's final status update failed:
  its run link contains an emoji that Python could not encode when the output was piped
  on Windows. They were marked `FAILED` by hand with a `failure.reason` tag, and the
  pilot was run again.
- In the first round, PyTorch kept about 5.7 GB of GPU memory cached after each
  training, so the evaluations of options 1 to 3 spilled into shared system memory.
  They took about 2,642 s and 0.074 kWh each instead of 259 s and 0.010 kWh. Their
  scores are valid, since the weights and sampling are unchanged, but **their timings
  and energy are not representative** and are excluded from the cost table. The round
  now releases the cache between runs.
- For the same reason, the evaluation of option 3 with seed 9,
  [`c1f11e6f`](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/c1f11e6fad204a07baf3386a9ab5feea),
  hit the one-hour evaluation limit and is `FAILED` with a `failure.reason` tag. Its
  checkpoint was scored again in a separate run, which the table links.

## Test plan and decision rule

Fixed in commit `903f7c4`, before the test block was opened, as requested in #13 and in
[PR #30](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/30) (point 6).

The test block is scored once, at a single clean commit, with the evaluator's
`--block test`:

| Run | Role |
| --- | --- |
| Option 1, seed 8 | Selected model |
| Control | Untuned Chronos, through the same local-checkpoint path on the GPU |
| Lag 48 | Primary baseline, selected on validation |
| Lag 336 | Weekly seasonal baseline, named in #13 |
| Option 1, seeds 7 and 9 | Reported only, to show seed variation; they do not change the selection or the conclusion |

**Decision rule:** fine-tuning is judged to improve held-out forecasts if the test mean
MASE of option 1 with seed 8 is lower than that of the control, lag 48 and lag 336, each
by more than 0.0091. That margin is the seed spread of option 1 on validation, the range
of its mean MASE over seeds 7 to 9, as proposed in #13: a smaller difference could come
from the seed alone. Otherwise, the result is reported as no clear improvement. Whether
the improvement justifies the training cost is assessed in the review, using the cost
above.

## Results on test

Pablo approved the test path (commit `e2cf097`) in #13 before the run. On 2026-10-12, each
model in the plan was scored once on the test block, at clean commit
[`0874b45`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/commit/0874b45142706df2838d42557af516733ecb90e0),
with 365 origins per state and the test target-table hash
`dd5945ccd8122c02c70be44250de5d2dfdc5a3359f10442b20078d2ac4d71f3e`. Links open the test runs.

| Run | Role | Test mean MASE | Validation mean MASE |
| --- | --- | ---: | ---: |
| [Option 1, seed 8](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/afb1ac0b57644101a3331b903588a4a2) | Selected model | **0.5571** | 0.5649 |
| [Option 1, seed 7](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/fbfafacfb04d49e78684ab17df57a4f3) | Reported only | 0.5618 | 0.5720 |
| [Option 1, seed 9](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/a72cfaed9c9d423eaabb0a22a2d7d4fb) | Reported only | 0.5669 | 0.5740 |
| [Control](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/89eb2214d4814a9cbb5f146165003462) | Untuned Chronos | 0.8175 | 0.8737 |
| [Lag 336](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/02804dbcf7654ff7b42ee1722078c8a2) | Weekly baseline | 0.9617 | 1.2020 |
| [Lag 48](https://dagshub.com/pauadal03/MLOps_Endesa.mlflow/#/experiments/1/runs/1d694eb077e54af4b815a74a4b1ecb2a) | Primary baseline | 0.9885 | 1.0422 |

Each cell below contains the test MASE per state.

| State | Option 1, seed 8 | Control | Lag 48 | Lag 336 |
| --- | ---: | ---: | ---: | ---: |
| NSW | 0.4566 | 0.7669 | 0.9066 | 0.8490 |
| VIC | 0.6110 | 0.9383 | 1.1868 | 1.0456 |
| QLD | 0.5690 | 0.8888 | 1.1021 | 1.0794 |
| SA | 0.6487 | 0.8444 | 0.9343 | 0.9692 |
| TAS | 0.5000 | 0.6489 | 0.8128 | 0.8654 |

**Decision rule: met.** The test mean MASE of option 1 with seed 8 is lower than the
control's by 0.2604, lag 48's by 0.4315 and lag 336's by 0.4046, each more than the
0.0091 margin. By the rule fixed above, fine-tuning improves held-out forecasts: its test
mean MASE is 31.9% below the untuned model and 43.6% below the primary baseline. It is
also lower in every state.

On test, option 1's three seeds span 0.0099, close to their validation spread of
0.0091, in the same order. Every method scores lower on test than on validation, so the
test period is easier to forecast in this protocol. Lag 336 scores below lag 48 on test,
the reverse of validation; lag 48 remains the primary baseline, since it was selected on
validation, and the rule requires beating both. The fine-tuned model's mean worst-origin
error share is 0.183, against 0.147 for the control: its remaining error is more
concentrated in the worst origins, as on validation. Each Chronos test pass took about
259 s and 0.0104 kWh.

These results come from one selected checkpoint on one test period, and the test block
has now been used. Holiday breakdowns remain unavailable until the civil timestamps in
[#6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6) are verified.
