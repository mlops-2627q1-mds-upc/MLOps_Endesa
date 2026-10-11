# Chronos-T5-small: project extension of the upstream model card

**Status:** Project extension of the [existing Chronos-T5-small model card](https://huggingface.co/amazon/chronos-t5-small/tree/a971ba21945c4f1796b17a91fe69214b5f4ad472), with intended use and a first [untuned validation result](experiments/001-baseline-chronos.md) awaiting peer reproduction and review. No project fine-tuned checkpoint exists. [Issue #7](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/7) tracks the remaining project-specific evidence.

## Model details and intended use

The proposed base is [Amazon Chronos-T5-small](https://huggingface.co/amazon/chronos-t5-small),
a 46-million-parameter pretrained forecasting model based on T5. The upstream
card describes scaling and quantizing time-series values into tokens, then
sampling future trajectories for probabilistic forecasts. Its usage example
shows how to load the checkpoint and call `predict`; we will adapt that pattern
to the selected demand data after its input contract is verified. Our planned
component will take a historical demand sequence for one Australian state and
forecast a fixed future horizon in the dataset's stored source scale.

The immediate use is a reproducible course comparison of the base checkpoint without project-specific training and a later version fine-tuned on the [selected dataset](DATASET_CARD.md). It is not validated for operational electricity scheduling, emergency decisions, or current demand forecasting.

## Base checkpoint provenance

| Item | Recorded value |
| --- | --- |
| Hub model revision | [`a971ba21945c4f1796b17a91fe69214b5f4ad472`](https://huggingface.co/amazon/chronos-t5-small/tree/a971ba21945c4f1796b17a91fe69214b5f4ad472) |
| Architecture | T5 encoder-decoder; the upstream model card lists 46M parameters. |
| License shown by the Hub | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Pinned configuration | Context length 512, default prediction length 64, 20 sampled trajectories. These are checkpoint defaults, not this project's chosen evaluation settings. |
| Project-trained artifact | Not created. |

## Project training and evaluation extension

The [M1 problem definition](problem-definition.md#evaluation-contract-for-m2) sets a common 48-step horizon, chronological train/validation/test indices, daily (lag 48) and weekly (lag 336) seasonal naive baselines, and per-state MAE plus equally weighted state-level MASE. The team [agreed these choices in #5](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/5#issuecomment-5972245779). The Monash Hub card's 60-step horizon and this checkpoint's 64-step default are different comparison settings. Report the same origins and targets for the baselines, pinned checkpoint, and later fine-tuned model.

The [2026-10-05 comparison](experiments/001-baseline-chronos.md) used the median of 20 sampled trajectories with seed 7. Untuned Chronos's validation mean MASE was 0.872259, compared with 1.042173 for the selected lag-48 baseline. The result record links per-state errors, run IDs, code/data versions and CPU wall-clock conditions. No final test evaluation or project training was performed. Prediction interval coverage and width remain unmeasured. A "zero-shot" label would mean no training *in this project*; it does not establish that the pretrained model never saw these public series.

## Limitations, risks, and future evidence

- The upstream paper lists Australian Electricity as a zero-shot evaluation dataset, separate from training datasets. This [disclosure](evaluation.md#interpretation-and-limitations) is not an independent audit of pretraining exposure.
- The selected demand data is historical and covers five Australian states. The [dataset card](DATASET_CARD.md#source-meaning-units-and-clock-limits) records MW evidence from the extraction package and limits on the archive's unit and civil-clock interpretation.
- Evidence covers untuned validation point errors and one CPU forecast-loop timing. Held-out accuracy, interval quality, energy use, fine-tuning gains and deployment remain unverified.
- A later card update must identify the exact fine-tuned artifact, training dataset version, evaluation runs, failure cases, and serving requirements. See [issue #7](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/7).

## License, citation, and update path

The upstream checkpoint is licensed under Apache 2.0; cite the [upstream Chronos
card](https://huggingface.co/amazon/chronos-t5-small) and [Chronos
paper](https://arxiv.org/abs/2403.07815). The [Hugging Face model-card
guide](https://huggingface.co/docs/hub/model-card-annotated) informs our
project-specific use, limitations, training, and evaluation sections. Update
those sections with observed results in #7; upstream performance claims are
not project results.
