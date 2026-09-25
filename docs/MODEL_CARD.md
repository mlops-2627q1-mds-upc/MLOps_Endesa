# Chronos-T5-small: project extension of the upstream model card

**Status:** Initial M1 extension of the [existing Chronos-T5-small model card](https://huggingface.co/amazon/chronos-t5-small/tree/a971ba21945c4f1796b17a91fe69214b5f4ad472). It carries forward the source model's architecture, usage, citation, and license, then adds our intended task and evaluation plan. No project fine-tuned checkpoint or evaluation result exists yet. [Issue #7](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/7) tracks the evidence needed to complete the project-specific parts.

## Model details and intended use

The proposed base is [Amazon Chronos-T5-small](https://huggingface.co/amazon/chronos-t5-small),
a 46-million-parameter pretrained forecasting model based on T5. The upstream
card describes scaling and quantizing time-series values into tokens, then
sampling future trajectories for probabilistic forecasts. Its usage example
shows how to load the checkpoint and call `predict`; we will adapt that pattern
to the selected demand data after its input contract is verified. Our planned
component will take a historical demand sequence for one Australian state and
forecast a fixed future horizon in the dataset's verified unit.

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

The [M1 problem definition](problem-definition.md#proposed-evaluation-contract-for-m2) proposes a common 48-step horizon, chronological train/validation/test indices, a weekly seasonal baseline, and per-state MAE plus equally weighted state-level MASE. These choices need team review before use. The Monash Hub card's 60-step horizon and this checkpoint's 64-step default are different comparison settings. Report the same origins and targets for the baseline, pinned checkpoint, and later fine-tuned model.

For the proposed point comparison, use the median of sampled trajectories and record the seed and sample count. Prediction interval quality can be reported once its coverage and width are measured; neither is available yet. Record run IDs, data and code versions, training settings, compute cost, and inference conditions when results exist. A "zero-shot" label would mean no training *in this project*; it does not establish that the pretrained model never saw these public series.

## Limitations, risks, and future evidence

- Training data for the upstream model includes public time-series collections and synthetic series, according to its card. Check overlap with the selected demand series when interpreting results.
- The selected demand data is historical, covers only five Australian states, and has unresolved unit and time-zone semantics in our card.
- No project accuracy, interval coverage, latency, energy use, model improvement, or deployment has been measured or verified.
- A later card update must identify the exact fine-tuned artifact, training dataset version, evaluation runs, failure cases, and serving requirements. See [issue #7](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/7).

## License, citation, and update path

The upstream checkpoint is licensed under Apache 2.0; cite the [upstream Chronos
card](https://huggingface.co/amazon/chronos-t5-small) and [Chronos
paper](https://arxiv.org/abs/2403.07815). The [Hugging Face model-card
guide](https://huggingface.co/docs/hub/model-card-annotated) informs our
project-specific use, limitations, training, and evaluation sections. Update
those sections with observed results in #7; upstream performance claims are
not project results.
