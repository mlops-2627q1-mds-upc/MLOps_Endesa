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

Compare a seasonal naive forecast, the pinned checkpoint without project fine-tuning, and the fine-tuned version on the same states, horizon, cutoffs, preprocessing, and metrics. A 48-step day is a candidate horizon; the Monash Hub card lists 60 steps for the selected configuration, and the model configuration defaults to 64. Select the comparison protocol before tuning or inspecting the held-out test period. Report per-state results as well as an aggregate so differences in scale do not disappear.

Possible outputs include point-forecast error and interval behavior; metric definitions, aggregation, and success criteria have not been chosen. Record run IDs, data and code versions, seeds, training settings, compute cost, and inference conditions when results exist. A "zero-shot" label would mean no training *in this project*; it does not by itself establish that the pretrained model never saw these public series.

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
