# M1 problem definition: Australian electricity-demand forecasting

**Status:** Inception draft for [issue #5](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/5). The team has proposed the dataset and model; the forecasting horizon, evaluation protocol, and success thresholds still need a recorded team choice.

## Goal and intended use

The project asks whether adapting [Chronos-T5-small](https://huggingface.co/amazon/chronos-t5-small) to the [Australian Electricity Demand](https://huggingface.co/datasets/Monash-University/monash_tsf) series improves forecasts over using the same pretrained checkpoint without project-specific training. A simple seasonal forecast should provide a second point of comparison. The course team and reviewers are the immediate users of this comparison. A grid operator or production service has not requested or validated it.

Given past half-hourly demand observations for one Australian state, the proposed ML component returns forecasts for a fixed future horizon. The five available series represent New South Wales, Victoria, Queensland, South Australia, and Tasmania. The first comparison will use past target values; any external features need a separate availability and leakage assessment.

The project-specific question is the trade-off among forecast error, uncertainty, training cost, and serving cost for these five historical series. Better results are an experiment outcome to test, not a premise of the project.

## Requirements to settle before evaluation

| Requirement | Proposed approach or open choice | Evidence needed |
| --- | --- | --- |
| Forecast horizon | Choose one horizon for all methods. A 48-step day is a candidate; the source dataset card lists a 60-step benchmark horizon, and the model configuration defaults to 64 steps. These are different comparisons. | Team decision and implementation configuration. |
| Input and output | Historical demand for one state to a future sequence in the same verified unit. Decide whether to report prediction intervals alongside point forecasts. | Data and model contract, examples, and unit verification. |
| Evaluation | Compare zero-shot Chronos, fine-tuned Chronos, and a seasonal naive baseline on identical chronological cutoffs and horizons. Keep the final test period out of model selection. | Split specification, run IDs, metrics, and reviewed results. |
| Metrics and success | Report errors per state and in aggregate, with compute and latency measurements where relevant. Select metrics and success criteria before reading test results. | Recorded decision and comparable evaluation outputs. |
| Traceability | Pin the source data, pretrained checkpoint, code, configuration, and trained artifacts. | Git/DVC/MLflow references as those systems are introduced. |

The dataset and checkpoint are described in the [dataset card](DATASET_CARD.md) and [model card](MODEL_CARD.md). [Issue #6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6) tracks data validation and the final dataset card; [issue #7](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/7) tracks the trained-model evidence. Their results must not be inferred from this inception draft.

## Course context

The [course M1 guidance](https://github.com/mlops-2627q1-mds-upc/MLOps-2627q1-demos#project-milestones) asks for problem selection, requirements engineering, and project-specific model and dataset cards. It links the [Hugging Face model-card guide](https://huggingface.co/docs/hub/model-card-annotated) and [dataset-card guide](https://github.com/huggingface/datasets/blob/main/templates/README_guide.md). This document and the two cards adapt those sections to the team's proposed component.
