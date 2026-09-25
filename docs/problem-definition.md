# M1 problem definition: Australian electricity-demand forecasting

**Status:** Inception draft for [issue #5](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/5). The dataset and base model are proposed. The evaluation contract below is a working proposal for team review, not an agreed protocol or a measured result.

## Goal and intended use

The project asks whether adapting [Chronos-T5-small](https://huggingface.co/amazon/chronos-t5-small) to the [Australian Electricity Demand](https://huggingface.co/datasets/Monash-University/monash_tsf) series improves forecasts over using the same pretrained checkpoint without project-specific training. A simple seasonal forecast should provide a second point of comparison. The course team and reviewers are the immediate users of this comparison. A grid operator or production service has not requested or validated it.

Given past half-hourly demand observations for one Australian state, the proposed ML component returns forecasts for a fixed future horizon. The five available series represent New South Wales, Victoria, Queensland, South Australia, and Tasmania. The first comparison will use past target values; any external features need a separate availability and leakage assessment.

The project-specific question is the trade-off among forecast error, uncertainty, training cost, and serving cost for these five historical series. Better results are an experiment outcome to test, not a premise of the project.

## Proposed evaluation contract for M2

| Requirement | Working proposal |
| --- | --- |
| Forecast | Use the previous 512 half-hourly target values to predict the next 48 values (one nominal day) for each state. Use no external features in the first comparison. Report the target in its source scale; its physical unit is still unverified. |
| Comparable history | Use the first 230,736 observations of each state. Set aside Queensland's remaining 1,536 and South Australia's remaining 48 observations so all states share the same index range. These are positional boundaries, not verified civil timestamps. |
| Chronological split | For each state, training indices `[0, 195696)`, validation `[195696, 213216)`, and held-out test `[213216, 230736)`. Validation and test each contain 365 × 48 observations. |
| Forecast origins | Evaluate every 48 steps within validation and test. At each origin, the 512-value context may include earlier observed validation or test targets, as a real rolling forecast would; it must never include a target at or after that origin. Training, preprocessing fits, and model selection cannot use test targets. |
| Comparators | Use a seasonal naive forecast from the same half-hour one week earlier (lag 336), the pinned Chronos checkpoint with no project fine-tuning, and a fine-tuned checkpoint. Give every method the same forecast origins and targets. |
| Metrics | Report MAE by state in the source scale. For cross-state comparison, compute each state's MASE using a lag-336 denominator estimated only on its training block, then average the five state-level MASE values equally. Use the median of Chronos samples as its point forecast; record sampling seed and count. |
| Decision rule | Before test evaluation, select the configuration on validation data. Report whether the fine-tuned model beats both comparators on held-out mean MASE, with per-state MAE and compute cost. An unchanged or worse result is still a valid finding. |
| Traceability | Pin source archive, base checkpoint, code, split configuration, seed, environment, and later trained artifact. Link DVC and MLflow records when M2 introduces them. |

This proposal uses a 48-step horizon for the project's one-day task. The [Monash dataset card](DATASET_CARD.md) lists 60 steps for its Hub configuration, and the [Chronos checkpoint](MODEL_CARD.md) defaults to 64; published scores under different horizons or splits are not directly comparable. The team must review the forecast horizon, split, metrics, and decision rule before implementing or tuning against them. The [dataset validation issue #6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6) tracks source semantics and checks; [issue #7](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/7) tracks the later trained-model evidence.

## Course context

The [course M1 guidance](https://github.com/mlops-2627q1-mds-upc/MLOps-2627q1-demos#project-milestones) asks for problem selection, requirements engineering, and model and dataset cards. It links the [Hugging Face model-card guide](https://huggingface.co/docs/hub/model-card-annotated) and [dataset-card guide](https://github.com/huggingface/datasets/blob/main/templates/README_guide.md). Our cards extend the existing Monash and Chronos cards with what this project selects, verifies, and later measures.
