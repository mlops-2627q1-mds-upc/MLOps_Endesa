# Dataset card: Australian electricity demand for the Endesa project

**Status:** Initial M1 card. Source facts and a limited archive inspection are recorded below. Units, timestamp semantics, and the project's evaluation splits remain open in [issue #6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6).

## Dataset and intended use

The project selects `australian_electricity_demand` from the [Monash Time Series Forecasting Repository](https://huggingface.co/datasets/Monash-University/monash_tsf). Its five series represent half-hourly electricity demand in New South Wales (`NSW`), Victoria (`VIC`), Queensland (`QUN` in the source file), South Australia (`SA`), and Tasmania (`TAS`). We intend to use past observations to compare short-term forecasts by a seasonal baseline, pretrained Chronos-T5-small, and a later fine-tuned version. This is a historical research benchmark, not a live feed.

The [Monash dataset card](https://huggingface.co/datasets/Monash-University/monash_tsf) describes the collection, its time-based validation/test splits, a 60-step prediction length for this configuration, and its research purpose. Those upstream splits and horizon do not define this project's final evaluation protocol.

## Source and inspection

| Item | Recorded value |
| --- | --- |
| Hub dataset revision | [`58aafbe2712ff481c014f562e42723f2820fd5d4`](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/58aafbe2712ff481c014f562e42723f2820fd5d4) |
| Selected archive | `data/australian_electricity_demand_dataset.zip` |
| Archive SHA-256 | `b1439c28a631766bd05ee327f1f430a887b9f53b38a8cd5c0769ded1fd4aaf5a` |
| Archive contents | One `australian_electricity_demand_dataset.tsf` file |
| License shown by the Hub | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); verify original-source conditions before redistributing a derived dataset. |

On 2026-09-25, the source archive was downloaded for inspection outside Git. The TSF header declares `half_hourly`, `@missing false`, and unequal series lengths. Parsing its five records found no `?` missing-value markers:

| State code | Start string in TSF | Observations | Missing markers |
| --- | --- | ---: | ---: |
| NSW | `2002-01-01 00-00-00` | 230,736 | 0 |
| VIC | `2002-01-01 00-00-00` | 230,736 | 0 |
| QUN | `2002-01-01 00-00-00` | 232,272 | 0 |
| SA | `2002-01-01 00-00-00` | 230,784 | 0 |
| TAS | `2002-01-01 00-00-00` | 230,736 | 0 |

Each raw record contains a series ID, state code, start string, and comma-separated numeric targets. The Hub loader exposes `start`, `target`, `feat_static_cat`, optional `feat_dynamic_real`, and `item_id`; its train/validation/test entries each contain five series. These are source-format and Hub metadata observations, not a validated project training table.

## Limits and checks before use

- The TSF file does not state a demand unit or a time zone. Do not label the targets as MW, kW, or UTC until the original source and conversion are checked.
- A half-hourly frequency declaration and zero missing markers do not prove that reconstructed civil timestamps are continuous. Check duplicate or skipped local times and daylight-saving behavior before calendar features or split boundaries are used.
- The five states have different series lengths. Split each series chronologically, document the exact cutoffs, and prevent preprocessing fitted on future observations from reaching training data.
- Check the value range and any anomalous periods after loading. Older historical demand may not represent current grid conditions.
- Confirm whether the pretrained Chronos checkpoint could have seen these series during pretraining before interpreting a zero-shot result as out-of-sample generalization.

No raw dataset, generated table, or DVC metadata is committed by this card. [Issue #6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6) covers the remaining validation; DVC tracking belongs to M2.

## Attribution

The archive credits `tsibbledata` as its extraction source. Cite the [Monash archive paper](https://openreview.net/forum?id=wEc1mgAjU-) and the source package when publishing results. This card adapts the sections in the [Hugging Face dataset-card guide](https://github.com/huggingface/datasets/blob/main/templates/README_guide.md) to our selected subset; it does not replace the upstream card.
