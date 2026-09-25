# Australian electricity demand: extension of the Monash dataset card

**Status:** Initial M1 extension of the [existing Monash dataset card](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/58aafbe2712ff481c014f562e42723f2820fd5d4). It retains the source card's account of the selected configuration and adds our subset inspection, intended use, and checks. Units, timestamp semantics, and project evaluation splits remain open in [issue #6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6).

## Dataset summary and intended use

The project selects `australian_electricity_demand` from the [Monash Time Series Forecasting Repository](https://huggingface.co/datasets/Monash-University/monash_tsf). Its five series represent half-hourly electricity demand in New South Wales (`NSW`), Victoria (`VIC`), Queensland (`QUN` in the source file), South Australia (`SA`), and Tasmania (`TAS`). We intend to use past observations to compare short-term forecasts by a seasonal baseline, pretrained Chronos-T5-small, and a later fine-tuned version. This is a historical research benchmark, not a live feed.

The upstream card covers the entire Monash collection. For this configuration it
records five series, a half-hourly frequency, a 60-step prediction length, and
time-based validation/test splits. It describes the collection as intended for
research. Our extension records the exact selected archive, what we inspected in
it, and how this team plans to use it. The upstream splits and horizon do not
define our final evaluation protocol.

## Dataset structure and provenance

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

Each raw record contains a series ID, state code, start string, and
comma-separated numeric targets. The upstream card describes the Hub loader's
`start`, `target`, `feat_static_cat`, optional `feat_dynamic_real`, and `item_id`
fields. The Hub metadata lists five series in each of its train, validation, and
test entries. These are source-format and Hub metadata observations, not a
validated project training table.

## Considerations for using the data

- The TSF file does not state a demand unit or a time zone. Do not label the targets as MW, kW, or UTC until the original source and conversion are checked.
- A half-hourly frequency declaration and zero missing markers do not prove that reconstructed civil timestamps are continuous. Check duplicate or skipped local times and daylight-saving behavior before calendar features or split boundaries are used.
- The five states have different series lengths. Split each series chronologically, document the exact cutoffs, and prevent preprocessing fitted on future observations from reaching training data.
- Check the value range and any anomalous periods after loading. Older historical demand may not represent current grid conditions.
- Confirm whether the pretrained Chronos checkpoint could have seen these series during pretraining before interpreting a zero-shot result as out-of-sample generalization.

No raw dataset, generated table, or DVC metadata is committed by this card.
[Issue #6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6)
covers the remaining validation; DVC tracking belongs to M2.

## License, citation, and update path

The upstream card lists CC BY 4.0. The archive credits `tsibbledata` as its
extraction source. Cite the [Monash archive paper](https://openreview.net/forum?id=wEc1mgAjU-)
and the source package when publishing results. The team will extend this card
with verified units, timestamp handling, chronological cutoffs, and data-quality
findings in #6. The [Hugging Face dataset-card guide](https://github.com/huggingface/datasets/blob/main/templates/README_guide.md)
informs these additional project-specific sections.
