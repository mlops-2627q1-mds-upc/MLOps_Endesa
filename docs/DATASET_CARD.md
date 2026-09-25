# Australian electricity demand: extension of the Monash dataset card

**Status:** M1 extension of the [existing Monash dataset card](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/58aafbe2712ff481c014f562e42723f2820fd5d4), with source-archive inspection for [issue #6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6). Physical units, civil-time semantics, and the final project evaluation protocol still require review.

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
| Archive size and MD5 | 5,770,526 bytes; `c9ec3ee8e81cd78d9edd2a29a7a2640b`, matching the [original Zenodo record](https://zenodo.org/records/4659727). |
| Archive contents | One `australian_electricity_demand_dataset.tsf` file |
| License shown by the Hub | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); verify original-source conditions before redistributing a derived dataset. |

On 2026-09-25, the pinned archive was downloaded and inspected outside Git. Its header declares `half_hourly`, `@missing false`, and unequal series lengths. Parsing all five records found no `?` markers, nonfinite numbers, or zeros. Ranges are in the source's **unverified unit**:

| State code | Observations | Minimum | Maximum | Negative values |
| --- | ---: | ---: | ---: | ---: |
| NSW | 230,736 | 3,498.38527 | 12,865.79582 | 0 |
| VIC | 230,736 | 2,688.51661 | 9,494.01099 | 0 |
| QUN | 232,272 | 2,008.62345 | 7,514.43652 | 0 |
| SA | 230,784 | 488.83538 | 3,182.47665 | 0 |
| TAS | 230,736 | -233.90682 | 1,093.50213 | 22 |

Each raw record contains a series ID (`T1`-`T5`), state code, the same start string (`2002-01-01 00-00-00`), and comma-separated targets. The upstream card describes the Hub loader's `start`, `target`, `feat_static_cat`, optional `feat_dynamic_real`, and `item_id` fields. Its train, validation, and test entries each list five series. Those are source-format and Hub metadata observations, not this project's training table or validated calendar.

## Retrieval and project split proposal

Download the pinned archive **outside the repository** and compare its SHA-256 with the value above:

```sh
curl -fL 'https://huggingface.co/datasets/Monash-University/monash_tsf/resolve/58aafbe2712ff481c014f562e42723f2820fd5d4/data/australian_electricity_demand_dataset.zip' -o /tmp/australian_electricity_demand_dataset.zip
shasum -a 256 /tmp/australian_electricity_demand_dataset.zip
```

The inspection counted comma-separated targets after each TSF `@data` record and checked every parsed number for finiteness, zero, and negative sign. It did not reconstruct per-observation timestamps. The [problem definition](problem-definition.md#proposed-evaluation-contract-for-m2) proposes a common 230,736-position prefix per state and train/validation/test index boundaries of 195,696 and 213,216. Queensland's additional 1,536 values and South Australia's additional 48 would be excluded from that comparison. This is a **proposal**, pending team review. Fit any scaler or transformation on training positions only; never use future targets when forming an origin's forecast context. DVC tracking and a versioned loader belong to M2.

## Considerations for using the data

- The TSF file does not state a demand unit or time zone. Its single start string per series provides no individual observation timestamps. Duplicate or skipped civil times and daylight-saving behavior cannot be verified from this archive alone. Use positional time until a more detailed primary source establishes calendar semantics.
- Tasmania has 22 negative values. Their meaning is unknown; preserve and investigate them rather than silently clipping or declaring them errors. Percentage errors would be hard to interpret around zero or negative targets.
- The state series have different lengths. The proposed common prefix and chronological split need team acceptance and implementation checks before evaluation.
- Historical demand may not represent current grid conditions. The base model's possible exposure to these public series also needs investigation before claims of unseen-data generalization.

No raw dataset, generated table, or DVC metadata is committed by this card.
[Issue #6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6)
tracks the unresolved source semantics and review; DVC tracking belongs to M2.

## License, citation, and update path

The Hub card lists CC BY 4.0. The TSF header and [Zenodo record](https://zenodo.org/records/4659727) credit the `tsibbledata` R package as the extraction source; the [Monash archive paper](https://openreview.net/forum?id=wEc1mgAjU-) describes the archive. Cite both when publishing results. The [Hugging Face dataset-card guide](https://github.com/huggingface/datasets/blob/main/templates/README_guide.md) informs these project-specific sections. Update the card when source semantics, approved split, and M2 data version are established.
