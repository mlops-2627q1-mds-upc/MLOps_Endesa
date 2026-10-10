# Australian electricity demand: extension of the Monash dataset card

**Status:** Updated 2026-10-10 for [issue #6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6). This extends the [Monash dataset card](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/58aafbe2712ff481c014f562e42723f2820fd5d4) with source evidence, explicit unit/clock limits, and the current DVC version. Peer acceptance of these limits is pending; the evaluation protocol was agreed in #5 and implemented in #17.

## Dataset summary and intended use

The project selects `australian_electricity_demand` from the [Monash Time Series Forecasting Repository](https://huggingface.co/datasets/Monash-University/monash_tsf). Its five half-hourly series cover New South Wales (`NSW`), Victoria (`VIC`), Queensland (`QUN` in TSF, `QLD` in project storage), South Australia (`SA`), and Tasmania (`TAS`). The source field is operational demand excluding a separately reported industrial load. We use past targets to compare seasonal baselines, pretrained Chronos-T5-small, and a later fine-tuned version. This is a historical research benchmark, not a live feed.

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
| Archive path within the pinned Hub revision | `data/australian_electricity_demand_dataset.zip` |
| Archive SHA-256 | `b1439c28a631766bd05ee327f1f430a887b9f53b38a8cd5c0769ded1fd4aaf5a` |
| Archive size and MD5 | 5,770,526 bytes; `c9ec3ee8e81cd78d9edd2a29a7a2640b`, matching the [original Zenodo record](https://zenodo.org/records/4659727). |
| Archive contents | One `australian_electricity_demand_dataset.tsf` file |
| License shown by the Hub | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); verify original-source conditions before redistributing a derived dataset. |

On 2026-09-25, the pinned archive was downloaded and inspected outside Git. Its header declares `half_hourly`, `@missing false`, and unequal series lengths. Parsing all five records found no `?` markers, nonfinite numbers, or zeros. These ranges describe the original text values in their source scale; the unit evidence below does not change or rescale them:

| State code | Observations | Minimum | Maximum | Negative values |
| --- | ---: | ---: | ---: | ---: |
| NSW | 230,736 | 3,498.38527 | 12,865.79582 | 0 |
| VIC | 230,736 | 2,688.51661 | 9,494.01099 | 0 |
| QUN | 232,272 | 2,008.62345 | 7,514.43652 | 0 |
| SA | 230,784 | 488.83538 | 3,182.47665 | 0 |
| TAS | 230,736 | -233.90682 | 1,093.50213 | 22 |

Each raw record contains a series ID (`T1`-`T5`), state code, the same start string (`2002-01-01 00-00-00`), and comma-separated targets. The upstream card describes the Hub loader's `start`, `target`, `feat_static_cat`, optional `feat_dynamic_real`, and `item_id` fields. Its train, validation, and test entries each list five series. Those are source-format and Hub metadata observations, not this project's training table or validated calendar.

## Source meaning, units, and clock limits

Pablo's [source investigation](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6#issuecomment-6013062180) and [reviewed EDA in PR #31](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/31) trace the series to `tsibbledata`'s [five-state source files at commit `58ea72462a4b34c4d50c2223b217a891a3d9416c`](https://github.com/tidyverts/tsibbledata/tree/58ea72462a4b34c4d50c2223b217a891a3d9416c/data-raw/aus_elec). We checked those files and the pinned package documentation on 2026-10-10.

| Question | Evidence and limit | Project use |
| --- | --- | --- |
| What is measured? | The [preparation script](https://github.com/tidyverts/tsibbledata/blob/58ea72462a4b34c4d50c2223b217a891a3d9416c/data-raw/aus_elec/aus_elec.R) selects `OperationalLessIndustrial`, leaving out `Industrial`. All five DVC training prefixes `[0, 195696)` exactly match that field after casting it to `float32`; the largest rounding difference is below 0.0005. | Describe the target as operational demand excluding the separately reported industrial load, rather than all electricity consumption. |
| Physical unit | The pinned [five-state package documentation](https://github.com/tidyverts/tsibbledata/blob/58ea72462a4b34c4d50c2223b217a891a3d9416c/man/aus_elec.Rd) states MW. Neither TSF nor CSV encodes a unit, and an AEMO specification for these exact historical files has not been located. The current [three-year Victorian subset page](https://tsibbledata.tidyverts.org/reference/vic_elec.html) labels demand MWh; it does not establish a conversion for this archive. | MW has support from the extraction package, but is not independently confirmed for the archive. Keep targets and MAE in stored source scale and make no MW/kW/MWh conversion. |
| Calendar and DST | Every upstream file has consecutive dates from 2002-01-01, exactly periods 1-48 per day, and no duplicate `(Date, Period)` pairs. Pablo's training-block ramp analysis is consistent with a fixed clock. [AEMO's glossary](https://tech-specs.docs.public.aemo.com.au/Content/TOC_common-topics-DOnotChange/Glossary.htm?TocPath=_____20) identifies market time as AEST; that general convention does not identify the clock of these exact CSVs. | Retain positional half-hourly steps. AEST is a supported interpretation, not a verified civil timestamp for each value. |
| Interval alignment | The preparation script adds `Period * 30` minutes to the date, placing period 1 at 00:30 and period 48 at the following midnight. TSF instead gives a 00:00 start string. The script sets no explicit time zone or interval-start/end definition. | Do not silently interpret position 0 as a midnight measurement. Local-hour, DST, holiday, and external-feature joins require a separately reviewed timestamp mapping. |

Calendar checks read only `Date` and `Period` over the whole upstream files;
demand comparisons used `nrows=195696`. The upstream files contain 230,784 rows
for NSW, VIC, SA and TAS, and 232,272 for QLD. Compared with their row counts,
Monash retains 48 fewer observations for NSW, VIC and TAS. Calendar continuity
in the source files does not independently verify every held-out TSF value.

## Current DVC version and storage

At Git base [`d623220b6eeda1eaf0c9d1fe5d236e96183a1ec3`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/commit/d623220b6eeda1eaf0c9d1fe5d236e96183a1ec3), the current versions are:

| Artifact | Version or representation |
| --- | --- |
| Raw directory, [`data/raw.dvc`](../data/raw.dvc) | MD5 `ab1300b4a36e9e1df9fabfeec15b0f90.dir`; 7,046,290 bytes; six files: five state Parquets and `.gitkeep`. |
| Raw schema, [`src/dataset.py`](../src/dataset.py) | One `float32` column per state file; integer `timestep` index beginning at 0. IDs map `T1→NSW`, `T2→VIC`, `T3→QLD`, `T4→SA`, `T5→TAS`. No civil timestamps or industrial-load column are stored. |
| Processed blocks, [`dvc.lock`](../dvc.lock) | Train `76fc8921b85ad9d35e98b9b2451c1c09`; validation `7d3a853328e8e21438ef90a4a17fcf03`; test `f07bd910922aa740f36f6c0b91516a4f`; origins `dcfc26b69903b4d59a72b1b7b6ffcc00` (MD5). |
| Remote, [`.dvc/config`](../.dvc/config) | `origin`: `https://dagshub.com/pauadal03/MLOps_Endesa.dvc`. Authentication setup is local and must not be committed. |
| DVC software, [`uv.lock`](../uv.lock) | `3.67.1`; the artifact hashes above identify the data version. |

Raw versioning was merged in [PR #16](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/16) for #11; processed blocks in [PR #27](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/27) for #17. On 2026-10-10, the local raw directory manifest and all six file hashes matched `data/raw.dvc`. Remote retrieval was not repeated for this documentation update; earlier [fresh-cache retrieval evidence](experiments/001-baseline-chronos.md#verification-and-reproduction) is dated 2026-10-05. Changing storage precision would require a new DVC version and comparable evaluation; this card retains the current `float32` artifacts.

## Retrieval and project split

For the project's versioned data, follow the [clean-checkout setup](setup.md#setup-from-a-clean-checkout), configure your authorized local DVC credentials, and run:

```sh
uv sync --locked --dev
uv run dvc pull -r origin
```

This restores raw and processed artifacts. It does not train or evaluate a model.

Download the pinned archive **outside the repository** and compare its SHA-256 with the value above:

```sh
curl -fL 'https://huggingface.co/datasets/Monash-University/monash_tsf/resolve/58aafbe2712ff481c014f562e42723f2820fd5d4/data/australian_electricity_demand_dataset.zip' -o /tmp/australian_electricity_demand_dataset.zip
shasum -a 256 /tmp/australian_electricity_demand_dataset.zip
```

The [evaluation contract](problem-definition.md#evaluation-contract-for-m2), [agreed in #5](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/5#issuecomment-5972245779) and implemented in [#17 / PR #27](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/27), uses the following positional boundaries:

| Block | Positions per state |
| --- | --- |
| Training | `[0, 195696)` |
| Validation | `[195696, 213216)` |
| Held-out test | `[213216, 230736)` |

The common prefix excludes QLD's additional 1,536 values and SA's additional 48.
[`params.yaml`](../params.yaml) sets 512 context values, a 48-step horizon and
48-step origin stride. The DVC `split` stage applies those settings;
[`tests/test_splits.py`](../tests/test_splits.py) checks boundaries and causal
contexts. Fit transformations and MASE scales on training only, choose
configurations on validation, and keep final test targets out of model selection.

## Considerations for using the data

- Tasmania has 22 negative values. Pablo's upstream inspection shows operational demand falling below the subtracted industrial load during a March 2005 episode; the cause is unverified. Preserve the values. Percentage errors are difficult to interpret around zero or negative targets; the agreed metrics are MAE and MASE.
- Historical demand may not represent current grid conditions. The base model's possible exposure to these public series also needs investigation before claims of unseen-data generalization.

Peer review of [EDN-004](edn/0004-dataset-source-limits.md) must explicitly accept
source-scale, positional-only use with the limits above before #6 is complete.
This update changes documentation; raw data and DVC references remain unchanged.

## License, citation, and update path

The Hub card lists CC BY 4.0. The TSF header and [Zenodo record](https://zenodo.org/records/4659727) credit `tsibbledata` as the extraction source; the [Monash archive paper](https://openreview.net/forum?id=wEc1mgAjU-) describes the archive. Cite the archive and extraction package when publishing results. The [Hugging Face dataset-card guide](https://github.com/huggingface/datasets/blob/main/templates/README_guide.md) informs these project-specific sections. Update this card when new source evidence, a reviewed calendar mapping, or a new DVC version changes the recorded limits.
