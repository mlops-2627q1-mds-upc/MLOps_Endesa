# Data validation

Automated quality checks for the raw and processed data produced by the DVC pipeline.
Related issue: [#36](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/36).

## Pipeline stages

Two DVC stages handle validation:

| Stage | Script | Runs when |
|---|---|---|
| `configure_gx` | `src/gx_context_configuration.py` | `params.yaml` or the script changes |
| `validate-data` | `src/validate_data.py` | Data or GX config changes |

`configure_gx` writes the GX suites and checkpoint to `gx/` and stamps
`reports/gx_configured.json`. `validate-data` depends on that stamp, so DVC
always runs configure before validate.

## What is checked

### Great Expectations suites (9 tables)

| Suite | File | Key expectations |
|---|---|---|
| `raw_NSW/VIC/QLD/SA/TAS` | `data/raw/<STATE>.parquet` | Row count matches dataset card; column exists; float32; no nulls; non-negative (all except TAS) |
| `train` | `data/processed/train.parquet` | Row count; column order; no nulls; float32; values within training-block range (no leakage) |
| `validation` | `data/processed/validation.parquet` | Row count; column order; no nulls; float32 |
| `test` | `data/processed/test.parquet` | Row count; column order; no nulls; float32 |
| `origins` | `data/processed/origins.parquet` | Row count; column order; no nulls; valid state and block values |

### Python checks (GX cannot express these)

| Check | Function | What it catches |
|---|---|---|
| No infinite values | `_check_finite` | `np.inf` in any numeric column; GX `ExpectColumnValuesToBeBetween(-inf, inf)` always passes |
| TAS negative positions | `_check_tas_negative_positions` | Checks the full raw TAS series; the 22 documented negatives must fall at exactly the positions in `TAS_NEGATIVE_POSITIONS` |
| Complete block indexes | `_check_block_index_ranges` | Each processed block must have a contiguous `RangeIndex(start, end)`; catches interior gaps and duplicates that row-count checks miss |
| Origins grid | `_check_origins_windows` | Integer and non-negative positions; exact context length and horizon; target within block; complete unique `(block, state)` grid with no duplicates or missing windows |

## Output

`reports/validation.json` (DVC metric, not cached):

```json
{
  "success": true,
  "evaluated_expectations": 118,
  "failed_expectations": 0,
  "failures": []
}
```

On the current valid data, the total is 118: 74 GX expectations and 44 Python
checks (20 finite-value, one TAS-position, three block-index, and 20 origin-window
checks). The origin-window count includes three integer checks, three non-negative
checks, context length, horizon, two block bounds, and two origin grids per state.

The pipeline exits with code 1 and logs each failure if any check fails.

## Running locally

```sh
# Configure GX (only needed once or after params.yaml changes)
uv run python -m src.gx_context_configuration

# Run all checks
uv run python -m src.validate_data

# Or run the full pipeline
uv run dvc repro validate-data
```

## Tests

`tests/test_data.py` runs the nine GX validation definitions against the real
DVC data. It is skipped automatically in CI where data is absent.

`tests/test_validation_checks.py` tests the four Python check functions with
small synthetic fixtures: infinity rejection, wrong TAS positions, index gaps,
duplicate origins, and missing origin windows. It also calls the production
`main()` with temporary Parquet files and deterministic GX results, checking the
passing report and normal return (CLI exit 0), and the failed report, failure
details, and `SystemExit(1)`. These tests run in ordinary CI without DVC data,
credentials, or model downloads.

```sh
uv run --locked pytest tests/test_validation_checks.py -q
```

To verify the exit regression locally, temporarily remove `sys.exit(1)` from
`main()` and run `test_validation_command_gx_failure_exits_1`. It must fail with
`DID NOT RAISE`; restore the line before running the checks again.
