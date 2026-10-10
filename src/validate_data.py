"""Run the Great Expectations checkpoint and fail if any expectation is not met.

This script is the DVC validate-data stage. It loads the file data context
built by gx_context_configuration.py, runs the checkpoint that covers all
nine tables (5 raw + train + validation + test + origins), then runs a set of
Python checks for things GX cannot express directly (infinity, exact TAS negative
positions, block contiguity and forecast window validity). It writes a summary
to reports/validation.json and exits with a non-zero code if any check fails,
so that DVC stops the pipeline before evaluation or training.
"""

import json
import sys

import great_expectations as gx
import numpy as np
import pandas as pd
from loguru import logger

from src.config import PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR, REPORTS_DIR
from src.gx_context_configuration import CHECKPOINT, TAS_NEGATIVE_POSITIONS
from src.splits import SplitConfig

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_REPORT = REPORTS_DIR / "validation.json"


def _check_finite(df: pd.DataFrame, suite_name: str) -> list[dict]:
    # GX's ExpectColumnValuesToBeBetween(-inf, inf) always passes and does not
    # catch infinite values. This function checks each numeric column explicitly.
    failures = []
    for col in df.select_dtypes(include="number").columns:
        vals = df[col].to_numpy()
        if np.any(np.isinf(vals[~np.isnan(vals)])):
            failures.append(
                {
                    "suite": suite_name,
                    "expectation": "custom_no_infinite_values",
                    "column": col,
                }
            )
    return failures


def _check_tas_negative_positions(train_df: pd.DataFrame) -> list[dict]:
    # The 22 TAS negatives must fall exactly at the documented positions (EDA §5).
    # Any negative outside those positions, or any missing position, is a failure.
    tas = train_df["TAS"]
    actual = set(int(i) for i in tas.index[tas < 0])
    expected = set(TAS_NEGATIVE_POSITIONS)
    if actual != expected:
        return [
            {
                "suite": "train",
                "expectation": "custom_tas_negative_positions",
                "column": "TAS",
            }
        ]
    return []


def _check_block_index_ranges(cfg: SplitConfig) -> list[dict]:
    # Checks that each processed block starts and ends at the positions defined in
    # params.yaml. This catches truncated or mis-aligned parquet files that would
    # still pass the row-count expectation if the wrong rows were kept.
    failures = []
    for name in ("train", "validation", "test"):
        df = pd.read_parquet(PROCESSED_DATA_DIR / f"{name}.parquet")
        start, end = cfg.bounds(name)
        if int(df.index[0]) != start or int(df.index[-1]) != end - 1:
            failures.append(
                {
                    "suite": name,
                    "expectation": "custom_block_index_range",
                    "column": "table-level",
                }
            )
    return failures


def _check_origins_windows(cfg: SplitConfig) -> list[dict]:
    # Checks that every forecast origin has a context of exactly context_length steps
    # ending at the origin, a target of exactly horizon steps, and that the target
    # does not extend beyond the end of its block.
    failures = []
    origins = pd.read_parquet(PROCESSED_DATA_DIR / "origins.parquet")

    if not (origins["origin"] - origins["context_start"] == cfg.context_length).all():
        failures.append(
            {
                "suite": "origins",
                "expectation": "custom_context_length",
                "column": "context_start",
            }
        )
    if not (origins["target_end"] - origins["origin"] == cfg.horizon).all():
        failures.append(
            {
                "suite": "origins",
                "expectation": "custom_target_length",
                "column": "target_end",
            }
        )
    for block_name in ("validation", "test"):
        _, block_end = cfg.bounds(block_name)
        if (origins.loc[origins["block"] == block_name, "target_end"] > block_end).any():
            failures.append(
                {
                    "suite": "origins",
                    "expectation": "custom_target_in_block",
                    "column": "target_end",
                }
            )
    return failures


if __name__ == "__main__":
    cfg = SplitConfig.from_params()

    # Load the file data context that gx_context_configuration.py built.
    context = gx.get_context(mode="file", project_root_dir=PROJ_ROOT)
    checkpoint = context.checkpoints.get(CHECKPOINT)

    # Run all 9 GX validation definitions at once.
    checkpoint_result = checkpoint.run()

    total_evaluated = 0
    total_failed = 0
    failures = []

    for result in checkpoint_result.run_results.values():
        stats = result["statistics"]
        total_evaluated += stats["evaluated_expectations"]
        total_failed += stats["unsuccessful_expectations"]
        if stats["unsuccessful_expectations"] > 0:
            for er in result["results"]:
                if not er["success"]:
                    failures.append(
                        {
                            "suite": result["meta"]["expectation_suite_name"],
                            "expectation": er["expectation_config"]["type"],
                            "column": er["expectation_config"]["kwargs"].get(
                                "column", "table-level"
                            ),
                        }
                    )

    # Python checks for things GX cannot express directly.
    for state in cfg.states:
        df = pd.read_parquet(RAW_DATA_DIR / f"{state}.parquet")
        inf_failures = _check_finite(df, f"raw_{state}")
        total_evaluated += 1
        total_failed += len(inf_failures)
        failures.extend(inf_failures)

    for name in ("train", "validation", "test"):
        df = pd.read_parquet(PROCESSED_DATA_DIR / f"{name}.parquet")
        inf_failures = _check_finite(df, name)
        total_evaluated += len(cfg.states)
        total_failed += len(inf_failures)
        failures.extend(inf_failures)

    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    tas_failures = _check_tas_negative_positions(train_df)
    total_evaluated += 1
    total_failed += len(tas_failures)
    failures.extend(tas_failures)

    block_failures = _check_block_index_ranges(cfg)
    total_evaluated += 3
    total_failed += len(block_failures)
    failures.extend(block_failures)

    origin_failures = _check_origins_windows(cfg)
    total_evaluated += 4
    total_failed += len(origin_failures)
    failures.extend(origin_failures)

    success = total_failed == 0

    # Write the summary to reports/validation.json as a DVC metric.
    report = {
        "success": success,
        "evaluated_expectations": total_evaluated,
        "failed_expectations": total_failed,
        "failures": failures,
    }
    VALIDATION_REPORT.write_text(json.dumps(report, indent=2))
    logger.info(f"Validation report written to {VALIDATION_REPORT}")

    if success:
        logger.success(f"All {total_evaluated} expectations passed.")
    else:
        # Exit with code 1 so DVC stops the pipeline before training or evaluation.
        logger.error(f"{total_failed} of {total_evaluated} expectations failed.")
        for f in failures:
            logger.error(f"  [{f['suite']}] {f['expectation']} on '{f['column']}'")
        sys.exit(1)
