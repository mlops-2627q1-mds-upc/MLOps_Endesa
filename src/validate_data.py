"""Run the Great Expectations checkpoint and fail if any expectation is not met.

This script is the DVC validate-data stage. It loads the file data context
built by gx_context_configuration.py, runs the checkpoint that covers all
nine tables (5 raw + train + validation + test + origins), writes a summary
to reports/validation.json, and exits with a non-zero code if any expectation
fails so that DVC stops the pipeline before evaluation or training.
"""

import json
import sys

import great_expectations as gx
from loguru import logger

from src.config import PROJ_ROOT, REPORTS_DIR
from src.gx_context_configuration import CHECKPOINT

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_REPORT = REPORTS_DIR / "validation.json"


if __name__ == "__main__":
    # Load the file data context that gx_context_configuration.py built.
    # This reads the suites and checkpoint from the gx/ folder.
    context = gx.get_context(mode="file", project_root_dir=PROJ_ROOT)
    checkpoint = context.checkpoints.get(CHECKPOINT)

    # Run all 9 validation definitions at once. GX loads each parquet,
    # applies its suite of expectations and returns the results.
    checkpoint_result = checkpoint.run()

    # Go through every validation result and count how many expectations
    # passed and how many failed across all tables.
    total_evaluated = 0
    total_failed = 0
    failures = []

    for result in checkpoint_result.run_results.values():
        stats = result["statistics"]
        evaluated = stats["evaluated_expectations"]
        failed = stats["unsuccessful_expectations"]
        total_evaluated += evaluated
        total_failed += failed

        if failed > 0:
            # For each failed expectation, record the suite name, the type of
            # expectation and the column it was applied to so we can report
            # exactly what went wrong.
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

    success = total_failed == 0

    # Write the summary to reports/validation.json. DVC tracks this file as a
    # metric so the result is visible in dvc metrics show and in the pipeline DAG.
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
        # Exit with code 1 so DVC marks the stage as failed and stops the
        # pipeline before any evaluation or training stage runs on bad data.
        logger.error(f"{total_failed} of {total_evaluated} expectations failed.")
        for f in failures:
            logger.error(f"  [{f['suite']}] {f['expectation']} on '{f['column']}'")
        sys.exit(1)
