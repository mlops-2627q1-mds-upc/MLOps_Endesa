"""Build and persist the Great Expectations file data context.

Run once (or after params.yaml changes) to write the suites, validation
definitions and checkpoint to gx/. The validate_data stage runs the
checkpoint; this stage only configures it.

One suite per table:
  - raw_<STATE>  x5  (src/dataset.py output)
  - train / validation / test  (split stage output)
  - origins                    (split stage output)
"""

import great_expectations as gx
from loguru import logger
import numpy as np

from src.config import PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR
from src.splits import SplitConfig

# Names imported by validate_data.py and tests/test_data.py
DATASOURCE_NAME = "pandas"
CHECKPOINT = "validation_checkpoint"

# Number of rows expected per state in data/raw/, taken from the dataset card (EDA section 1).
# QLD and SA have extra steps beyond the common prefix; those are intentional and documented.
CARD_LENGTHS = {
    "NSW": 230_736,
    "VIC": 230_736,
    "QLD": 232_272,
    "SA":  230_784,
    "TAS": 230_736,
}

# The 22 positions in TAS where the value goes negative (EDA section 5).
# They correspond to three nominal days in March 2005 where total operational demand
# dropped while industrial load stayed near its usual level. We keep them as-is.
TAS_NEGATIVE_POSITIONS = [
    55440, 55441, 55442, 55443, 55444, 55445, 55446, 55447,
    55448, 55449, 55450, 55451, 55452, 55457, 55458, 55460,
    55461, 56139, 56226, 56227, 56228, 56229,
]

# Min and max values observed in the training block for each state (EDA section 2).
# We only derive these from train to avoid leaking information from validation or test.
# TAS lower bound covers the 22 known negatives; all other states are strictly positive.
TRAIN_RANGES = {
    "NSW": (3_498.0, 12_866.0),
    "VIC": (2_688.0,  9_495.0),
    "QLD": (2_008.0,  7_515.0),
    "SA":  (  696.0,  3_183.0),
    "TAS": ( -234.0,  1_094.0),
}


def _build_raw_suite(state: str, n_rows: int) -> gx.ExpectationSuite:
    # Builds the expectation suite for one raw parquet file (data/raw/<STATE>.parquet).
    # We check that the file has the exact number of rows recorded in the dataset card,
    # that the demand column exists with the right type (float32), has no null values,
    # and contains only finite numbers. For states other than TAS we also require
    # that all values are non-negative, since only TAS has documented negative readings.
    suite = gx.ExpectationSuite(
        name=f"raw_{state}",
        meta={"source": "notebooks/1.0-pp-demand-eda.ipynb §1"},
    )
    suite.add_expectation(gx.expectations.ExpectTableRowCountToEqual(value=n_rows))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column=state))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(column=state, type_="float32")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column=state)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column=state,
            min_value=float(-np.inf),
            max_value=float(np.inf),
            meta={"note": "rejects NaN and Inf; finite check"},
        )
    )
    if state != "TAS":
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=state,
                min_value=0.0,
                meta={
                    "source": "notebooks/1.0-pp-demand-eda.ipynb §2",
                    "note": "no negatives outside TAS",
                },
            )
        )
    return suite


def _build_block_suite(name: str, cfg: SplitConfig) -> gx.ExpectationSuite:
    # Builds the expectation suite for one of the three processed blocks:
    # train, validation or test (data/processed/<name>.parquet).
    # All three blocks are checked for the correct number of rows (derived from
    # params.yaml), the right column order, no nulls and float32 types.
    # For the training block only, we also check that every column stays within
    # the value range observed during training. We deliberately skip range checks
    # on validation and test to avoid any form of data leakage.
    start, end = cfg.bounds(name)
    n_rows = end - start

    suite = gx.ExpectationSuite(
        name=name,
        meta={"source": "notebooks/1.0-pp-demand-eda.ipynb §9"},
    )
    suite.add_expectation(gx.expectations.ExpectTableRowCountToEqual(value=n_rows))
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=list(cfg.states)
        )
    )

    for state in cfg.states:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=state)
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeOfType(column=state, type_="float32")
        )
        if name == "train":
            lo, hi = TRAIN_RANGES[state]
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=state,
                    min_value=float(lo),
                    max_value=float(hi),
                    meta={
                        "source": "notebooks/1.0-pp-demand-eda.ipynb §2",
                        "note": "range derived from training block only",
                    },
                )
            )

    return suite


def _build_origins_suite(cfg: SplitConfig) -> gx.ExpectationSuite:
    # Builds the expectation suite for the forecast origins table
    # (data/processed/origins.parquet). Each row represents a forecast window:
    # one origin position per state per day in validation and test.
    # We check the total number of rows, that all required columns are present
    # and non-null, and that the state and block columns only contain valid values.
    n_rows = sum(
        len(range(cfg.bounds(b)[0], cfg.bounds(b)[1] - cfg.horizon + 1, cfg.origin_stride))
        * len(cfg.states)
        for b in ("validation", "test")
    )
    suite = gx.ExpectationSuite(
        name="origins",
        meta={"source": "notebooks/1.0-pp-demand-eda.ipynb §9"},
    )
    suite.add_expectation(gx.expectations.ExpectTableRowCountToEqual(value=n_rows))
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=["state", "block", "origin", "context_start", "target_end"]
        )
    )
    for col in ["state", "block", "origin", "context_start", "target_end"]:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="state", value_set=list(cfg.states)
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="block", value_set=["validation", "test"]
        )
    )
    return suite


if __name__ == "__main__":
    # Read the split parameters from params.yaml so the suite boundaries
    # always stay in sync with what the split stage actually produces.
    cfg = SplitConfig.from_params()

    # Create or open the GX file data context rooted at the project directory.
    # On the first run this creates the gx/ folder with great_expectations.yml.
    context = gx.get_context(mode="file", project_root_dir=PROJ_ROOT)

    # Store the Data Docs HTML report inside gx/data_docs/ so it is committed
    # to Git and can be reviewed without running GX again.
    data_docs_config = {
        "class_name": "SiteBuilder",
        "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
        "store_backend": {
            "class_name": "TupleFilesystemStoreBackend",
            "base_directory": "data_docs",
        },
    }
    context.update_data_docs_site("local_site", data_docs_config)

    datasource = context.data_sources.add_or_update_pandas(name=DATASOURCE_NAME)

    validation_definitions = []

    # Register one asset, suite and validation definition per raw state file.
    for state in cfg.states:
        asset = datasource.add_parquet_asset(
            name=f"raw_{state}",
            path=RAW_DATA_DIR / f"{state}.parquet",
        )
        batch_def = asset.add_batch_definition(name=f"raw_{state}_batch")
        suite = _build_raw_suite(state, CARD_LENGTHS[state])
        context.suites.add_or_update(suite)
        suite.save()
        val_def = gx.ValidationDefinition(
            name=f"raw_{state}",
            data=batch_def,
            suite=suite,
        )
        context.validation_definitions.add_or_update(val_def)
        validation_definitions.append(val_def)
        logger.info(f"Configured suite for raw_{state}")

    # Register one asset, suite and validation definition per processed block.
    for block_name in ("train", "validation", "test"):
        asset = datasource.add_parquet_asset(
            name=block_name,
            path=PROCESSED_DATA_DIR / f"{block_name}.parquet",
        )
        batch_def = asset.add_batch_definition(name=f"{block_name}_batch")
        suite = _build_block_suite(block_name, cfg)
        context.suites.add_or_update(suite)
        suite.save()
        val_def = gx.ValidationDefinition(
            name=block_name,
            data=batch_def,
            suite=suite,
        )
        context.validation_definitions.add_or_update(val_def)
        validation_definitions.append(val_def)
        logger.info(f"Configured suite for {block_name}")

    # Register the origins table.
    asset = datasource.add_parquet_asset(
        name="origins",
        path=PROCESSED_DATA_DIR / "origins.parquet",
    )
    batch_def = asset.add_batch_definition(name="origins_batch")
    suite = _build_origins_suite(cfg)
    context.suites.add_or_update(suite)
    suite.save()
    val_def = gx.ValidationDefinition(
        name="origins",
        data=batch_def,
        suite=suite,
    )
    context.validation_definitions.add_or_update(val_def)
    validation_definitions.append(val_def)
    logger.info("Configured suite for origins")

    # Create a single checkpoint that runs all validation definitions and
    # regenerates the Data Docs HTML report after each run.
    checkpoint = gx.Checkpoint(
        name=CHECKPOINT,
        validation_definitions=validation_definitions,
        actions=[gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")],
        result_format="SUMMARY",
    )
    context.checkpoints.add_or_update(checkpoint)
    logger.success("GX context configured successfully.")
