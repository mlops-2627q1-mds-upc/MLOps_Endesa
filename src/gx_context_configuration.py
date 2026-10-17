"""Build and persist the Great Expectations file data context.

Run once (or after params.yaml changes) to write the suites, validation
definitions and checkpoint to gx/. The validate_data stage runs the
checkpoint; this stage only configures it.

One suite per table:
  - raw_<STATE>  x5  (src/dataset.py output)
  - train / validation / test  (split stage output)
  - origins                    (split stage output)
"""

import json
from pathlib import Path

import great_expectations as gx
from great_expectations.exceptions import DataContextError
from loguru import logger

from src.config import PROJ_ROOT
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
    "SA": 230_784,
    "TAS": 230_736,
}

# The 22 positions in TAS where the value goes negative (EDA section 5).
# They correspond to three nominal days in March 2005 where total operational demand
# dropped while industrial load stayed near its usual level. We keep them as-is.
TAS_NEGATIVE_POSITIONS = [
    55440,
    55441,
    55442,
    55443,
    55444,
    55445,
    55446,
    55447,
    55448,
    55449,
    55450,
    55451,
    55452,
    55457,
    55458,
    55460,
    55461,
    56139,
    56226,
    56227,
    56228,
    56229,
]

# Min and max values observed in the training block for each state (EDA section 2).
# We only derive these from train to avoid leaking information from validation or test.
# TAS lower bound covers the 22 known negatives; all other states are strictly positive.
TRAIN_RANGES = {
    "NSW": (3_498.0, 12_866.0),
    "VIC": (2_688.0, 9_495.0),
    "QLD": (2_008.0, 7_515.0),
    "SA": (696.0, 3_183.0),
    "TAS": (-234.0, 1_094.0),
}


def _build_raw_suite(state: str, n_rows: int) -> gx.ExpectationSuite:
    # Builds the expectation suite for one raw parquet file (data/raw/<STATE>.parquet).
    # We check that the file has the exact number of rows recorded in the dataset card,
    # that the demand column exists with the right type (float32), has no null values,
    # and contains only finite numbers. For states other than TAS we also require
    # that all values are non-negative, since only TAS has documented negative readings.
    # add_expectation() can write immediately when a same-named suite exists.
    # Pass the complete list to the constructor so candidate rules stay in memory.
    expectations = []
    expectations.append(gx.expectations.ExpectTableRowCountToEqual(value=n_rows))
    expectations.append(gx.expectations.ExpectColumnToExist(column=state))
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeOfType(column=state, type_="float32")
    )
    expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column=state))
    # Infinity check is handled by a Python function in validate_data.py because
    # ExpectColumnValuesToBeBetween(-inf, inf) always passes and does not catch inf.
    if state != "TAS":
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=state,
                min_value=0.0,
                meta={
                    "source": "notebooks/1.0-pp-demand-eda.ipynb §2",
                    "note": "no negatives outside TAS",
                },
            )
        )
    return gx.ExpectationSuite(
        name=f"raw_{state}",
        expectations=expectations,
        meta={"source": "notebooks/1.0-pp-demand-eda.ipynb §1"},
    )


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

    expectations = []
    expectations.append(gx.expectations.ExpectTableRowCountToEqual(value=n_rows))
    expectations.append(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(column_list=list(cfg.states))
    )

    for state in cfg.states:
        expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column=state))
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeOfType(column=state, type_="float32")
        )
        if name == "train":
            lo, hi = TRAIN_RANGES[state]
            expectations.append(
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

    return gx.ExpectationSuite(
        name=name,
        expectations=expectations,
        meta={"source": "notebooks/1.0-pp-demand-eda.ipynb §9"},
    )


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
    expectations = []
    expectations.append(gx.expectations.ExpectTableRowCountToEqual(value=n_rows))
    expectations.append(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=["state", "block", "origin", "context_start", "target_end"]
        )
    )
    for col in ["state", "block", "origin", "context_start", "target_end"]:
        expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(column="state", value_set=list(cfg.states))
    )
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="block", value_set=["validation", "test"]
        )
    )
    return gx.ExpectationSuite(
        name="origins",
        expectations=expectations,
        meta={"source": "notebooks/1.0-pp-demand-eda.ipynb §9"},
    )


def _suite_definition(suite: gx.ExpectationSuite) -> dict:
    """Compare validation rules and annotations without opaque object identifiers."""
    definition = suite.to_json_dict()
    definition.pop("id", None)
    for expectation in definition["expectations"]:
        expectation.pop("id", None)
    return definition


def _configure_suite(context, wanted: gx.ExpectationSuite) -> gx.ExpectationSuite:
    try:
        existing = context.suites.get(wanted.name)
    except DataContextError:
        return context.suites.add(wanted)
    if _suite_definition(existing) == _suite_definition(wanted):
        return existing
    return context.suites.add_or_update(wanted)


def _configure_definition(context, datasource, suite, path):
    name = suite.name
    try:
        asset = datasource.get_asset(name)
    except LookupError:
        asset = datasource.add_parquet_asset(name=name, path=path)
    if asset.type != "parquet":
        raise ValueError(f"GX asset {name!r} must read Parquet data")
    path_changed = asset.path != Path(path)
    if path_changed:
        asset.path = Path(path)
    try:
        batch = asset.get_batch_definition(f"{name}_batch")
    except KeyError:
        batch = asset.add_batch_definition(name=f"{name}_batch")
    if batch.partitioner is not None:
        raise ValueError(f"GX batch {name!r} must cover the complete file")
    if path_changed:
        batch.save()
    try:
        existing = context.validation_definitions.get(name)
    except DataContextError:
        existing = None
    if existing is not None and existing.data.id == batch.id and existing.suite.id == suite.id:
        return existing
    return context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name=name, data=batch, suite=suite)
    )


def _configure_checkpoint(context, definitions):
    wanted = gx.Checkpoint(
        name=CHECKPOINT,
        validation_definitions=definitions,
        actions=[gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")],
        result_format="SUMMARY",
    )
    try:
        existing = context.checkpoints.get(CHECKPOINT)
    except DataContextError:
        existing = None
    if existing is not None:
        same_definitions = [definition.id for definition in existing.validation_definitions] == [
            definition.id for definition in definitions
        ]
        same_actions = [action.dict() for action in existing.actions] == [
            action.dict() for action in wanted.actions
        ]
        if same_definitions and same_actions and existing.result_format == wanted.result_format:
            return existing
    return context.checkpoints.add_or_update(wanted)


def configure_context(project_root: Path = PROJ_ROOT, cfg: SplitConfig | None = None) -> None:
    """Persist changed GX resources and preserve equivalent saved resources."""
    if cfg is None:
        cfg = SplitConfig.from_params(project_root / "params.yaml")
    context = gx.get_context(mode="file", project_root_dir=project_root)
    data_docs_config = {
        "class_name": "SiteBuilder",
        "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
        "store_backend": {
            "class_name": "TupleFilesystemStoreBackend",
            "base_directory": "uncommitted/data_docs",
        },
    }
    context.update_data_docs_site("local_site", data_docs_config)
    datasource = context.data_sources.all().get(DATASOURCE_NAME)
    if datasource is None:
        datasource = context.data_sources.add_pandas(name=DATASOURCE_NAME)
    if datasource.type != "pandas":
        raise ValueError(f"GX datasource {DATASOURCE_NAME!r} must use pandas")

    tables = [
        (_build_raw_suite(state, CARD_LENGTHS[state]), f"data/raw/{state}.parquet")
        for state in cfg.states
    ]
    tables.extend(
        (_build_block_suite(name, cfg), f"data/processed/{name}.parquet")
        for name in ("train", "validation", "test")
    )
    tables.append((_build_origins_suite(cfg), "data/processed/origins.parquet"))
    definitions = []
    for wanted, path in tables:
        suite = _configure_suite(context, wanted)
        definitions.append(_configure_definition(context, datasource, suite, path))
        logger.info(f"Configured suite for {wanted.name}")
    _configure_checkpoint(context, definitions)

    # GX may serialize native Windows separators; persisted paths stay portable.
    gx_yml = project_root / "gx" / "great_expectations.yml"
    gx_yml.write_text(
        gx_yml.read_text(encoding="utf-8").replace("\\", "/"),
        encoding="utf-8",
        newline="\n",
    )
    reports = project_root / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "gx_configured.json").write_text(
        json.dumps({"configured": True, "suites": len(definitions)}), encoding="utf-8"
    )
    logger.success("GX context configured successfully.")


if __name__ == "__main__":
    configure_context()
