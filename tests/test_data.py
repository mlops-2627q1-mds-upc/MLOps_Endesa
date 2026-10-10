"""Data validation tests using the stored Great Expectations validation definitions.

These tests load the file data context built by src/gx_context_configuration.py
and run each validation definition against the real DVC data. They are skipped
automatically when the data files are not present (e.g. in CI, where DVC
credentials are not available). Run them locally after `uv run dvc pull -r origin`.
"""

import great_expectations as gx
import pytest

from src.config import PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR
from src.splits import SplitConfig

# Build the list of validation names from params.yaml so they stay in sync
# with the configuration without hardcoding state names here.
CFG = SplitConfig.from_params()
VALIDATION_NAMES = [f"raw_{state}" for state in CFG.states] + [
    "train",
    "validation",
    "test",
    "origins",
]

# Skip the whole module when data files are not on disk (e.g. in CI).
data_available = (
    all((RAW_DATA_DIR / f"{state}.parquet").exists() for state in CFG.states)
    and (PROCESSED_DATA_DIR / "train.parquet").exists()
)
pytestmark = pytest.mark.skipif(
    not data_available,
    reason="DVC data not present; run `uv run dvc pull -r origin` first",
)


@pytest.fixture(scope="module")
def gx_context():
    # Load the context once per module to avoid re-reading gx/ for every test.
    return gx.get_context(mode="file", project_root_dir=PROJ_ROOT)


@pytest.mark.parametrize("name", VALIDATION_NAMES)
def test_validation_definition(gx_context, name):
    # Run the validation definition and check that no expectation failed.
    val_def = gx_context.validation_definitions.get(name)
    result = val_def.run(result_format="BOOLEAN_ONLY")
    failed = result["statistics"]["unsuccessful_expectations"]
    assert failed == 0, (
        f"Validation '{name}' had {failed} failing expectation(s). "
        "Run `uv run python -m src.validate_data` for details."
    )
