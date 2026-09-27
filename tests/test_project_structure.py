"""Smoke tests: the package imports and the CCDS directory layout is in place."""

from src import config


def test_config_paths_point_inside_the_project():
    for path in (
        config.RAW_DATA_DIR,
        config.INTERIM_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.EXTERNAL_DATA_DIR,
        config.MODELS_DIR,
        config.FIGURES_DIR,
    ):
        assert config.PROJ_ROOT in path.parents


def test_expected_directories_exist():
    for path in (
        config.RAW_DATA_DIR,
        config.INTERIM_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.EXTERNAL_DATA_DIR,
        config.MODELS_DIR,
        config.FIGURES_DIR,
    ):
        assert path.is_dir(), f"missing directory: {path}"
