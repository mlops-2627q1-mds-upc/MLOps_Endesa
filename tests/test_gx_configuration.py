"""Persisted GX setup follows configuration changes without identifier churn."""

from dataclasses import replace
import json

import great_expectations as gx
import pytest

from src.gx_context_configuration import configure_context
from src.splits import SplitConfig


def _snapshot(root):
    files = [
        root / "gx/great_expectations.yml",
        root / "reports/gx_configured.json",
        *sorted((root / "gx/expectations").glob("*.json")),
        *sorted((root / "gx/validation_definitions").glob("*.json")),
        *sorted((root / "gx/checkpoints").glob("*.json")),
    ]
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in files}


def _row_count(root, name):
    suite = json.loads((root / f"gx/expectations/{name}.json").read_text(encoding="utf-8"))
    return next(
        expectation["kwargs"]["value"]
        for expectation in suite["expectations"]
        if expectation["type"] == "expect_table_row_count_to_equal"
    )


def test_repeated_setup_preserves_configuration_and_runtime_docs_are_separate(tmp_path):
    cfg = SplitConfig.from_params()
    configure_context(tmp_path, cfg)
    first = _snapshot(tmp_path)
    context = gx.get_context(mode="file", project_root_dir=tmp_path)
    context.build_data_docs()
    assert (tmp_path / "gx/uncommitted/data_docs/index.html").is_file()
    configure_context(tmp_path, cfg)
    assert _snapshot(tmp_path) == first
    assert len(context.checkpoints.get("validation_checkpoint").validation_definitions) == 9


def test_changed_split_updates_rules_and_then_stabilizes(tmp_path):
    cfg = SplitConfig.from_params()
    configure_context(tmp_path, cfg)
    before = _snapshot(tmp_path)
    changed = replace(cfg, train_end=cfg.train_end - 48)
    configure_context(tmp_path, changed)
    assert _row_count(tmp_path, "train") == 195_648
    assert _row_count(tmp_path, "validation") == 17_568
    assert _row_count(tmp_path, "origins") == 3_655
    after = _snapshot(tmp_path)
    for name in ("raw_NSW", "raw_VIC", "raw_QLD", "raw_SA", "raw_TAS", "test"):
        path = f"gx/expectations/{name}.json"
        assert after[path] == before[path]
    configure_context(tmp_path, changed)
    assert _snapshot(tmp_path) == after


@pytest.mark.parametrize("resource", ["suite", "asset", "batch", "definition"])
def test_setup_recreates_a_missing_resource(tmp_path, resource):
    cfg = SplitConfig.from_params()
    configure_context(tmp_path, cfg)
    context = gx.get_context(mode="file", project_root_dir=tmp_path)
    datasource = context.data_sources.get("pandas")
    if resource == "suite":
        (tmp_path / "gx/expectations/raw_NSW.json").unlink()
    elif resource == "asset":
        datasource.delete_asset("raw_NSW")
    elif resource == "batch":
        datasource.get_asset("raw_NSW").delete_batch_definition("raw_NSW_batch")
    else:
        (tmp_path / "gx/validation_definitions/raw_NSW.json").unlink()
    configure_context(tmp_path, cfg)
    context = gx.get_context(mode="file", project_root_dir=tmp_path)
    definition = context.validation_definitions.get("raw_NSW")
    assert definition.suite.name == "raw_NSW"
    assert definition.data.name == "raw_NSW_batch"
    assert definition.data.data_asset.name == "raw_NSW"
    assert _row_count(tmp_path, "raw_NSW") == 230_736


def test_setup_restores_a_modified_persisted_rule(tmp_path):
    cfg = SplitConfig.from_params()
    configure_context(tmp_path, cfg)
    path = tmp_path / "gx/expectations/train.json"
    suite = json.loads(path.read_text(encoding="utf-8"))
    suite["expectations"][0]["kwargs"]["value"] = 1
    path.write_text(json.dumps(suite), encoding="utf-8")
    configure_context(tmp_path, cfg)
    assert _row_count(tmp_path, "train") == 195_696


def test_malformed_saved_suite_fails_instead_of_accepting_a_stamp(tmp_path):
    cfg = SplitConfig.from_params()
    configure_context(tmp_path, cfg)
    (tmp_path / "gx/expectations/train.json").write_text("{invalid JSON", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        configure_context(tmp_path, cfg)
