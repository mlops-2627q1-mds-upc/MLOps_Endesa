"""Selected DVC artifact checks, excluding the final-test file."""

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.drift.core import DriftConfig
from src.drift.data import file_hash, load_data


@pytest.fixture
def artifacts(tmp_path):
    """A tiny local DVC manifest and two parquet outputs, without a test file."""
    cfg = DriftConfig(
        train_end=64, validation_end=256, cycle_steps=4, reference_cycles=8, max_windows=6
    )
    params = {
        "data": {"states": list(cfg.states), "common_length": 300},
        "split": {"train_end": cfg.train_end, "validation_end": cfg.validation_end},
        "forecast": {"context_length": 8, "horizon": 4, "origin_stride": 4},
    }
    (tmp_path / "src").mkdir()
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "src/splits.py").write_text("# fixture split source\n")
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    (tmp_path / "data/raw.dvc").write_text("outs:\n- md5: fixture.dir\n")
    outputs = []
    for name, start, end in (("train", 0, 64), ("validation", 64, 256)):
        frame = pd.DataFrame(
            {state: [1.0] * (end - start) for state in cfg.states}, index=range(start, end)
        )
        path = tmp_path / f"data/processed/{name}.parquet"
        frame.to_parquet(path)
        outputs.append({"path": str(path.relative_to(tmp_path)), "md5": file_hash(path, "md5")})
    lock = {
        "stages": {
            "split": {
                "params": {"params.yaml": params},
                "deps": [
                    {"path": "src/splits.py", "md5": file_hash(tmp_path / "src/splits.py", "md5")},
                    {"path": "data/raw", "md5": "fixture.dir"},
                ],
                "outs": outputs,
            }
        }
    }
    (tmp_path / "dvc.lock").write_text(yaml.safe_dump(lock))
    return tmp_path, cfg


def test_verified_parquet_roundtrip_opens_only_train_validation(artifacts, monkeypatch):
    root, cfg = artifacts
    read = pd.read_parquet
    opened = []

    def checked(path):
        opened.append(Path(path).name)
        return read(path)

    monkeypatch.setattr(pd, "read_parquet", checked)
    train, validation, source = load_data(root / "data/processed", cfg, root=root)
    assert len(train) == cfg.train_end
    assert validation.index[0] == cfg.train_end
    assert opened == ["train.parquet", "validation.parquet"]
    assert source["kind"] == "dvc"
    assert set(source["file_hashes"]) == {"train.parquet", "validation.parquet", "params.yaml"}


@pytest.mark.parametrize("changed", ["data", "params", "source", "raw"])
def test_dvc_mismatches_stop_before_evidence(artifacts, changed):
    root, cfg = artifacts
    if changed == "data":
        (root / "data/processed/train.parquet").write_bytes(b"changed")
    elif changed == "params":
        params = yaml.safe_load((root / "params.yaml").read_text())
        params["split"]["train_end"] += 1
        (root / "params.yaml").write_text(yaml.safe_dump(params))
    elif changed == "source":
        (root / "src/splits.py").write_text("# changed source\n")
    else:
        (root / "data/raw.dvc").write_text("outs:\n- md5: changed.dir\n")
    with pytest.raises(ValueError, match="differ"):
        load_data(root / "data/processed", cfg, root=root)


def test_missing_artifact_and_wrong_config_fail(artifacts):
    root, cfg = artifacts
    (root / "data/processed/train.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        load_data(root / "data/processed", cfg, root=root)
    changes = {**asdict(cfg), "train_end": cfg.train_end - 1}
    with pytest.raises(ValueError, match="boundaries/states"):
        load_data(root / "data/processed", DriftConfig(**changes), root=root)
