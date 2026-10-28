"""Read only verified training/validation data; never inspect final-test targets."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.drift.core import DETECTOR, DriftConfig, identity, validate_frame

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "monitoring/drift/config.json"


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Hash a selected file without reading adjacent data or DVC outputs."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> DriftConfig:
    """Load a declared configuration, rejecting unknown or missing fields."""
    value = json.loads(path.read_text(encoding="utf-8"))
    value["states"] = tuple(value["states"])
    return DriftConfig(**value)


def load_data(data_dir: Path, cfg: DriftConfig, *, root: Path = ROOT) -> tuple:
    """Verify the split lock, source implementation, parameters and two artifacts.

    This is an artifact-level DVC checksum check, not a claim of successful
    remote retrieval or reproduction of the complete DVC pipeline.
    """
    lock = yaml.safe_load((root / "dvc.lock").read_text(encoding="utf-8"))
    split = lock["stages"]["split"]
    params_sha256 = file_hash(root / "params.yaml")
    params = yaml.safe_load((root / "params.yaml").read_text(encoding="utf-8"))
    locked_params = split["params"]["params.yaml"]
    if any(params[name] != value for name, value in locked_params.items()):
        raise ValueError("split parameters differ from dvc.lock")
    if (
        tuple(params["data"]["states"]) != cfg.states
        or params["split"]["train_end"] != cfg.train_end
        or params["split"]["validation_end"] != cfg.validation_end
    ):
        raise ValueError("drift boundaries/states differ from the versioned split")
    deps = {item["path"]: item["md5"] for item in split["deps"]}
    if file_hash(root / "src/splits.py", "md5") != deps["src/splits.py"]:
        raise ValueError("split source differs from dvc.lock")
    raw = yaml.safe_load((root / "data/raw.dvc").read_text(encoding="utf-8"))
    if raw["outs"][0]["md5"] != deps["data/raw"]:
        raise ValueError("raw source identity differs from the split dependency")
    outputs = {item["path"]: item["md5"] for item in split["outs"]}
    frames, hashes, file_hashes = {}, {}, {"params.yaml": params_sha256}
    for name, start, end in (
        ("train", 0, cfg.train_end),
        ("validation", cfg.train_end, cfg.validation_end),
    ):
        path = data_dir / f"{name}.parquet"
        before = file_hash(path)
        md5 = file_hash(path, "md5")
        if md5 != outputs[f"data/processed/{name}.parquet"]:
            raise ValueError(f"{name} artifact differs from dvc.lock")
        frame = pd.read_parquet(path)
        if file_hash(path) != before:
            raise ValueError(f"{name} artifact changed while loading")
        validate_frame(frame, cfg)
        if list(frame.columns) != list(cfg.states) or not frame.index.equals(
            pd.RangeIndex(start, end)
        ):
            raise ValueError(f"{name} artifact has unexpected states or position boundaries")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} artifact contains missing/nonfinite demand")
        frames[name], hashes[name] = frame, md5
        file_hashes[path.name] = before
    if file_hash(root / "params.yaml") != params_sha256:
        raise ValueError("split parameters changed while loading")
    source = {
        "kind": "dvc",
        "raw_md5": deps["data/raw"],
        "split_sha256": identity(split),
        "train_md5": hashes["train"],
        "validation_md5": hashes["validation"],
        "split_source_md5": deps["src/splits.py"],
        "split_parameters": locked_params,
        "file_hashes": file_hashes,
    }
    return frames["train"], frames["validation"], source


def config_record(cfg: DriftConfig) -> dict:
    """Expose the exact bounded schedule and excluded positional edges."""
    last_end = cfg.bounds(cfg.max_windows - 1)[1]
    return {
        "settings": asdict(cfg),
        "config_sha256": identity(asdict(cfg)),
        "detector": dict(DETECTOR),
        "reference_bounds": [
            cfg.reference_end - cfg.reference_cycles * cfg.cycle_steps,
            cfg.reference_end,
        ],
        "windows": [list(cfg.bounds(index)) for index in range(cfg.max_windows)],
        "excluded_validation_edges": [
            [cfg.train_end, cfg.schedule_start],
            [last_end, cfg.validation_end],
        ],
        "nominal_state_window_alpha": cfg.family_alpha / (len(cfg.states) * cfg.max_windows),
        "nominal_feature_alpha": cfg.family_alpha / (len(cfg.states) * cfg.max_windows * 2),
        "alpha_interpretation": "nominal allocation; autocorrelated cycle features are not iid",
    }
