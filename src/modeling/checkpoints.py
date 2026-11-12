"""Keep a checkpoint's training lineage tied to its saved weights."""

import hashlib
import json
from pathlib import Path
import re

METADATA_FILENAME = "training_metadata.json"


def weights_sha256(checkpoint: Path) -> str:
    """Identify a full local checkpoint by its weights."""
    digest = hashlib.sha256()
    with (checkpoint / "model.safetensors").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_training_metadata(checkpoint: Path, training_run_id: str) -> dict:
    """Save the originating MLflow run with the weights, so moving them keeps lineage."""
    if not re.fullmatch(r"[0-9a-f]{32}", training_run_id):
        raise ValueError("training_run_id must be a full MLflow run ID")
    metadata = {
        "schema_version": 1,
        "training_run_id": training_run_id,
        "model_safetensors_sha256": weights_sha256(checkpoint),
    }
    (checkpoint / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return metadata


def describe_checkpoint(checkpoint: Path, *, provenance: Path | None = None) -> dict:
    """Read hash-matched training metadata, or record unknown lineage explicitly.

    New checkpoints carry their metadata inside the directory. A versioned sibling
    `<checkpoint>.provenance.json` supports artifacts published before that metadata
    was introduced, without changing the existing DVC artifact. An explicit
    `provenance` file is required and checked too, for a selected artifact's identity.
    """
    checkpoint = checkpoint.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise ValueError(f"No model.safetensors in checkpoint {checkpoint}")
    weights_hash = weights_sha256(checkpoint)
    sources = (
        checkpoint / METADATA_FILENAME,
        checkpoint.with_name(checkpoint.name + ".provenance.json"),
    )
    if provenance is not None:
        provenance = provenance.resolve()
        if not provenance.is_file():
            raise ValueError(f"Missing required checkpoint provenance: {provenance}")
        sources = tuple(dict.fromkeys((*sources, provenance)))
    run_id = None
    verified_sources = []
    for source in sources:
        if not source.is_file():
            continue
        metadata = json.loads(source.read_text())
        if (
            not isinstance(metadata, dict)
            or type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 1
            or not isinstance(metadata.get("training_run_id"), str)
            or not re.fullmatch(r"[0-9a-f]{32}", metadata["training_run_id"])
        ):
            raise ValueError(f"Invalid training metadata in {source}")
        if metadata.get("model_safetensors_sha256") != weights_hash:
            raise ValueError(f"Training metadata does not match checkpoint weights: {source}")
        if run_id is not None and metadata["training_run_id"] != run_id:
            raise ValueError(f"Conflicting training run IDs for checkpoint {checkpoint}")
        run_id = metadata["training_run_id"]
        verified_sources.append(str(source))
    return {
        "path": str(checkpoint),
        "sha256": weights_hash,
        "training_run_id": run_id,
        "training_metadata": verified_sources,
    }
