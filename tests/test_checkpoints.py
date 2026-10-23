"""Checkpoint lineage survives promotion without trusting directory names."""

import hashlib
import json
import shutil

import pytest

from src.modeling import train
from src.modeling.checkpoints import (
    METADATA_FILENAME,
    describe_checkpoint,
    write_training_metadata,
)

RUN_ID = "a" * 32


def test_training_output_retains_its_run_when_promoted(tmp_path):
    """Exercise the actual final-checkpoint writer without model dependencies."""

    class SavedModel:
        def save_pretrained(self, directory):
            directory.mkdir(parents=True)
            (directory / "model.safetensors").write_bytes(b"trained weights")

    original = tmp_path / "fine-tuning" / RUN_ID
    saved = train.save_final(SavedModel(), original, lora=False, training_run_id=RUN_ID)
    assert saved["training_run_id"] == RUN_ID
    assert saved["model_safetensors_sha256"] == hashlib.sha256(b"trained weights").hexdigest()
    promoted = tmp_path / "models" / "selected-checkpoint"
    shutil.copytree(original / "final", promoted)
    recorded = describe_checkpoint(promoted)
    assert recorded["training_run_id"] == RUN_ID
    assert recorded["sha256"] == saved["model_safetensors_sha256"]
    assert recorded["training_metadata"] == [str(promoted / METADATA_FILENAME)]


def test_legacy_promoted_checkpoint_uses_a_hash_matched_sidecar(tmp_path):
    checkpoint = tmp_path / "models" / "selected-checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"published weights")
    metadata = write_training_metadata(checkpoint, RUN_ID)
    # Older published DVC directories have no metadata inside; keep their bytes intact.
    sidecar = checkpoint.with_name(checkpoint.name + ".provenance.json")
    (checkpoint / METADATA_FILENAME).rename(sidecar)
    recorded = describe_checkpoint(checkpoint)
    assert recorded["training_run_id"] == RUN_ID
    assert recorded["sha256"] == metadata["model_safetensors_sha256"]
    assert recorded["training_metadata"] == [str(sidecar)]


@pytest.mark.parametrize("parent", ["models", RUN_ID])
def test_missing_metadata_does_not_infer_a_run_from_the_path(tmp_path, parent):
    checkpoint = tmp_path / parent / "final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"unknown weights")
    recorded = describe_checkpoint(checkpoint)
    assert recorded["training_run_id"] is None
    assert recorded["training_metadata"] == []


def test_changed_weights_cannot_reuse_training_metadata(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"original weights")
    write_training_metadata(tmp_path, RUN_ID)
    (tmp_path / "model.safetensors").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="does not match checkpoint weights"):
        describe_checkpoint(tmp_path)


def test_conflicting_metadata_cannot_silently_choose_a_run(tmp_path):
    checkpoint = tmp_path / "selected-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    metadata = write_training_metadata(checkpoint, RUN_ID)
    sidecar = checkpoint.with_name(checkpoint.name + ".provenance.json")
    sidecar.write_text(json.dumps({**metadata, "training_run_id": "b" * 32}))
    with pytest.raises(ValueError, match="Conflicting training run IDs"):
        describe_checkpoint(checkpoint)


def test_required_provenance_verifies_selected_weights_after_relocation(tmp_path):
    checkpoint = tmp_path / "relocated"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"selected weights")
    metadata = write_training_metadata(checkpoint, RUN_ID)
    tracked = tmp_path / "selected.provenance.json"
    (checkpoint / METADATA_FILENAME).rename(tracked)
    recorded = describe_checkpoint(checkpoint, provenance=tracked)
    assert recorded["training_run_id"] == RUN_ID
    assert recorded["sha256"] == metadata["model_safetensors_sha256"]
    assert recorded["training_metadata"] == [str(tracked)]
    (checkpoint / "model.safetensors").write_bytes(b"replacement weights")
    write_training_metadata(checkpoint, RUN_ID)
    with pytest.raises(ValueError, match="does not match checkpoint weights"):
        describe_checkpoint(checkpoint, provenance=tracked)


def test_required_provenance_cannot_be_missing_even_with_internal_metadata(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    write_training_metadata(tmp_path, RUN_ID)
    with pytest.raises(ValueError, match="Missing required checkpoint provenance"):
        describe_checkpoint(tmp_path, provenance=tmp_path / "absent.json")


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": 2},
        {"training_run_id": "invalid"},
    ],
)
def test_malformed_training_metadata_is_rejected(tmp_path, metadata):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    valid = write_training_metadata(tmp_path, RUN_ID)
    payload = {**valid, **metadata} if isinstance(metadata, dict) else metadata
    if metadata == {}:
        payload = {}
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid training metadata"):
        describe_checkpoint(tmp_path)
