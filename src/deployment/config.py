"""Validate an operator-selected deployment without discovering hosts or credentials."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
import fcntl
import hashlib
import json
from pathlib import Path
import re
import socket

ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = ROOT / "models/chronos-t5-small-fine-tuned.provenance.json"
COMMAND = [
    "uvicorn",
    "src.api.app:create_app",
    "--factory",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--workers",
    "1",
]


def fingerprint(value: dict) -> str:
    """Hash canonical JSON; configuration paths and resource limits are part of identity."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash files in bounded chunks, including weights without loading them into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(path: Path) -> dict:
    """Identify every checkpoint file, refusing symlinks and special files."""
    if not path.is_dir() or path.is_symlink():
        raise ValueError("Checkpoint must be an existing regular directory")
    manifest = {}
    for entry in sorted(path.rglob("*")):
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise ValueError("Checkpoint may contain only regular files and directories")
        if entry.is_file():
            manifest[entry.relative_to(path).as_posix()] = file_sha256(entry)
    if not {"model.safetensors", "config.json"} <= manifest.keys():
        raise ValueError("Checkpoint requires model.safetensors and config.json")
    return manifest


def write_new(path: Path, payload: dict):
    """Never overwrite evidence, config or another worker's output."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def absolute_path(value) -> bool:
    """Accept explicit local paths that cannot change Docker's mount arguments."""
    return (
        isinstance(value, str)
        and Path(value).is_absolute()
        and not any(c in value for c in (",", "\n", "\r", "\x00"))
        and ".." not in Path(value).parts
    )


@contextmanager
def compute_lock(path: Path):
    """Hold an advisory lock across the entire operation and its child lifetimes."""
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Compute lock is busy; retry after its owner finishes") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@dataclass(frozen=True)
class DeploymentConfig:  # pylint: disable=too-many-instance-attributes
    """No host, image, model, config or capacity default can select a real deployment."""

    schema_version: int
    deployment_id: str
    target_hostname: str
    source_commit: str
    image_id: str
    checkpoint: str
    checkpoint_sha256: str
    checkpoint_tree_sha256: str
    training_run_id: str
    params: str
    params_sha256: str
    cpu_threads: int
    memory_mb: int
    port: int
    docker_socket: str
    release: dict | None = None

    @classmethod
    def load(cls, path: Path):
        """Reject incomplete, extra and template fields before executing commands."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        expected = {f.name for f in fields(cls)} - {"release"}
        if isinstance(raw, dict) and raw.get("schema_version") == 2:
            expected.add("release")
        if not isinstance(raw, dict) or raw.keys() != expected:
            raise ValueError("Deployment config must contain exactly the documented fields")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self):
        """Validate syntax and the tracked selection, without network access."""
        if type(self.schema_version) is not int or self.schema_version not in (1, 2):
            raise ValueError("Expected deployment schema_version 1 or 2")
        if self.schema_version == 1 and self.release is not None:
            raise ValueError("Candidate schema 1 cannot contain a release binding")
        if self.schema_version == 2:
            if not isinstance(self.release, dict) or self.release.keys() != {
                "manifest",
                "manifest_sha256",
                "validation_dir",
                "image_reference",
            }:
                raise ValueError("Release schema 2 requires exactly the documented release fields")
            if any(not absolute_path(self.release[n]) for n in ("manifest", "validation_dir")):
                raise ValueError("Release bundle paths must be absolute local paths")
            if not isinstance(self.release["manifest_sha256"], str) or not re.fullmatch(
                r"[0-9a-f]{64}", self.release["manifest_sha256"]
            ):
                raise ValueError("Release manifest requires a SHA-256 pin")
            if not isinstance(self.release["image_reference"], str) or not re.fullmatch(
                r"[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}", self.release["image_reference"]
            ):
                raise ValueError("Release image requires an explicit immutable repository digest")
        patterns = {
            "deployment_id": r"endesa75-[a-z0-9][a-z0-9-]{0,31}",
            "target_hostname": r"[a-zA-Z0-9][a-zA-Z0-9.-]*",
            "source_commit": r"[0-9a-f]{40}",
            "image_id": r"sha256:[0-9a-f]{64}",
            "checkpoint_sha256": r"[0-9a-f]{64}",
            "checkpoint_tree_sha256": r"[0-9a-f]{64}",
            "training_run_id": r"[0-9a-f]{32}",
            "params_sha256": r"[0-9a-f]{64}",
        }
        for name, pattern in patterns.items():
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(pattern, value):
                raise ValueError(f"Invalid or missing deployment field: {name}")
        for name in ("checkpoint", "params", "docker_socket"):
            value = getattr(self, name)
            if not absolute_path(value):
                raise ValueError(f"{name} must be an absolute local path without mount separators")
        for name, minimum, maximum in (
            ("cpu_threads", 1, 2),
            ("memory_mb", 1, 1048576),
            ("port", 1024, 65535),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        selected = json.loads(PROVENANCE.read_text())
        if (self.checkpoint_sha256, self.training_run_id) != (
            selected["model_safetensors_sha256"],
            selected["training_run_id"],
        ):
            raise ValueError(
                "Config must select the tracked fine-tuned checkpoint and training run"
            )

    def as_dict(self) -> dict:
        """Serialize only the declared, non-secret configuration fields."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "release" or self.schema_version == 2
        }

    @property
    def config_sha256(self):
        """Return the ownership fence used for every container operation."""
        return fingerprint(self.as_dict())

    def verify_target(self):
        """Refuse a different host, including accidental execution on a developer laptop."""
        if self.target_hostname != socket.getfqdn():
            raise ValueError(
                "Configured target_hostname differs from this host's socket.getfqdn()"
            )

    def verify_artifacts(self, checkpoint: Path | None = None) -> dict:
        """Reuse serving validation and verify all selected bytes before mounting."""
        checkpoint = checkpoint if checkpoint is not None else Path(self.checkpoint)
        manifest = checkpoint_files(checkpoint)
        if manifest["model.safetensors"] != self.checkpoint_sha256 or (
            fingerprint(manifest) != self.checkpoint_tree_sha256
        ):
            raise ValueError("Checkpoint bytes differ from selected deployment configuration")
        # Older DVC artifacts carry lineage only in the tracked sibling provenance.
        # If embedded metadata exists, it must agree as the serving loader requires.
        metadata = checkpoint / "training_metadata.json"
        if metadata.exists():
            value = json.loads(metadata.read_text())
            if value.get("training_run_id") != self.training_run_id or (
                value.get("model_safetensors_sha256") != self.checkpoint_sha256
            ):
                raise ValueError("Embedded training metadata differs from the selected artifact")
        params = Path(self.params)
        if file_sha256(params) != self.params_sha256:
            raise ValueError("Params bytes differ from selected deployment configuration")
        return {
            "checkpoint": {
                "sha256": self.checkpoint_sha256,
                "training_run_id": self.training_run_id,
            },
            "checkpoint_files": manifest,
            "checkpoint_tree_sha256": fingerprint(manifest),
            "params_sha256": self.params_sha256,
        }
