"""Durable release identities and an inherited lock for target-local transactions."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
from uuid import uuid4

from src.deployment.config import DeploymentConfig, absolute_path, file_sha256


def read_json(path: Path):
    """Refuse ambiguous JSON fields and non-finite values in operational state."""

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("Non-finite JSON value")

    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant
    )


def sync_directory(path: Path):
    """Persist directory entries as well as the bytes they name."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_json(path: Path, value: dict, *, replace=False):
    """Publish complete JSON with fsync; only the owned state pointer is replaceable."""
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    destination = path.with_name(".writing-" + uuid4().hex)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if replace:
        os.replace(destination, path)
    else:
        # Publish immutable evidence atomically without replacing another file.
        os.link(destination, path, follow_symlinks=False)
        destination.unlink()
    sync_directory(path.parent)


def local_identity(expected_machine_id: str) -> dict:
    """Bind execution to an operator-checked native Linux machine, account and hostname."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_machine_id):
        raise ValueError("Supply the SHA-256 of the verified target's /etc/machine-id")
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise ValueError("Release transactions require native Linux amd64")
    if file_sha256(Path("/etc/machine-id")) != expected_machine_id:
        raise ValueError("Target machine identity differs from the authorized pin")
    return {
        "machine_id_sha256": expected_machine_id,
        "hostname": socket.getfqdn(),
        "uid": os.geteuid(),
    }


class Store:
    """The operator creates one private directory for a service and retains it across releases."""

    def __init__(self, directory: Path, lock_file: Path, identity: dict):
        if not absolute_path(str(directory)) or directory.resolve() != directory:
            raise ValueError("State directory must be absolute without symlink components")
        metadata = directory.stat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or (stat.S_IMODE(metadata.st_mode) != 0o700)
        ):
            raise ValueError("State directory must be owned by this account with mode 0700")
        if lock_file != directory / "transaction.lock":
            raise ValueError(
                "Use the stable STATE/transaction.lock for every release and recovery"
            )
        self.directory, self.lock_file, self.identity = directory, lock_file, identity

    @contextmanager
    def locked(self):
        """Keep the same inode locked; helper children inherit its open file description."""
        descriptor = os.open(self.lock_file, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if (
                metadata.st_uid != os.geteuid()
                or not stat.S_ISREG(metadata.st_mode)
                or (stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_nlink != 1)
            ):
                raise ValueError("Unsafe transaction lock ownership or permissions")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(
                    "Deployment lock is busy, including recovery or an orphan helper"
                ) from error
            # Do not explicitly LOCK_UN: a surviving child must retain the lock after parent death.
            yield descriptor
        finally:
            os.close(descriptor)

    def load(self) -> dict | None:
        """Read the atomic state pointer without touching the target or taking its lock."""
        path = self.directory / "state.json"
        if not path.exists():
            return None
        if path.is_symlink():
            raise ValueError("State pointer may not be a symlink")
        value = read_json(path)
        if (
            not isinstance(value, dict)
            or value.keys()
            != {
                "schema_version",
                "identity",
                "generation",
                "current",
                "previous",
                "pending",
                "last_result",
            }
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or type(value["generation"]) is not int
            or (value["generation"] < 1 or value["identity"] != self.identity)
            or not isinstance(value["current"], dict)
        ):
            raise ValueError("State identity/schema mismatch")
        for accepted in (value["current"], value["previous"]):
            if accepted is not None:
                self.accepted(accepted)
        pending = value["pending"]
        if pending is not None:
            if not isinstance(pending, dict) or pending.keys() != {"id", "baseline", "candidate"}:
                raise ValueError("Malformed interrupted transaction")
            if not re.fullmatch(r"[0-9a-f]{32}", pending["id"]):
                raise ValueError("Malformed transaction identity")
            self.accepted(pending["baseline"])
            self.config(pending["candidate"])
            if pending["baseline"] != value["current"]:
                raise ValueError("Pending baseline differs from the durable accepted release")
        return value

    def save(self, value: dict):
        """Replace only this store's pointer, after immutable evidence has been persisted."""
        durable_json(self.directory / "state.json", value, replace=True)

    def snapshot(self, descriptor: Path) -> dict:
        """Pin a release's own source checkout and complete immutable deployment config."""
        selection = read_json(descriptor)
        if not isinstance(selection, dict) or selection.keys() != {"checkout", "config"}:
            raise ValueError("Slot descriptor requires exactly checkout and config")
        if any(not absolute_path(selection[k]) for k in selection):
            raise ValueError("Slot paths must be explicit and absolute")
        checkout = Path(selection["checkout"])
        if checkout.resolve() != checkout or not checkout.is_dir():
            raise ValueError("Checkout must be a regular absolute directory without symlinks")
        raw = read_json(Path(selection["config"]))
        config = DeploymentConfig(**raw)
        # Model/provenance validation belongs to this slot's source-specific helper.
        # The controller's tracked model may differ from the retained release's model.
        if type(config.schema_version) is not int or config.schema_version != 2:
            raise ValueError("Only verified published schema-2 releases can be accepted")
        if config.target_hostname != self.identity["hostname"]:
            raise ValueError("Release targets a different host")
        slots = self.directory / "slots"
        slots.mkdir(mode=0o700, exist_ok=True)
        path = slots / (config.config_sha256 + ".json")
        if not path.exists():
            durable_json(path, config.as_dict())
        slot = {"checkout": str(checkout), "config_sha256": config.config_sha256}
        self.config(slot)
        return slot

    def config_path(self, slot: dict) -> Path:
        """Derive owned snapshot paths rather than trusting arbitrary journal paths."""
        if (
            not isinstance(slot, dict)
            or slot.keys() != {"checkout", "config_sha256"}
            or (
                not isinstance(slot["config_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", slot["config_sha256"])
                or not absolute_path(slot["checkout"])
            )
        ):
            raise ValueError("Missing or malformed release identity")
        return self.directory / "slots" / (slot["config_sha256"] + ".json")

    def config(self, slot: dict) -> DeploymentConfig:
        """Detect edited snapshots before any lifecycle operation."""
        path = self.config_path(slot)
        if path.is_symlink():
            raise ValueError("Release snapshot may not be a symlink")
        config = DeploymentConfig(**read_json(path))
        if (
            type(config.schema_version) is not int
            or config.schema_version != 2
            or (config.config_sha256 != slot["config_sha256"])
            or (config.target_hostname != self.identity["hostname"])
        ):
            raise ValueError("Release snapshot identity mismatch")
        return config

    def reference(self, path: Path) -> dict:
        """Retain a relocatable hash-bound reference to immutable operation evidence."""
        return {"path": path.relative_to(self.directory).as_posix(), "sha256": file_sha256(path)}

    def accepted(self, value: dict):
        """An accepted release always has inspection and actual forecast verification evidence."""
        if not isinstance(value, dict) or value.keys() != {"slot", "inspection", "forecast"}:
            raise ValueError("Missing accepted release proof")
        config = self.config(value["slot"])
        for key in ("inspection", "forecast"):
            reference = value[key]
            if not isinstance(reference, dict) or reference.keys() != {"path", "sha256"}:
                raise ValueError("Missing evidence identity")
            path = self.directory / reference["path"]
            if (
                not path.resolve().is_relative_to(self.directory)
                or path.is_symlink()
                or (file_sha256(path) != reference["sha256"])
            ):
                raise ValueError("Accepted evidence identity mismatch")
            evidence = read_json(path)
            if evidence["status"] != "passed" or evidence["config_sha256"] != config.config_sha256:
                raise ValueError("Accepted evidence did not pass for this release")

    def attempt(self) -> Path:
        """Retain a separate operation directory for each attempt, including recovery retries."""
        attempts = self.directory / "attempts"
        attempts.mkdir(mode=0o700, exist_ok=True)
        path = attempts / uuid4().hex
        path.mkdir(mode=0o700)
        sync_directory(path.parent)
        return path
