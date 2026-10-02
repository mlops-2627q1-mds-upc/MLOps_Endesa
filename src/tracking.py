"""Explicit MLflow runs for forecasting scripts and notebooks."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib.metadata import distributions
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Literal
from urllib.parse import urlparse

from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException

from src.config import PROJ_ROOT


def tracking_client(tracking_uri: str | None = None) -> MlflowClient:
    """Require a chosen store, rather than silently recording runs locally."""
    uri = (tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")).strip()
    if not uri:
        raise ValueError("Set MLFLOW_TRACKING_URI or pass tracking_uri explicitly.")
    if urlparse(uri).hostname == "dagshub.com":
        missing = [
            name
            for name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
            if not os.environ.get(name, "").strip()
        ]
        if missing:
            raise ValueError(f"Set your DagsHub credentials: {', '.join(missing)}.")
    return MlflowClient(tracking_uri=uri)


def _experiment_id(client: MlflowClient, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        try:
            return client.create_experiment(name)
        except MlflowException:
            # Another contributor may have created the experiment in the meantime.
            experiment = client.get_experiment_by_name(name)
            if experiment is None:
                raise
    if experiment.lifecycle_stage != "active":
        raise ValueError(f"Experiment {name!r} is deleted; choose an active experiment.")
    return experiment.experiment_id


def _provenance(data_version: str, model_revision: str) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=PROJ_ROOT, text=True, stderr=subprocess.PIPE
        ).strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "data_version": data_version,
        "model_revision": model_revision,
        "uv_lock_sha256": hashlib.sha256((PROJ_ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": dict(
            sorted((package.metadata["Name"], package.version) for package in distributions())
        ),
    }


@dataclass(frozen=True)
class TrackedRun:
    """A run and its client; no global active-run state is changed."""

    client: MlflowClient
    run_id: str

    def log_metrics(self, values: Mapping[str, float], *, step: int = 0) -> None:
        timestamp = int(time.time() * 1000)
        metrics = []
        for name, value in values.items():
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Metric {name!r} must be finite.")
            metrics.append(Metric(name, value, timestamp, step))
        self.client.log_batch(self.run_id, metrics=metrics, synchronous=True)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        self.client.log_artifact(self.run_id, str(path), artifact_path=artifact_path)


@contextmanager
def tracked_run(
    *,
    name: str,
    config: Mapping[str, Any],
    data_version: str,
    model_revision: str,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    purpose: Literal["evaluation", "smoke"] = "evaluation",
) -> Iterator[TrackedRun]:
    """Record configuration and versions once; propagate failures to the caller.

    Config must contain only JSON-serializable experiment settings, never secrets.
    The caller supplies the version of the data/model it actually used; this helper
    does not download data, select splits, or load a model.
    """
    if not data_version.strip() or not model_revision.strip():
        raise ValueError("Provide explicit data_version and model_revision values.")
    config = dict(config)
    params = [
        Param(
            key,
            value
            if isinstance(value, str)
            else json.dumps(value, sort_keys=True, allow_nan=False),
        )
        for key, value in config.items()
    ]
    provenance = _provenance(data_version, model_revision)
    client = tracking_client(tracking_uri)
    experiment = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME", "MLOps_Endesa")
    run = client.create_run(
        _experiment_id(client, experiment),
        run_name=name,
        tags={
            "mlflow.source.git.commit": provenance["git_commit"],
            "git.dirty": str(provenance["git_dirty"]).lower(),
            "data.version": data_version,
            "model.revision": model_revision,
            "uv.lock.sha256": provenance["uv_lock_sha256"],
            "run.purpose": purpose,
        },
    )
    recorder = TrackedRun(client, run.info.run_id)
    try:
        client.log_batch(recorder.run_id, params=params, synchronous=True)
        client.log_dict(recorder.run_id, config, "config.json")
        client.log_dict(recorder.run_id, provenance, "provenance.json")
        client.log_dict(recorder.run_id, _environment(), "environment.json")
        recorder.log_artifact(PROJ_ROOT / "uv.lock")
        start = time.perf_counter()
        yield recorder
        recorder.log_metrics({"run_body_seconds": time.perf_counter() - start})
    except BaseException as error:
        try:
            client.set_terminated(recorder.run_id, status="FAILED")
        except MlflowException:
            error.add_note(f"Could not mark MLflow run {recorder.run_id} as FAILED.")
        raise
    else:
        client.set_terminated(recorder.run_id, status="FINISHED")
