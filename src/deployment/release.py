"""Bind local deployment selections to #78's offline manifest verifier."""

import hashlib
import json
from pathlib import Path

from src.deployment.config import ROOT, DeploymentConfig


def prediction_settings(manifest: dict) -> dict:
    """Translate the release prediction record to the unchanged HTTP settings shape."""
    prediction = manifest["configuration"]["prediction"]
    return {
        **{n: prediction[n] for n in ("num_samples", "seed", "device", "cpu_threads", "sampling")},
        "dtype": "float32",
        "quantile_method": "linear",
        "point_statistic": "median",
    }


def verify_release(
    config: DeploymentConfig,
    *,
    manifest_path: Path | None = None,
    validation_dir: Path | None = None,
) -> dict | None:
    """Verify evidence once through the producer; compare only deployment-specific fields.

    The caller obtains the bundle from a trusted successful publication run.
    Client-side paths may differ from the target's paths; pinned bytes may not.
    No network, Docker operation, model retrieval or inference runs here.
    """
    if config.release is None:
        if manifest_path is not None or validation_dir is not None:
            raise ValueError("A release bundle requires deployment config schema 2")
        return None
    if (manifest_path is None) != (validation_dir is None):
        raise ValueError("Supply both client release bundle paths")
    # Candidate operations and --help retain the minimal bootstrap environment.
    # Release operations require the exact source's full locked host environment.
    from src.release_manifest import verify_manifest  # pylint: disable=import-outside-toplevel

    selection = config.release
    manifest_path = manifest_path if manifest_path is not None else Path(selection["manifest"])
    validation_dir = (
        validation_dir if validation_dir is not None else Path(selection["validation_dir"])
    )
    try:
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != selection["manifest_sha256"]:
            raise ValueError("Release manifest bytes differ from the config pin")
        manifest = verify_manifest(
            json.loads(raw), validation_dir, selection["image_reference"], "amd64", root=ROOT
        )
        if (
            config.source_commit != manifest["source"]["git_commit"]
            or config.params_sha256 != manifest["configuration"]["params_sha256"]
            or config.checkpoint_sha256 != manifest["model"]["weights_sha256"]
            or config.training_run_id != manifest["model"]["training_run_id"]
            or config.cpu_threads != manifest["configuration"]["prediction"]["cpu_threads"]
        ):
            raise ValueError("Deployment source/model/params/threads differ from the release")
    except Exception as error:  # pylint: disable=broad-exception-caught
        # Dependency/report errors may contain untrusted content; keep it out of logs.
        raise ValueError(
            "Release binding rejected; check pinned source, bundle and configuration"
        ) from error
    return {"manifest_sha256": selection["manifest_sha256"], "manifest": manifest}
