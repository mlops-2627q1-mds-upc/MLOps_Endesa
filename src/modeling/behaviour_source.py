"""Identify the model used by bounded behaviour checks before inference."""

from pathlib import Path

from src.modeling.checkpoints import describe_checkpoint
from src.modeling.evaluate import EvaluationConfig


def describe_model(settings: EvaluationConfig, checkpoint: Path | None = None) -> dict:
    """Record the executed local artifact separately from the default Hub settings."""
    if checkpoint is not None:
        return {"source": "local", **describe_checkpoint(checkpoint)}
    return {
        "source": "hub",
        "model_id": settings.model_id,
        "revision": settings.model_revision,
    }


def checkpoint_for_model(model: dict) -> Path | None:
    """Verify the prepared artifact has not changed before loading it in a worker."""
    if model["source"] == "hub":
        return None
    if model["source"] != "local":
        raise ValueError("Model source must be hub or local")
    checkpoint = Path(model["path"])
    current = describe_checkpoint(checkpoint)
    if current["sha256"] != model["sha256"]:
        raise ValueError("Checkpoint weights changed after preparing the behaviour checks")
    return checkpoint
