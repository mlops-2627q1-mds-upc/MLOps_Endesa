"""Exercise selected-model discovery without downloading data or weights."""

from pathlib import Path
import subprocess

from dvc.repo import Repo

from src.config import PROJ_ROOT


def test_default_dvc_discovery_includes_selected_model(tmp_path):
    project_dir = tmp_path.resolve()
    subprocess.run(["git", "init", "--quiet", str(project_dir)], check=True)
    Repo.init(str(project_dir)).close()
    (project_dir / ".gitignore").write_text(
        (PROJ_ROOT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
    )
    models = project_dir / "models"
    models.mkdir()
    name = "chronos-t5-small-fine-tuned"
    metadata = models / f"{name}.dvc"
    metadata.write_text(
        (PROJ_ROOT / "models" / metadata.name).read_text(encoding="utf-8"), encoding="utf-8"
    )
    weights = models / name / "model.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"fixture only")
    with Repo(str(project_dir)) as repo:
        discovered = {
            Path(stage.path).relative_to(project_dir).as_posix() for stage in repo.index.stages
        }
        assert f"models/{metadata.name}" in discovered
        assert repo.scm.is_ignored(str(weights))
        assert not repo.scm.is_ignored(str(metadata))
