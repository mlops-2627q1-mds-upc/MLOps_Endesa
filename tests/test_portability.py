"""Keep DVC-tracked source bytes identical under Windows Git checkout settings."""

import subprocess

import pytest

from src.config import PROJ_ROOT


@pytest.mark.parametrize("autocrlf", ["true", "false"])
def test_checkout_preserves_dvc_input_bytes_and_binary_files(tmp_path, autocrlf):
    repository = tmp_path / "repository"
    repository.mkdir()
    files = {
        path: (PROJ_ROOT / path).read_bytes()
        for path in ("src/splits.py", "params.yaml", "dvc.lock", ".gitattributes")
    }
    files["figure.png"] = b"\x89PNG\r\n\x1a\n\x00\xff"
    for name, content in files.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    # core.eol=crlf simulates Windows' native checkout on any host.
    git = ["git", "-c", f"core.autocrlf={autocrlf}", "-c", "core.eol=crlf"]
    subprocess.run([*git, "init", "-q"], cwd=repository, check=True)
    subprocess.run([*git, "add", "."], cwd=repository, check=True)
    checkout = tmp_path / "checkout"
    subprocess.run(
        [*git, "checkout-index", "--all", f"--prefix={checkout.as_posix()}/"],
        cwd=repository,
        check=True,
    )
    for name, content in files.items():
        assert (checkout / name).read_bytes() == content, f"Checkout changed {name}"
