"""Check the image source boundary and HTTP harness without weights or Docker."""

import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

from tests.container.smoke import run_smoke

ROOT = Path(__file__).resolve().parents[1]


def _stage_runtime(directory):
    """Reproduce the Dockerfile's explicit local COPY inputs in an empty directory."""
    copied = set()
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        *sources, target = shlex.split(line)[1:]
        destination = directory / target
        destination.mkdir(parents=True, exist_ok=True)
        for source in sources:
            assert not any(character in source for character in "*?[")
            assert (ROOT / source).is_file(), source
            shutil.copyfile(ROOT / source, destination / Path(source).name)
            copied.add(source)
    return copied


def _environment(directory):
    return {
        **os.environ,
        "PYTHONPATH": str(directory),
        "PYTHONDONTWRITEBYTECODE": "1",
        "FORECAST_PARAMS_PATH": str(directory / "params.yaml"),
        "FORECAST_CHECKPOINT": str(directory / "models/chronos-t5-small-fine-tuned"),
        "FORECAST_CPU_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }


def test_build_context_admits_only_explicit_runtime_files(tmp_path):
    copied = _stage_runtime(tmp_path)
    rules = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert rules[0] == "**"
    allowed = set()
    for rule in rules[1:]:
        assert rule.startswith("!") and not any(char in rule for char in "*?[")
        path = rule[1:]
        # Re-including a directory could admit secrets and caches under that parent.
        assert (ROOT / path).is_file() and not (ROOT / path).is_symlink(), path
        allowed.add(path)
    assert allowed == copied | {"Dockerfile", ".dockerignore"}
    assert not (tmp_path / "src/modeling/train.py").exists()
    assert not (tmp_path / "tests").exists()
    assert list((tmp_path / "models").iterdir()) == [
        tmp_path / "models/chronos-t5-small-fine-tuned.provenance.json"
    ]


def test_production_factory_exits_without_selected_weights(tmp_path):
    _stage_runtime(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "uvicorn", "src.api.app:create_app", "--factory", "--port", "0"],
        cwd=tmp_path,
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "Could not verify local checkpoint" in result.stderr
    assert "Application startup failed" in result.stderr


def test_fixture_harness_exercises_actual_http_from_staged_runtime(tmp_path):
    _stage_runtime(tmp_path)
    environment = _environment(tmp_path)
    environment["PYTHONPATH"] += os.pathsep + str(ROOT / "tests/container")
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    with (tmp_path / "uvicorn.log").open("w+") as log:
        server = subprocess.Popen(  # pylint: disable=consider-using-with
            [
                sys.executable,
                "-m",
                "uvicorn",
                "fixture_app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
            ],
            cwd=tmp_path,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            report = run_smoke(f"http://127.0.0.1:{port}", timeout=10)
            assert report["fixture_only"]
            assert report["forecast"] == 200
            assert report["invalid_request"] == 422
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            log.seek(0)
            print(log.read())
