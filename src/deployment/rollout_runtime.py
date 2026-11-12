"""Execute #75 helpers in each release's own pinned checkout and locked environment."""

import os
from pathlib import Path
import subprocess

from src.deployment.rollout_state import durable_json, read_json

# This small protocol runs from the selected source checkout. It does not import
# the controller package from another release or acquire the inherited lock again.
HELPER = r"""
import json, os, signal, sys
from pathlib import Path
from src.deployment.config import DeploymentConfig
from src.deployment.release import verify_release
from src.deployment.runtime import Deployment
from src.deployment.verify import verify

config_path, expected, action, request_path, output_path, lock_fd = sys.argv[1:]
output = Path(output_path)
result = {"schema_version": 1, "evidence_kind": "release-operation",
          "action": action, "config_sha256": expected, "status": "failed"}

def interrupted(_signum, _frame):
    raise InterruptedError("Helper interrupted")

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    # The parent passed its locked open file description; never take another lock.
    os.fstat(int(lock_fd))
    config = DeploymentConfig.load(Path(config_path))
    if config.schema_version != 2 or config.config_sha256 != expected:
        raise ValueError("Release snapshot identity mismatch")
    deployment = Deployment(config)
    if action != "quiesce":
        verify_release(config)
    deployment.target()
    target_verified = True
    if action == "quiesce":
        names = deployment.docker("container", "ls", "--all", "--format", "{{.Names}}").splitlines()
        # A killed preflight can leave its owned parser. Retain its exit evidence
        # before removing only that transient parser, as #75's normal finally does.
        parsers = []
        name = config.deployment_id + "-preflight"
        if name in names:
            parser = deployment.owned(name)
            if parser["State"]["Running"]:
                deployment.docker("stop", "--time", "10", parser["Id"])
            parser = deployment.owned(name)
            parsers.append({"id": parser["Id"], "state": parser["State"]})
            deployment.docker("rm", parser["Id"])
        value = deployment.lifecycle("stop") if config.deployment_id in names else {
            "status": "passed", "absent": True, "running": False}
        value["interrupted_parsers"] = parsers
    elif action == "preflight":
        value = deployment.preflight()
    elif action == "start":
        value = deployment.start()
    elif action == "restore":
        verified = deployment.preflight()
        names = deployment.docker("container", "ls", "--all", "--format", "{{.Names}}").splitlines()
        if config.deployment_id in names and deployment.owned()["State"]["Running"]:
            value = deployment.receipt(verified)
        else:
            value = deployment.start()
    elif action == "inspect":
        value = deployment.receipt()
    elif action == "forecast":
        request = json.loads(Path(request_path).read_text())
        value = verify(f"http://127.0.0.1:{config.port}", request["receipt"], request["payload"], timeout=30)
    else:
        raise ValueError("Unsupported release operation")
    result.update(result=value, status=value.get("status", "passed"))
except BaseException as error:
    # Never retain arbitrary subprocess stderr, ambient environment or exception text.
    result["error_type"] = type(error).__name__
    if locals().get("target_verified"):
        try:
            observed = deployment.owned()
            result["observed_container"] = {"id": observed["Id"], "image": observed["Image"],
                                             "state": observed["State"]}
        except Exception as inspection_error:
            result["inspection_error_type"] = type(inspection_error).__name__
finally:
    data = (json.dumps(result, allow_nan=False, indent=2) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(fd)
    os.close(fd)
sys.exit(0 if result["status"] == "passed" else 1)
"""


class SourceRunner:
    """No SSH, pull, provisioning, training, environment installation or image publication."""

    def __init__(self, store):
        self.store = store

    def run(self, slot: dict, action: str, request: dict, output: Path, lock_fd: int) -> dict:
        """Wait through the helper lifetime; its Docker/HTTP operations have bounded waits."""
        if output.exists() or output.is_symlink() or not output.parent.is_dir():
            raise ValueError("Use a fresh operation evidence path")
        config = self.store.config(slot)
        checkout = Path(slot["checkout"])
        python = checkout / ".venv/bin/python"
        if checkout.resolve() != checkout or not python.is_file():
            raise ValueError("Keep each source checkout and its full locked .venv available")
        request_path = output.with_suffix(".request.json")
        durable_json(request_path, request)
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "PYTHONNOUSERSITE": "1",
        }
        # Never let ambient Python search paths substitute another release's code.
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        argv = [
            str(python),
            "-c",
            HELPER,
            str(self.store.config_path(slot)),
            config.config_sha256,
            action,
            str(request_path),
            str(output),
            str(lock_fd),
        ]
        completed = subprocess.run(
            argv,
            cwd=checkout,
            env=environment,
            pass_fds=(lock_fd,),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not output.exists():
            durable_json(
                output,
                {
                    "schema_version": 1,
                    "evidence_kind": "release-operation",
                    "action": action,
                    "config_sha256": config.config_sha256,
                    "status": "failed",
                    "error_type": "HelperExit",
                    "exit_code": completed.returncode,
                },
            )
        response = read_json(output)
        if (
            response.get("config_sha256") != config.config_sha256
            or response.get("action") != action
        ):
            raise ValueError("Helper response identity mismatch")
        if completed.returncode != 0 and response["status"] == "passed":
            raise ValueError("Helper exit contradicts operation evidence")
        return response
