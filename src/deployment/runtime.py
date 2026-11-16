"""Operate only an explicitly selected container on the current course target."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import stat
import subprocess
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from src.deployment.config import COMMAND, ROOT, DeploymentConfig, fingerprint
from src.deployment.release import prediction_settings, verify_release

ISSUE_LABEL = "org.mlops-endesa.issue"
CONFIG_LABEL = "org.mlops-endesa.deployment-config"
MODEL_PATH = "/app/models/chronos-t5-small-fine-tuned"
PARAMS_PATH = "/app/params.yaml"


def run_command(argv: list[str], timeout: int = 30, *, include_stderr=False) -> str:
    """Wait for bounded commands; never print environment or raw service logs."""
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} command failed (exit {result.returncode})")
    return (result.stdout + (result.stderr if include_stderr else "")).strip()


def verify_source(commit: str):
    """The build label must identify this clean source tree, not an unrelated revision."""
    if (
        run_command(
            ["git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT), "rev-parse", "HEAD"]
        )
        != commit
    ):
        raise ValueError("source_commit does not match this checkout")
    if run_command(
        [
            "git",
            "-c",
            "safe.directory=" + str(ROOT),
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ]
    ):
        raise ValueError(
            "Deployment requires a clean source checkout; put config/evidence in temporary/"
        )


class Deployment:
    """No SSH, host discovery, VM creation, package installation or image publication."""

    def __init__(self, config: DeploymentConfig):
        self.config = config

    def docker(self, *args: str, timeout: int = 30, include_stderr=False) -> str:
        """Address an explicit local Unix socket, bypassing ambient Docker contexts."""
        return run_command(
            ["docker", "--host", "unix://" + self.config.docker_socket, *args],
            timeout,
            include_stderr=include_stderr,
        )

    def target(self) -> dict:
        """Check the current host and daemon before any container operation."""
        self.config.verify_target()
        if not stat.S_ISSOCK(Path(self.config.docker_socket).stat().st_mode):
            raise ValueError("docker_socket must be an existing local Unix socket")
        info = json.loads(self.docker("info", "--format", "{{json .}}"))
        if info.get("OSType") != "linux" or info.get("Name") != socket.gethostname():
            raise ValueError("Docker daemon is not the selected local Linux host")
        if self.config.release is not None and info.get("Architecture") not in ("amd64", "x86_64"):
            raise ValueError("Published releases require a native Linux amd64 target")
        return {
            key: info.get(key)
            for key in (
                "Name",
                "ServerVersion",
                "Architecture",
                "NCPU",
                "MemTotal",
                "CgroupDriver",
                "CgroupVersion",
                "OperatingSystem",
                "KernelVersion",
            )
        }

    def image(self, release: dict | None = None) -> dict:
        """Require immutable content, revision, normal entry point and non-root runtime."""
        image = json.loads(self.docker("image", "inspect", self.config.image_id))[0]
        cfg = image["Config"]
        revision = (cfg.get("Labels") or {}).get("org.opencontainers.image.revision")
        if (
            image["Id"] != self.config.image_id
            or revision != self.config.source_commit
            or cfg.get("User") != "10001:10001"
            or cfg.get("Cmd") != COMMAND
            or cfg.get("Entrypoint")
        ):
            raise ValueError(
                "Image ID/revision/user/entrypoint differs from the selected candidate"
            )
        if release is not None and (
            image.get("Os") != "linux"
            or image.get("Architecture") != "amd64"
            or self.config.release["image_reference"] not in (image.get("RepoDigests") or [])
            or (cfg.get("Labels") or {}).get("org.opencontainers.image.source")
            != "https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa"
        ):
            raise ValueError(
                "Pulled image platform/source/repository digest differs from the release"
            )
        return {"id": image["Id"], "revision": revision, "user": cfg["User"]}

    def mounts(self, *, model: bool = True) -> list[str]:
        """Use read-only, explicit paths; missing bind sources must never become directories."""
        mounts = [(self.config.params, PARAMS_PATH)]
        if model:
            mounts.append((self.config.checkpoint, MODEL_PATH))
        result = []
        for source, target in mounts:
            result += ["--mount", f"type=bind,src={source},dst={target},readonly"]
        return result

    def environment(self, *, model: bool = True) -> list[str]:
        """Set a one/two-thread budget without forwarding ambient secrets or environment."""
        result = []
        for name, value in {
            "FORECAST_CHECKPOINT": MODEL_PATH if model else "/missing-selected-checkpoint",
            "FORECAST_PARAMS_PATH": PARAMS_PATH,
            "FORECAST_CPU_THREADS": str(self.config.cpu_threads),
            "OMP_NUM_THREADS": str(self.config.cpu_threads),
            "MKL_NUM_THREADS": str(self.config.cpu_threads),
            "OPENBLAS_NUM_THREADS": str(self.config.cpu_threads),
        }.items():
            result += ["--env", name + "=" + value]
        return result

    def limits(self) -> list[str]:
        """Private, bounded runtime; capacity is selected from measured target resources."""
        return [
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=128",
            "--memory",
            f"{self.config.memory_mb}m",
            "--memory-swap",
            f"{self.config.memory_mb}m",
            "--cpus",
            str(min(self.config.cpu_threads, os.cpu_count() or 1)),
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=67108864",
            "--restart=no",
        ]

    def labels(self) -> list[str]:
        """Fence lifecycle actions by the complete selected deployment configuration."""
        return [
            "--label",
            ISSUE_LABEL + "=75",
            "--label",
            CONFIG_LABEL + "=" + self.config.config_sha256,
        ]

    def owned(self, name: str | None = None) -> dict:
        """Refuse unrelated resources even when a container name happens to match."""
        name = name or self.config.deployment_id
        container = json.loads(self.docker("container", "inspect", name))[0]
        labels = container["Config"].get("Labels") or {}
        if (
            container["Name"] != "/" + name
            or labels.get(ISSUE_LABEL) != "75"
            or labels.get(CONFIG_LABEL) != self.config.config_sha256
            or container["Image"] != self.config.image_id
        ):
            raise ValueError("Refusing to operate on a container without matching ownership/image")
        return container

    def preflight(self) -> dict:
        """Verify bytes and reuse the image's actual serving configuration parser."""
        release = verify_release(self.config)
        target = self.target()
        if release is None:
            verify_source(self.config.source_commit)
        artifacts = self.config.verify_artifacts()
        image = self.image(release)
        code = (
            "import json; from src.api.settings import Settings; "
            "from src.modeling.predict import ForecastConfig,SAMPLING; "
            "from src.modeling.checkpoints import describe_checkpoint; "
            "from src.modeling.predict import SELECTED_PROVENANCE; "
            "s=Settings.from_env(); "
            "c=ForecastConfig.from_params(s.params_path,checkpoint=s.checkpoint,cpu_threads=s.cpu_threads); "
            "d=describe_checkpoint(s.checkpoint,provenance=SELECTED_PROVENANCE); "
            "print(json.dumps({'checkpoint':{k:d[k] for k in ('sha256','training_run_id')},"
            "'settings':{'num_samples':c.num_samples,'seed':c.seed,'device':c.device,"
            "'cpu_threads':c.cpu_threads,'dtype':'float32','sampling':SAMPLING,"
            "'quantile_method':'linear','point_statistic':'median'}}))"
        )
        observed = self.parse_settings(code)
        if observed["checkpoint"] != artifacts["checkpoint"]:
            raise ValueError("Image's serving loader reported a different checkpoint identity")
        if release is not None and observed["settings"] != prediction_settings(
            release["manifest"]
        ):
            raise ValueError("Image's executed prediction settings differ from the release")
        return {
            "target": target,
            "image": image,
            **artifacts,
            "settings": observed["settings"],
            **({"release": release} if release is not None else {}),
        }

    def parse_settings(self, code: str) -> dict:
        """Bound the parser container lifetime, including Docker-client timeouts."""
        name = self.config.deployment_id + "-preflight"
        self.docker(
            "run",
            "--detach",
            "--network=none",
            "--name",
            name,
            *self.labels(),
            *self.limits(),
            *self.mounts(),
            *self.environment(),
            self.config.image_id,
            "python",
            "-c",
            code,
        )
        try:
            if self.docker("wait", name, timeout=90) != "0":
                raise ValueError("Image's serving configuration parser failed")
            return json.loads(self.docker("logs", name))
        finally:
            container = self.owned(name)
            if container["State"]["Running"]:
                self.docker("stop", "--time", "10", container["Id"])
            self.docker("rm", container["Id"])

    def provision(self, source: Path) -> dict:
        """Copy only verified selected bytes to a new slot; never overwrite a checkpoint."""
        release = verify_release(self.config)
        self.config.verify_target()
        verified = self.config.verify_artifacts(source)
        destination = Path(self.config.checkpoint)
        destination.mkdir(exist_ok=False)
        for relative in verified["checkpoint_files"]:
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with (source / relative).open("rb") as incoming, output.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            output.chmod(0o444)
        for folder in sorted(destination.rglob("*"), reverse=True):
            if folder.is_dir():
                folder.chmod(0o555)
        destination.chmod(0o555)
        return {**self.config.verify_artifacts(), **({"release": release} if release else {})}

    def start(self) -> dict:
        """Create a new isolated name, or resume only its matching stopped container."""
        verified = self.preflight()
        existing = self.docker("container", "ls", "--all", "--format", "{{.Names}}")
        if self.config.deployment_id in existing.splitlines():
            container = self.owned()
            if container["State"]["Running"]:
                raise ValueError("Selected container is already running; inspect it")
            self.receipt(verified)
            self.docker("start", container["Id"])
        else:
            self.docker(
                "run",
                "--detach",
                "--name",
                self.config.deployment_id,
                *self.labels(),
                *self.limits(),
                *self.mounts(),
                *self.environment(),
                "--publish",
                f"127.0.0.1:{self.config.port}:8000",
                self.config.image_id,
            )
        self.wait_ready()
        try:
            return self.receipt(verified)
        except (ValueError, RuntimeError, OSError):
            self.docker("stop", "--time", "20", self.owned()["Id"], timeout=45)
            raise

    def wait_ready(self):
        """Keep the operation locked through model startup; stop only its own failed instance."""
        deadline = time.monotonic() + 90
        opener = build_opener(ProxyHandler({}))
        try:
            while time.monotonic() < deadline:
                if not self.owned()["State"]["Running"]:
                    raise RuntimeError("Owned container exited before readiness")
                try:
                    with opener.open(
                        f"http://127.0.0.1:{self.config.port}/ready", timeout=2
                    ) as response:
                        if response.status == 200 and json.loads(response.read(4096)) == {
                            "status": "ready"
                        }:
                            return
                except (URLError, OSError, ValueError):
                    pass
                time.sleep(0.5)
            raise RuntimeError("Owned container did not become ready within 90 seconds")
        except (RuntimeError, OSError):
            self.docker("stop", "--time", "10", self.owned()["Id"])
            raise

    def receipt(self, verified: dict | None = None) -> dict:
        """Record selected and inspected identities; HTTP readiness is a separate check."""
        container = self.owned()
        cfg, host = container["Config"], container["HostConfig"]
        expected_mounts = {(self.config.checkpoint, MODEL_PATH), (self.config.params, PARAMS_PATH)}
        actual_mounts = {
            (m["Source"], m["Destination"])
            for m in container["Mounts"]
            if m["Type"] == "bind" and not m["RW"]
        }
        ports = host.get("PortBindings") or {}
        if (
            cfg.get("Cmd") != COMMAND
            or cfg.get("Entrypoint")
            or cfg.get("User") != "10001:10001"
            or expected_mounts != actual_mounts
            or len(container["Mounts"]) != len(expected_mounts)
            or ports != {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(self.config.port)}]}
            or not host.get("ReadonlyRootfs")
            or host.get("RestartPolicy", {}).get("Name") != "no"
            or host.get("Memory") != self.config.memory_mb * 1024 * 1024
        ):
            raise ValueError("Inspected runtime differs from the selected private container spec")
        expected_env = self.environment()[1::2]
        if not set(expected_env) <= set(cfg.get("Env") or []):
            raise ValueError("Container inference environment differs from selected settings")
        release = Path("/etc/os-release")
        return {
            "schema_version": 1,
            "evidence_kind": "container-inspection",
            "recorded_at": datetime.now(UTC).isoformat(),
            "config": self.config.as_dict(),
            "config_sha256": self.config.config_sha256,
            "source_commit": self.config.source_commit,
            "verified": verified if verified is not None else self.preflight(),
            "container": {
                "id": container["Id"],
                "name": container["Name"],
                "image": container["Image"],
                "running": container["State"]["Running"],
                "pid": container["State"]["Pid"],
                "exit_code": container["State"]["ExitCode"],
                "oom_killed": container["State"]["OOMKilled"],
                "started_at": container["State"]["StartedAt"],
                "restart_count": container["RestartCount"],
            },
            "host": {
                "hostname": socket.getfqdn(),
                "architecture": platform.machine(),
                "os_release": release.read_text() if release.exists() else None,
                "cpu_count": os.cpu_count(),
                "disk": dict(zip(("total", "used", "free"), shutil.disk_usage(ROOT))),
            },
            "base_url_on_target": f"http://127.0.0.1:{self.config.port}",
        }

    def lifecycle(self, action: str) -> dict:
        """Stop/remove safely even when model files are broken; restart requires preflight."""
        self.target()
        container = self.owned()
        if action == "restart":
            verified = self.preflight()
            self.receipt(verified)
            self.docker("restart", "--time", "20", container["Id"], timeout=45)
            self.wait_ready()
            return self.receipt(verified)
        if action == "remove":
            if container["State"]["Running"]:
                raise ValueError("Stop the owned container before removing it")
            self.docker("rm", container["Id"])
            return {"removed_container": container["Id"]}
        self.docker("stop", "--time", "20", container["Id"], timeout=45)
        stopped = self.owned()
        if stopped["State"]["Running"] or stopped["Id"] != container["Id"]:
            raise RuntimeError("Container did not stop")
        clean = stopped["State"]["ExitCode"] == 0 and not stopped["State"]["OOMKilled"]
        return {
            "status": "passed" if clean else "failed",
            "stopped_container": stopped["Id"],
            "running": False,
            "was_running": container["State"]["Running"],
            "exit_code": stopped["State"]["ExitCode"],
            "oom_killed": stopped["State"]["OOMKilled"],
            "clean_shutdown": clean,
        }

    def missing_model(self) -> dict:
        """Normal entry point, no weights/network/ports, unique name; always remove that probe."""
        self.preflight()
        name = self.config.deployment_id + "-missing-model"
        self.docker(
            "run",
            "--detach",
            "--network=none",
            "--name",
            name,
            *self.labels(),
            *self.limits(),
            *self.mounts(model=False),
            *self.environment(model=False),
            self.config.image_id,
        )
        try:
            code = self.docker("wait", name, timeout=120)
            container = self.owned(name)
            diagnostic = "Forecast model initialization failed" in self.docker(
                "logs", name, include_stderr=True
            )
            # Uvicorn writes diagnostics to stderr, which docker logs forwards there.
            # Exit 3 + no OOM + the image's unmodified entry point prove startup failure.
            if (
                code != "3"
                or container["State"]["OOMKilled"]
                or container["State"]["Running"]
                or not diagnostic
            ):
                raise ValueError("Missing-model probe did not fail normal startup with exit 3")
            return {
                "exit_code": 3,
                "oom_killed": False,
                "published_ports": [],
                "normal_entrypoint": True,
                "model_initialization_diagnostic": diagnostic,
            }
        finally:
            container = self.owned(name)
            if container["State"]["Running"]:
                self.docker("stop", "--time", "10", container["Id"])
            self.docker("rm", container["Id"])


def receipt_sha256(receipt: dict) -> str:
    """Link an HTTP client report to its operator-provided container inspection."""
    return fingerprint(receipt)
