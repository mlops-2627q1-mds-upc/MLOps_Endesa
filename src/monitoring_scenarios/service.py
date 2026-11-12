"""Service fault injection confined to an injected app or two owned containers."""

from contextlib import ExitStack
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
PROMETHEUS_IMAGE = (
    "prom/prometheus:v3.15.1@sha256:"
    "fa3302279710be2f21a8301ec17a8b4f2301fd829bd113839b3b6b02b6f191a2"
)
OWNER_LABEL = "org.endesa.scenario-owner"
PAYLOAD = {"state": "NSW", "history": [10.0] * 512}


def timestamp():
    """Record an explicit UTC observation timestamp."""
    return datetime.now(UTC).isoformat()


def docker(*args, timeout=10):
    """Call only a local daemon, with finite command waits and no shell expansion."""
    context = subprocess.run(
        ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    ).stdout.strip()
    endpoint = os.environ.get("DOCKER_HOST", context)
    if not endpoint.startswith(("unix://", "npipe://")):
        raise ValueError("Scenario Docker execution requires a local daemon endpoint")
    result = subprocess.run(
        ["docker", *args], text=True, capture_output=True, check=True, timeout=timeout
    )
    return (result.stdout + result.stderr if args[0] == "logs" else result.stdout).strip()


def publish_fault(directory, delay):
    """Replace the complete fixture instruction; readers never see a truncated file."""
    staged = directory / "fault-next.json"
    with staged.open("w") as stream:
        json.dump({"delay_seconds": delay}, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    staged.replace(directory / "fault.json")


def inspect_container(identifier):
    """Inspect a full immutable container ID; mutable names cannot authorize cleanup."""
    if not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise ValueError("Cleanup requires an exact full container ID")
    return json.loads(docker("inspect", identifier))[0]


def cleanup(directory):
    """Stop only durable, matching owned IDs; retain containers, logs and volumes."""
    path = directory / "runtime.json"
    if not path.exists():
        return []
    runtime = json.loads(path.read_text())
    results = []
    identifiers = list(runtime["containers"])
    for role in ("api", "prometheus"):
        cidfile = directory / f"{role}.cid"
        if cidfile.exists():
            identifier = cidfile.read_text().strip()
            if identifier not in identifiers:
                identifiers.append(identifier)
    for identifier in reversed(identifiers):
        record = inspect_container(identifier)
        if record["Config"]["Labels"].get(OWNER_LABEL) != runtime["owner"]:
            raise ValueError("Container ownership changed; refusing stop")
        if record["State"]["Running"]:
            docker("stop", "--time", "5", identifier)
        record = inspect_container(identifier)
        results.append({"id": identifier, "state": record["State"]})
        if record["State"]["Running"]:
            raise RuntimeError("Owned container did not stop")
    return results


class OwnedStack:
    """Allocate a private bridge and loopback ports, never attaching a shared target."""

    def __init__(self, directory, owner, image):
        self.directory, self.owner, self.image = directory, owner, image
        self.runtime = {"owner": owner, "containers": [], "network": None}
        self.path = directory / "runtime.json"
        self.api = self.prometheus = None
        self.api_url = self.prometheus_url = None

    def persist(self):
        """Publish durable ownership before faults so the supervisor can stop survivors."""
        staged = self.path.with_suffix(".tmp")
        staged.write_text(json.dumps(self.runtime, indent=2) + "\n")
        staged.replace(self.path)

    def launch(self, role, image, options, command):
        """Persist a created ID before starting it, including partially failed startup."""
        identifier = docker(
            "create",
            "--cidfile",
            str(self.directory / f"{role}.cid"),
            "--pull",
            "never",
            "--name",
            f"{self.owner}-{role}",
            "--label",
            f"{OWNER_LABEL}={self.owner}",
            "--network",
            self.owner,
            "--read-only",
            "--cpus",
            "2",
            "--memory",
            "768m",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            *options,
            image,
            *command,
        )
        self.runtime["containers"].append(identifier)
        self.persist()
        docker("start", identifier)
        return identifier

    def start(self):
        """Use cached immutable images; overlay this checkout's API source read-only."""
        if self.path.exists():
            raise ValueError("Runtime metadata exists; use a fresh report directory")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.image or ""):
            raise ValueError("Supply the inspected cached fixture image's immutable ID")
        if not re.fullmatch(r"endesa-84-[a-z0-9-]{1,45}", self.owner):
            raise ValueError("Target identity must be a unique endesa-84-* local project")
        image = json.loads(docker("image", "inspect", self.image))[0]
        prometheus = json.loads(docker("image", "inspect", PROMETHEUS_IMAGE))[0]
        revision = image["Config"]["Labels"].get("org.opencontainers.image.revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Fixture base must identify its source revision")
        for path in ("pyproject.toml", "uv.lock", "params.yaml"):
            image_source = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
            if image_source != (ROOT / path).read_bytes():
                raise ValueError(f"Fixture base {path} differs from the scenario checkout")
        self.runtime.update({"fixture_image": image, "prometheus_image": prometheus})
        self.persist()
        self.runtime["network"] = docker(
            "network",
            "create",
            "--driver",
            "bridge",
            "--label",
            f"{OWNER_LABEL}={self.owner}",
            self.owner,
        )
        self.persist()
        publish_fault(self.directory, 0)
        self.api = self.launch(
            "api",
            self.image,
            [
                "--platform",
                "linux/amd64",
                "--network-alias",
                "api",
                "--publish",
                "127.0.0.1::8000",
                "--mount",
                f"type=bind,src={ROOT / 'src'},dst=/app/src,readonly",
                "--mount",
                f"type=bind,src={self.directory},dst=/scenario,readonly",
                "--env",
                "ENDESA_SCENARIO_FAULT=/scenario/fault.json",
                "--env",
                "OMP_NUM_THREADS=2",
                "--env",
                "OPENBLAS_NUM_THREADS=2",
                "--env",
                "MKL_NUM_THREADS=2",
            ],
            [
                "uvicorn",
                "src.monitoring_scenarios.fixture_service:create_app",
                "--factory",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--workers",
                "1",
            ],
        )
        config = self.directory / "prometheus.yml"
        config.write_text(
            "global:\n  scrape_interval: 2s\n  evaluation_interval: 2s\n"
            "rule_files: [/rules/alerts.yml]\nscrape_configs:\n"
            "  - job_name: forecast-api\n    static_configs:\n      - targets: ['api:8000']\n"
        )
        self.prometheus = self.launch(
            "prometheus",
            PROMETHEUS_IMAGE,
            [
                "--publish",
                "127.0.0.1::9090",
                "--tmpfs",
                "/prometheus:rw,size=64m,mode=1777",
                "--mount",
                f"type=bind,src={config},dst=/etc/prometheus/prometheus.yml,readonly",
                "--mount",
                f"type=bind,src={ROOT / 'monitoring/service/rules'},dst=/rules,readonly",
            ],
            [
                "--config.file=/etc/prometheus/prometheus.yml",
                "--storage.tsdb.path=/prometheus",
                "--storage.tsdb.retention.time=30m",
            ],
        )
        self.api_url = self.url(self.api, "8000/tcp")
        self.prometheus_url = self.url(self.prometheus, "9090/tcp")
        self.runtime.update({"api_url": self.api_url, "prometheus_url": self.prometheus_url})
        self.runtime["api_urls"] = [self.api_url]
        self.persist()

    @staticmethod
    def url(identifier, port):
        """Verify the actual loopback mapping returned by the Docker daemon."""
        bindings = inspect_container(identifier)["NetworkSettings"]["Ports"][port]
        if len(bindings) != 1 or bindings[0]["HostIp"] != "127.0.0.1":
            raise ValueError("Expected exactly one loopback publication")
        return f"http://127.0.0.1:{int(bindings[0]['HostPort'])}"

    def change_running(self, running):
        """Change only the exact fixture API allocated by this object."""
        if inspect_container(self.api)["Config"]["Labels"].get(OWNER_LABEL) != self.owner:
            raise ValueError("Fixture API ownership mismatch")
        if running:
            docker("start", self.api)
            self.api_url = self.url(self.api, "8000/tcp")
            self.runtime["api_url"] = self.api_url
            self.runtime["api_urls"].append(self.api_url)
            self.persist()
        else:
            docker("stop", "--time", "5", self.api)

    def set_delay(self, delay):
        """Acknowledge the exact instruction inside this owned container before traffic."""
        if inspect_container(self.api)["Config"]["Labels"].get(OWNER_LABEL) != self.owner:
            raise ValueError("Fixture API ownership mismatch")
        publish_fault(self.directory, delay)
        observed = json.loads(
            docker(
                "exec",
                self.api,
                "python",
                "-c",
                "import json; print(json.dumps(json.load(open('/scenario/fault.json'))))",
            )
        )
        if observed != {"delay_seconds": delay}:
            raise ValueError("Container did not acknowledge the declared fixture delay")
        return observed


def _forecast(client):
    start = time.monotonic()
    response = client.post("/forecast", json=PAYLOAD)
    return {"http_status": response.status_code, "wall_seconds": time.monotonic() - start}


def fixture_service(directory, scenario, declaration):
    """Observe production handlers/metrics in process; no alert evaluator is simulated."""
    from fastapi.testclient import TestClient  # pylint: disable=import-outside-toplevel

    from src.api.app import create_app  # pylint: disable=import-outside-toplevel
    from src.monitoring_scenarios.fixture_service import (  # pylint: disable=import-outside-toplevel
        ScenarioPredictor,
    )

    fault = directory / "fault.json"
    publish_fault(directory, 0)
    predictor = ScenarioPredictor(fault)
    app = create_app(predictor_factory=lambda _settings: predictor)
    with TestClient(app) as client:
        control = _forecast(client)
        if scenario == "service-unavailable":
            app.state.predictor = None
        if scenario == "slow-request":
            publish_fault(directory, declaration["delay_seconds"])
        observations = [_forecast(client) for _ in range(declaration["requests"])]
        metrics = client.get("/metrics").text
        (directory / "fault-metrics.prom").write_text(metrics)
        app.state.predictor = predictor
        publish_fault(directory, 0)
        recovery = _forecast(client)
    passed = control["http_status"] == recovery["http_status"] == 200
    expected_status = 503 if scenario == "service-unavailable" else 200
    passed &= all(row["http_status"] == expected_status for row in observations)
    if scenario == "slow-request":
        passed &= all(row["wall_seconds"] >= declaration["delay_seconds"] for row in observations)
    return {
        "evidence_kind": "in_process_service_fixture",
        "procedure_passed": bool(passed),
        "observed_signal": {"control": control, "fault": observations},
        "recovery": recovery,
        "sample_sufficiency": {"completed_requests": len(observations)},
        "missed_detections": [],
        "limitations": [
            "No Prometheus process in fixture mode; alert timing and delivery unmeasured"
        ],
    }


def docker_service(directory, scenario, declaration, owner, image):
    """Observe real HTTP and Prometheus firing/clearing against an owned synthetic API."""
    stack = OwnedStack(directory, owner, image)
    trace = directory / "observations.jsonl"
    start = time.monotonic()
    end = start + declaration["max_seconds"] - 40

    def record(kind, value):
        row = {
            "utc": timestamp(),
            "elapsed_seconds": time.monotonic() - start,
            "kind": kind,
            "value": value,
        }
        with trace.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def query(client, expression):
        response = client.get("/api/v1/query", params={"query": expression})
        response.raise_for_status()
        payload = response.json()
        if payload["status"] != "success":
            raise ValueError("Prometheus query failed")
        return record(expression, payload["data"]["result"])

    def wait_for(client, expression, predicate):
        while time.monotonic() < end:
            try:
                row = query(client, expression)
                if predicate(row["value"]):
                    return row
            except (httpx.HTTPError, ValueError) as error:
                record("wait_error", str(error))
            time.sleep(1)
        raise TimeoutError(f"Scenario budget exhausted waiting for {expression}")

    try:
        stack.start()
        with ExitStack() as clients:
            api = clients.enter_context(
                httpx.Client(base_url=stack.api_url, timeout=5, trust_env=False)
            )
            prom = clients.enter_context(
                httpx.Client(base_url=stack.prometheus_url, timeout=5, trust_env=False)
            )
            wait_for(
                prom,
                'endesa_model_ready{job="forecast-api"}',
                lambda rows: len(rows) == 1 and rows[0]["value"][1] == "1",
            )
            control = record("control", _forecast(api))
            if control["value"]["http_status"] != 200:
                raise RuntimeError("Normal control forecast failed")
            firing = clearing = None
            fault_started = record("fault_start", scenario)
            if scenario == "service-unavailable":
                stack.change_running(False)
                wait_for(
                    prom,
                    'up{job="forecast-api"}',
                    lambda rows: len(rows) == 1 and rows[0]["value"][1] == "0",
                )
                firing = wait_for(
                    prom,
                    'ALERTS{alertname="ForecastApiDown",alertstate="firing"}',
                    lambda rows: bool(rows),
                )
                stack.change_running(True)
                record("api_restart", {"id": stack.api, "api_url": stack.api_url})
                api = clients.enter_context(
                    httpx.Client(base_url=stack.api_url, timeout=5, trust_env=False)
                )
                wait_for(
                    prom,
                    'endesa_model_ready{job="forecast-api"}',
                    lambda rows: len(rows) == 1 and rows[0]["value"][1] == "1",
                )
                clearing = wait_for(
                    prom,
                    'ALERTS{alertname="ForecastApiDown",alertstate="firing"}',
                    lambda rows: not rows,
                )
                fault = []
            else:
                delay = declaration.get("delay_seconds", 0)
                record("fault_acknowledged", stack.set_delay(delay))
                fault = [
                    record("forecast", _forecast(api)) for _ in range(declaration["requests"])
                ]
                record("recovery_acknowledged", stack.set_delay(0))
            recovery = record("recovery", _forecast(api))
            metrics = api.get("/metrics")
            metrics.raise_for_status()
            (directory / "service-metrics.prom").write_text(metrics.text)
            alerts = prom.get("/api/v1/alerts")
            alerts.raise_for_status()
            record("all_alerts", alerts.json())
            if scenario == "normal" and alerts.json()["data"]["alerts"]:
                raise RuntimeError("Normal control has unexpected pending/firing service alerts")
            targets = prom.get("/api/v1/targets")
            targets.raise_for_status()
            record("targets", targets.json())
            passed = recovery["value"]["http_status"] == 200 and all(
                row["value"]["http_status"] == 200 for row in fault
            )
            if scenario == "slow-request":
                passed &= all(
                    row["value"]["wall_seconds"] >= declaration["delay_seconds"] for row in fault
                )
            return {
                "evidence_kind": "owned_docker_service_fixture",
                "procedure_passed": bool(passed),
                "observed_signal": {
                    "control": control,
                    "fault_start": fault_started,
                    "forecasts": fault,
                },
                "recovery": recovery,
                "missed_detections": [],
                "alerts": {"firing": firing, "clearing": clearing, "delivery": "not_configured"},
                "sample_sufficiency": {
                    "completed_requests": len(fault),
                    "latency_slo": "not_adopted",
                },
                "limitations": [
                    "Cached fixture base with mounted source; no selected model or final-source image",
                    "Slow requests have no numerical latency alert; notification receiver unconfigured",
                ],
            }
    finally:
        for identifier in stack.runtime["containers"]:
            try:
                (directory / f"{identifier[:12]}.log").write_text(docker("logs", identifier))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
        cleanup(directory)
