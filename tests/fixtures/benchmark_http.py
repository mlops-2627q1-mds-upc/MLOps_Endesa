"""Explicit synthetic HTTP fixture and local declaration generator; never a real model."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys


def fixture_plan(port, cwd):
    """Declare an unmistakably synthetic lifecycle with actual local machine facts."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    if sys.platform == "darwin":
        memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    else:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "evidence_kind": "fixture",
        "purpose": "discovery",
        "base_url": f"http://127.0.0.1:{port}",
        "source": {
            "service_commit": commit,
            "benchmark_commit": commit,
            "image_id": None,
            "params_sha256": hashlib.sha256((cwd / "params.yaml").read_bytes()).hexdigest(),
        },
        "hardware": {
            "host_identity": platform.node(),
            "os": platform.platform(),
            "cpu": platform.machine(),
            "cpu_count": os.cpu_count(),
            "memory_bytes": memory,
        },
        "request": {
            "state": "TAS",
            "history": [15 + 20 * math.sin(2 * math.pi * i / 48) for i in range(512)],
        },
        "expected": {
            "model_id": "chronos-t5-small-fine-tuned",
            "model_revision": "local-sha256:" + "0" * 64,
            "checkpoint": {"sha256": "0" * 64, "training_run_id": "0" * 32},
            "value_scale": "source",
            "context_length": 512,
            "horizon": 48,
            "settings": {
                "num_samples": 20,
                "seed": 7,
                "device": "cpu",
                "cpu_threads": 2,
                "dtype": "float32",
                "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
                "quantile_method": "linear",
                "point_statistic": "median",
            },
        },
        "measurement": {
            "budget_seconds": 60,
            "startup_seconds": 5,
            "request_timeout_seconds": 2,
            "warmup_requests": 3,
            "requests_per_level": 20,
            "concurrency": [1, 2],
            "overload_requests": 6,
            "consecutive_timeout_limit": 3,
            "memory_limit_bytes": 128 * 1024 * 1024,
            "proposal_margin_fraction": 0.2,
        },
        "startup": {
            "argv": [sys.executable, str(Path(__file__).resolve()), "--serve", str(port)],
            "cwd": str(cwd),
            "stop_argv": [],
        },
        "resource": {"scope": "process"},
    }


class FixtureHandler(BaseHTTPRequestHandler):
    """Serve a distinct zero-hash identity; no inference, input data or model loading."""

    def log_message(self, _format, *args):
        """Keep fixture output quiet."""

    def do_GET(self):  # pylint: disable=invalid-name
        """Answer only the declared readiness route."""
        self._send(200, {"status": "ready"})

    def do_POST(self):  # pylint: disable=invalid-name
        """Return synthetic quantiles after reading the request body."""
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        response = dict(self.server.expected)  # pylint: disable=no-member
        response.update(state=request["state"], p10=[-1.0] * 48, p50=[0.0] * 48, p90=[1.0] * 48)
        self._send(200, response)

    def _send(self, status, body):
        content = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main():
    """Create a plan exclusively, or launch the synthetic server in the foreground."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-plan", type=Path)
    parser.add_argument("--serve", type=int)
    args = parser.parse_args()
    cwd = Path(__file__).resolve().parents[2]
    if args.write_plan:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        args.write_plan.parent.mkdir(parents=True, exist_ok=True)
        with args.write_plan.open("x") as stream:
            json.dump(fixture_plan(port, cwd), stream, indent=2)
            stream.write("\n")
    elif args.serve:
        with HTTPServer(("127.0.0.1", args.serve), FixtureHandler) as server:
            server.expected = fixture_plan(args.serve, cwd)["expected"]
            server.serve_forever()
    else:
        parser.error("Choose --write-plan or --serve")


if __name__ == "__main__":
    main()
