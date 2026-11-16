"""Exercise a fixture-backed Uvicorn server over HTTP using only the standard library."""

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base_url: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    """Read success and validation-error JSON from the actual HTTP socket."""
    data = None if body is None else json.dumps(body).encode()
    message = Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(message, timeout=2) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        with error:
            return error.code, json.load(error)


def check_metrics(base_url: str, require_process_metrics: bool = False) -> dict:
    """Check bounded service samples and distinguish unsupported process telemetry."""
    with urlopen(base_url + "/metrics", timeout=2) as response:
        assert response.status == 200
        assert response.headers.get_content_type() == "text/plain"
        samples = {
            key: float(value)
            for line in response.read().decode().splitlines()
            if line and not line.startswith("#")
            for key, value in [line.rsplit(" ", 1)]
        }
    assert samples["endesa_model_ready"] == 1
    assert samples["endesa_forecasts_in_flight"] == 0
    for status_class in ("2xx", "4xx", "5xx"):
        name = (
            'endesa_http_requests_total{method="POST",route="/forecast",'
            f'status_class="{status_class}"}}'
        )
        assert samples[name] >= (0 if status_class == "5xx" else 1)
    assert (
        samples['endesa_http_request_duration_seconds_count{method="POST",route="/forecast"}'] >= 2
    )
    available = samples["endesa_process_metrics_available"]
    assert available in (0, 1)
    if require_process_metrics:
        assert available == 1, "Linux container must expose process CPU and RSS"
    for name in ("process_cpu_seconds_total", "process_resident_memory_bytes"):
        if available:
            assert samples[name] > 0
        else:
            assert name not in samples, "Unsupported process samples must be absent"
    return {"ready": 1, "process_metrics_available": int(available)}


def run_smoke(base_url: str, timeout: float = 30, require_process_metrics: bool = False) -> dict:
    """Wait for startup and verify probes, a forecast and request rejection."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            health = request(base_url, "/health")
            break
        except (URLError, TimeoutError, ConnectionError):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Fixture server did not start within the smoke deadline"
                ) from None
            time.sleep(0.1)
    assert health == (200, {"status": "ok"}), health
    ready = request(base_url, "/ready")
    assert ready == (200, {"status": "ready"}), ready
    payload = {"state": "TAS", "history": [-3, 0, 2.5, 4] * 128}
    status, result = request(base_url, "/forecast", payload)
    assert status == 200, (status, result)
    assert result["state"] == "TAS"
    assert result["value_scale"] == "source"
    assert (result["context_length"], result["horizon"]) == (512, 48)
    for name, expected in (("p10", 3.0), ("p50", 4.0), ("p90", 5.0)):
        assert result[name] == [expected] * 48, name
    assert result["model_id"] == "chronos-t5-small-fine-tuned"
    assert result["model_revision"] == "local-sha256:" + "0" * 64
    assert result["checkpoint"] == {"sha256": "0" * 64, "training_run_id": "0" * 32}
    assert result["settings"]["device"] == "cpu"
    assert result["settings"]["cpu_threads"] == 2
    invalid_status, invalid = request(base_url, "/forecast", {**payload, "history": [1]})
    assert invalid_status == 422 and invalid["detail"], (invalid_status, invalid)
    metrics = check_metrics(base_url, require_process_metrics)
    return {
        "fixture_only": True,
        "health": health[0],
        "ready": ready[0],
        "forecast": status,
        "invalid_request": invalid_status,
        "model_revision": result["model_revision"],
        "metrics": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("--require-process-metrics", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(args.base_url, require_process_metrics=args.require_process_metrics),
            indent=2,
        )
    )
