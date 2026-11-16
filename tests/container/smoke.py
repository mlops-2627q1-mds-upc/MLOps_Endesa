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


def run_smoke(base_url: str, timeout: float = 30) -> dict:
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
    return {
        "fixture_only": True,
        "health": health[0],
        "ready": ready[0],
        "forecast": status,
        "invalid_request": invalid_status,
        "model_revision": result["model_revision"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    print(json.dumps(run_smoke(parser.parse_args().base_url), indent=2))
