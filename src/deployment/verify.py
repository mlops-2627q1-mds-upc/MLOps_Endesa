"""Verify HTTP from a separate client against an operator-provided identity receipt."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import errno
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from src.deployment.config import DeploymentConfig, compute_lock, fingerprint, write_new


class NoRedirect(HTTPRedirectHandler):
    """Do not silently verify a different service after a redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_request(payload: dict):
    """Match the unchanged source-scale API input without loading model dependencies."""
    if not isinstance(payload, dict) or payload.keys() != {"state", "history"}:
        raise ValueError("Request must contain only state and history")
    if payload["state"] not in ("NSW", "VIC", "QLD", "SA", "TAS"):
        raise ValueError("Invalid state")
    history = payload["history"]
    if (
        not isinstance(history, list)
        or len(history) != 512
        or any(
            type(v) not in (int, float)
            or not math.isfinite(v)
            or abs(v) > float.fromhex("0x1.fffffep+127")
            for v in history
        )
    ):
        raise ValueError("Request requires exactly 512 finite source-scale float32 numbers")


def validate_receipt(receipt: dict) -> dict:
    """Require a coherent selected-model inspection; source/image identity is not in HTTP."""
    config = DeploymentConfig(**receipt["config"])
    config.validate()
    verified = receipt["verified"]
    if (
        receipt.get("schema_version") != 1
        or receipt.get("evidence_kind") != "container-inspection"
        or receipt.get("config_sha256") != config.config_sha256
        or receipt["source_commit"] != config.source_commit
        or receipt["host"]["hostname"] != config.target_hostname
        or receipt["container"]["image"] != config.image_id
        or receipt["container"]["running"] is not True
        or receipt["container"]["oom_killed"]
        or verified["image"]
        != {"id": config.image_id, "revision": config.source_commit, "user": "10001:10001"}
        or verified["checkpoint"]
        != {"sha256": config.checkpoint_sha256, "training_run_id": config.training_run_id}
        or verified["params_sha256"] != config.params_sha256
        or verified["checkpoint_tree_sha256"] != config.checkpoint_tree_sha256
        or fingerprint(verified["checkpoint_files"]) != config.checkpoint_tree_sha256
        or verified["checkpoint_files"].get("model.safetensors") != config.checkpoint_sha256
    ):
        raise ValueError("Inconsistent or non-selected container inspection receipt")
    settings = verified["settings"]
    if settings["cpu_threads"] != config.cpu_threads or settings["device"] != "cpu":
        raise ValueError("Receipt inference settings differ from deployment configuration")
    return verified


def validate_forecast(body: dict, payload: dict, verified: dict):
    """Check actual quantiles, selected model and every executed prediction setting."""
    if not isinstance(body, dict):
        raise TypeError("Forecast is not a JSON object")
    expected = {
        "state": payload["state"],
        "context_length": 512,
        "horizon": 48,
        "value_scale": "source",
        "model_id": "chronos-t5-small-fine-tuned",
        "model_revision": "local-sha256:" + verified["checkpoint"]["sha256"],
        "checkpoint": verified["checkpoint"],
        "settings": verified["settings"],
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise ValueError("HTTP forecast model/config identity differs from deployment receipt")
    for name in ("p10", "p50", "p90"):
        values = body.get(name)
        if (
            not isinstance(values, list)
            or len(values) != 48
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
        ):
            raise ValueError("Expected 48 finite numeric values per quantile")
    if any(
        not low <= mid <= high for low, mid, high in zip(body["p10"], body["p50"], body["p90"])
    ):
        raise ValueError("Forecast quantiles are unordered")


def request_json(base_url: str, route: str, payload: dict | None, timeout: float) -> dict:
    """Bound responses, bypass ambient proxies and retain actual status/body or transport error."""
    opener = build_opener(ProxyHandler({}), NoRedirect())
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = Request(
        base_url.rstrip("/") + route,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    started = time.monotonic()
    result = {"route": route, "method": request.get_method(), "status": None}
    try:
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as error:
            response = error
        with response:
            result["status"] = response.code
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("HTTP response exceeds 1 MiB")
            result["body_sha256"] = hashlib.sha256(raw).hexdigest()
            result["body"] = json.loads(raw)
    except (URLError, TimeoutError, OSError, ValueError) as error:
        reason = getattr(error, "reason", error)
        result["error"] = type(reason).__name__
        result["connection_refused"] = getattr(reason, "errno", None) == errno.ECONNREFUSED
    result["seconds"] = time.monotonic() - started
    return result


def verify(base_url: str, receipt: dict, payload: dict, *, timeout=30, stopped=False) -> dict:
    """One valid forecast and one invalid request; never evaluate accuracy or tune sampling."""
    url = urlsplit(base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ("", "/")
    ):
        raise ValueError("Use an explicit base HTTP(S) URL without credentials, query or subpath")
    if not 0 < timeout <= 120:
        raise ValueError("HTTP timeout must be in (0,120] seconds")
    verified = validate_receipt(receipt)
    validate_request(payload)
    checks = []
    for route, expected in (("/health", "ok"), ("/ready", "ready")):
        result = request_json(base_url, route, None, timeout)
        result["passed"] = (
            result.get("connection_refused") is True
            if stopped
            else result["status"] == 200 and result.get("body") == {"status": expected}
        )
        checks.append(result)
    if not stopped:
        forecast = request_json(base_url, "/forecast", payload, timeout)
        forecast["passed"] = False
        if forecast["status"] == 200 and "body" in forecast:
            try:
                validate_forecast(forecast["body"], payload, verified)
                forecast["passed"] = True
            except (ValueError, TypeError) as error:
                forecast["validation_error"] = str(error)
        checks.append(forecast)
        invalid = request_json(
            base_url, "/forecast", {**payload, "history": payload["history"][:-1]}, timeout
        )
        invalid_body = invalid.get("body")
        invalid["passed"] = (
            invalid["status"] == 422
            and isinstance(invalid_body, dict)
            and isinstance(invalid_body.get("detail"), list)
        )
        checks.append(invalid)
    return {
        "schema_version": 1,
        "evidence_kind": "http-verification",
        "recorded_at": datetime.now(UTC).isoformat(),
        "inspection_receipt_sha256": fingerprint(receipt),
        "source_commit": receipt["source_commit"],
        "base_url": base_url,
        "request_sha256": fingerprint(payload),
        "checkpoint": verified["checkpoint"],
        "settings": verified["settings"],
        "expect_stopped": stopped,
        "checks": checks,
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "scope": "HTTP behaviour and model/config agreement with the attached operator inspection",
    }


def main(argv=None):
    """Run without installing serving/model dependencies on the client."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lock-file", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--expect-stopped", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or not args.output.parent.is_dir():
            raise ValueError("Output must be a new file in an existing evidence directory")
        with compute_lock(args.lock_file):
            result = verify(
                args.base_url,
                json.loads(args.receipt.read_text()),
                json.loads(args.request.read_text()),
                timeout=args.timeout,
                stopped=args.expect_stopped,
            )
            write_new(args.output, result)
        print(json.dumps({"status": result["status"], "output": str(args.output)}))
        return 0 if result["status"] == "passed" else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"HTTP verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
