"""Read-only guards for the selected private GHCR publication destination."""

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ORGANIZATION = "mlops-2627q1-mds-upc"
SOURCE_REPOSITORY = f"{ORGANIZATION}/MLOps_Endesa"
PACKAGE = "endesa-forecast"
DESTINATION = f"ghcr.io/{ORGANIZATION}/{PACKAGE}"
PACKAGE_PATH = f"orgs/{ORGANIZATION}/packages/container/{PACKAGE}"
POST_ATTEMPTS = 5


class PrivacyError(ValueError):
    """A controlled error that contains no token or raw API response."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrivacyError(message)


def _get(path: str) -> tuple[int, dict | None]:
    token = os.environ.get("GH_TOKEN")
    _require(bool(token), "GH_TOKEN is required for authenticated visibility checks")
    request = Request(
        "https://api.github.com/" + path,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "endesa-private-publication",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = response.read(1024 * 1024 + 1)
            _require(response.status == 200, "Expected HTTP 200 for GitHub metadata")
            _require(len(payload) <= 1024 * 1024, "GitHub metadata exceeds the size limit")
            metadata = json.loads(payload)
            _require(isinstance(metadata, dict), "Malformed GitHub metadata object")
            return 200, metadata
    except HTTPError as error:
        status = error.code
        error.close()
        if status == 404:
            return 404, None
        raise PrivacyError(
            f"GitHub metadata HTTP {status}; authorization/visibility unverified"
        ) from None
    except (URLError, TimeoutError, OSError):
        raise PrivacyError(
            "GitHub metadata request failed; authorization/visibility unverified"
        ) from None


def check_private_package(
    destination: str, *, phase: str, first_publication: bool = False
) -> dict:
    """Observe identity/privacy; a 404 never establishes absence or reader access."""
    receipt = {
        "schema_version": 1,
        "destination": DESTINATION,
        "phase": phase,
        "observed_utc": datetime.now(UTC).isoformat(),
        "status": "failed",
        "package_state": "UNOBSERVED",
        "private_default_bootstrap": False,
    }
    try:
        _require(
            destination == DESTINATION, "Only the selected private GHCR destination is permitted"
        )
        _require(phase in ("before", "after"), "Privacy phase must be before or after")
        _require(
            type(first_publication) is bool, "First-publication mode must be an explicit boolean"
        )
        status, organization = _get(f"orgs/{ORGANIZATION}")
        _require(
            status == 200
            and organization["login"] == ORGANIZATION
            and organization["type"] == "Organization"
            and type(organization["id"]) is int
            and organization["id"] > 0,
            "Authenticated organization identity could not be verified",
        )
        status, repository = _get(f"repos/{SOURCE_REPOSITORY}")
        _require(
            status == 200
            and repository["full_name"] == SOURCE_REPOSITORY
            and type(repository["id"]) is int
            and repository["id"] > 0
            and type(repository["owner"]["id"]) is int
            and repository["owner"]["id"] == organization["id"]
            and repository["owner"]["login"] == ORGANIZATION
            and repository["owner"]["type"] == "Organization",
            "Authenticated source repository/namespace could not be verified",
        )
        receipt.update(
            organization_id=organization["id"],
            source_repository=SOURCE_REPOSITORY,
            source_repository_id=repository["id"],
        )
        for attempt in range(POST_ATTEMPTS if phase == "after" else 1):
            status, package = _get(PACKAGE_PATH)
            receipt.update(package_http_status=status, observed_utc=datetime.now(UTC).isoformat())
            if status == 200:
                _require(
                    package["name"] == PACKAGE
                    and package["package_type"] == "container"
                    and type(package["id"]) is int
                    and package["id"] > 0
                    and type(package["owner"]["id"]) is int
                    and package["owner"]["id"] == organization["id"]
                    and package["owner"]["login"] == ORGANIZATION
                    and package["owner"]["type"] == "Organization",
                    "Observed package identity does not match the selected destination",
                )
                _require(
                    package["visibility"] in ("private", "public", "internal"),
                    "Malformed package visibility metadata",
                )
                receipt.update(
                    package_id=package["id"],
                    package_state=package["visibility"].upper(),
                    visibility=package["visibility"],
                )
                _require(
                    package["visibility"] == "private",
                    "Observed package visibility must be private",
                )
                receipt.update(
                    status="passed",
                    package_state="PRIVATE",
                    package_id=package["id"],
                    visibility="private",
                )
                return receipt
            _require(status == 404, "Unexpected package API response; visibility unverified")
            if phase == "before":
                _require(
                    first_publication,
                    "Package is unobserved (404); normal publication requires verified private metadata",
                )
                receipt.update(status="bootstrap-permitted", private_default_bootstrap=True)
                return receipt
            if attempt + 1 < POST_ATTEMPTS:
                time.sleep(2)
        raise PrivacyError(
            "Private visibility was not observed after push; publication remains unverified"
        )
    except PrivacyError as error:
        receipt["error"] = str(error)
    except (KeyError, TypeError, ValueError):
        receipt["error"] = (
            "Malformed GitHub identity/visibility metadata; publication is unverified"
        )
    return receipt


def main() -> None:
    """Write sanitized pre/post receipts; never change or publish a package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--first-publication", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = check_private_package(
        args.repository, phase=args.phase, first_publication=args.first_publication
    )
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    if receipt["status"] not in ("passed", "bootstrap-permitted"):
        parser.exit(1, f"Private GHCR guard rejected: {receipt['error']}\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
