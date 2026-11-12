"""Run explicit target-local deployment operations; every result is a new receipt."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from src.deployment.config import DeploymentConfig, compute_lock, write_new
from src.deployment.runtime import Deployment


def main(argv=None):
    """Keep deployment mutation separate from read-only HTTP verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "action",
        choices=[
            "preflight",
            "provision",
            "start",
            "inspect",
            "restart",
            "stop",
            "remove",
            "missing-model",
        ],
    )
    parser.add_argument("--from-checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or not args.output.parent.is_dir():
            raise ValueError("Output must be a new file in an existing evidence directory")
        config = DeploymentConfig.load(args.config)
        with compute_lock(args.lock_file):
            deployment = Deployment(config)
            if args.action == "provision":
                if args.from_checkpoint is None:
                    raise ValueError("provision requires --from-checkpoint")
                result = deployment.provision(args.from_checkpoint)
            elif args.action in ("restart", "stop", "remove"):
                result = deployment.lifecycle(args.action)
            else:
                method = {
                    "inspect": deployment.receipt,
                    "missing-model": deployment.missing_model,
                    "preflight": deployment.preflight,
                    "start": deployment.start,
                }[args.action]
                result = method()
            status = result.get("status", "passed")
            write_new(args.output, {"action": args.action, "status": status, **result})
        print(json.dumps({"status": status, "action": args.action, "output": str(args.output)}))
        return 0 if status == "passed" else 1
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Deployment failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
