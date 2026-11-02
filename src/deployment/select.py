"""Create an explicit deployment config from verified local checkpoint/params bytes."""

import argparse
import json
from pathlib import Path
import sys

from src.deployment.config import (
    PROVENANCE,
    DeploymentConfig,
    checkpoint_files,
    file_sha256,
    fingerprint,
    write_new,
)


def main(argv=None):
    """Select without provisioning services, copying weights or overwriting evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("deployment-id", "target-hostname", "source-commit", "image-id", "docker-socket"):
        parser.add_argument("--" + name, required=True)
    for name in ("checkpoint", "from-checkpoint", "params", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("cpu-threads", "memory-mb", "port"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args(argv)
    try:
        selected = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        manifest = checkpoint_files(args.from_checkpoint)
        config = DeploymentConfig(
            schema_version=1,
            deployment_id=args.deployment_id,
            target_hostname=args.target_hostname,
            source_commit=args.source_commit,
            image_id=args.image_id,
            docker_socket=args.docker_socket,
            checkpoint=str(args.checkpoint),
            checkpoint_tree_sha256=fingerprint(manifest),
            checkpoint_sha256=selected["model_safetensors_sha256"],
            training_run_id=selected["training_run_id"],
            params=str(args.params),
            params_sha256=file_sha256(args.params),
            cpu_threads=args.cpu_threads,
            memory_mb=args.memory_mb,
            port=args.port,
        )
        config.validate()
        config.verify_artifacts(args.from_checkpoint)
        write_new(args.output, config.as_dict())
        print(json.dumps({"config_sha256": config.config_sha256, "output": str(args.output)}))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Config selection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
