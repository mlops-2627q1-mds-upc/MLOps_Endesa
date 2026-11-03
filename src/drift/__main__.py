"""Bounded local drift assessment and read-only historical replay monitoring."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from src.drift.assessment import assess_controls, fixture_data
from src.drift.core import build_reference, is_integer
from src.drift.data import DEFAULT_CONFIG, ROOT, load_config, load_data
from src.drift.replay import assess_replay


def write_report(path: Path, report: dict, *, replace: bool = False) -> None:
    """Publish complete JSON atomically; preserve existing assessments by default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _run_replay(args):
    from filelock import FileLock  # pylint: disable=import-outside-toplevel

    from src.monitoring import JournalReader  # pylint: disable=import-outside-toplevel
    from src.monitoring.source import ReplaySource  # pylint: disable=import-outside-toplevel

    cfg = load_config(args.config)
    train, validation, source = load_data(args.data_dir, cfg)
    replay_source = ReplaySource.from_validation(args.data_dir, ROOT / "params.yaml")
    expected = ReplaySource(
        pd.concat([train, validation]),
        replay_source.split,
        "validation:train.parquet+validation.parquet",
        source["file_hashes"],
    )
    if expected.source_sha256 != replay_source.source_sha256:
        raise ValueError("replay artifacts changed after DVC verification")
    source = {
        **source,
        "replay_source_sha256": expected.source_sha256,
        "replay_source_reference": expected.reference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output) + ".lock", timeout=0):
        previous = None
        if args.output.exists():
            previous = json.loads(args.output.read_text())
            if (
                not isinstance(previous, dict)
                or not isinstance(previous.get("checkpoint"), dict)
                or not is_integer(previous.get("snapshot_clock_position"))
                or previous["snapshot_clock_position"] < 0
            ):
                raise ValueError("Existing drift report requires a valid checkpoint and clock")
        checkpoint = previous["checkpoint"] if previous is not None else None
        with JournalReader(args.journal) as reader:
            if (
                previous is not None
                and reader.counts().clock_position < previous["snapshot_clock_position"]
            ):
                raise ValueError("replay clock moved backwards from the saved checkpoint")
            report = assess_replay(
                reader,
                build_reference(train, cfg, source),
                replay_source,
                checkpoint,
                max_seconds=args.max_seconds,
            )
        write_report(args.output, report, replace=True)
    return report


def main() -> None:
    """Never train, infer a forecast, contact a service or read final-test data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fixture", "assess", "replay"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--max-seconds", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 300:
        parser.error("--max-seconds must be 1..300")
    if args.command == "fixture":
        report = assess_controls(*fixture_data(), max_seconds=args.max_seconds)
    elif args.command == "assess":
        cfg = load_config(args.config)
        train, validation, source = load_data(args.data_dir, cfg)
        report = assess_controls(train, validation, source, cfg, args.max_seconds)
    else:
        if args.journal is None:
            parser.error("replay requires --journal")
        report = _run_replay(args)
    if args.command != "replay":
        write_report(args.output, report)
    print(json.dumps({"output": str(args.output), "evidence_kind": report["evidence_kind"]}))


if __name__ == "__main__":
    main()
