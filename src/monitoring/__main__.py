"""Bounded historical replay CLI: python -m src.monitoring --help."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

from src.monitoring import JournalReader, ModelIdentity, ReplayConfig
from src.monitoring.replay import Replay
from src.monitoring.source import ReplaySource


def main(argv: list[str] | None = None) -> int:
    """Resume an explicit run or inspect a journal without opening source data."""
    parser = argparse.ArgumentParser(description="Historical positional replay in source scale")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Read committed identities and counts")
    status.add_argument("--journal", type=Path, required=True)
    run = commands.add_parser("run", help="Resume bounded forecasts and optionally release truth")
    run.add_argument("--journal", type=Path, required=True)
    run.add_argument("--source-dir", type=Path, required=True)
    run.add_argument("--params", type=Path, default=Path("params.yaml"))
    run.add_argument("--identity", type=Path, required=True)
    run.add_argument("--endpoint", default="http://127.0.0.1:8000")
    run.add_argument("--origin-start", type=int, required=True)
    run.add_argument("--origin-stop", type=int, required=True)
    run.add_argument("--states", nargs="+", required=True)
    run.add_argument("--origin-stride", type=int, default=48)
    run.add_argument("--observation-delay", type=int, default=0)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--timeout-seconds", type=float, default=10)
    run.add_argument("--retry-delay-seconds", type=float, default=0.1)
    run.add_argument("--max-forecasts", type=int, default=1)
    run.add_argument("--max-seconds", type=float, default=60)
    run.add_argument("--release-through", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            identity = ModelIdentity.from_api(
                json.loads(args.identity.read_text(encoding="utf-8"))
            )
            source = ReplaySource.from_validation(args.source_dir, args.params)
            config = ReplayConfig(
                args.origin_start,
                args.origin_stop,
                tuple(args.states),
                identity,
                args.endpoint,
                args.origin_stride,
                args.observation_delay,
                args.max_attempts,
                args.timeout_seconds,
                args.retry_delay_seconds,
            )
            with Replay(args.journal, source, config) as replay:
                replay.run(max_forecasts=args.max_forecasts, max_seconds=args.max_seconds)
                if args.release_through is not None:
                    replay.release(args.release_through)
        with JournalReader(args.journal) as reader, reader.snapshot():
            print(
                json.dumps(
                    {"run": asdict(reader.run_info()), "counts": asdict(reader.counts())},
                    indent=2,
                    allow_nan=False,
                )
            )
    except (ValueError, KeyError, TypeError, RuntimeError, sqlite3.Error, OSError) as error:
        # Pydantic and transport messages may contain raw data or URL credentials.
        print(
            json.dumps(
                {
                    "error": type(error).__name__,
                    "detail": "Replay stopped; verify declared inputs and inspect journal status",
                }
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
