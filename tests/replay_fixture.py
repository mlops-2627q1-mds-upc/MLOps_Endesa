"""Generate declared synthetic replay inputs and a journal for consumer checks."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import yaml

from src.monitoring import SCHEMA_VERSION, JournalReader, ModelIdentity, ReplayConfig
from src.monitoring.replay import Replay
from src.monitoring.source import ReplaySource


@dataclass(frozen=True)
class ReplayFixture:
    """Synthetic inputs and writer configuration; no real artifact is referenced."""

    directory: Path
    source: ReplaySource
    config: ReplayConfig

    @property
    def journal(self) -> Path:
        """Journal consumed through the public JournalReader interface."""
        return self.directory / "replay.sqlite"


def create_replay_fixture(directory: Path) -> ReplayFixture:
    """Write four forecasts and six delayed joins, refusing to overwrite a directory.

    NSW[t] = t and TAS[t] = -t-1. Origins 512 and 513 overlap; delay is 2.
    The declared predictor extrapolates these lines with p10/p90 one unit apart
    from p50. It uses MockTransport explicitly and never opens a network socket.
    """
    directory.mkdir(parents=True, exist_ok=False)
    params = {
        "data": {"states": ["NSW", "TAS"], "common_length": 656},
        "split": {"train_end": 512, "validation_end": 608},
        "forecast": {"context_length": 512, "horizon": 48, "origin_stride": 1},
    }
    params_path = directory / "params.yaml"
    params_path.write_text(yaml.safe_dump(params), encoding="utf-8")
    frame = pd.DataFrame({"NSW": np.arange(608), "TAS": -np.arange(608) - 1})
    frame.iloc[:512].to_parquet(directory / "train.parquet")
    frame.iloc[512:].to_parquet(directory / "validation.parquet")
    identity = {
        "model_id": "chronos-t5-small-fine-tuned",
        "model_revision": "local-sha256:" + "a" * 64,
        "checkpoint": {"sha256": "a" * 64, "training_run_id": "b" * 32},
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
    }
    (directory / "identity.json").write_text(json.dumps(identity) + "\n", encoding="utf-8")
    source = ReplaySource.from_validation(directory, params_path)
    config = ReplayConfig(
        512,
        514,
        ("NSW", "TAS"),
        ModelIdentity.from_api(identity),
        endpoint="http://replay-fixture.invalid",
        origin_stride=1,
        observation_delay=2,
    )
    fixture = ReplayFixture(directory, source, config)

    def predict(request):
        payload = json.loads(request.content)
        direction = 1 if payload["state"] == "NSW" else -1
        medians = [payload["history"][-1] + direction * step for step in range(1, 49)]
        return httpx.Response(
            200,
            json={
                **identity,
                "state": payload["state"],
                "value_scale": "source",
                "context_length": 512,
                "horizon": 48,
                "p10": [mid - 1 for mid in medians],
                "p50": medians,
                "p90": [mid + 1 for mid in medians],
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(predict)) as client,
        Replay(fixture.journal, source, config, client=client) as replay,
    ):
        replay.run(max_forecasts=4)
        replay.release(516)
    return fixture


def main(argv: list[str] | None = None) -> int:
    """Create an explicitly requested fixture and print its public snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    fixture = create_replay_fixture(args.directory)
    with JournalReader(fixture.journal) as reader, reader.snapshot():
        print(
            json.dumps(
                {
                    "evidence_kind": "deterministic_fixture",
                    "schema_version": SCHEMA_VERSION,
                    "journal": str(fixture.journal),
                    "run": asdict(reader.run_info()),
                    "counts": asdict(reader.counts()),
                },
                indent=2,
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
