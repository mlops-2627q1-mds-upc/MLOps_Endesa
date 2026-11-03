"""Local CLI output preservation and bounded argument handling."""

from contextlib import nullcontext
import json
from types import SimpleNamespace

import pandas as pd
import pytest

import src.drift.__main__ as cli
from src.drift.__main__ import main, write_report
from src.drift.core import DriftConfig
from src.monitoring.source import ReplaySource
from src.splits import SplitConfig


def test_atomic_reports_preserve_prior_assessments(tmp_path):
    path = tmp_path / "output/report.json"
    write_report(path, {"evidence_kind": "fixture", "value": 1})
    with pytest.raises(FileExistsError):
        write_report(path, {"value": 2})
    assert json.loads(path.read_text())["value"] == 1
    assert list(path.parent.iterdir()) == [path]
    write_report(path, {"value": 3}, replace=True)
    assert json.loads(path.read_text()) == {"value": 3}
    with pytest.raises(ValueError):
        write_report(tmp_path / "nan.json", {"value": float("nan")})
    assert not (tmp_path / "nan.json").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["fixture", "--max-seconds", "0"],
        ["fixture", "--max-seconds", "301"],
        ["replay"],
    ],
)
def test_cli_rejects_invalid_budget_or_missing_journal(tmp_path, monkeypatch, arguments):
    monkeypatch.setattr(
        "sys.argv", ["drift", *arguments, "--output", str(tmp_path / "report.json")]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize(
    "corrupt",
    [{}, None, [], False, 0, {"checkpoint": None, "snapshot_clock_position": 544}],
)
def test_replay_cli_preserves_corrupt_checkpoint_without_retesting(tmp_path, monkeypatch, corrupt):
    cfg = DriftConfig(train_end=512, validation_end=704, cycle_steps=4, reference_cycles=8)
    frame = pd.DataFrame({state: [1.0] * 704 for state in cfg.states})
    source = ReplaySource(
        frame,
        SplitConfig(cfg.states, 896, 512, 704, 512, 48, 48),
        "validation:train.parquet+validation.parquet",
        {},
    )
    info = SimpleNamespace(
        run_id="cli-fixture",
        config_sha256="config",
        source_sha256=source.source_sha256,
        source_reference=source.reference,
    )
    rows = [
        SimpleNamespace(
            id=index + 1,
            state=state,
            position=position,
            value=1.0,
            released_position=position + 1,
            config_sha256=info.config_sha256,
            source_reference=info.source_reference,
        )
        for index, (position, state) in enumerate(
            (position, state) for position in range(512, 544) for state in cfg.states
        )
    ]
    reader = SimpleNamespace(
        snapshot=nullcontext,
        run_info=lambda: info,
        counts=lambda: SimpleNamespace(clock_position=544),
        released_observations=lambda after_id, limit: [row for row in rows if row.id > after_id][
            :limit
        ],
    )
    calls = []

    def compare(*_args):
        calls.append(1)
        return {"status": "no_drift"}

    monkeypatch.setattr(cli, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        cli,
        "load_data",
        lambda *_: (frame.iloc[:512], frame.iloc[512:], {"kind": "fixture", "file_hashes": {}}),
    )
    monkeypatch.setattr(ReplaySource, "from_validation", lambda *_: source)
    monkeypatch.setattr("src.monitoring.JournalReader", lambda _: nullcontext(reader))
    monkeypatch.setattr("src.drift.core.ks_compare", compare)
    output = tmp_path / "checkpoint.json"
    args = SimpleNamespace(config=None, data_dir=None, output=output, journal=None, max_seconds=30)
    first = cli._run_replay(args)  # pylint: disable=protected-access
    assert len(calls) == 5
    assert cli._run_replay(args) == first  # pylint: disable=protected-access
    assert len(calls) == 5
    original = json.dumps(corrupt)
    output.write_text(original)
    with pytest.raises(ValueError, match="checkpoint"):
        cli._run_replay(args)  # pylint: disable=protected-access
    assert len(calls) == 5
    assert output.read_text() == original
