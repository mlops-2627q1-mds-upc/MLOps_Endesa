"""Local CLI output preservation and bounded argument handling."""

import json

import pytest

from src.drift.__main__ import main, write_report


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
