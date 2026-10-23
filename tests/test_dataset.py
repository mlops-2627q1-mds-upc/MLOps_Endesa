"""Exercise source ingestion with small offline ZIP/TSF fixtures."""

import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
import requests

from src import dataset

RECORDS = [
    ("T1", "NSW", "3498.38527,12865.79582,4000"),
    ("T2", "VIC", "2500.25,5000.5"),
    ("T3", "QUN", "3000,?,6000.75,4500"),
    ("T4", "SA", "1000.25"),
    ("T5", "TAS", "-61.8178845,1252.9437,-0.5"),
]


def _archive(records, newline="\n"):
    lines = [
        "# Synthetic source fixture: caf\u00e9",
        "@attribute series_name string",
        "@attribute state string",
        "@attribute start_timestamp date",
        "@frequency half_hourly",
        "@horizon 48",
        "@missing true",
        "@equallength false",
        "@data",
        *(f"{name}:{state}:2002-01-01 00:00:00:{values}" for name, state, values in records),
    ]
    content = (newline.join(lines) + newline).encode("cp1252")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("australian_electricity_demand_dataset.tsf", content)
    return buffer.getvalue()


@pytest.fixture
def serve_archive(monkeypatch):
    """Replace HTTP and the source pin with the digest of a synthetic archive."""

    def serve(content):
        response = requests.Response()
        response.status_code = 200
        response._content = content
        monkeypatch.setattr(dataset.requests, "get", lambda *args, **kwargs: response)
        monkeypatch.setattr(dataset, "ZENODO_SHA256", hashlib.sha256(content).hexdigest())

    return serve


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["LF", "CRLF"])
@pytest.mark.parametrize("reverse", [False, True], ids=["source-order", "reversed-order"])
def test_download_preserves_state_values_and_positional_indexes(
    tmp_path, serve_archive, newline, reverse
):
    records = RECORDS[::-1] if reverse else RECORDS
    serve_archive(_archive(records, newline))
    output_dir = tmp_path / "raw"

    dataset.main(output_dir=output_dir)

    expected = {
        "NSW": [3498.38527, 12865.79582, 4000],
        "VIC": [2500.25, 5000.5],
        "QLD": [3000, np.nan, 6000.75, 4500],
        "SA": [1000.25],
        "TAS": [-61.8178845, 1252.9437, -0.5],
    }
    assert {path.name for path in output_dir.iterdir()} == {
        f"{state}.parquet" for state in expected
    }
    for state, values in expected.items():
        actual = pd.read_parquet(output_dir / f"{state}.parquet")
        series = pd.Series(
            values,
            dtype=np.float32,
            index=pd.RangeIndex(len(values), name="timestep"),
            name=state,
        )
        pd.testing.assert_frame_equal(actual, series.to_frame())


@pytest.mark.parametrize("existing_output", [False, True], ids=["new-output", "existing-output"])
@pytest.mark.parametrize(
    "rejection, records, message",
    [
        ("checksum", RECORDS, "Archive SHA-256 mismatch"),
        ("missing", RECORDS[:-1], "missing: T5"),
        ("duplicate", [*RECORDS, RECORDS[-1]], "duplicate: T5"),
        ("unknown", [*RECORDS, ("T6", "OTHER", "1,2")], "unknown: T6"),
        ("substitution", [*RECORDS[:-1], ("T6", "TAS", "1,2")], "missing: T5; unknown: T6"),
        ("empty", [], "missing: T1, T2, T3, T4, T5"),
    ],
    ids=["checksum", "missing", "duplicate", "unknown", "substitution", "empty"],
)
def test_rejected_input_does_not_create_or_change_outputs(
    tmp_path, monkeypatch, serve_archive, existing_output, rejection, records, message
):
    output_dir = tmp_path / "raw"
    if existing_output:
        output_dir.mkdir()
        for state in ("NSW", "VIC", "QLD", "SA", "TAS"):
            pd.Series([42], dtype=np.float32, name=state).to_frame().to_parquet(
                output_dir / f"{state}.parquet"
            )
        (output_dir / ".gitkeep").touch()
    before = (
        {path.name: path.read_bytes() for path in output_dir.iterdir()} if existing_output else {}
    )

    # Invalid ZIP bytes prove the checksum check runs before ZIP decoding/parsing.
    content = b"not a ZIP archive" if rejection == "checksum" else _archive(records)
    serve_archive(content)
    if rejection == "checksum":
        monkeypatch.setattr(dataset, "ZENODO_SHA256", "0" * 64)

    with pytest.raises(ValueError, match=message) as error:
        dataset.main(output_dir=output_dir)

    if rejection == "checksum":
        assert "0" * 64 in str(error.value)
        assert hashlib.sha256(content).hexdigest() in str(error.value)
    if existing_output:
        assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == before
    else:
        assert not output_dir.exists()
