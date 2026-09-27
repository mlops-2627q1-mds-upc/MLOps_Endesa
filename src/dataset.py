import io
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger
import typer

from src.config import RAW_DATA_DIR

app = typer.Typer()

STATE_NAMES = ["VIC", "NSW", "QLD", "SA", "TAS"]

# Original source — Zenodo record pinned by DATASET_CARD.md SHA-256
ZENODO_URL = (
    "https://zenodo.org/records/4659727/files/"
    "australian_electricity_demand_dataset.zip"
)


def _parse_tsf(content: bytes) -> tuple[list[str], list[datetime], list[np.ndarray]]:
    """Parse raw .tsf bytes; return (names, starts, value_arrays)."""
    col_names, col_types = [], []
    names, starts, series_list = [], [], []
    found_data = False

    for line in content.decode("cp1252").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("@attribute"):
            parts = line.split()
            col_names.append(parts[1])
            col_types.append(parts[2])
        elif line.startswith("@data"):
            found_data = True
        elif found_data:
            parts = line.split(":")
            # attributes: parts[0..n-2], values: parts[-1]
            for i, (name, typ) in enumerate(zip(col_names, col_types)):
                if typ == "string" and name == "series_name":
                    names.append(parts[i])
                elif typ == "date" and name == "start_timestamp":
                    starts.append(datetime.strptime(parts[i], "%Y-%m-%d %H-%M-%S"))
            values = np.array(
                [float(v) if v != "?" else np.nan for v in parts[-1].split(",")],
                dtype=np.float32,
            )
            series_list.append(values)

    return names, starts, series_list


@app.command()
def main(
    output_dir: Path = RAW_DATA_DIR,
):
    logger.info(f"Downloading from Zenodo: {ZENODO_URL}")
    resp = requests.get(ZENODO_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        tsf_name = next(n for n in zf.namelist() if n.endswith(".tsf"))
        tsf_bytes = zf.read(tsf_name)

    names, starts, series_list = _parse_tsf(tsf_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (start, values) in enumerate(zip(starts, series_list)):
        name = STATE_NAMES[i]
        index = pd.date_range(start=start, periods=len(values), freq="30min")
        series = pd.Series(values, index=index, name=name)
        out_path = output_dir / f"{name}.parquet"
        series.to_frame().to_parquet(out_path)
        logger.info(f"Saved {name}: {len(series):,} observations -> {out_path}")

    logger.success("Dataset download complete.")


if __name__ == "__main__":
    app()
