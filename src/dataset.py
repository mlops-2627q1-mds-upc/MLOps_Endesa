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

# Original source — Zenodo record pinned by DATASET_CARD.md SHA-256
ZENODO_URL = (
    "https://zenodo.org/records/4659727/files/"
    "australian_electricity_demand_dataset.zip"
)

# Mapping from TSF series IDs to state codes.
# Order and IDs derived from the source archive; QUN renamed to QLD.
TSF_ID_TO_STATE = {
    "T1": "NSW",
    "T2": "VIC",
    "T3": "QLD",  # source uses "QUN"
    "T4": "SA",
    "T5": "TAS",
}


def _parse_tsf(content: bytes) -> tuple[list[str], list[np.ndarray]]:
    """Parse raw .tsf bytes; return (tsf_ids, value_arrays)."""
    col_names, col_types = [], []
    tsf_ids, series_list = [], []
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
            for i, (col, typ) in enumerate(zip(col_names, col_types)):
                if typ == "string" and col == "series_name":
                    tsf_ids.append(parts[i])
            values = np.array(
                [float(v) if v != "?" else np.nan for v in parts[-1].split(",")],
                dtype=np.float32,
            )
            series_list.append(values)

    return tsf_ids, series_list


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

    tsf_ids, series_list = _parse_tsf(tsf_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tsf_id, values in zip(tsf_ids, series_list):
        state = TSF_ID_TO_STATE[tsf_id]
        # Positional index: timestamps from the source are unverified
        # for timezone and DST — use step index to avoid misleading callers.
        index = pd.RangeIndex(len(values), name="timestep")
        series = pd.Series(values, index=index, name=state)
        out_path = output_dir / f"{state}.parquet"
        series.to_frame().to_parquet(out_path)
        logger.info(f"Saved {tsf_id} -> {state}: {len(series):,} observations -> {out_path}")

    logger.success("Dataset download complete.")


if __name__ == "__main__":
    app()
