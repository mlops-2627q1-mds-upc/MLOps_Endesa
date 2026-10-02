"""Check MLflow write/read/artifact access without data or model downloads."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import typer

from src.tracking import tracked_run

app = typer.Typer(pretty_exceptions_show_locals=False)


def smoke_run(
    tracking_uri: str | None = None,
    experiment_name: str = "MLOps_Endesa-smoke",
) -> str:
    targets = [10, 11, 12, 13]
    predictions = [10, 12, 11, 14]
    mae = sum(abs(target - prediction) for target, prediction in zip(targets, predictions)) / 4
    with TemporaryDirectory(prefix="mlflow-smoke-") as directory:
        output = Path(directory) / "predictions.csv"
        with output.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestep", "target", "prediction"])
            writer.writerows(zip(range(4), targets, predictions))
        with tracked_run(
            name="tracking-smoke",
            config={"fixture": "four-point-series-v1", "seed": 0},
            data_version="synthetic-fixture-v1",
            model_revision="none-smoke",
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            purpose="smoke",
        ) as run:
            run.log_metrics({"smoke.mae": mae, "smoke.rows": 4})
            run.log_artifact(output)

        recorded = run.client.get_run(run.run_id)
        if (
            recorded.info.status != "FINISHED"
            or recorded.data.params.get("fixture") != "four-point-series-v1"
            or recorded.data.metrics.get("smoke.mae") != mae
            or recorded.data.metrics.get("smoke.rows") != 4
            or recorded.data.tags.get("run.purpose") != "smoke"
        ):
            raise RuntimeError(f"MLflow smoke run {run.run_id} did not round-trip correctly.")
        download_dir = Path(directory) / "download"
        download_dir.mkdir()
        downloaded = Path(
            run.client.download_artifacts(run.run_id, "predictions.csv", str(download_dir))
        )
        if downloaded.read_bytes() != output.read_bytes():
            raise RuntimeError(f"Artifact contents differ for MLflow smoke run {run.run_id}.")
    return run.run_id


@app.command()
def main(
    tracking_uri: str | None = typer.Option(None, help="Override MLFLOW_TRACKING_URI explicitly."),
    experiment_name: str = "MLOps_Endesa-smoke",
) -> None:
    """Create a synthetic run and verify its values and downloaded artifact."""
    run_id = smoke_run(tracking_uri, experiment_name)
    typer.echo(json.dumps({"run_id": run_id, "experiment": experiment_name, "verified": True}))


if __name__ == "__main__":
    app()
