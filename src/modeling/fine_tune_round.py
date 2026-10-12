"""Train and score fine-tuning options and seeds one after another for #13."""

from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Annotated

from loguru import logger
import typer

from src.config import REPORTS_DIR
from src.modeling import evaluate, train
from src.splits import PARAMS_PATH

app = typer.Typer(pretty_exceptions_show_locals=False)


def run_round(
    options: list[str],
    seeds: list[int],
    *,
    params_path: Path = PARAMS_PATH,
    summary_dir: Path = REPORTS_DIR / "fine-tuning",
    tracking_uri: str | None = None,
    max_hours: float = 8.0,
) -> list[dict]:
    """Train and score every option with every seed on validation.

    A failed run is recorded in its MLflow run and in the summary, and the round goes on
    with the next one. No run starts after `max_hours`. The summary file is rewritten
    after every run, so it is complete up to the last finished run.
    """
    if not options or not seeds or max_hours <= 0:
        raise ValueError("Give at least one option and one seed, and a positive time budget")
    for option in options:
        # Fails before any training if an option name is unknown.
        train.FineTuneConfig.from_params(params_path, option)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"round-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    deadline = time.monotonic() + max_hours * 3600
    results = []
    for option in options:
        for seed in seeds:
            if time.monotonic() >= deadline:
                logger.warning(f"Time budget used; {option} with seed {seed} was not started")
                results.append({"option": option, "seed": seed, "skipped": "time budget"})
                continue
            entry = {"option": option, "seed": seed}
            try:
                trained = train.fine_tune(
                    params_path=params_path, tracking_uri=tracking_uri, option=option, seed=seed
                )
                entry["training_run_id"] = trained["run_id"]
                comparison = evaluate.evaluate(
                    params_path=params_path,
                    method="chronos",
                    tracking_uri=tracking_uri,
                    checkpoint=Path(trained["checkpoint"]["path"]),
                    device=train.FineTuneConfig.from_params(params_path, option, seed).device,
                )
                scored = comparison["results"]["chronos"]
                entry["evaluation_run_id"] = scored["run_id"]
                entry["mean_mase"] = scored["metrics"]["mean.mase"]
            except Exception as error:  # noqa: BLE001 - recorded, and the round continues
                logger.exception(f"{option} with seed {seed} failed")
                entry["error"] = f"{type(error).__name__}: {error}"
            results.append(entry)
            summary_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    logger.info(f"Round summary: {summary_path}")
    return results


@app.command()
def main(
    option: Annotated[list[str], typer.Option(help="Option names from params.yaml; repeatable")],
    seed: Annotated[list[int], typer.Option(help="Seeds; repeatable")],
    params_path: Path = PARAMS_PATH,
    tracking_uri: str | None = None,
    max_hours: float = 8.0,
) -> None:
    """Train and score each option with each seed, then print the summary."""
    results = run_round(
        option, seed, params_path=params_path, tracking_uri=tracking_uri, max_hours=max_hours
    )
    typer.echo(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    app()
