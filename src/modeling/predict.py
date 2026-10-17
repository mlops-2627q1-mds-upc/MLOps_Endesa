"""Reserved standalone prediction entry point for the M4 serving work."""

import typer

app = typer.Typer()


@app.command()
def main():
    """Explain the pending interface instead of reporting a successful forecast."""
    typer.echo(
        "Standalone prediction is not implemented yet (M4). "
        "For the recorded validation comparison, use python -m src.modeling.evaluate "
        "as described in docs/evaluation.md.",
        err=True,
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
