# Project structure and development setup

This repository follows the [Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/)
(CCDS) layout, generated with the options used in the
[course setup guide](https://github.com/mlops-2627q1-mds-upc/MLOps-2627q1-demos/blob/main/docs/project-setup.md):
`uv` as environment manager, `pyproject.toml` as dependency file, Python 3.11,
`pytest` and `ruff`. Tracked in [issue #10](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/10).

## Setup from a clean checkout

Requirements: Git, [uv](https://docs.astral.sh/uv/getting-started/installation/)
(`pip install uv` also works), and **Python 3.11** installed from
[python.org](https://www.python.org/downloads/) (on Windows, `pymanager install 3.11`
also works). uv can download Python 3.11 itself, but that build is blocked on Windows
machines with Smart App Control (see [Troubleshooting on Windows](#troubleshooting-on-windows)).

```sh
git clone https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa.git
cd MLOps_Endesa
uv sync          # creates .venv with the exact versions in uv.lock
uv run pytest    # smoke test: package imports, layout, and DVC metadata
```

Other useful commands:

| Command | Purpose |
| --- | --- |
| `uv add <package>` | Add a project dependency (updates `pyproject.toml` and `uv.lock`) |
| `uv add --group dev <package>` | Add a development-only dependency |
| `uv run ruff check` / `uv run ruff format` | Lint and format the code |
| `uv run python -m src.dataset` | Run a module inside the environment |

Commit `pyproject.toml` and `uv.lock` together whenever dependencies change.
`make` targets (`make test`, `make lint`) are available on systems with GNU Make.

## Pull request checks

The [CI workflow](../.github/workflows/ci.yml) runs on pull requests to `main`
and pushes to `main`. It uses Ubuntu 24.04 and Python 3.11, installs from the
committed lockfile, then runs `make lint` and `make test` in separate jobs. Run
the same checks locally:

```sh
uv sync --locked --dev
make lint
make test
```

If a job fails, open its GitHub Actions log and run the failing command locally.
The current pytest suite only checks package imports, checked-in directories,
and the presence of raw-data DVC metadata;
data, forecast, model, and API tests belong with their future implementations.
CI does not download datasets or model weights or run training. Requiring these
checks before merge is tracked separately in [issue #20](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/20).

### Troubleshooting on Windows

If Python 3.11 is not installed, `uv sync` downloads its own Python 3.11. On Windows 11,
**Smart App Control** can block that interpreter and `uv sync` fails with `os error 4551`.
Install Python 3.11 from [python.org](https://www.python.org/downloads/) and run
`uv sync` again: uv detects and uses the installed interpreter automatically
(`uv python find 3.11` shows which one it uses).

If `uv` is not recognised after `pip install uv`, the Python `Scripts` folder is not on
`PATH`; use `py -m uv sync` and `py -m uv run pytest`, or install uv with the
[official installer](https://docs.astral.sh/uv/getting-started/installation/).

## Directory layout

```text
data/
  raw/            Original, immutable source data (DVC, issue #11)
  interim/        Intermediate transformed data
  processed/      Final datasets used for modelling
  external/       Third-party data
models/           Model checkpoints and predictions (DVC / MLflow)
notebooks/        Exploration notebooks, named like 1.0-tl-initial-eda.ipynb
references/       Data dictionaries and explanatory material
reports/figures/  Generated figures for the reports
src/              Python package with the project code
  config.py       Project paths and configuration
  dataset.py      Data download and preparation
  features.py     Feature engineering
  modeling/       train.py (training) and predict.py (inference)
  plots.py        Visualisation code
tests/            Pytest suite
docs/             Project documentation, cards and EDN
pyproject.toml    Package metadata, dependencies and tool configuration
uv.lock           Exact, locked dependency versions
Makefile          Shortcuts for common tasks
```

## Adaptations from the CCDS template

| Change | Reason |
| --- | --- |
| Template generated in a separate folder and copied in, instead of initialising the repository with `ccds` | The repository already existed with the README and M1 documentation, which had to be preserved |
| Package named `src` | Same name as the course demo repository, so its examples apply directly |
| `pytest` and `ruff` moved to a `dev` dependency group | Keeps runtime dependencies separate from development tools, as in the course setup guide |
| Template `README.md`, `docs/` (mkdocs) and `LICENSE` not used | The project README and `docs/` already exist; the licence is a team decision |
| `.env` not committed | It is meant for local secrets; it is listed in `.gitignore` |
| `.gitignore` excludes generated data and model files; `data/raw.dvc` tracks raw data | Checked-in placeholders retain the other layout directories. `data/raw/` is created when DVC data is pulled and is absent from a clean checkout. |
| `.gitattributes` normalises line endings | The team works on Windows and Unix systems; avoids whole-file diffs caused by CRLF/LF |
| Placeholder failing test replaced by smoke tests | Gives a working check from a clean checkout; real data and model tests come in M3 |
