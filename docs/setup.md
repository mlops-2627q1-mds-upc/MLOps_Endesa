# Project structure and development setup

This repository follows the [Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/)
(CCDS) layout, generated with the options used in the
[course setup guide](https://github.com/mlops-2627q1-mds-upc/MLOps-2627q1-demos/blob/main/docs/project-setup.md):
`uv` as environment manager, `pyproject.toml` as dependency file, Python 3.11,
`pytest` and `ruff`. Tracked in [issue #10](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/10).

## Setup from a clean checkout

Requirements: Git and [uv](https://docs.astral.sh/uv/getting-started/installation/)
(`pip install uv` also works). uv downloads Python 3.11 if it is not installed.

```sh
git clone https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa.git
cd MLOps_Endesa
uv sync          # creates .venv with the exact versions in uv.lock
uv run pytest    # smoke test: package imports and directory layout
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
| `.gitignore` keeps the `data/` and `models/` folders but ignores their contents | Folders exist after cloning; their contents will be versioned with DVC (issue #11) |
| `.gitattributes` normalises line endings | The team works on Windows and Unix systems; avoids whole-file diffs caused by CRLF/LF |
| Placeholder failing test replaced by smoke tests | Gives a working check from a clean checkout; real data and model tests come in M3 |
