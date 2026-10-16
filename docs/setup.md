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
also works). uv can download Python 3.11 itself, but Smart App Control on Windows 11
blocks that build and the project's compiled dependencies (see [Troubleshooting on Windows](#troubleshooting-on-windows)).

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
| `uv run pylint src tests` | Static analysis and quality score (see below) |
| `uv run python -m src.dataset` | Run a module inside the environment |

Commit `pyproject.toml` and `uv.lock` together whenever dependencies change.
`make` targets (`make test`, `make lint`) are available on systems with GNU Make.

## Static analysis with Pylint

`make lint` runs Ruff first and then Pylint. Both tools are complementary:
Ruff is a fast linter and formatter (style, import order, common errors),
while Pylint does deeper static analysis (docstrings, unused arguments, design
smells such as too many locals or arguments, duplicated code) and gives a global
score out of 10. Tracked in [issue #34](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/34).

The configuration lives in `pyproject.toml` under `[tool.pylint.*]`:

| Setting | Reason |
| --- | --- |
| `max-line-length = 99` | Same limit as Ruff |
| `line-too-long`, `wrong-import-order` disabled | Already enforced by Ruff, avoids duplicated warnings |
| `redefined-outer-name` disabled | pytest injects fixtures by parameter name |
| `no-docstring-rgx = "^(_\|test_)"` | Test names already describe the behaviour; private helpers are exempt |
| `fail-under = 9.0` | Quality gate: the lint job fails if the score drops below 9.0 |
| `fail-on = ["E"]` | Any error message (for example an undefined name) fails the job, even when the score stays above 9.0 |

Score history: 8.89/10 with the default configuration, 9.43/10 after the
configuration above and three small fixes (explicit UTF-8 encoding when reading
files, which matters on Windows, and an explicit exception for the optional
Chronos/torch imports). The remaining messages (missing docstrings in `src`,
unused arguments in template stubs, long evaluation functions) are kept visible
on purpose as technical debt to reduce in later milestones.

The score is a global average, so it depends on the size of the codebase: the
`fail-under` threshold catches large regressions, not a single new warning.
That is why errors are gated separately with `fail-on`.

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
The pytest suite checks package imports, checked-in directories, raw-data DVC
metadata, archive ingestion with offline ZIP/TSF fixtures, and the split and
forecast-origin rules on synthetic series.
CI does not download datasets or model weights or run training. Requiring these
checks before merge is tracked separately in [issue #20](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/20).

### Troubleshooting on Windows

With **Smart App Control** on, Windows 11 blocks the project's compiled code, so the
project does not run natively. If Python 3.11 is not installed, uv's downloaded
interpreter is blocked and `uv sync` fails with `os error 4551`. A Python 3.11 from
python.org runs, but compiled dependencies such as pyarrow and SQLAlchemy are still
blocked: imports fail with `DLL load failed while importing …: An Application Control
policy has blocked this file.`

Either turn Smart App Control off (turning it back on may require resetting Windows),
or run the project inside WSL.

If `uv` is not recognised after `pip install uv`, the Python `Scripts` folder is not on
`PATH`; use `py -m uv sync` and `py -m uv run pytest`, or install uv with the
[official installer](https://docs.astral.sh/uv/getting-started/installation/).

## Source archive verification

[EDN-007](edn/0007-source-archive-integrity.md) records the decision, review and
implementation evidence for this safeguard.

The [downloader](../src/dataset.py) verifies the ZIP's SHA-256 before opening it,
then requires T1-T5 exactly once before writing any Parquet files. A checksum
mismatch or missing, duplicate, or unknown ID raises an error before the output
directory is created or existing outputs are changed. Values remain `float32`,
with the existing state mapping and positional `timestep` index.

The expected ZIP SHA-256 is
`b1439c28a631766bd05ee327f1f430a887b9f53b38a8cd5c0769ded1fd4aaf5a`,
matching the [dataset card](DATASET_CARD.md#dataset-structure-and-provenance).
To download, verify, and inspect the source using an ignored scratch directory:

```sh
uv run python -m src.dataset --output-dir temporary/archive-check/raw
```

This command accesses Zenodo and writes to the specified scratch directory.
The offline ingestion tests use synthetic ZIP/TSF fixtures, require no network or
credentials, and check both new and existing output directories:

```sh
uv run pytest tests/test_dataset.py -q
```

The decision is recorded in [EDN-007 / PR #47](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/47);
implementation and peer review are tracked in [#46](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/46).

## Directory layout

```text
data/
  raw/            Original, immutable source data (DVC, issue #11)
  interim/        Intermediate transformed data
  processed/      Train/validation/test blocks and forecast origins (DVC stage `split`)
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
