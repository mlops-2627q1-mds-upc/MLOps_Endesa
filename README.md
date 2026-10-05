# Electricity demand forecasting with Amazon Chronos

MLOps course project on half-hourly electricity demand in Australia. We plan to
compare Chronos-T5-small's zero-shot forecasts with a fine-tuned version, then
build the training, serving, and monitoring workflow around that model.

The repository contains the development environment, working agreement, M1
documentation, DVC chronological splits, and MLflow tracking. The first
[baseline/Chronos validation comparison](docs/experiments/001-baseline-chronos.md)
is recorded; peer reproduction and review are pending. Project fine-tuning,
serving, and monitoring remain later work.

## Dataset and model

The planned dataset is **Australian Electricity Demand** from the
[Monash Time Series Forecasting Repository](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/main/data):
five Australian states, half-hourly measurements, and roughly 231,000 observations
per state. Splits are versioned in `params.yaml`; the dataset card tracks the
remaining source-unit and calendar checks.

The base model is [amazon/chronos-t5-small](https://huggingface.co/amazon/chronos-t5-small),
a pretrained time-series model with 46 million parameters. It takes a historical
sequence and produces forecasts with prediction intervals. The comparison will
cover forecasting accuracy, compute cost, and serving requirements.

## Working on the project

Tasks are tracked in the [organization Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3).
Use a task branch and a reviewed PR for changes to `main`.

- [Working agreement](CONTRIBUTING.md): planning, ownership, branches, and reviews.
- [Development setup](docs/setup.md): environment installation, directory layout, and adaptations from the CCDS template.
- [MLflow tracking](docs/mlflow.md): shared setup, a small verification command, and run records for experiments.
- [Validation comparison](docs/evaluation.md): comparable seasonal-baseline and pinned Chronos runs, metrics, and reproduction commands.
- [M1 problem definition](docs/problem-definition.md): forecasting scope and open requirements.
- [Dataset card](docs/DATASET_CARD.md) and [model card](docs/MODEL_CARD.md): M1 extensions of the selected upstream cards, with project evidence still needed.
- [Engineering Decision Notebook](docs/edn/README.md): decisions and their rationale.
- [Course evidence](docs/rubric-evidence.md): links to work for each assessed practice.
- [GitHub setup status](docs/project-setup.md): configured settings and open tasks.

## Repository structure

The layout follows [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/),
adapted to this repository as described in [docs/setup.md](docs/setup.md).

```text
.github/            Issue and PR templates
data/               Raw, interim, processed, and external data (contents versioned with DVC)
docs/               Problem definition, dataset and model cards, setup guide, EDN
models/             Model checkpoints and predictions
notebooks/          Exploration notebooks
references/         Data dictionaries and explanatory material
reports/figures/    Generated figures
src/                Python package: config, dataset, features, modeling, plots
tests/              Pytest suite
AGENTS.md           Instructions for AI-assisted contributions
CONTRIBUTING.md     Working agreement
Makefile            Shortcuts for common tasks
pyproject.toml      Package metadata, dependencies, and tool configuration
uv.lock             Locked dependency versions
```

Quick start: `uv sync` to create the environment, then `uv run pytest`.

## Team members

| Name          | GitHub Profile                                           |
| :------------- | :-------------------------------------------------------- |
| Sindri Másson | [@sindrimasson-upc](https://github.com/sindrimasson-upc) |
| Didac Cayuela | [@didicayu](https://github.com/didicayu)                 |
| Pablo Perez   | [@PabloPerezCano](https://github.com/PabloPerezCano)     |
| Pau Adal      | [@pauadal03](https://github.com/NIU1638529)             |
| Antoni Lopera | [@toni646](https://github.com/toni646)                   |

## Citations

```text
@InProceedings{godahewa2021monash,
    author = "Godahewa, Rakshitha and Bergmeir, Christoph and Webb, Geoffrey I. and Hyndman, Rob J. and Montero-Manso, Pablo",
    title = "Monash Time Series Forecasting Archive",
    booktitle = "Neural Information Processing Systems Track on Datasets and Benchmarks",
    year = "2021",
    note = "forthcoming"
}
```

```text
@article{ansari2024chronos,
    title={Chronos: Learning the Language of Time Series},
    author={Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan, and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Syndar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2024},
    url={https://openreview.net/forum?id=gerNCVqqtR}
}
```
