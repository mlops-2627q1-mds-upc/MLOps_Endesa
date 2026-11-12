# Electricity demand forecasting with Amazon Chronos

MLOps course project comparing seasonal baselines, pretrained Chronos-T5-small
and a fine-tuned checkpoint on five Australian electricity-demand series.
Each forecast uses the previous 512 values to predict the next 48 positions.

The reviewed first delivery and local FastAPI implementation are merged on
`main`. This candidate branch integrates the pending container, release,
deployment and monitoring implementations, including the
[local monitoring scenarios](docs/monitoring-scenarios.md). Their source checks,
historical model experiments and local synthetic runtime results have different
scopes. Human peer review, exact-source release validation in Actions, private
image publication, independent selected-model startup and final target rollout
remain pending. See the [delivery status and checklist](docs/final-delivery.md).

## Dataset and model

The dataset is **Australian Electricity Demand** from the
[Monash Time Series Forecasting Repository](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/58aafbe2712ff481c014f562e42723f2820fd5d4):
five Australian states, half-hourly measurements, and roughly 231,000 observations
per state. Splits are versioned in `params.yaml`; the dataset card tracks the
remaining source-unit and calendar checks.

The base model is [amazon/chronos-t5-small](https://huggingface.co/amazon/chronos-t5-small/tree/a971ba21945c4f1796b17a91fe69214b5f4ad472),
a pretrained time-series model with approximately 46 million parameters.
The evaluator reports the median of 20 sampled trajectories. The comparison
records forecasting error and compute cost. A bounded historical validation
subset measures interval coverage, and a historical VM experiment measures
serving performance; neither establishes an operating target or a final release.

## Working on the project

Tasks are tracked in the [organization Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3).
Use a task branch and a reviewed PR for changes to `main`.

- [Working agreement](CONTRIBUTING.md): planning, ownership, branches, and reviews.
- [Development setup](docs/setup.md): environment installation, directory layout, and adaptations from the CCDS template.
- [Replication](docs/replication.md): fast setup, artifact identities, historical reproduction and release gates.
- [Final delivery drafts](docs/delivery/README.md): editable report, PDF presentation, speaker guide and local build instructions.
- [Final execution checklist](docs/final-delivery.md): candidate source, dependency PRs, reviewer access and remaining acceptance evidence.
- [API](docs/api.md), [container](docs/container.md) and [release validation](docs/release-validation.md): separate interface, fixture and selected-model checks.
- [Private image publication](docs/image-publication.md) and [deployment/recovery](docs/deployment-release.md): reviewed prerequisites and later execution procedures.
- [Service monitoring](docs/monitoring-service.md), [chronological replay](docs/replay.md), [drift](docs/drift.md) and [forecast quality](docs/forecast-quality.md): signals, sample sufficiency and response procedures.
- [MLflow tracking](docs/mlflow.md): shared setup, a small verification command, and run records for experiments.
- [Validation comparison](docs/evaluation.md): comparable seasonal-baseline and pinned Chronos runs, metrics, and reproduction commands.
- [Minimum functionality checks](docs/minimum-functionality.md): known synthetic continuations, declared error limits, and bounded real-model tests.
- [M1 problem definition](docs/problem-definition.md): forecasting scope and open requirements.
- [Dataset card](docs/DATASET_CARD.md) and [model card](docs/MODEL_CARD.md): versions, project results and limitations.
- [Engineering Decision Notebook](docs/edn/README.md): decisions and their rationale.
- [Course evidence](docs/rubric-evidence.md): links to work for each assessed practice.
- [GitHub setup status](docs/project-setup.md): configured settings and open tasks.

## Repository structure

The layout follows [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/),
adapted to this repository as described in [docs/setup.md](docs/setup.md).

```text
.github/            Issue and PR templates
data/               Raw data and processed splits (contents versioned with DVC)
docs/               Problem definition, dataset and model cards, setup guide, EDN
models/             DVC checkpoint references and ignored training outputs
notebooks/          Exploration notebooks
reports/figures/    Generated figures
src/                Python package: data preparation, validation, evaluation and tracking
monitoring/         Prometheus/Grafana configuration, drift lock and local evidence
tests/              Pytest suite
AGENTS.md           Instructions for AI-assisted contributions
CONTRIBUTING.md     Working agreement
Makefile            Shortcuts for common tasks
pyproject.toml      Package metadata, dependencies, and tool configuration
uv.lock             Locked dependency versions
```

Quick start: `uv sync --locked --dev`, then `make lint` and `make coverage`.
Follow the replication guide for real data and optional model inference.

## Team members

| Name          | GitHub Profile                                           |
| :------------- | :-------------------------------------------------------- |
| Sindri Másson | [@sindrimasson-upc](https://github.com/sindrimasson-upc) |
| Didac Cayuela | [@didicayu](https://github.com/didicayu)                 |
| Pablo Perez   | [@PabloPerezCano](https://github.com/PabloPerezCano)     |
| Pau Adal      | [@pauadal03](https://github.com/pauadal03)             |
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
