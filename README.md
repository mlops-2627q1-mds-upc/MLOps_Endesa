# Electricity demand forecasting with Amazon Chronos

MLOps course project on half-hourly electricity demand in Australia. We plan to
compare Chronos-T5-small's zero-shot forecasts with a fine-tuned version, then
build the training, serving, and monitoring workflow around that model.

The repository currently contains the project outline and working agreement.
Implementation and experiments are still to come.

## Dataset and model

The planned dataset is **Australian Electricity Demand** from the
[Monash Time Series Forecasting Repository](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/main/data):
five Australian states, half-hourly measurements, and roughly 231,000 observations
per state. The dataset card will confirm units and evaluation splits before use.

The base model is [amazon/chronos-t5-small](https://huggingface.co/amazon/chronos-t5-small),
a pretrained time-series model with 46 million parameters. It takes a historical
sequence and produces forecasts with prediction intervals. The comparison will
cover forecasting accuracy, compute cost, and serving requirements.

## Working on the project

Tasks are tracked in the [organization Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3).
Use a task branch and a reviewed PR for changes to `main`.

- [Working agreement](CONTRIBUTING.md): planning, ownership, branches, and reviews.
- [Engineering Decision Notebook](docs/edn/README.md): decisions and their rationale.
- [Course evidence](docs/rubric-evidence.md): links to work for each assessed practice.
- [GitHub setup status](docs/project-setup.md): configured settings and open tasks.

## Repository structure

```text
.github/                Issue and PR templates
docs/edn/               Decision records and template
docs/project-setup.md   GitHub configuration status
docs/rubric-evidence.md Course evidence
AGENTS.md               Instructions for AI-assisted contributions
CONTRIBUTING.md         Working agreement
README.md               Project outline
```

The implementation structure and dependency environment will be introduced in M2,
following the course's Cookiecutter Data Science guidance.

## Team members

| Name          | GitHub Profile                                           |
| :------------- | :-------------------------------------------------------- |
| Sindri Másson | [@sindrimasson-upc](https://github.com/sindrimasson-upc) |
| Didac Cayuela | [@didicayu](https://github.com/didicayu)                 |
| Pablo Perez   | [@PabloPerezCano](https://github.com/PabloPerezCano)     |
| Pau Adal      | [@NIU1638529](https://github.com/NIU1638529)             |
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
