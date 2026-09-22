# Electricity Demand Forecasting with Amazon Chronos

An end-to-end Machine Learning project for short-term electricity demand forecasting using pre-trained and fine-tuned Foundation Models for time-series forecasting.

---

## Project Overview

Accurate electricity demand forecasting is crucial for grid stability, efficient energy distribution, and cost optimization. This project leverages **Amazon Chronos-T5-small**, a modern pretrained probabilistic time-series forecasting model, to predict half-hourly electricity demand across multiple states in Australia.

We evaluate the base zero-shot performance of Chronos, fine-tune it on domain-specific historical demand data, and perform a comparative analysis to assess accuracy improvements, computational trade-offs, and deployment feasibility.

---

## Dataset

We use the **Australian Electricity Demand Dataset** from the [Monash Time Series Forecasting Repository](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/main/data).

- **Granularity:** Half-hourly measurements ($2 \text{ records/hour}$)
- **Scope:** 5 Australian states
- **Volume:** $\approx 231,000$ observations per state
- **Target Variable:** Electricity demand (in Megawatts/kW)
- **Source:** [Monash TSF Repository on Hugging Face](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/main/data)

---

## Model Architecture

### Base Model: Amazon Chronos-T5-Small

- **Model:** [`amazon/chronos-t5-small`](https://huggingface.co/amazon/chronos-t5-small)
- **Parameters:** 46M
- **Type:** Pretrained language-model-based time-series forecaster (based on T5)
- **Input:** Historical sequence of electricity demand values.
- **Output:** Forecasted demand values over a predefined future horizon (with prediction intervals).

---

## Collaboration and Engineering Decisions

The initial working method uses GitHub Flow, issue-based planning across the six
course milestones, peer review, and evidence linked to completed work.

- [Organization Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3)
and [documentation task #1](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/1).
- [Team working agreement](CONTRIBUTING.md): board states, priorities, sizes,
responsibilities, and Git workflow.
- [Project setup record](docs/project-setup.md): verified configuration and
remaining access-dependent setup.
- [Engineering Decision Notebook](docs/edn/README.md): selected decisions and how
the team used and assessed AI contributions, beginning with
[EDN-001: team workflow](docs/edn/0001-team-workflow.md).
- [Course practice evidence](docs/rubric-evidence.md): current evidence and gaps
for the assessed practices.
- [Agent instructions](AGENTS.md): repository rules for AI-assisted work.

The Project and task are live. The native repository link and Project access setup
remain incomplete because of GitHub permissions. Branch protection, CI, and wider
team review are tracked separately from the documented policy.

## Repository Structure

Current repository documentation structure:

```text
├── .github/
│   ├── ISSUE_TEMPLATE/         # Engineering task and experiment templates
│   └── pull_request_template.md
├── docs/
│   ├── edn/                    # Decision index, entries, and reusable template
│   ├── project-setup.md        # Verified GitHub setup and access limitations
│   └── rubric-evidence.md      # Evidence for the course practices
├── AGENTS.md                   # Instructions for AI-assisted contributions
├── CONTRIBUTING.md             # Team working agreement
└── README.md                   # Project overview
```

The implementation structure and dependency environment will be introduced in
M2, following the course's Cookiecutter Data Science guidance and documenting any
project-specific adaptations.

## Team Members


| Name          | GitHub Profile                                           |
| :------------- | :-------------------------------------------------------- |
| Sindri Másson | [@sindrimasson-upc](https://github.com/sindrimasson-upc) |
| Didac Cayuela | [@didicayu](https://github.com/didicayu)                 |
| Pablo Perez   | [@PabloPerezCano](https://github.com/PabloPerezCano)     |
| Pau Adal      | [@NIU1638529](https://github.com/NIU1638529)             |
| Antoni Lopera | [@toni646](https://github.com/toni464)                   |


---

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
