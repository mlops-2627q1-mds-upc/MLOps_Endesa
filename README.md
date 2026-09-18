# Electricity Demand Forecasting with Amazon Chronos

An end-to-end Machine Learning project for short-term electricity demand forecasting using pre-trained and fine-tuned Foundation Models for time-series forecasting.

---

## Project Overview

Accurate electricity demand forecasting is crucial for grid stability, efficient energy distribution, and cost optimization. This project leverages **Amazon Chronos-T5-small**, a modern pretrained probabilistic time-series forecasting model, to predict half-hourly electricity demand across multiple states in Australia.

We evaluate the base zero-shot performance of Chronos, fine-tune it on domain-specific historical demand data, and perform a comparative analysis to assess accuracy improvements, computational trade-offs, and deployment feasibility.

---

## Dataset

We use the **Australian Electricity Demand Dataset** from the [Monash Time Series Forecasting Repository](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/main/data).

* **Granularity:** Half-hourly measurements ($2 \text{ records/hour}$)
* **Scope:** 5 Australian states
* **Volume:** $\approx 231,000$ observations per state
* **Target Variable:** Electricity demand (in Megawatts/kW)
* **Source:** [Monash TSF Repository on Hugging Face](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/main/data)

---

## Model Architecture

### Base Model: Amazon Chronos-T5-Small
* **Model:** [`amazon/chronos-t5-small`](https://huggingface.co/amazon/chronos-t5-small)
* **Parameters:** 46M
* **Type:** Pretrained language-model-based time-series forecaster (based on T5)
* **Input:** Historical sequence of electricity demand values.
* **Output:** Forecasted demand values over a predefined future horizon (with prediction intervals).

---

## Repository Structure

```text

├── data/                  # Raw and processed datasets (git-ignored)
├── notebooks/             # Exploratory Data Analysis & experimentation
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

```
---

## Team Members

| Name | GitHub Profile |
| :--- | :--- |
| Sindri Másson | [@sindrimasson-upc](https://github.com/sindrimasson-upc) |
| Didac Cayuela | [@didicayu](https://github.com/didicayu) |
|  Pablo Perez  | [@PabloPerezCano](https://github.com/PabloPerezCano) |
| Pau Adal | [@NIU1638529](https://github.com/NIU1638529) |
| Antoni Lopera | [@toni464](https://github.com/toni464) |

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
