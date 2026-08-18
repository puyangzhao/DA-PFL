# DA-PFL

PyTorch implementation accompanying **Demography-Aware Personalized Federated Learning for fair, private, and efficient clinical risk prediction** (Biomedical Signal Processing and Control 114, 109312; DOI: [10.1016/j.bspc.2025.109312](https://doi.org/10.1016/j.bspc.2025.109312)).

This repository provides the methodological and experimental code described in the article. The analytic dataset is not included. Users are responsible for obtaining the publicly available NHANES data from the CDC and preparing the required input data.

## What is included

- Demography-aware six-client partition using `RIDRETH3`
- Balanced random ten-client robustness partition
- MLP, single-timestep LSTM, and feature-token self-attention architectures
- FedAvg, FedProx, FedRep, and SCAFFOLD
- Diabetes and hypertension tasks
- AUC, accuracy, balanced accuracy, and gender/race TPR gaps
- Bidirectional parameter-transmission accounting
- Multi-seed experiment configuration and per-round output

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Prepare the data

Download and preprocess the required public NHANES component files:

```bash
python scripts/download_nhanes.py
python scripts/preprocess_nhanes.py
```

The preprocessing script creates `data/nhanes_cleaned.csv`, which is the input expected by the experiment configuration. Users should review the official NHANES documentation and applicable CDC data-use guidance before conducting analyses.

## Run the experiments

```bash
python -m dapfl.experiment --config configs/paper.yaml
```

Results are written incrementally to `results/paper/summary.csv`, with a JSON learning curve for every run. A quick validation is available with:

```bash
python -m dapfl.experiment --config configs/smoke.yaml
```

## Data availability

The dataset is not distributed with this repository. NHANES data are publicly available from the [CDC National Center for Health Statistics](https://www.cdc.gov/nchs/nhanes/).

## Citation

Zhao P, Yue Z, Mi N, Zhang H. Demography-Aware Personalized Federated Learning for fair, private, and efficient clinical risk prediction. *Biomedical Signal Processing and Control*. 2026;114:109312. doi:10.1016/j.bspc.2025.109312.
