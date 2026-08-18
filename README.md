# DA-PFL

Reproducible PyTorch implementation accompanying **Demography-Aware Personalized Federated Learning for fair, private, and efficient clinical risk prediction** (Biomedical Signal Processing and Control 114, 109312; DOI: [10.1016/j.bspc.2025.109312](https://doi.org/10.1016/j.bspc.2025.109312)).

## What is included

- Demography-aware six-client partition using `RIDRETH3`
- Balanced random ten-client robustness partition
- MLP, single-timestep LSTM, and feature-token self-attention architectures
- FedAvg, FedProx, FedRep, and SCAFFOLD
- Diabetes and hypertension tasks
- AUC, accuracy, balanced accuracy, gender/race TPR gaps
- Exact bidirectional parameter-transmission accounting
- Multi-seed experiment configuration and per-round output

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp /path/to/nhanes_cleaned.csv data/nhanes_cleaned.csv
python -m dapfl.experiment --config configs/paper.yaml
```

Results are written incrementally to `results/paper/summary.csv`, with a JSON learning curve for every run. A quick validation is available with `configs/smoke.yaml`.

To rebuild the analytic table from official public component files:

```bash
python scripts/download_nhanes.py
python scripts/preprocess_nhanes.py
```

Because the article does not enumerate participant-level IQR exclusions or every recoding decision, the supplied cleaned CSV is the exact analysis input. The rebuilding script is an auditable reconstruction from the stated variables and criteria and reports its resulting sample size for comparison.

## Data and privacy

The analytic CSV is intentionally ignored by Git. NHANES is publicly available, but distributing a derived file should follow the applicable CDC terms. The code simulates cross-silo federated optimization: raw client records remain in client-specific `TensorDataset` objects and only model states are aggregated. This is not, by itself, a formal differential-privacy guarantee.

## Reproducibility notes

The released CSV contains 83 diabetes-positive participants among 1,119 (7.4%). One sentence in Section 3.1 reports 17.4%, while Section 4.1 and the CSV indicate 7.4%; this implementation follows the data.

The article describes local standardization but also a global training scaler for the held-out test set. To avoid incompatible feature spaces across clients, this implementation fits one preprocessing pipeline on training data and applies it to both client and test records. One-hot categories are learned only from the training split.

Communication cost is reported here as actual parameters transmitted in both server-to-client and client-to-server directions, multiplied by participating clients and rounds. This is stricter than counting a single model vector once per round.

For the hypertension generalizability task, hypertension is defined as systolic BP >=130 mmHg or diastolic BP >=80 mmHg (2017 ACC/AHA threshold), and the two BP measurements are excluded from predictors to prevent target leakage.

## Citation

Zhao P, Yue Z, Mi N, Zhang H. Demography-Aware Personalized Federated Learning for fair, private, and efficient clinical risk prediction. *Biomedical Signal Processing and Control*. 2026;114:109312. doi:10.1016/j.bspc.2025.109312.
