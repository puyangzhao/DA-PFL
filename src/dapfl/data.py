from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import TensorDataset

CATEGORICAL = ["RIAGENDR", "RIDRETH3", "SMQ020", "ALQ111"]
CONTINUOUS = ["RIDAGEYR", "INDFMPIR", "BMXBMI", "BPXOSY1", "BPXODI1", "LBXGLU", "LBXIN", "PAD680", "PAD820"]
RACE_NAMES = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other/Multiracial"}


@dataclass
class PreparedData:
    clients: dict[int, TensorDataset]
    test_x: torch.Tensor
    test_y: torch.Tensor
    test_gender: np.ndarray
    test_race: np.ndarray
    input_dim: int
    feature_names: list[str]


def load_clean_csv(path: str | Path, task: str = "diabetes") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(CATEGORICAL + CONTINUOUS + ["diabetes"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if task == "diabetes":
        df = df.rename(columns={"diabetes": "target"})
    elif task == "hypertension":
        # ACC/AHA 2017 threshold. BP variables are removed from predictors below.
        df["target"] = ((df["BPXOSY1"] >= 130) | (df["BPXODI1"] >= 80)).astype(int)
    else:
        raise ValueError("task must be 'diabetes' or 'hypertension'")
    return df


def _transformer(task: str) -> ColumnTransformer:
    continuous = [c for c in CONTINUOUS if not (task == "hypertension" and c in {"BPXOSY1", "BPXODI1"})]
    return ColumnTransformer([
        ("continuous", StandardScaler(), continuous),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def prepare_data(path: str | Path, task: str, partition: str, n_clients: int, test_size: float, seed: int) -> PreparedData:
    df = load_clean_csv(path, task)
    train, test = train_test_split(df, test_size=test_size, stratify=df["target"], random_state=seed)
    prep = _transformer(task)
    x_train = prep.fit_transform(train)
    x_test = prep.transform(test)
    y_train = train["target"].to_numpy(np.float32)

    if partition == "demographic":
        raw_ids = train["RIDRETH3"].astype(int).to_numpy()
        values = sorted(np.unique(raw_ids).tolist())
        id_map = {v: i for i, v in enumerate(values)}
        client_ids = np.array([id_map[v] for v in raw_ids])
    elif partition == "random":
        rng = np.random.default_rng(seed)
        client_ids = np.arange(len(train)) % n_clients
        rng.shuffle(client_ids)
    else:
        raise ValueError("partition must be 'demographic' or 'random'")

    clients = {}
    for cid in sorted(np.unique(client_ids)):
        mask = client_ids == cid
        clients[int(cid)] = TensorDataset(torch.tensor(x_train[mask], dtype=torch.float32), torch.tensor(y_train[mask]))
    names = prep.get_feature_names_out().tolist()
    return PreparedData(
        clients=clients,
        test_x=torch.tensor(x_test, dtype=torch.float32),
        test_y=torch.tensor(test["target"].to_numpy(np.float32)),
        test_gender=test["RIAGENDR"].astype(int).to_numpy(),
        test_race=test["RIDRETH3"].astype(int).to_numpy(),
        input_dim=x_train.shape[1], feature_names=names,
    )

