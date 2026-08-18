"""Build the article's 14-column analytic table from public NHANES XPT files."""
from functools import reduce
from pathlib import Path

import pandas as pd

COMPONENTS = {
    "DEMO_L": ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDFMPIR"],
    "BMX_L": ["SEQN", "BMXBMI"],
    "BPXO_L": ["SEQN", "BPXOSY1", "BPXODI1"],
    "GLU_L": ["SEQN", "LBXGLU"],
    "INS_L": ["SEQN", "LBXIN"],
    "SMQ_L": ["SEQN", "SMQ020"],
    "ALQ_L": ["SEQN", "ALQ111"],
    "PAQ_L": ["SEQN", "PAD680", "PAD820"],
    "DIQ_L": ["SEQN", "DIQ010"],
}
FEATURES = ["RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDFMPIR", "BMXBMI", "BPXOSY1", "BPXODI1", "LBXGLU", "LBXIN", "SMQ020", "ALQ111", "PAD680", "PAD820"]


def main():
    raw = Path("data/raw")
    frames = []
    for stem, columns in COMPONENTS.items():
        path = raw / f"{stem}.xpt"
        frame = pd.read_sas(path, format="xport")
        absent = set(columns) - set(frame.columns)
        if absent:
            raise ValueError(f"{path} is missing {sorted(absent)}")
        frames.append(frame[columns])
    merged = reduce(lambda left, right: left.merge(right, on="SEQN", how="inner", validate="one_to_one"), frames)
    merged = merged.drop_duplicates("SEQN")
    merged = merged.loc[merged[FEATURES + ["DIQ010"]].isna().mean(axis=1) <= 0.20]
    for col in ["SMQ020", "ALQ111", "DIQ010"]:
        merged.loc[merged[col].isin([7, 9]), col] = pd.NA
    clean = merged.dropna(subset=FEATURES + ["DIQ010"]).copy()
    clean["diabetes"] = ((clean["LBXGLU"] >= 126) | (clean["DIQ010"] == 1)).astype(int)
    clean[FEATURES + ["diabetes"]].to_csv("data/nhanes_cleaned.csv", index=False)
    print(f"wrote data/nhanes_cleaned.csv: {len(clean)} rows; diabetes positives={clean['diabetes'].sum()}")


if __name__ == "__main__":
    main()

