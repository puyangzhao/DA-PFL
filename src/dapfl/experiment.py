from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .data import prepare_data
from .federated import train


def run(config_path: str):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output = Path(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in cfg["tasks"]:
        for partition in cfg["partitions"]:
            n_clients = 6 if partition == "demographic" else cfg["random_clients"]
            for seed in cfg["seeds"]:
                data = prepare_data(cfg["data_path"], task, partition, n_clients, cfg["test_size"], seed)
                for model in cfg["models"]:
                    for algorithm in cfg["algorithms"]:
                        result = train(data, model, algorithm, cfg["rounds"], cfg["local_epochs"], cfg["batch_size"],
                                       cfg["learning_rate"], cfg["fedprox_mu"], cfg["early_stopping_patience"],
                                       cfg["early_stopping_min_delta"], seed, cfg["weighted_aggregation"])
                        stem = f"{task}_{partition}_{model}_{algorithm}_seed{seed}"
                        with open(output / f"{stem}.json", "w", encoding="utf-8") as f:
                            json.dump(result.history, f, indent=2, allow_nan=True)
                        rows.append({"task": task, "partition": partition, "seed": seed, "model": model,
                                     "algorithm": algorithm, "input_dim": data.input_dim, **result.best_metrics})
                        pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args(); run(args.config)


if __name__ == "__main__":
    main()

