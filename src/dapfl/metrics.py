from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def tpr_gap(y_true, probability, group, threshold=0.5):
    pred = np.asarray(probability) >= threshold
    y = np.asarray(y_true).astype(bool)
    rates = {}
    for g in np.unique(group):
        positive = (np.asarray(group) == g) & y
        rates[str(g)] = float(pred[positive].mean()) if positive.any() else float("nan")
    valid = [v for v in rates.values() if not np.isnan(v)]
    return (float(max(valid) - min(valid)) if len(valid) >= 2 else float("nan")), rates


def evaluate(y_true, probability, gender, race):
    y = np.asarray(y_true)
    p = np.asarray(probability)
    gender_gap, gender_tpr = tpr_gap(y, p, gender)
    race_gap, race_tpr = tpr_gap(y, p, race)
    return {
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
        "gender_tpr_gap": gender_gap,
        "race_tpr_gap": race_gap,
        "gender_tpr": gender_tpr,
        "race_tpr": race_tpr,
    }

