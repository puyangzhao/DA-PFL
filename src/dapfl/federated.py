from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import evaluate
from .models import build_model, shared_parameter_names


@dataclass
class TrainResult:
    history: list[dict]
    best_metrics: dict
    communication_parameters: int


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _average_states(states, weights, keys=None):
    keys = keys or states[0].keys()
    total = float(sum(weights))
    return {k: sum(s[k] * (w / total) for s, w in zip(states, weights)) for k in keys}


def _probabilities(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x)).cpu().numpy()


def train(data, model_name: str, algorithm: str, rounds: int, local_epochs: int, batch_size: int,
          lr: float, mu: float, patience: int, min_delta: float, seed: int, weighted_aggregation: bool = False):
    set_seed(seed)
    global_model = build_model(model_name, data.input_dim)
    client_heads = {cid: deepcopy(global_model.head.state_dict()) for cid in data.clients}
    server_c = {n: torch.zeros_like(p) for n, p in global_model.named_parameters()}
    client_c = {cid: {n: torch.zeros_like(p) for n, p in global_model.named_parameters()} for cid in data.clients}
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_metrics, stale, history, comm = -np.inf, {}, 0, [], 0

    for round_idx in range(rounds):
        local_states, sizes, control_deltas = [], [], []
        for cid, dataset in data.clients.items():
            local = deepcopy(global_model)
            if algorithm == "fedrep":
                local.head.load_state_dict(client_heads[cid])
            optimizer = torch.optim.Adam(local.parameters(), lr=lr)
            start = {n: p.detach().clone() for n, p in global_model.named_parameters()}
            steps = 0
            for _ in range(local_epochs):
                for xb, yb in DataLoader(dataset, batch_size=batch_size, shuffle=True):
                    optimizer.zero_grad()
                    loss = criterion(local(xb), yb)
                    if algorithm == "fedprox":
                        loss = loss + 0.5 * mu * sum((p - start[n]).pow(2).sum() for n, p in local.named_parameters())
                    loss.backward()
                    if algorithm == "scaffold":
                        for n, p in local.named_parameters():
                            p.grad.add_(server_c[n] - client_c[cid][n])
                    optimizer.step(); steps += 1
            if algorithm == "fedrep":
                client_heads[cid] = deepcopy(local.head.state_dict())
            if algorithm == "scaffold":
                new_c, delta_c = {}, {}
                for n, p in local.named_parameters():
                    new_c[n] = client_c[cid][n] - server_c[n] + (start[n] - p.detach()) / (steps * lr)
                    delta_c[n] = new_c[n] - client_c[cid][n]
                client_c[cid] = new_c; control_deltas.append(delta_c)
            local_states.append(deepcopy(local.state_dict())); sizes.append(len(dataset))

        weights = sizes if weighted_aggregation else [1] * len(sizes)
        keys = shared_parameter_names(global_model) if algorithm == "fedrep" else set(global_model.state_dict())
        averaged = _average_states(local_states, weights, keys)
        state = global_model.state_dict(); state.update(averaged); global_model.load_state_dict(state)
        if algorithm == "scaffold":
            for n in server_c:
                server_c[n] += sum(d[n] for d in control_deltas) / len(control_deltas)

        sent_per_client = sum(global_model.state_dict()[k].numel() for k in keys)
        comm += 2 * len(data.clients) * sent_per_client
        probabilities = _probabilities(global_model, data.test_x)
        metrics = evaluate(data.test_y.numpy(), probabilities, data.test_gender, data.test_race)
        metrics.update({"round": round_idx + 1, "communication_parameters": comm})
        history.append(metrics)
        if metrics["auc"] > best_auc + min_delta:
            best_auc, best_metrics, stale = metrics["auc"], deepcopy(metrics), 0
        else:
            stale += 1
            if stale >= patience:
                break
    return TrainResult(history, best_metrics, comm)

