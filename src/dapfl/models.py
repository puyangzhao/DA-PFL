from __future__ import annotations

import math
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.representation = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        return self.head(self.representation(x)).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.representation = nn.LSTM(input_dim, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.representation(x.unsqueeze(1))
        return self.head(out[:, -1]).squeeze(-1)


class Attention(nn.Module):
    """Single-head feature-token self-attention for tabular data."""
    def __init__(self, input_dim: int, embed_dim: int = 8):
        super().__init__()
        self.feature_scale = nn.Parameter(torch.randn(input_dim, embed_dim) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, embed_dim))
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.representation = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.head = nn.Linear(embed_dim, 1)
        self.embed_dim = embed_dim

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.feature_scale + self.feature_bias
        q, k, v = self.q(tokens), self.k(tokens), self.v(tokens)
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.embed_dim), dim=-1)
        context = (weights @ v).mean(dim=1)
        return self.head(self.representation(context)).squeeze(-1)


def build_model(name: str, input_dim: int) -> nn.Module:
    return {"mlp": MLP, "lstm": LSTM, "attention": Attention}[name.lower()](input_dim)


def shared_parameter_names(model: nn.Module) -> set[str]:
    return {name for name, _ in model.named_parameters() if not name.startswith("head.")}

