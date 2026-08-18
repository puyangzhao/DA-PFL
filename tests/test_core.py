import numpy as np
import torch

from dapfl.metrics import tpr_gap
from dapfl.models import build_model, shared_parameter_names


def test_all_models_shape():
    x = torch.randn(5, 22)
    for name in ["mlp", "lstm", "attention"]:
        assert build_model(name, 22)(x).shape == (5,)


def test_tpr_gap():
    gap, rates = tpr_gap([1, 1, 1, 1], [.9, .1, .8, .7], [1, 1, 2, 2])
    assert np.isclose(gap, 0.5)
    assert rates == {"1": 0.5, "2": 1.0}


def test_fedrep_excludes_head():
    model = build_model("mlp", 10)
    assert all(not n.startswith("head.") for n in shared_parameter_names(model))

