"""
PyTorch version of the Module 8 network (the course calls it the "Module 9 MLP"; in this repository
it lives in ../module-8/mlp.py).

    nn.Sequential(nn.Linear(D, H), nn.Tanh(), nn.Linear(H, 1))   -> output LOGITS
Loss: nn.BCEWithLogitsLoss (numerically stable, the same log(1 + e^z) - y z as np.logaddexp).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-8"))

from mlp import MLP as NumpyMLP  # noqa: E402,F401


class TorchMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16):
        super().__init__()
        self.config = {"input_dim": input_dim, "hidden_dim": hidden_dim}
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)          # logits, shape (N,)


def copy_from_numpy(torch_model, numpy_model):
    """nn.Linear stores weight as (out, in) and computes x W^T + b, so W1 (in x out) is transposed.

    dtype is given explicitly: the NumPy b2 is a plain Python float, and torch.tensor([float]) would
    silently create a FLOAT32 tensor (PyTorch's default dtype), rounding b2 before the copy."""
    f64 = torch.float64
    with torch.no_grad():
        torch_model.network[0].weight.copy_(torch.tensor(numpy_model.W1.T, dtype=f64))
        torch_model.network[0].bias.copy_(torch.tensor(numpy_model.b1, dtype=f64))
        torch_model.network[2].weight.copy_(torch.tensor(numpy_model.W2[None, :], dtype=f64))
        torch_model.network[2].bias.copy_(torch.tensor([numpy_model.b2], dtype=f64))
    return torch_model


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
