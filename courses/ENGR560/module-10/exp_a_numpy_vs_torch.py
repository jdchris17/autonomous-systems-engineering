"""
Experiment A: does PyTorch compute exactly the same forward pass and gradients as the NumPy network?

Reference: the Module 8 NumPy MLP (2 -> 4 -> 1, tanh, sigmoid / BCE from logits) at a random
initialisation and again after 2000 training steps, on the XOR data. Its parameters are copied into
nn.Linear / nn.Tanh layers in float64; autograd gradients are compared with the analytical NumPy
backpropagation gradients.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from torch_mlp import NumpyMLP, TorchMLP, copy_from_numpy

OUT = Path(__file__).parent / "outputs"
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
Y = np.array([0, 1, 1, 0], float)


def compare(np_model, label):
    tm = copy_from_numpy(TorchMLP(2, 4).double(), np_model)
    xt, yt = torch.tensor(X), torch.tensor(Y)
    logits_t = tm(xt)
    loss_t = nn.BCEWithLogitsLoss()(logits_t, yt)
    tm.zero_grad()
    loss_t.backward()
    g_np = np_model.backward(X, Y)
    logits_np = np_model.forward(X)
    rows = [
        ("output logits", np.max(np.abs(logits_t.detach().numpy() - logits_np))),
        ("predicted probabilities", np.max(np.abs(torch.sigmoid(logits_t).detach().numpy() - np_model.predict_proba(X)))),
        ("binary cross-entropy loss", abs(loss_t.item() - np_model.loss(X, Y))),
        ("dJ/dW1 (2 x 4)", np.max(np.abs(tm.network[0].weight.grad.numpy().T - g_np["W1"]))),
        ("dJ/db1 (4)", np.max(np.abs(tm.network[0].bias.grad.numpy() - g_np["b1"]))),
        ("dJ/dW2 (4)", np.max(np.abs(tm.network[2].weight.grad.numpy().ravel() - g_np["W2"]))),
        ("dJ/db2 (scalar)", abs(tm.network[2].bias.grad.item() - g_np["b2"])),
    ]
    return rows, loss_t.item(), tm.network[0].weight.dtype


def main():
    OUT.mkdir(exist_ok=True)
    print(f"Experiment A: NumPy MLP vs PyTorch (torch {torch.__version__}), XOR data, float64")
    m0 = NumpyMLP(2, 4, seed=0)
    r0, l0, dt = compare(m0, "init")
    m1 = NumpyMLP(2, 4, seed=0)
    for _ in range(2000):
        m1.step(m1.backward(X, Y), 0.5)
    r1, l1, _ = compare(m1, "trained")
    print(f"  PyTorch parameter dtype: {dt}; loss at initialisation {l0:.6f}, after 2000 steps {l1:.6f}")
    print(f"\n    {'quantity':<28}{'max |torch - numpy|, at init':>30}{'after 2000 steps':>20}{'within 1e-10':>14}")
    for (name, a), (_, b) in zip(r0, r1):
        print(f"    {name:<28}{a:>30.2e}{b:>20.2e}{str(max(a, b) <= 1e-10):>14}")
    print("  (weights copied with W1 transposed: nn.Linear stores (out, in) and computes x W^T + b)")
    # the pitfall found while building this experiment
    b2 = m1.b2
    print(f"  dtype pitfall: torch.tensor([b2]) with a Python float -> {torch.tensor([b2]).dtype};"
          f" rounding error {abs(torch.tensor([b2]).item() - b2):.1e}. With float32 the trained-network logits"
          f" differed by ~5e-8, so every parameter is now copied with dtype=torch.float64.")


if __name__ == "__main__":
    main()
