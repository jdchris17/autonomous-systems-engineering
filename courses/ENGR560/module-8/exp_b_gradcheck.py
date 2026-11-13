"""
Experiment B: verify backpropagation numerically with Module 2's numerical-gradient checker.

    dJ/dtheta_j ~ (J(theta + h e_j) - J(theta - h e_j)) / (2h),   h = 1e-5
    relative error E = ||g_backprop - g_numerical|| / max(1, ||g_backprop||, ||g_numerical||)
                       (Module 2's relative_discrepancy, applied per parameter block)

Test cases (double precision):
  1. the XOR network at its random initialisation (2 -> 4 -> 1)
  2. a well-conditioned random problem: 3 -> 16 -> 1, 64 random observations, random labels
Then four deliberately wrong backward passes, to confirm the checker catches each one.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-2"))

from gd_core import numerical_gradient, relative_discrepancy  # noqa: E402

from mlp import MLP, PARAMS, sigmoid  # noqa: E402

H = 1e-5


def blocks(model, g_flat):
    out, i = {}, 0
    for k in PARAMS:
        size = int(np.prod(np.shape(getattr(model, k)))) if np.shape(getattr(model, k)) else 1
        out[k] = g_flat[i:i + size]
        i += size
    return out


def check(model, X, y, backward=None, label=""):
    theta0 = model.get_flat()

    def J(theta):
        model.set_flat(theta)
        v = model.loss(X, y)
        model.set_flat(theta0)
        return v
    g_num = numerical_gradient(J, theta0, H)
    g = (backward or model.backward)(X, y)
    g_bp = np.concatenate([np.ravel(g[k]) for k in PARAMS])
    nb = blocks(model, g_num)
    rows = []
    for k in PARAMS:
        a, n = np.ravel(g[k]), nb[k]
        rows.append((k, a, n, relative_discrepancy(a, n)))
    return rows, relative_discrepancy(g_bp, g_num)


def show(rows, total, label):
    print(f"\n  {label}")
    print(f"    {'parameter':<10}{'shape':<9}{'analytical (first 3 entries)':<38}{'numerical (first 3 entries)':<38}{'relative error':>15}")
    for k, a, n, e in rows:
        shape = "scalar" if a.size == 1 else str(a.size)
        fa = np.array2string(a[:3], precision=6, suppress_small=False)
        fn = np.array2string(n[:3], precision=6, suppress_small=False)
        print(f"    {k:<10}{shape:<9}{fa:<38}{fn:<38}{e:>15.2e}")
    print(f"    all parameters together: relative error {total:.2e}  ->  {'PASS' if total < 1e-6 else 'FAIL'} (threshold 1e-6)")


def main():
    print(f"Experiment B: backpropagation vs central differences (h = {H}), Module 2's numerical_gradient")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0], float)
    m = MLP(2, 4, seed=0)
    rows, tot = check(m, X, y)
    show(rows, tot, "Case 1 - XOR network 2 -> 4 -> 1 at initialisation (17 parameters)")

    rng = np.random.default_rng(1)
    Xr = rng.normal(size=(64, 3))
    yr = (rng.random(64) < 0.5).astype(float)
    m2 = MLP(3, 16, seed=2)
    rows, tot = check(m2, Xr, yr)
    show(rows, tot, f"Case 2 - random well-conditioned problem 3 -> 16 -> 1, 64 observations ({m2.n_params()} parameters)")

    print("\n  Step size h (Case 2): relative error of all parameters")
    theta0 = m2.get_flat()
    g_bp = np.concatenate([np.ravel(m2.backward(Xr, yr)[k]) for k in PARAMS])
    for h in (1e-1, 1e-3, 1e-5, 1e-7, 1e-9):
        def J(t):
            m2.set_flat(t)
            v = m2.loss(Xr, yr)
            m2.set_flat(theta0)
            return v
        print(f"    h = {h:<7g} E = {relative_discrepancy(g_bp, numerical_gradient(J, theta0, h)):.2e}")

    # ---------------------------------------------------------------- deliberate bugs
    def make_bug(kind):
        def bw(Xb, yb, model=m2):
            z2 = model.forward(Xb)
            a1 = model.cache["a1"]
            n = len(yb)
            dz2 = (sigmoid(z2) - yb) / n
            if kind == "sign":
                dz2 = (yb - sigmoid(z2)) / n                 # wrong sign of the output error
            if kind == "mean":
                dz2 = (sigmoid(z2) - yb)                     # forgot the 1/N of the mean
            dW2, db2 = a1.T @ dz2, float(dz2.sum())
            da1 = np.outer(dz2, model.W2)
            dz1 = da1 * (1 - a1**2)
            if kind == "tanh":
                dz1 = da1 * (1 - a1)                         # wrong tanh derivative
            if kind == "relu":
                dz1 = da1 * (a1 > 0)                         # derivative of the wrong activation
            return {"W1": Xb.T @ dz1, "b1": dz1.sum(0), "W2": dW2, "b2": db2}
        return bw

    print("\n  Deliberately incorrect backward passes (Case 2): the checker must flag each one")
    print(f"    {'bug':<46}{'W1':>10}{'b1':>10}{'W2':>10}{'b2':>10}{'overall':>10}   verdict")
    for kind, desc in (("tanh", "tanh' written as (1 - a) instead of (1 - a^2)"),
                       ("relu", "used a ReLU derivative for a tanh layer"),
                       ("sign", "output error (y - p) instead of (p - y)"),
                       ("mean", "forgot to divide by N (sum instead of mean)")):
        rows, tot = check(m2, Xr, yr, backward=make_bug(kind))
        print(f"    {desc:<46}" + "".join(f"{r[3]:>10.1e}" for r in rows) + f"{tot:>10.1e}   "
              + ("DETECTED" if tot > 1e-6 else "missed"))
    print("    (a bug in the hidden layer leaves the W2 / b2 rows clean - the checker also localises the error)")


if __name__ == "__main__":
    main()
