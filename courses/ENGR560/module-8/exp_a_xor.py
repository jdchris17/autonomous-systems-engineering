"""
Experiment A: a 2 -> 4 -> 1 network (tanh hidden, sigmoid output, BCE, gradient descent) on XOR.

Also shown: the best a single linear classifier (logistic regression, no hidden layer) can do, and
how often random initialisations succeed for several learning rates (the objective is nonconvex).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlp import MLP, sigmoid, train

OUT = Path(__file__).parent / "outputs"
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
Y = np.array([0, 1, 1, 0], float)
ITERS = 5000


def linear_baseline():
    """Logistic regression on XOR by gradient descent: the optimum is p = 0.5 everywhere."""
    w, b = np.zeros(2), 0.0
    for _ in range(20000):
        p = sigmoid(X @ w + b)
        w -= 1.0 * X.T @ (p - Y) / 4
        b -= 1.0 * np.mean(p - Y)
    p = sigmoid(X @ w + b)
    loss = np.mean(np.logaddexp(0, X @ w + b) - Y * (X @ w + b))
    return w, b, p, loss


def main():
    OUT.mkdir(exist_ok=True)
    print("Experiment A: MLP 2 -> 4 -> 1 on XOR (tanh hidden, sigmoid output, binary cross-entropy, full-batch GD)")
    w, b, p, loss = linear_baseline()
    print(f"\n  Linear classifier (no hidden layer): w = {np.round(w, 4)}, b = {b:.4f}, p = {np.round(p, 3)},"
          f" loss {loss:.4f} (= ln 2 = {np.log(2):.4f}); correct at threshold 0.5: {np.sum((p >= 0.5) == Y)}/4")
    best_possible = max(np.sum(((X @ np.array([c, d]) + e) > 0) == Y) for c in (-1, 1) for d in (-1, 1) for e in np.linspace(-2.5, 1.5, 9))
    print(f"  (scanning sign patterns of any line: at most {best_possible}/4 points can be on the correct side)")

    print(f"\n  Random initialisations: 20 seeds per learning rate, {ITERS} iterations; success = all 4 correct")
    print(f"    {'learning rate':>14}{'success':>9}{'median final loss':>19}{'worst final loss':>18}")
    runs = {}
    for lr in (0.1, 0.5, 1.0, 3.0):
        fin = []
        for s in range(20):
            m = MLP(2, 4, seed=s)
            h, _ = train(m, X, Y, lr, ITERS)
            ok = np.all((m.predict_proba(X) >= 0.5) == Y)
            fin.append((h["train"][-1], ok, s))
        runs[lr] = fin
        losses = np.array([f[0] for f in fin])
        print(f"    {lr:>14}{sum(f[1] for f in fin):>6}/20{np.median(losses):>19.2e}{losses.max():>18.2e}")

    fails = []
    for s in range(50):
        m = MLP(2, 2, seed=s)
        hh, _ = train(m, X, Y, 0.5, ITERS)
        if not np.all((m.predict_proba(X) >= 0.5) == Y):
            fails.append((s, hh["train"][-1]))
    print(f"    for contrast, the minimal 2 -> 2 -> 1 network (lr 0.5, 50 seeds): {50 - len(fails)}/50 succeed;"
          f" failed runs stall at loss {np.round(sorted({round(f[1], 3) for f in fails}), 3).tolist()}"
          " (local minima / plateaus of the nonconvex objective)")

    model = MLP(2, 4, seed=0)
    h, _ = train(model, X, Y, 0.5, ITERS)
    p = model.predict_proba(X)
    print("\n  Selected run (seed 0, learning rate 0.5):")
    for xi, yi, pi in zip(X, Y, p):
        print(f"    x = {xi.astype(int)}  target {int(yi)}  P(y = 1) = {pi:.4f}  predicted {int(pi >= 0.5)}")
    print(f"    final loss {h['train'][-1]:.2e}; all four correct: {np.all((p >= 0.5) == Y)}")
    print(f"    hidden activations tanh(X W1 + b1):\n{np.round(np.tanh(X @ model.W1 + model.b1), 3)}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    for lr, col in ((0.1, "tab:green"), (0.5, "tab:blue"), (3.0, "tab:red")):
        m = MLP(2, 4, seed=0)
        hh, _ = train(m, X, Y, lr, ITERS)
        ax.semilogy(hh["iter"], hh["train"], color=col, label=f"seed 0, lr {lr}")
    for s in range(1, 6):
        m = MLP(2, 4, seed=s)
        hh, _ = train(m, X, Y, 0.5, ITERS)
        ax.semilogy(hh["iter"], hh["train"], color="tab:blue", alpha=0.3, lw=0.8)
    ax.axhline(np.log(2), color="k", ls=":", label="ln 2 (best linear model)")
    ax.set_xlabel("gradient-descent iteration")
    ax.set_ylabel("binary cross-entropy")
    ax.set_title("Loss vs iteration (faint: seeds 1-5, lr 0.5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 300), np.linspace(-0.5, 1.5, 300))
    G = np.column_stack([xx.ravel(), yy.ravel()])
    ax = axes[1]
    P = model.predict_proba(G).reshape(xx.shape)
    cs = ax.contourf(xx, yy, P, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.8)
    ax.contour(xx, yy, P, levels=[0.5], colors="k", linewidths=2)
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap="RdBu_r", edgecolor="k", s=200, vmin=0, vmax=1)
    fig.colorbar(cs, ax=ax, label="P(y = 1)")
    ax.set_title("MLP decision boundary (black: P = 0.5)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax = axes[2]
    H = np.tanh(G @ model.W1 + model.b1)
    for j in range(4):
        ax.contour(xx, yy, H[:, j].reshape(xx.shape), levels=[0], colors=[f"C{j}"], linewidths=1.5)
        ax.plot([], [], color=f"C{j}", label=f"hidden unit {j + 1}: z = 0")
    ax.contour(xx, yy, P, levels=[0.5], colors="k", linewidths=2.5)
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap="RdBu_r", edgecolor="k", s=200, vmin=0, vmax=1)
    ax.set_title("Each hidden unit draws a line; the output combines them")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_xor.png", dpi=130)
    plt.close(fig)
    print("\n  saved", OUT / "exp_a_xor.png")


if __name__ == "__main__":
    main()
