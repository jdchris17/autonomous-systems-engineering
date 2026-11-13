"""
Experiment B: gradient descent vs direct least squares on Module 1's data.

Data: Module 1 starter experiment (rng 560), 30 training points, y = 3 + 2x + eps, sigma = 1.5.
Start: theta = [0, 0].  Learning rate 0.05 (below the stability limit 2/lambda_max ~= 0.12,
found in Experiment D), tolerance ||grad|| <= 1e-8, budget 10,000 updates.
Reference: Module 1 fit_linear() (numpy.linalg.lstsq) - used only for the comparison.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gd_core import design_matrix, fit_linear, gradient_descent, mse_loss, starter_datasets

OUT = Path(__file__).parent / "outputs"
ETA = 0.05
MAX_IT = 10_000
TOL = 1e-8


def main():
    OUT.mkdir(exist_ok=True)
    x, y, _, _ = starter_datasets()
    X = design_matrix(x)
    theta_gd, loss_h, theta_h, info = gradient_descent(X, y, np.zeros(2), ETA, MAX_IT, TOL)
    theta_ls = fit_linear(x, y)
    j_ls = mse_loss(X, y, theta_ls)
    diff = np.linalg.norm(theta_gd - theta_ls)

    print(f"Experiment B: gradient descent (eta = {ETA}, start [0, 0]) vs least squares, N = 30")
    print(f"  stop: {info['status']} after {info['iterations']} updates,"
          f" final ||grad|| = {info['grad_norm']:.2e}")
    print(f"  {'quantity':<28}{'gradient descent':>18}{'direct least squares':>22}")
    print(f"  {'learned intercept':<28}{theta_gd[0]:>18.10f}{theta_ls[0]:>22.10f}")
    print(f"  {'learned slope':<28}{theta_gd[1]:>18.10f}{theta_ls[1]:>22.10f}")
    print(f"  {'final training MSE':<28}{loss_h[-1]:>18.10f}{j_ls:>22.10f}")
    print(f"  {'parameter difference norm':<28}{diff:>18.2e}{'-':>22}")
    print(f"  agreement within 1e-4: {diff < 1e-4}")
    first = {thr: int(np.argmax(np.linalg.norm(theta_h - theta_ls, axis=1) < thr))
             for thr in (1e-1, 1e-2, 1e-4, 1e-6)}
    print("  first iteration within distance d of the LS solution: "
          + ", ".join(f"d={k:g}: {v}" for k, v in first.items()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    it = np.arange(loss_h.size)
    ax.semilogy(it, loss_h, color="tab:blue", label="training loss J(theta_k)")
    ax.semilogy(it, loss_h - j_ls, color="tab:orange", label="J(theta_k) - J(theta_LS)")
    ax.axhline(j_ls, color="k", ls=":", label=f"least-squares loss {j_ls:.4f}")
    ax.set_xlabel("iteration k")
    ax.set_ylabel("MSE")
    ax.set_title(f"Loss vs iteration (eta = {ETA})")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    t0 = np.linspace(-0.5, 4.0, 300)
    t1 = np.linspace(-0.5, 2.6, 300)
    T0, T1 = np.meshgrid(t0, t1)
    # J on a grid, vectorized: residuals for every grid point
    R = T0[..., None] + T1[..., None] * x - y
    J = np.mean(R**2, axis=-1)
    cs = ax.contour(T0, T1, J, levels=np.geomspace(j_ls * 1.01, J.max(), 18), cmap="viridis",
                    linewidths=0.8)
    ax.clabel(cs, fontsize=6, fmt="%.0f")
    ax.plot(theta_h[:, 0], theta_h[:, 1], ".-", color="tab:red", ms=3, lw=1,
            label="gradient-descent path")
    ax.plot(*theta_h[0], "ks", label="start [0, 0]")
    ax.plot(*theta_ls, "*", color="gold", mec="k", ms=15, label="least-squares solution")
    ax.set_xlabel("theta0 (intercept)")
    ax.set_ylabel("theta1 (slope)")
    ax.set_title("Path through parameter space")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_gd_vs_lstsq.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_b_gd_vs_lstsq.png'}")


if __name__ == "__main__":
    main()
