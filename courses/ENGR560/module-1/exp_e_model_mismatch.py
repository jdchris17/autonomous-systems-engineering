"""
Optional advanced challenge: deliberate model mismatch.

Truth:   y = 3 + 2x + 0.25x^2 + eps,  eps ~ N(0, 1.5^2),  x ~ U(-5, 5)
Model:   y_hat = theta0 + theta1 x            (still a straight line)

With infinite data, least squares converges to the best straight line in the
mean-square sense over x ~ U(-5,5), i.e. the projection of the truth onto {1, x}:
    theta1* = 2                          (x^2 is even, uncorrelated with x)
    theta0* = 3 + 0.25 E[x^2] = 3 + 0.25 * 25/3 = 5.0833
and the error that can never be removed by more data is
    bias^2 = 0.25^2 Var(x^2) = 0.0625 (E[x^4] - E[x^2]^2) = 0.0625 (125 - 625/9) = 3.4722
so E[test MSE] -> sigma^2 + bias^2 = 2.25 + 3.47 = 5.72, not 2.25.

For contrast the same data are also fitted with a quadratic model (columns 1, x, x^2),
whose functional form is correct and therefore approaches 2.25.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learning_core import (NOISE_STD, SEED, TRUE_INTERCEPT, TRUE_SLOPE, fit_linear, generate_data,
                           mse, predict)

OUT = Path(__file__).parent / "outputs"
QUAD = 0.25
N_VALUES = (5, 20, 100, 1000, 10000)
N_TEST = 2000
N_TRIALS = 500

EX2 = 25.0 / 3.0
EX4 = 125.0
BEST_LINE = np.array([TRUE_INTERCEPT + QUAD * EX2, TRUE_SLOPE])
BIAS2 = QUAD**2 * (EX4 - EX2**2)
SIGMA2 = NOISE_STD**2


def truth(x):
    return TRUE_INTERCEPT + TRUE_SLOPE * x + QUAD * x**2


def main():
    OUT.mkdir(exist_ok=True)
    print("Optional challenge: straight-line model fitted to y = 3 + 2x + 0.25x^2 + eps")
    print(f"  best possible line (infinite data): theta* = [{BEST_LINE[0]:.4f}, {BEST_LINE[1]:.4f}]")
    print(f"  irreducible noise sigma^2 = {SIGMA2:.4f},  model bias^2 = {BIAS2:.4f},"
          f"  floor for the line = {SIGMA2 + BIAS2:.4f}\n")

    stats = {}
    for i_n, n in enumerate(N_VALUES):
        lin = {"th0": [], "th1": [], "train": [], "test": []}
        quad_test = []
        for t in range(N_TRIALS):
            rng = np.random.default_rng([t, i_n, 99])
            x, y = generate_data(n, rng, quad_coef=QUAD)
            xt, yt = generate_data(N_TEST, rng, quad_coef=QUAD)
            th = fit_linear(x, y, degree=1)
            lin["th0"].append(th[0])
            lin["th1"].append(th[1])
            lin["train"].append(mse(y, predict(th, x)))
            lin["test"].append(mse(yt, predict(th, xt)))
            if n > 3:
                quad_test.append(mse(yt, predict(fit_linear(x, y, degree=2), xt)))
        stats[n] = {k: np.array(v) for k, v in lin.items()}
        stats[n]["quad_test"] = np.array(quad_test)

    print(f"  {N_TRIALS} trials per N, test sets of {N_TEST} points")
    print(f"  {'N':>6} | {'theta0':>15} | {'theta1':>15} | {'line train':>10} {'line test':>10}"
          f" {'test-floor':>10} | {'quad test':>10}")
    print("  " + "-" * 92)
    for n in N_VALUES:
        s = stats[n]
        print(f"  {n:>6} | {s['th0'].mean():>7.3f}+-{s['th0'].std():<6.3f} |"
              f" {s['th1'].mean():>7.3f}+-{s['th1'].std():<6.3f} | {s['train'].mean():>10.3f}"
              f" {s['test'].mean():>10.3f} {s['test'].mean() - SIGMA2 - BIAS2:>+10.3f} |"
              f" {np.median(s['quad_test']):>10.3f}*")
    print("  * quadratic-model test MSE shown as the median (at N = 5 a 3-parameter fit to 5 points"
          " has a heavy tail)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    rng = np.random.default_rng(SEED)
    x, y = generate_data(1000, rng, quad_coef=QUAD)
    th = fit_linear(x, y)
    xs = np.linspace(-5, 5, 200)
    ax = axes[0]
    ax.scatter(x, y, s=6, alpha=0.35, color="tab:gray", label="observations (N = 1000)")
    ax.plot(xs, truth(xs), "k", lw=2, label="true curve 3 + 2x + 0.25x^2")
    ax.plot(xs, predict(th, xs), "--", color="tab:orange", lw=2,
            label=f"best-fit line {th[0]:.2f} + {th[1]:.2f}x")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("A line cannot bend: systematic error at the edges and centre")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    lin_mean = [stats[n]["test"].mean() for n in N_VALUES]
    lin_tr = [stats[n]["train"].mean() for n in N_VALUES]
    quad_med = [np.median(stats[n]["quad_test"]) for n in N_VALUES]
    ax.plot(N_VALUES, lin_mean, "o-", color="tab:orange", label="straight line: test MSE")
    ax.plot(N_VALUES, lin_tr, "s--", color="tab:orange", alpha=0.6, label="straight line: train MSE")
    ax.plot(N_VALUES, quad_med, "o-", color="tab:blue", label="quadratic model: test MSE (median)")
    ax.axhline(SIGMA2 + BIAS2, color="tab:orange", ls=":", label=f"sigma^2 + bias^2 = {SIGMA2 + BIAS2:.2f}")
    ax.axhline(SIGMA2, color="k", ls=":", label=f"sigma^2 = {SIGMA2:.2f}")
    ax.set_xscale("log")
    ax.set_ylim(0, 12)
    ax.set_xlabel("training observations N")
    ax.set_ylabel("mean squared error")
    ax.set_title("More data removes variance, not bias")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT / "exp_e_model_mismatch.png", dpi=150)
    plt.close(fig)
    print(f"\n  saved {OUT / 'exp_e_model_mismatch.png'}")


if __name__ == "__main__":
    main()
