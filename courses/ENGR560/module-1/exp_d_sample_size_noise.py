"""
Experiment D: sample size and measurement noise.

Grid: N_train in {5, 20, 100, 1000}  x  sigma in {0.5, 1.5, 4}  -> 12 configurations.
For every configuration and every trial, a fresh training set AND a fresh 500-point
test set are generated with that configuration's noise level, so each noise setting
is scored on its own independently generated test observations.

Outputs
  1. Single-run table (trial seed 560) - what one experiment shows.
  2. Repeated-trial table (N_TRIALS seeds) - mean and spread of the fitted parameters,
     training MSE, test MSE, and the exact excess risk E_x[(f_hat - f_true)^2].
  3. outputs/exp_d_results.csv with every trial, and summary figures.

Reference values (large-N theory for x ~ U(-5,5), Var(x) = 25/3):
  sd(theta1) ~= sigma / sqrt(N Var(x)),  sd(theta0) ~= sigma / sqrt(N)
  E[train MSE] = sigma^2 (N - 2) / N,     E[test MSE] ~= sigma^2 (1 + 2/N)
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learning_core import (SEED, TRUE_INTERCEPT, TRUE_SLOPE, X_VAR, exact_excess_risk, fit_linear,
                           generate_data, mse, predict)

OUT = Path(__file__).parent / "outputs"
N_VALUES = (5, 20, 100, 1000)
SIGMAS = (0.5, 1.5, 4.0)
N_TEST = 500
N_TRIALS = 1000


def run_trial(n, sigma, trial_seed, i_n, i_s):
    """One full experiment. The seed mixes trial, N and sigma so every cell is independent."""
    rng = np.random.default_rng([trial_seed, i_n, i_s])
    x_tr, y_tr = generate_data(n, rng, noise_std=sigma)
    x_te, y_te = generate_data(N_TEST, rng, noise_std=sigma)
    theta = fit_linear(x_tr, y_tr)
    return {
        "theta0": theta[0],
        "theta1": theta[1],
        "train_mse": mse(y_tr, predict(theta, x_tr)),
        "test_mse": mse(y_te, predict(theta, x_te)),
        "baseline_mse": mse(y_te, np.full_like(y_te, y_tr.mean())),
        "excess": exact_excess_risk(theta),
    }


def single_run_table():
    print(f"D.1  One experiment per configuration (trial seed {SEED})")
    hdr = (f"  {'N':>5} {'sigma':>5} {'th0':>8} {'th1':>8} {'train MSE':>10} {'test MSE':>9}"
           f" {'sigma^2':>8} {'test/s^2':>9} {'baseline':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i_s, s in enumerate(SIGMAS):
        for i_n, n in enumerate(N_VALUES):
            r = run_trial(n, s, SEED, i_n, i_s)
            print(f"  {n:>5} {s:>5.1f} {r['theta0']:>8.3f} {r['theta1']:>8.3f} {r['train_mse']:>10.3f}"
                  f" {r['test_mse']:>9.3f} {s**2:>8.2f} {r['test_mse'] / s**2:>9.3f}"
                  f" {r['baseline_mse']:>9.3f}")
    print()


def repeated_trials():
    rows = []
    summary = {}
    for i_s, s in enumerate(SIGMAS):
        for i_n, n in enumerate(N_VALUES):
            trials = [run_trial(n, s, t, i_n, i_s) for t in range(N_TRIALS)]
            arr = {k: np.array([tr[k] for tr in trials]) for k in trials[0]}
            summary[(n, s)] = arr
            for t, tr in enumerate(trials):
                rows.append({"N": n, "sigma": s, "trial": t, **tr})

    print(f"D.2  Repeated experiments: {N_TRIALS} independent trials per configuration")
    print("     mean +- std across trials; 'th' columns are large-N theory for the std")
    hdr = (f"  {'N':>5} {'sigma':>5} | {'theta0':>15} {'th sd':>6} | {'theta1':>15} {'th sd':>6} |"
           f" {'train MSE':>15} | {'test MSE':>15} {'median':>7} | {'test/s^2':>8} {'excess':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in SIGMAS:
        for n in N_VALUES:
            a = summary[(n, s)]
            sd0 = s / np.sqrt(n)
            sd1 = s / np.sqrt(n * X_VAR)
            print(f"  {n:>5} {s:>5.1f} | {a['theta0'].mean():>7.3f}+-{a['theta0'].std():<6.3f}{sd0:>7.3f} |"
                  f" {a['theta1'].mean():>7.3f}+-{a['theta1'].std():<6.3f}{sd1:>7.3f} |"
                  f" {a['train_mse'].mean():>7.3f}+-{a['train_mse'].std():<6.3f} |"
                  f" {a['test_mse'].mean():>7.3f}+-{a['test_mse'].std():<6.3f} {np.median(a['test_mse']):>7.3f} |"
                  f" {a['test_mse'].mean() / s**2:>8.3f} {a['excess'].mean():>8.4f}")
        print()

    with open(OUT / "exp_d_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT / 'exp_d_results.csv'} ({len(rows)} rows)")
    return summary


def plot_errors(summary):
    colors = {0.5: "tab:green", 1.5: "tab:blue", 4.0: "tab:red"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) absolute MSEs: each noise level has its own floor sigma^2
    ax = axes[0]
    for s in SIGMAS:
        tr = [summary[(n, s)]["train_mse"].mean() for n in N_VALUES]
        te = [summary[(n, s)]["test_mse"].mean() for n in N_VALUES]
        ax.plot(N_VALUES, te, "o-", color=colors[s], label=f"test, sigma={s}")
        ax.plot(N_VALUES, tr, "s--", color=colors[s], alpha=0.7, label=f"train, sigma={s}")
        ax.axhline(s**2, color=colors[s], lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 400)   # headroom so the legend clears the sigma = 4 curves
    ax.set_xlabel("training observations N")
    ax.set_ylabel("mean MSE")
    ax.set_title("(a) train vs test MSE (dotted = sigma^2)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")

    # (b) normalized by sigma^2: all noise levels collapse onto one curve
    ax = axes[1]
    nn = np.logspace(np.log10(4), 3.1, 100)
    for s in SIGMAS:
        tr = [summary[(n, s)]["train_mse"].mean() / s**2 for n in N_VALUES]
        te = [summary[(n, s)]["test_mse"].mean() / s**2 for n in N_VALUES]
        ax.plot(N_VALUES, te, "o-", color=colors[s], label=f"test / s^2, sigma={s}")
        ax.plot(N_VALUES, tr, "s--", color=colors[s], alpha=0.7, label=f"train / s^2, sigma={s}")
    ax.plot(nn, (nn - 2) / nn, "k:", lw=1, label="(N-2)/N")
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 2.6)
    ax.set_xlabel("training observations N")
    ax.set_ylabel("MSE / sigma^2")
    ax.set_title("(b) normalized: noise level drops out")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # (c) excess risk = how well the relationship itself was learned
    ax = axes[2]
    for s in SIGMAS:
        ex = [summary[(n, s)]["excess"].mean() for n in N_VALUES]
        ax.plot(N_VALUES, ex, "o-", color=colors[s], label=f"sigma={s}")
        ax.plot(nn, 2 * s**2 / nn, ":", color=colors[s], lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training observations N")
    ax.set_ylabel("E[(f_hat(x) - f_true(x))^2]")
    ax.set_title("(c) excess risk (dotted = 2 sigma^2 / N)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Experiment D: effect of sample size and measurement noise", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_errors.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_d_errors.png'}")


def plot_parameter_spread(summary):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey="row")
    for j, s in enumerate(SIGMAS):
        for i, (key, true, label) in enumerate((("theta1", TRUE_SLOPE, "slope theta1"),
                                                ("theta0", TRUE_INTERCEPT, "intercept theta0"))):
            ax = axes[i, j]
            data = [summary[(n, s)][key] for n in N_VALUES]
            ax.boxplot(data, tick_labels=[str(n) for n in N_VALUES], whis=(2.5, 97.5),
                       showfliers=False)
            ax.axhline(true, color="tab:red", lw=1.2, label=f"true {label}")
            ax.set_title(f"{label}, sigma = {s}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("training observations N")
            if j == 0:
                ax.set_ylabel(label)
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Experiment D: spread of fitted parameters over {N_TRIALS} trials "
                 "(box = IQR, whiskers = 2.5-97.5%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_parameter_spread.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_d_parameter_spread.png'}")


def plot_example_fits():
    """One trial per corner of the grid, to see what the numbers mean."""
    xs = np.linspace(-5, 5, 100)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, (n, s) in zip(axes.flat, ((5, 0.5), (1000, 0.5), (5, 4.0), (1000, 4.0))):
        rng = np.random.default_rng([SEED, n, int(s * 10)])
        x, y = generate_data(n, rng, noise_std=s)
        theta = fit_linear(x, y)
        ax.scatter(x, y, s=8 if n > 100 else 30, alpha=0.4 if n > 100 else 0.9, color="tab:gray")
        ax.plot(xs, TRUE_INTERCEPT + TRUE_SLOPE * xs, "k", lw=2, label="true")
        ax.plot(xs, predict(theta, xs), "--", color="tab:orange", lw=2,
                label=f"fit: {theta[0]:.2f} + {theta[1]:.2f}x")
        ax.set_title(f"N = {n}, sigma = {s}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("Experiment D: noisy points vs learned line at the corners of the grid")
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_example_fits.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_d_example_fits.png'}")


def main():
    OUT.mkdir(exist_ok=True)
    single_run_table()
    summary = repeated_trials()
    plot_errors(summary)
    plot_parameter_spread(summary)
    plot_example_fits()


if __name__ == "__main__":
    main()
