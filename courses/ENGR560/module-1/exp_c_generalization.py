"""
Experiment C: generalization.

Part 1 (the fixed course experiment, seed 560):
    training MSE, test MSE on 500 new observations, and the test MSE of a baseline
    that always predicts the mean of the training targets.

Part 2 (what happens on average): repeat the whole experiment with many seeds
    (new training set, new test set each time) and average. The irreducible noise
    level is sigma^2 = 2.25; individual runs scatter around their expectation and
    can land below 2.25, but the average test MSE sits slightly ABOVE 2.25 because
    the 30-point fit is never exactly right.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learning_core import (NOISE_STD, exact_excess_risk, fit_linear, generate_data, mse, predict,
                           starter_datasets)

OUT = Path(__file__).parent / "outputs"
N_REPEATS = 5000


def one_experiment(x_train, y_train, x_test, y_test):
    theta = fit_linear(x_train, y_train)
    baseline = np.mean(y_train)
    return {
        "theta": theta,
        "train": mse(y_train, predict(theta, x_train)),
        "test": mse(y_test, predict(theta, x_test)),
        "baseline": mse(y_test, np.full_like(y_test, baseline)),
        "excess": exact_excess_risk(theta),
    }


def main():
    OUT.mkdir(exist_ok=True)
    sigma2 = NOISE_STD**2

    r = one_experiment(*starter_datasets())
    print("Experiment C: generalization (seed 560, N_train = 30, N_test = 500)")
    print(f"  learned theta              = [{r['theta'][0]:.4f}, {r['theta'][1]:.4f}]")
    print(f"  training MSE               = {r['train']:.4f}")
    print(f"  test MSE                   = {r['test']:.4f}")
    print(f"  baseline test MSE (mean y) = {r['baseline']:.4f}")
    print(f"  irreducible noise sigma^2  = {sigma2:.4f}")
    print(f"  exact excess risk of fit   = {r['excess']:.4f}   (E[test MSE] for this theta = "
          f"{sigma2 + r['excess']:.4f})")

    # Repeated experiments: independent train/test sets each time.
    rng = np.random.default_rng(56000)
    res = {k: np.empty(N_REPEATS) for k in ("train", "test", "baseline", "excess")}
    for i in range(N_REPEATS):
        x_tr, y_tr = generate_data(30, rng)
        x_te, y_te = generate_data(500, rng)
        ri = one_experiment(x_tr, y_tr, x_te, y_te)
        for k in res:
            res[k][i] = ri[k]

    print(f"\n  Averages over {N_REPEATS} independent repetitions (N_train = 30, N_test = 500):")
    print(f"  {'quantity':<22}{'mean':>9}{'std':>9}{'5th pct':>9}{'95th pct':>9}")
    for k, label in (("train", "training MSE"), ("test", "test MSE"),
                     ("baseline", "baseline test MSE"), ("excess", "excess risk")):
        v = res[k]
        print(f"  {label:<22}{v.mean():>9.4f}{v.std():>9.4f}"
              f"{np.percentile(v, 5):>9.4f}{np.percentile(v, 95):>9.4f}")
    print(f"  fraction of runs with test MSE < 2.25:     {np.mean(res['test'] < sigma2):.3f}")
    print(f"  fraction of runs with training MSE < 2.25: {np.mean(res['train'] < sigma2):.3f}")
    print(f"  theory: E[train MSE] = sigma^2 (N-2)/N = {sigma2 * 28 / 30:.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(1.0, 4.0, 70)
    ax.hist(res["train"], bins=bins, alpha=0.55, color="tab:blue", label="training MSE")
    ax.hist(res["test"], bins=bins, alpha=0.55, color="tab:orange", label="test MSE")
    ax.axvline(sigma2, color="k", lw=2, label="irreducible noise sigma^2 = 2.25")
    ax.axvline(res["train"].mean(), color="tab:blue", ls="--", lw=1.5,
               label=f"mean training MSE = {res['train'].mean():.3f}")
    ax.axvline(res["test"].mean(), color="tab:orange", ls="--", lw=1.5,
               label=f"mean test MSE = {res['test'].mean():.3f}")
    ax.axvline(r["test"], color="tab:red", ls=":", lw=2, label=f"seed 560 test MSE = {r['test']:.3f}")
    ax.set_xlabel("MSE")
    ax.set_ylabel("count")
    ax.set_title(f"Experiment C: {N_REPEATS} repeated experiments (N_train = 30)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_mse_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_c_mse_distribution.png'}")


if __name__ == "__main__":
    main()
