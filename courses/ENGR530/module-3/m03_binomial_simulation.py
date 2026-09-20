"""Empirical vs. theoretical: simulate Binomial(20, 0.3) at growing sample
sizes and watch the empirical frequencies converge to the PMF."""

import matplotlib.pyplot as plt
import numpy as np

from m03_pmf_visualizer import binomial_pmf_table

N_TRIALS = 20
P_SUCCESS = 0.3
SAMPLE_SIZES = [100, 1_000, 10_000, 100_000]


def empirical_frequencies(samples, n_trials):
    """Normalized count of each outcome k = 0, ..., n_trials in `samples`."""
    counts = np.bincount(samples, minlength=n_trials + 1)
    return counts / len(samples)


def plot_convergence(sample_sizes, n_trials, p, path, rng):
    theoretical = binomial_pmf_table(n_trials, p)
    ks = np.arange(n_trials + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, n_samples in zip(axes.flat, sample_sizes):
        samples = rng.binomial(n_trials, p, size=n_samples)
        empirical = empirical_frequencies(samples, n_trials)

        max_error = np.max(np.abs(empirical - theoretical))

        ax.bar(ks, empirical, color="steelblue", width=0.8, label="empirical")
        ax.plot(ks, theoretical, "o-", color="firebrick", markersize=3, linewidth=1, label="theoretical PMF")
        ax.set_title(f"n = {n_samples:,} samples  |  max |error| = {max_error:.4f}")
        ax.set_xlabel("k")
        ax.set_ylabel("P(X = k)")
        ax.legend(fontsize=8)

    fig.suptitle(f"Binomial({n_trials}, {p}): empirical frequency vs. theoretical PMF")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    theoretical = binomial_pmf_table(N_TRIALS, P_SUCCESS)

    print(f"Binomial(n={N_TRIALS}, p={P_SUCCESS})\n")
    for n_samples in SAMPLE_SIZES:
        samples = rng.binomial(N_TRIALS, P_SUCCESS, size=n_samples)
        empirical = empirical_frequencies(samples, N_TRIALS)
        max_error = np.max(np.abs(empirical - theoretical))
        mean_abs_error = np.mean(np.abs(empirical - theoretical))
        print(
            f"samples={n_samples:>7,}   "
            f"max |empirical - theoretical| = {max_error:.5f}   "
            f"mean |error| = {mean_abs_error:.5f}"
        )

    rng = np.random.default_rng(0)
    plot_convergence(SAMPLE_SIZES, N_TRIALS, P_SUCCESS, "binomial_convergence.png", rng)
