"""Bernoulli and Binomial PMFs, visualized across several (n, p) cases."""

from math import comb

import matplotlib.pyplot as plt


def bernoulli_pmf(k, p):
    """P(X = k) for X ~ Bernoulli(p), k in {0, 1}."""
    if k == 1:
        return p
    elif k == 0:
        return 1 - p
    else:
        return 0.0


def binomial_pmf(k, n, p):
    """P(X = k) for X ~ Binomial(n, p), via n independent Bernoulli(p) trials.

    C(n, k) counts the arrangements of k successes among n trials; each
    arrangement has probability p^k (1-p)^(n-k).
    """
    if k < 0 or k > n:
        return 0.0
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def binomial_pmf_table(n, p):
    """P(X = k) for every k = 0, ..., n."""
    return [binomial_pmf(k, n, p) for k in range(n + 1)]


CASES = [
    (10, 0.1),
    (10, 0.5),
    (10, 0.9),
    (100, 0.1),
]


def plot_cases(cases, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, (n, p) in zip(axes.flat, cases):
        ks = list(range(n + 1))
        pmf = binomial_pmf_table(n, p)
        mean = n * p
        std = (n * p * (1 - p)) ** 0.5

        ax.bar(ks, pmf, color="steelblue", width=0.8)
        ax.axvline(mean, color="firebrick", linestyle="--", linewidth=1, label=f"mean = {mean:.1f}")
        ax.set_title(f"Binomial(n={n}, p={p})  |  std = {std:.2f}")
        ax.set_xlabel("k")
        ax.set_ylabel("P(X = k)")
        ax.legend(fontsize=8)
        if n == 100:
            ax.set_xlim(mean - 4 * std, mean + 4 * std)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    for n, p in CASES:
        pmf = binomial_pmf_table(n, p)
        mean = n * p
        std = (n * p * (1 - p)) ** 0.5
        mode = max(range(n + 1), key=lambda k: pmf[k])
        print(f"Binomial(n={n:>3}, p={p}):  mean={mean:6.2f}  std={std:5.2f}  mode={mode:>3}  sum(pmf)={sum(pmf):.6f}")

    plot_cases(CASES, "pmf_visualizer.png")
