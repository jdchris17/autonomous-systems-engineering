"""PMF and CDF of a Binomial random variable, plotted separately, with the
CDF used to answer verbal probability questions."""

import matplotlib.pyplot as plt
import numpy as np

from m03_pmf_visualizer import binomial_pmf_table

N_TRIALS = 20
P_SUCCESS = 0.3


def cdf_by_direct_sum(pmf):
    """F_X(k) = sum_{j=0}^{k} p_X(j), computed literally term by term."""
    return [sum(pmf[: k + 1]) for k in range(len(pmf))]


def cdf_by_cumsum(pmf):
    """Same definition, via numpy's running total."""
    return np.cumsum(pmf)


def p_le(cdf, k):
    """P(X <= k)."""
    return cdf[k]


def p_gt(cdf, k):
    """P(X > k) = 1 - P(X <= k)."""
    return 1 - cdf[k]


def p_between(cdf, a, b):
    """P(a <= X <= b) = F_X(b) - F_X(a - 1)."""
    return cdf[b] - (cdf[a - 1] if a > 0 else 0)


def plot_pmf_and_cdf(pmf, cdf, n, p, path):
    ks = np.arange(len(pmf))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(ks, pmf, color="steelblue", width=0.8)
    ax1.set_title(f"PMF: P(X = k)  |  Binomial({n}, {p})")
    ax1.set_xlabel("k")
    ax1.set_ylabel("P(X = k)")

    ax2.step(ks, cdf, where="post", color="darkorange", linewidth=2)
    ax2.plot(ks, cdf, "o", color="darkorange", markersize=3)
    ax2.set_title(f"CDF: F_X(k) = P(X <= k)")
    ax2.set_xlabel("k")
    ax2.set_ylabel("F_X(k)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    pmf = binomial_pmf_table(N_TRIALS, P_SUCCESS)

    cdf_direct = cdf_by_direct_sum(pmf)
    cdf_numpy = cdf_by_cumsum(pmf)
    max_diff = np.max(np.abs(np.array(cdf_direct) - cdf_numpy))
    print(f"max |direct sum - cumsum| = {max_diff:.2e}  (should be ~0, both compute F_X(k) = sum_{{j=0}}^k p_X(j))")

    cdf = cdf_numpy

    print(f"\nBinomial(n={N_TRIALS}, p={P_SUCCESS})")
    print(f"P(X <= 5)        = {p_le(cdf, 5):.4f}")
    print(f"P(X > 5)         = {p_gt(cdf, 5):.4f}  (= 1 - F_X(5) = 1 - {cdf[5]:.4f})")
    print(f"P(3 <= X <= 7)   = {p_between(cdf, 3, 7):.4f}  (= F_X(7) - F_X(2) = {cdf[7]:.4f} - {cdf[2]:.4f})")

    plot_pmf_and_cdf(pmf, cdf, N_TRIALS, P_SUCCESS, "pmf_and_cdf.png")
