"""Engineering challenge: how a tiny per-bit error rate compounds across a
packet, modeled as X ~ Binomial(n, p) = number of bit errors."""

import matplotlib.pyplot as plt
import numpy as np

from m03_pmf_visualizer import binomial_pmf, binomial_pmf_table

N_BITS = 100
P_BIT_ERROR = 0.002
N_PACKETS = 1_000_000


def packet_probabilities(n, p):
    """P(X=0), P(X=1), P(X>=1), P(X>=2) for X ~ Binomial(n, p)."""
    p0 = binomial_pmf(0, n, p)
    p1 = binomial_pmf(1, n, p)
    p_at_least_1 = 1 - p0
    p_at_least_2 = 1 - p0 - p1
    return p0, p1, p_at_least_1, p_at_least_2


def simulate_packets(n, p, n_packets, rng):
    """Bit-error counts for n_packets independent packets."""
    return rng.binomial(n, p, size=n_packets)


def plot_packet_pmf(n, p, error_counts, path):
    pmf = binomial_pmf_table(n, p)
    max_k = 8
    ks = np.arange(max_k + 1)

    counts = np.bincount(error_counts, minlength=max_k + 1)[: max_k + 1]
    empirical = counts / len(error_counts)

    fig, ax = plt.subplots(figsize=(9, 6))
    width = 0.4
    ax.bar(ks - width / 2, pmf[: max_k + 1], width=width, color="firebrick", label="theoretical PMF")
    ax.bar(ks + width / 2, empirical, width=width, color="steelblue", label=f"empirical ({len(error_counts):,} packets)")
    ax.set_xlabel("X = number of bit errors in packet")
    ax.set_ylabel("P(X = k)")
    ax.set_title(f"Bit errors per packet:  Binomial(n={n}, p={p})")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    p0, p1, p_at_least_1, p_at_least_2 = packet_probabilities(N_BITS, P_BIT_ERROR)

    print(f"X = number of bit errors, X ~ Binomial(n={N_BITS}, p={P_BIT_ERROR})\n")
    print(f"P(X = 0)   = {p0:.6f}   (packet arrives with zero bit errors)")
    print(f"P(X = 1)   = {p1:.6f}")
    print(f"P(X >= 1)  = {p_at_least_1:.6f}   (packet contains at least one error)")
    print(f"P(X >= 2)  = {p_at_least_2:.6f}   (more than a single-bit error)")

    rng = np.random.default_rng(0)
    error_counts = simulate_packets(N_BITS, P_BIT_ERROR, N_PACKETS, rng)

    emp_p0 = np.mean(error_counts == 0)
    emp_p1 = np.mean(error_counts == 1)
    emp_at_least_1 = np.mean(error_counts >= 1)
    emp_at_least_2 = np.mean(error_counts >= 2)

    print(f"\nSimulated {N_PACKETS:,} packets:\n")
    print(f"{'quantity':<12}{'theory':>12}{'empirical':>12}{'abs diff':>12}")
    for label, theory, empirical in [
        ("P(X=0)", p0, emp_p0),
        ("P(X=1)", p1, emp_p1),
        ("P(X>=1)", p_at_least_1, emp_at_least_1),
        ("P(X>=2)", p_at_least_2, emp_at_least_2),
    ]:
        print(f"{label:<12}{theory:>12.6f}{empirical:>12.6f}{abs(theory - empirical):>12.6f}")

    print(
        "\nBit-level reliability: 1 - p = "
        f"{1 - P_BIT_ERROR:.4f} ({(1 - P_BIT_ERROR) * 100:.2f}% of individual bits are correct)."
    )
    print(
        f"Packet-level reliability: P(X=0) = (1-p)^n = {p0:.4f} ({p0 * 100:.2f}% of packets are "
        f"error-free) - {N_BITS} independent chances for a 0.2% failure to occur compound "
        f"multiplicatively, so roughly 1 in {round(1 / p_at_least_1)} packets contains an error "
        "even though each individual bit is correct 99.8% of the time."
    )

    plot_packet_pmf(N_BITS, P_BIT_ERROR, error_counts, "packet_reliability.png")
