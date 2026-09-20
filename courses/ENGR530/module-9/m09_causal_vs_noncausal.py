"""Engineering challenge: causal vs. noncausal 3-point averaging.
    h1[n] = (1/3)(d[n] + d[n-1] + d[n-2])   causal (present and past)
    h2[n] = (1/3)(d[n+1] + d[n] + d[n-1])   noncausal (uses one future sample)
Also repeated with 11 points, where the difference is much larger."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module-8", "signal_toolkit"))
from convolution import convolve_discrete

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N = 200
STEP_AT = 100
NOISE_STD = 0.3


def convolve_aligned(x, h, start):
    """Convolve, where h[0] sits at time n = start (start < 0 means h has future taps).

    Full-convolution index m corresponds to time m + start, so the output at
    time n is full[n - start]. Returns len(x) samples aligned with x.
    """
    full = convolve_discrete(x, h)
    off = -start
    return full[off:off + len(x)]


def averagers(M):
    """(h1, start1), (h2, start2): M-point causal and centered averagers."""
    h = np.full(M, 1.0 / M)
    return (h, 0), (h, -((M - 1) // 2))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = np.arange(N)
    s = np.sin(2 * np.pi * n / 80) + 1.5 * (n >= STEP_AT)
    x = s + rng.normal(0, NOISE_STD, N)
    clean_step = (n >= STEP_AT).astype(float)

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1, 1])
    (h1, st1), (h2, st2) = averagers(3)
    ax = fig.add_subplot(gs[0, 0])
    ax.stem(np.arange(st1, st1 + 3), h1)
    ax.set_title("h1: causal, taps at n = 0, 1, 2")
    ax = fig.add_subplot(gs[0, 1])
    ax.stem(np.arange(st2, st2 + 3), h2)
    ax.set_title("h2: noncausal, taps at n = -1, 0, 1")
    for ax in fig.axes:
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("n")
        ax.grid(alpha=0.3)

    print("Edge = a clean step at n = 100 (its true 50% point is n = 99.5).\n")
    print(f"{'M':>3}{'filter':>14}{'edge 50% point':>16}{'edge shift':>12}{'RMSE vs s (noisy data)':>25}")
    for row, M in enumerate([3, 11], start=1):
        (h1, st1), (h2, st2) = averagers(M)
        y1 = convolve_aligned(x, h1, st1)
        y2 = convolve_aligned(x, h2, st2)
        interior = slice(M, N - M)   # skip boundary effects at the record ends

        for name, h, st, y in [("causal", h1, st1, y1), ("noncausal", h2, st2, y2)]:
            ys = convolve_aligned(clean_step, h, st)
            edge = np.interp(0.5, ys[90:120], n[90:120])
            print(f"{M:>3}{name:>14}{edge:>16.2f}{edge - 99.5:>12.2f}{rmse(y[interior], s[interior]):>25.3f}")

        ax = fig.add_subplot(gs[row, :])
        ax.plot(n, x, color="lightgray", linewidth=1, label="recorded x = s + w")
        ax.plot(n, s, "k--", linewidth=1.2, label="true s")
        ax.plot(n, y1, color="firebrick", linewidth=2, label=f"causal {M}-pt")
        ax.plot(n, y2, color="steelblue", linewidth=2, label=f"noncausal {M}-pt")
        ax.axvline(STEP_AT - 0.5, color="gray", linestyle=":")
        ax.set_xlim(60, 150)
        ax.set_title(f"{M}-point averaging on recorded data (zoom on the step)")
        ax.set_xlabel("n")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=4)
    fig.suptitle("Causal vs. noncausal averaging")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "causal_vs_noncausal.png"), dpi=150)
    print("\nSaved causal_vs_noncausal.png")

    print(
        """
What this means
---------------
Both filters average the same number of samples and smooth the noise equally
well. The difference is WHERE the window sits:
  - Causal: the window covers the present and past, so it is centered in the
    past. Every feature is reported late: the step's midpoint shows up (M-1)/2
    samples after it really happened (1 sample for M = 3, 5 for M = 11).
  - Noncausal: the window is centered on the current sample, so features stay
    exactly where they occurred (shift 0), and edges stay symmetric.
Why noncausal is better offline: when the whole recording already exists, the
'future' sample is just the next entry in the array, so using it costs
nothing. You get smoothing without delay, and lower error against the true
signal, because the estimate at time n is built from data on both sides of n.
Why we cannot always do it: a real-time system (control loop, live display)
has not seen sample n+1 yet. It must use a causal filter and accept the lag,
or wait, which is the same lag. The choice is dictated by whether the
future is available."""
    )
