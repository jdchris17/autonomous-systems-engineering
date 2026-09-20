"""Impulse response library, plus a demo feeding one input through each system.

Each function returns h[n] for n = 0 .. length-1. The first-order decay is
infinite in principle, so it is truncated at `length` samples.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from convolution import convolve_discrete

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def pure_gain(K):
    """h[n] = K * delta[n]."""
    return np.array([K], dtype=float)


def delay(D):
    """h[n] = delta[n - D]."""
    h = np.zeros(D + 1)
    h[D] = 1.0
    return h


def moving_average_response(M):
    """h[n] = 1/M for 0 <= n < M, else 0."""
    return np.full(M, 1.0 / M)


def first_order_decay(a, length=30):
    """h[n] = a^n u[n], |a| < 1, truncated to `length` samples."""
    if abs(a) >= 1:
        raise ValueError("need |a| < 1")
    return a ** np.arange(length)


if __name__ == "__main__":
    n_in = 40
    x = np.zeros(n_in)
    x[5:15] = 1.0   # a 10-sample block
    x[25] = 2.0     # an isolated spike

    systems = [
        ("pure gain, K = 2", pure_gain(2.0)),
        ("delay, D = 6", delay(6)),
        ("moving average, M = 5", moving_average_response(5)),
        ("first-order decay, a = 0.8", first_order_decay(0.8, 30)),
    ]

    x_max = n_in + 30
    fig, axes = plt.subplots(5, 2, figsize=(14, 13), sharex="col")

    axes[0, 0].axis("off")
    axes[0, 1].stem(np.arange(n_in), x)
    axes[0, 1].set_title("input x[n]: 10-sample block + one spike")

    print(f"{'system':<28}{'sum h':>8}{'len(y)':>8}{'peak y':>9}   max|ours-np|")
    for row, (name, h) in enumerate(systems, start=1):
        y = convolve_discrete(x, h)
        err = np.max(np.abs(y - np.convolve(x, h)))
        print(f"{name:<28}{h.sum():>8.3f}{len(y):>8}{y.max():>9.3f}   {err:.1e}")

        axes[row, 0].stem(np.arange(len(h)), h)
        axes[row, 0].set_title(f"h[n]: {name}", fontsize=10)
        axes[row, 1].stem(np.arange(len(y)), y)
        axes[row, 1].set_title(f"y = x * h", fontsize=10)

    for ax in axes.flatten():
        ax.grid(alpha=0.3)
    axes[0, 1].set_xlim(0, x_max)
    axes[1, 1].set_xlim(0, x_max)
    axes[-1, 0].set_xlabel("n")
    axes[-1, 1].set_xlabel("n")
    fig.suptitle("Same input through four systems: read h, predict y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "impulse_response.png"), dpi=150)
    print("\nSaved impulse_response.png")

    print(
        """
What this means
---------------
The output is the input convolved with h, so h tells you what the system does
before you run anything:
  - Gain (a single spike of height 2 at n = 0): the output is the input
    scaled by 2. Nothing moves, nothing smears.
  - Delay (a single spike at n = 6): the output is the input, unchanged in
    shape, 6 samples later.
  - Moving average (a flat block of height 1/5): each output is an average of
    the last 5 inputs, so edges become ramps, the block gets wider, and the
    lone spike is spread into a low, wide bump.
  - First-order decay (a tail that shrinks by 0.8 each step): the system has
    memory that fades. Every input leaves an echo that dies out gradually, so
    edges rise and fall slowly, the block builds toward a higher level, and the
    spike is followed by a decaying trail.
Rule of thumb: where h has mass tells you WHEN the system responds (delay),
how spread out h is tells you how much it smooths, and the sum of h (2, 1, 1,
and 5 here) is how much a constant input gets scaled once things settle."""
    )
