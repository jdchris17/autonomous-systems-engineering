"""Validate our hand-written convolution sum against np.convolve."""

import os

import matplotlib.pyplot as plt
import numpy as np

from convolution import convolve_discrete

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Hand-checkable: x = [1,2,3], h = [1,1]
    #   y[0]=1, y[1]=1+2=3, y[2]=2+3=5, y[3]=3
    x_small, h_small = [1, 2, 3], [1, 1]
    y_small = convolve_discrete(x_small, h_small)
    print(f"x = {x_small}, h = {h_small}")
    print(f"  ours:        {y_small.tolist()}")
    print(f"  by hand:     [1.0, 3.0, 5.0, 3.0]")
    print(f"  np.convolve: {np.convolve(x_small, h_small).astype(float).tolist()}\n")

    print(f"{'Nx':>4}{'Nh':>4}{'len(y)':>8}{'Nx+Nh-1':>9}{'max |ours - np|':>18}")
    worst = 0.0
    for nx, nh in [(1, 1), (5, 3), (3, 5), (20, 7), (50, 50), (100, 4)]:
        x = rng.standard_normal(nx)
        h = rng.standard_normal(nh)
        ours = convolve_discrete(x, h)
        ref = np.convolve(x, h)
        err = np.max(np.abs(ours - ref))
        worst = max(worst, err)
        print(f"{nx:>4}{nh:>4}{len(ours):>8}{nx + nh - 1:>9}{err:>18.2e}")

    x = rng.standard_normal(12)
    h = rng.standard_normal(5)
    comm = np.max(np.abs(convolve_discrete(x, h) - convolve_discrete(h, x)))
    delta = np.array([1.0])
    ident = np.max(np.abs(convolve_discrete(x, delta) - x))
    delay3 = np.concatenate([np.zeros(3), [1.0]])
    shifted = convolve_discrete(x, delay3)
    delay_err = np.max(np.abs(shifted[3:3 + len(x)] - x))
    print(f"\nCommutativity  x*h vs h*x:              {comm:.2e}")
    print(f"Identity       x*delta[n] vs x:         {ident:.2e}")
    print(f"Delay          x*delta[n-3] is x moved: {delay_err:.2e}")

    # Visual: rectangle * rectangle = trapezoid
    xr = np.ones(8)
    hr = np.ones(5)
    yr = convolve_discrete(xr, hr)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, sig, title in [
        (axes[0], xr, f"x[n]: 8-sample block"),
        (axes[1], hr, f"h[n]: 5-sample block"),
        (axes[2], yr, f"y = x * h: {len(yr)} samples (8 + 5 - 1)"),
    ]:
        ax.stem(np.arange(len(sig)), sig)
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "manual_convolution.png"), dpi=150)
    print("\nSaved manual_convolution.png")

    print(
        f"""
What this means
---------------
Our double loop is the convolution sum, and it agrees with np.convolve to
floating-point precision (worst error {worst:.1e}) for every length mix,
including the output length Nx + Nh - 1. Each output sample y[n] is a weighted
sum of input samples, with the weights being h flipped and slid to position n.
The picture shows it: a block convolved with a block ramps up while the
windows overlap more, plateaus when one block sits fully inside the other, and
ramps down as they separate. The identity and delay checks preview the big
idea for the next build: h fully describes an LTI system, and convolving with
h is how the system acts on any input."""
    )
