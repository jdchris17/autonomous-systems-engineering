"""The Module 8 moving average, revisited as convolution y = x * h."""

import os

import matplotlib.pyplot as plt
import numpy as np

from convolution import convolve_discrete
from signals import sinusoid
from systems import moving_average

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 100
N = 100


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = np.arange(N)

    # Identify h: the response of the Module 8 system to a unit impulse.
    impulse = np.zeros(N)
    impulse[0] = 1.0
    h_measured = moving_average(impulse)[:8]
    h = np.full(5, 1 / 5)
    print("Impulse response of the Module 8 system (first 8 samples):")
    print(f"  {np.round(h_measured, 3).tolist()}")
    print(f"  = (1/5) * {{1, 1, 1, 1, 1}} = {h.tolist()} followed by zeros\n")

    step = np.ones(N)
    step[:10] = 0.0
    _, sine = sinusoid(1.0, 5.0, 0.0, N / FS, FS)
    noise = rng.standard_normal(N)
    signals = [
        ("constant (3)", np.full(N, 3.0)),
        ("impulse", impulse),
        ("step (starts at n = 10)", step),
        ("sinusoid (5 Hz)", sine),
        ("random noise", noise),
        ("signal + noise", sine + 0.5 * noise),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    print(f"{'input':<26}{'len(y)':>8}{'max |conv - Module 8|':>24}")
    for ax, (name, x) in zip(axes.flatten(), signals):
        y_conv = convolve_discrete(x, h)          # length N + 5 - 1
        y_sys = moving_average(x)                 # length N (Module 8 system)
        err = np.max(np.abs(y_conv[:N] - y_sys))
        print(f"{name:<26}{len(y_conv):>8}{err:>24.2e}")

        ax.plot(n, x, color="lightgray", linewidth=1.5, label="input x[n]")
        ax.plot(np.arange(len(y_conv)), y_conv, color="steelblue", linewidth=2, label="y = x * h")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("n")
    fig.suptitle("Moving average as convolution with h = (1/5){1,1,1,1,1}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "moving_average_convolution.png"), dpi=150)
    print("\nSaved moving_average_convolution.png")

    print(
        """
What this means
---------------
The Module 8 "smoothing box" is fully described by five numbers. Convolving
any input with h = (1/5){1,1,1,1,1} reproduces the old moving_average output
exactly (differences ~1e-16). The only extra thing convolution returns is the
4-sample tail after the input ends, where the window drains out.
Reading the results through h:
  - Impulse in, h out: the impulse response is literally the system's output
    for a single spike, which is how h was identified.
  - Step: the output climbs in five equal steps of 1/5 as the window fills,
    then sits at 1. The window is 5 wide, so a sudden change takes 5 samples
    to fully register.
  - Constant: passes through unchanged once the window is full (h sums to 1).
  - Noise / signal + noise: each output averages 5 inputs, so rapid random
    jumps partly cancel while slower structure survives.
  - Sinusoid: still a sinusoid at the same frequency, just smaller and slightly
    delayed. Filtering rescales and shifts it, it does not change its shape.
What was a black box in Module 8 is now a single operation: slide h across
the input and take weighted sums."""
    )
