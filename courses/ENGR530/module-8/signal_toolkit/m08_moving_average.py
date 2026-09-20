"""Engineering challenge: the 5-sample moving average
    y[n] = (1/5) * sum_{k=0}^{4} x[n - k]
Classify it analytically, verify numerically, then feed it four signals."""

import os

import matplotlib.pyplot as plt
import numpy as np

from signals import sinusoid
from systems import moving_average

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 100
DURATION = 2.0
WARMUP = 4  # first 4 outputs are the start-up transient (window not yet full)


def steady(y):
    return y[WARMUP:]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = np.arange(int(FS * DURATION))

    print("ANALYTIC CLASSIFICATION of y[n] = (1/5) sum_{k=0}^{4} x[n-k]")
    print("-" * 64)
    print("Linear:            YES. Scaling and adding inputs scales and adds outputs,")
    print("                   because the system is a fixed weighted sum of samples.")
    print("Time invariant:    YES. The weights (1/5) do not depend on n, so delaying")
    print("                   x by n0 delays y by n0.")
    print("Causal:            YES. Only x[n], x[n-1], ..., x[n-4] appear: no future samples.")
    print("Memoryless:        NO.  y[n] depends on 4 past inputs, not just x[n].")
    print("BIBO stable:       YES. |y[n]| <= (1/5)*5*max|x| = max|x|. Equivalently the")
    print("                   impulse response h[n] = 1/5 for n = 0..4 has sum|h| = 1 < inf.")
    print("Invertible:        NO.  x1[n] = 0 and x2[n] = cos(2*pi*n/5) give the same")
    print("                   output: any 5 consecutive samples of x2 sum to 0. Two")
    print("                   different inputs, one output, so x cannot be recovered.")

    print("\nNUMERIC SPOT CHECKS")
    print("-" * 64)
    x1 = rng.standard_normal(200)
    x2 = rng.standard_normal(200)
    a, b = 2.5, -1.3
    lin_err = np.max(np.abs(moving_average(a * x1 + b * x2) - (a * moving_average(x1) + b * moving_average(x2))))
    print(f"Linearity:        max |S(a x1 + b x2) - (a S(x1) + b S(x2))| = {lin_err:.2e}")

    n0 = 7
    x_shift = np.concatenate([np.zeros(n0), x1])
    ti_err = np.max(np.abs(moving_average(x_shift)[n0:] - moving_average(x1)))
    print(f"Time invariance:  max |S(x[n - {n0}]) - S(x)[n - {n0}]|          = {ti_err:.2e}")

    impulse = np.zeros(12)
    impulse[0] = 1
    print(f"Impulse response: {np.round(moving_average(impulse), 3).tolist()}")
    print(f"                  sum |h| = {np.sum(np.abs(moving_average(impulse))):.3f}  (finite -> BIBO stable)")

    x_zero = np.zeros(200)
    x_cos5 = np.cos(2 * np.pi * np.arange(200) / 5)
    diff = np.max(np.abs(steady(moving_average(x_zero)) - steady(moving_average(x_cos5))))
    print(f"Invertibility:    max |S(0) - S(cos(2 pi n/5))| after start-up  = {diff:.2e}  (same output)")

    # Four test signals.
    const = np.full(len(n), 5.0)
    _, sine_slow = sinusoid(1.0, 2.0, 0.0, DURATION, FS)
    _, sine_fast = sinusoid(1.0, 10.0, 0.0, DURATION, FS)
    noise = rng.standard_normal(len(n))
    signal_plus_noise = sine_slow + 0.5 * noise

    cases = [
        ("1. constant (5)", const),
        ("2. sinusoid (10 Hz)", sine_fast),
        ("3. random noise", noise),
        ("4. 2 Hz sinusoid + noise", signal_plus_noise),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (title, sig) in zip(axes.flatten(), cases):
        out = moving_average(sig)
        ax.plot(n / FS, sig, color="lightgray", linewidth=1.5, label="input x[n]")
        ax.plot(n / FS, out, color="steelblue", linewidth=2, label="output y[n]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("t (s)   (n / 100)")
    fig.suptitle("5-sample moving average applied to four inputs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "moving_average.png"), dpi=150)

    print("\nWHAT HAPPENED (steady state, after the 4-sample start-up)")
    print("-" * 64)
    print(f"constant 5:            output mean {steady(moving_average(const)).mean():.3f}  (input 5.000)")
    for name, sig in [("2 Hz sinusoid", sine_slow), ("10 Hz sinusoid", sine_fast)]:
        ratio = np.max(np.abs(steady(moving_average(sig)))) / np.max(np.abs(sig))
        print(f"{name + ':':<23}output/input peak amplitude = {ratio:.3f}")
    print(f"random noise:          output/input std          = {steady(moving_average(noise)).std() / noise.std():.3f}")
    print("Saved moving_average.png")

    print(
        """
What this means
---------------
Analytically this system is a well-behaved LTI filter: linear, time invariant,
causal, stable, with memory, and not invertible (it throws information away,
so you cannot undo it).
What we saw: the constant passes through unchanged, the slow sinusoid barely
changes, the fast sinusoid shrinks a lot, and the noise shrinks hard. The
pattern is that slowly varying content survives while rapidly varying content
is suppressed, and the signal-plus-noise case shows the practical payoff: the
smooth 2 Hz shape is recovered from the jagged input. Why this happens, and
how much each speed is suppressed, is a frequency-domain question we leave
for later."""
    )
