"""Engineering challenge: smoothing noise with moving averages of length
M = 3, 5, 15, 51 and watching the trade-off."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module-8", "signal_toolkit"))
from convolution import convolve_discrete
from impulse_response import moving_average_response
from signals import sinusoid

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 100
DURATION = 10.0
NOISE_STD = 0.4
LENGTHS = [3, 5, 15, 51]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t, slow = sinusoid(1.0, 0.4, -np.pi / 2, DURATION, FS)        # slow sine
    pulse = np.exp(-0.5 * ((t - 7.0) / 0.1) ** 2)                  # brief genuine event at t = 7 s
    s = slow + pulse
    w = rng.normal(0.0, NOISE_STD, size=len(t))
    x = s + w
    N = len(t)

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True, sharey=True)
    print(f"s[n] = slow 0.4 Hz sine + brief pulse at t = 7 s;  w[n] ~ Normal(0, {NOISE_STD});  N = {N}\n")
    print(f"{'M':>4}{'noise std ratio':>17}{'1/sqrt(M)':>11}{'delay (samples)':>17}{'(M-1)/2':>9}{'pulse peak kept':>17}{'RMSE vs s':>11}{'RMSE, delay removed':>21}")
    for ax, M in zip(axes, LENGTHS):
        h = moving_average_response(M)
        y = convolve_discrete(x, h)[:N]
        y_w = convolve_discrete(w, h)[:N]
        y_s = convolve_discrete(s, h)[:N]
        y_p = convolve_discrete(pulse, h)[:N]

        noise_ratio = np.std(y_w[M - 1:]) / np.std(w)

        lags = np.arange(0, 60)
        errs = [np.mean((y_s[60:] - s[60 - L:N - L]) ** 2) for L in lags]
        delay = int(lags[np.argmin(errs)])

        pulse_kept = y_p.max() / pulse.max()
        d = (M - 1) // 2
        err_raw = rmse(y[60:], s[60:])
        err_aligned = rmse(y[60 + d:], s[60:N - d])
        print(f"{M:>4}{noise_ratio:>17.3f}{1 / np.sqrt(M):>11.3f}{delay:>17d}{d:>9d}{pulse_kept:>17.2f}{err_raw:>11.3f}{err_aligned:>21.3f}")

        ax.plot(t, x, color="lightgray", linewidth=1, label="x = s + w")
        ax.plot(t, s, "k--", linewidth=1.2, label="true s")
        ax.plot(t, y, color="steelblue", linewidth=2, label=f"moving average, M = {M}")
        ax.set_title(f"M = {M}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle("Noise smoothing with causal moving averages: less noise, more delay, less detail")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "noise_smoothing.png"), dpi=150)
    print("\nSaved noise_smoothing.png")

    print(
        """
What this means
---------------
Widening the window does three things at once, and you cannot pick just one:
  - Noise reduction: noise falls roughly like 1/sqrt(M). Going from M = 3 to
    51 cuts it from about 0.58 to 0.14 of its original size. The blue curve
    gets calmer.
  - Delay: the causal filter reports the average of the last M samples, which
    is centered (M-1)/2 samples in the past. At M = 51 the output is half a
    second behind the truth.
  - Loss of rapid signal changes: the brief pulse at t = 7 s is a real feature,
    not noise. It keeps ~100% of its height at M = 3, ~91% at M = 15, and only
    ~49% at M = 51, because the window becomes wider than the pulse.
The error columns show the cost. Against the true s, the raw error is best near
M = 15 and then jumps up at M = 51, because the delay alone is now a big error.
Even with the delay removed, the gain from M = 15 to 51 is tiny (0.100 -> 0.093)
while half the pulse is destroyed. Smoothing is not free: past some point you
are paying in real signal detail for almost no extra noise reduction, and the
right M depends on how fast the real signal can change. The frequency domain
will quantify exactly this."""
    )
