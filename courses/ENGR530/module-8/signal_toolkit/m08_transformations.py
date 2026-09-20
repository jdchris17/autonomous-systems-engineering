"""Signal transformations built from the independent variable: y(t) = x(g(t)).

x is defined as a function of t, so each transformation just changes the
argument we feed it. No array shifting or reversing.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PEAK_TIME = 2.0


def x(t):
    """Slow linear rise 0 -> 2 over [0, 2], fast fall 2 -> 0 over [2, 3], else 0.

    Non-symmetric on purpose: the sharp fall on the right makes shifts,
    reversal, and scaling all visually unambiguous.
    """
    t = np.asarray(t, dtype=float)
    rise = (t >= 0) & (t <= 2)
    fall = (t > 2) & (t <= 3)
    return np.where(rise, t, 0.0) + np.where(fall, 2.0 * (3.0 - t), 0.0)


# (label, g(t), g^-1(u)): g^-1 gives the time at which the original feature
# at u = PEAK_TIME lands, since y(t) = x(g(t)) has its peak where g(t) = 2.
TRANSFORMS = [
    ("x(t - 2)", lambda t: t - 2, lambda u: u + 2, "delay: slides RIGHT by 2"),
    ("x(t + 2)", lambda t: t + 2, lambda u: u - 2, "advance: slides LEFT by 2"),
    ("x(-t)", lambda t: -t, lambda u: -u, "time reversal: mirrors about t = 0"),
    ("x(2t)", lambda t: 2 * t, lambda u: u / 2, "compression: twice as fast, half as wide"),
    ("x(t/2)", lambda t: t / 2, lambda u: 2 * u, "stretch: half as fast, twice as wide"),
]


if __name__ == "__main__":
    t = np.linspace(-8, 8, 3201)
    original = x(t)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    axes[0].plot(t, original, color="black", linewidth=2)
    axes[0].plot(PEAK_TIME, 2.0, "o", color="firebrick")
    axes[0].set_title("original x(t)  (peak at t = 2)")

    print(f"{'transform':<10}{'expected peak t':>17}{'measured peak t':>17}   meaning")
    for ax, (label, g, g_inv, meaning) in zip(axes[1:], TRANSFORMS):
        y = x(g(t))
        expected_peak = g_inv(PEAK_TIME)
        measured_peak = t[np.argmax(y)]

        ax.plot(t, original, color="lightgray", linewidth=1.5, label="original x(t)")
        ax.plot(t, y, color="steelblue", linewidth=2, label=label)
        ax.plot(expected_peak, 2.0, "o", color="firebrick")
        ax.set_title(f"{label}\n{meaning}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)

        print(f"{label:<10}{expected_peak:>17.2f}{measured_peak:>17.2f}   {meaning}")

    for ax in axes:
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("t")
        ax.set_ylim(-0.3, 2.6)
        ax.set_xlim(-8, 8)
    for ax in axes[::3]:
        ax.set_ylabel("amplitude")

    fig.suptitle("Transformations as x(g(t)): red dot tracks the original peak (t = 2)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "transformations.png"), dpi=150)
    print("\nSaved transformations.png")

    print(
        """
What this means
---------------
Every transformation is a change to the INPUT of x, not to its output values.
To find where a feature lands, solve g(t) = (where it was): the original peak
sits at u = 2, so y(t) = x(g(t)) peaks where g(t) = 2.
  - x(t - 2): g(t) = 2 -> t = 4. Subtracting inside DELAYS the signal (moves it
    right). It feels backwards, but t must grow by 2 before x sees the argument
    it used to see.
  - x(t + 2): the opposite, peak at t = 0. It happens EARLIER.
  - x(-t): peak at t = -2. The sharp fall that was on the right is now on the
    left: the signal plays backwards.
  - x(2t): peak at t = 1. The argument runs twice as fast, so the whole shape
    is squeezed to half the width. Multiplying t by a > 1 compresses.
  - x(t/2): peak at t = 4. The argument runs half as fast, so the shape is
    stretched to twice the width.
Rule of thumb: inside the parentheses, the effect on the plot is the inverse
of what the arithmetic suggests. That is why we solve g(t) = u instead of
guessing."""
    )
