"""Time-frequency scaling: rectangular pulses of width T = 0.1, 0.5, 1, 5 and
how the main lobe of their spectra narrows as T grows."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module-8", "signal_toolkit"))
from m11_numerical_transform import rect_pulse
from transforms import fourier_transform

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
WIDTHS = [0.1, 0.5, 1.0, 5.0]
COLORS = ["firebrick", "darkorange", "steelblue", "darkgreen"]
OMEGA = np.linspace(0.0, 80.0, 1601)            # rad/s, positive side (X is real and even)
SAMPLES_PER_PULSE = 1000
SQRT_HALF_SINC_ARG = 0.44295                    # sinc(x) = 1/sqrt(2) at x = 0.44295


def spectrum(width):
    """Numerical X(w) of a unit-height pulse of the given width, on a grid with 1000 samples per pulse."""
    dt = width / SAMPLES_PER_PULSE
    t = np.arange(-2 * width, 2 * width + dt / 2, dt)
    return t, rect_pulse(t, width), fourier_transform(t, rect_pulse(t, width), OMEGA)


def first_null(omega, X):
    """First zero crossing of the (real) spectrum for w > 0, by linear interpolation."""
    x = X.real
    i = np.argmax((x[:-1] > 0) & (x[1:] <= 0))
    return omega[i] + (omega[i + 1] - omega[i]) * x[i] / (x[i] - x[i + 1])


def half_power_freq(omega, X):
    """Frequency where |X| falls to |X(0)|/sqrt(2), by linear interpolation."""
    m = np.abs(X)
    level = m[0] / np.sqrt(2)
    i = np.argmax(m < level)
    return omega[i - 1] + (omega[i] - omega[i - 1]) * (m[i - 1] - level) / (m[i - 1] - m[i])


if __name__ == "__main__":
    results = []
    data = {}
    for T in WIDTHS:
        t, x, X = spectrum(T)
        data[T] = (t, x, X)
        null = first_null(OMEGA, X)
        half = half_power_freq(OMEGA, X)
        results.append((T, null, half, abs(X[0])))

    print("Main-lobe width of the spectrum of a unit-height rectangular pulse of width T")
    print(f"{'T (s)':>7}{'X(0)':>8}{'first null (rad/s)':>20}{'expected 2pi/T':>16}{'null-to-null width':>20}{'half-power width':>18}{'expected 5.566/T':>18}{'null width x T':>16}")
    for T, null, half, x0 in results:
        print(f"{T:>7g}{x0:>8.4f}{null:>20.4f}{2 * np.pi / T:>16.4f}{2 * null:>20.4f}{2 * half:>18.4f}{2 * 2 * np.pi * SQRT_HALF_SINC_ARG / T:>18.4f}{2 * null * T:>16.3f}")
    print("\n(width x T is constant, about 4 pi = 12.566: doubling T halves the main lobe)")

    # Figure 1
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    for (T, _, _, _), c in zip(results, COLORS):
        t, x, _ = data[T]
        ax.plot(t, x, color=c, linewidth=2, label=f"T = {T:g} s")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-0.1, 1.2)
    ax.set_title("Time domain: pulses of increasing width")
    ax.set_xlabel("t (s)")
    ax.legend(loc="upper right")

    for ax, xmax, title in [(axes[0, 1], 80, "Spectrum |X(w)| / T, full range"),
                            (axes[1, 0], 15, "Same spectra, zoom on the low frequencies")]:
        for (T, null, _, _), c in zip(results, COLORS):
            X = data[T][2]
            ax.plot(OMEGA, np.abs(X) / T, color=c, linewidth=2, label=f"T = {T:g} s  (first null {null:.2f})")
            ax.plot([null], [0], "v", color=c)
        ax.set_xlim(0, xmax)
        ax.set_title(title)
        ax.set_xlabel("w (rad/s)")
        ax.set_ylabel("|X(w)| / T   (peak normalized to 1)")
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    for (T, _, _, _), c in zip(results, COLORS):
        X = data[T][2]
        ax.plot(OMEGA * T / (2 * np.pi), np.abs(X) / T, color=c, linewidth=2, label=f"T = {T:g} s")
    f_axis = np.linspace(0, 8, 800)
    ax.plot(f_axis, np.abs(np.sinc(f_axis)), "k--", linewidth=1, label="|sinc(w T / 2 pi)|")
    ax.set_xlim(0, 8)
    ax.set_title("Rescaled axis w T / 2 pi: all four collapse onto one sinc")
    ax.set_xlabel("w T / (2 pi)")
    ax.legend(fontsize=8)
    for ax in axes.flatten():
        ax.grid(alpha=0.3)
    fig.suptitle("Time-frequency scaling of a rectangular pulse")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "time_frequency_scaling.png"), dpi=150)

    # Figure 2: measured widths vs T
    Ts = np.array([r[0] for r in results])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(Ts, [2 * r[1] for r in results], "o-", label="measured null-to-null width", markersize=8)
    ax.loglog(Ts, [2 * r[2] for r in results], "s-", label="measured half-power width", markersize=7)
    tt = np.logspace(np.log10(Ts.min()), np.log10(Ts.max()), 50)
    ax.loglog(tt, 4 * np.pi / tt, "k--", linewidth=1, label="4 pi / T  (slope -1)")
    ax.set_xlabel("pulse width T (s)")
    ax.set_ylabel("main-lobe width (rad/s)")
    ax.set_title("Wider pulse -> proportionally narrower spectrum")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "time_frequency_scaling_widths.png"), dpi=150)
    print("Saved time_frequency_scaling.png and time_frequency_scaling_widths.png")

    print(
        """
What this means
---------------
Stretch the pulse in time and its spectrum shrinks by exactly the same factor:
the first null sits at w = 2 pi / T, so the main-lobe width is 4 pi / T. Going
from T = 0.1 s to 5 s (50x longer) narrows the lobe from about 126 rad/s to
about 2.5 rad/s (50x narrower). Plotted against w T / 2 pi, all four spectra
are the same sinc, so a pulse is just one shape, stretched.
  - T up  -> spectrum narrows: a long, slowly changing signal is concentrated
    near zero frequency.
  - T down -> spectrum broadens: a short signal contains a wide range of
    frequencies. A very short pulse (T -> 0) tends toward an impulse, whose
    spectrum is flat.
The product (duration) x (bandwidth) stays fixed at about 4 pi. You cannot make
a signal both short in time and narrow in frequency; that trade-off is the
reason a fast sensor needs wide bandwidth and why brief events need fast,
wide-band electronics to be captured faithfully."""
    )
