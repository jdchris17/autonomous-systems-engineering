"""Fourier series synthesizer demo: a cosine first, then square, triangle,
sawtooth, and pulse-train waves built purely from harmonic coefficients."""

import os

import matplotlib.pyplot as plt
import numpy as np

from fourier import (
    fourier_synthesis,
    pulse_train_coeffs,
    sawtooth_wave_coeffs,
    square_wave_coeffs,
    triangle_wave_coeffs,
)
from signals import sinusoid, time_axis

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
F0 = 5.0
FS = 1000
DURATION = 1.0
PLOT_WINDOW = 0.4  # seconds shown (2 periods)
DUTY = 0.25

# Sample at interval midpoints so no sample lands exactly on a jump, where
# the series converges to the midpoint of the jump instead of either side.
T = time_axis(DURATION, FS) + 0.5 / FS


def ideal_square(t):
    return np.sign(np.sin(2 * np.pi * F0 * t))


def ideal_triangle(t):
    return (2 / np.pi) * np.arcsin(np.cos(2 * np.pi * F0 * t))


def ideal_sawtooth(t):
    return 2 * ((t * F0 + 0.5) % 1.0) - 1


def ideal_pulse(t):
    phase = (t * F0 + 0.5) % 1.0 - 0.5   # in [-0.5, 0.5)
    return (np.abs(phase) < DUTY / 2).astype(float)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    # 1. A simple cosine: 3 cos(2 pi 5 t + pi/4) = a_1 e^{j w0 t} + a_-1 e^{-j w0 t}
    A, PHI = 3.0, np.pi / 4
    coeffs = {1: (A / 2) * np.exp(1j * PHI), -1: (A / 2) * np.exp(-1j * PHI)}
    x_syn = fourier_synthesis(coeffs, F0, T)
    x_ref = A * np.cos(2 * np.pi * F0 * T + PHI)
    _, x_toolkit = sinusoid(A, F0, PHI, DURATION, FS)
    x_syn0 = fourier_synthesis(coeffs, F0, time_axis(DURATION, FS))
    print("COSINE TEST: x(t) = 3 cos(2 pi 5 t + pi/4)")
    print(f"  a_1  = {coeffs[1]:.4f}   (magnitude A/2 = {A / 2}, angle = phase = {np.angle(coeffs[1]):.4f} rad)")
    print(f"  a_-1 = {coeffs[-1]:.4f}   (complex conjugate of a_1)")
    print(f"  max |synthesis - 3 cos(...)|            = {np.max(np.abs(x_syn - x_ref)):.2e}")
    print(f"  max |synthesis - toolkit sinusoid()|    = {np.max(np.abs(x_syn0 - x_toolkit)):.2e}")

    non_symmetric_ok = False
    try:
        fourier_synthesis({1: 1.0}, F0, T)
    except ValueError:
        non_symmetric_ok = True
    print(f"  real=True rejects a lone a_1 (not conjugate-symmetric): {non_symmetric_ok}")

    # 2. More complex periodic signals.
    waves = [
        ("square wave", square_wave_coeffs, ideal_square),
        ("triangle wave", triangle_wave_coeffs, ideal_triangle),
        ("sawtooth wave", sawtooth_wave_coeffs, ideal_sawtooth),
        (f"pulse train (duty {DUTY:g})", lambda K: pulse_train_coeffs(K, DUTY), ideal_pulse),
    ]
    K_LIST = [1, 3, 9, 51]

    print("\nRMSE of the K-harmonic synthesis against the ideal waveform")
    print(f"{'wave':<22}" + "".join(f"{'K = ' + str(K):>11}" for K in K_LIST))
    for name, coeff_fn, ideal in waves:
        ref = ideal(T)
        errs = [rmse(fourier_synthesis(coeff_fn(K), F0, T), ref) for K in K_LIST]
        print(f"{name:<22}" + "".join(f"{e:>11.4f}" for e in errs))

    t_fine = (np.arange(20000) + 0.5) / 20000 / F0   # one period, 20 kHz-equivalent grid
    print("\nSquare wave overshoot near a jump (fine grid; the ideal wave peaks at 1):")
    for K in [9, 51, 201]:
        peak = fourier_synthesis(square_wave_coeffs(K), F0, t_fine).max()
        print(f"  K = {K:>3}: peak = {peak:.3f}")

    win = T <= PLOT_WINDOW
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), gridspec_kw={"width_ratios": [2.2, 1]})
    for row, (name, coeff_fn, ideal) in enumerate(waves):
        ax_t, ax_c = axes[row]
        ax_t.plot(T[win], ideal(T)[win], color="lightgray", linewidth=4, label="ideal")
        ax_t.plot(T[win], fourier_synthesis(coeff_fn(3), F0, T)[win], color="darkorange", linewidth=1.5, label="K = 3")
        ax_t.plot(T[win], fourier_synthesis(coeff_fn(25), F0, T)[win], color="steelblue", linewidth=1.5, label="K = 25")
        ax_t.set_title(name)
        ax_t.set_xlabel("t (s)")
        ax_t.grid(alpha=0.3)
        ax_t.legend(loc="upper right", fontsize=8, ncol=3)

        c = coeff_fn(25)
        ks = np.arange(0, 26)
        ax_c.stem(ks, [abs(c[k]) for k in ks])
        ax_c.set_title(f"|a_k|, k = 0..25")
        ax_c.set_xlabel("k")
        ax_c.grid(alpha=0.3)
    fig.suptitle("Fourier series synthesis: harmonic coefficients -> waveform")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fourier_synthesis.png"), dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(T[win], x_ref[win], color="lightgray", linewidth=4, label="3 cos(2$\\pi$5t + $\\pi$/4)")
    axes[0].plot(T[win], x_syn[win], "b--", linewidth=1.5, label="synthesized from a$_{\\pm1}$")
    axes[0].set_xlabel("t (s)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("Cosine from two complex coefficients")
    axes[0].grid(alpha=0.3)
    axes[1].stem([-1, 1], [abs(coeffs[-1]), abs(coeffs[1])])
    axes[1].set_xticks([-2, -1, 0, 1, 2])
    axes[1].set_title("|a_k|: two spikes of height A/2 = 1.5")
    axes[1].set_xlabel("k")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cosine_synthesis.png"), dpi=150)
    print("Saved fourier_synthesis.png and cosine_synthesis.png")

    print(
        """
What this means
---------------
A periodic signal is just a list of numbers: its harmonic coefficients a_k.
  - Cosine: two coefficients, at k = +1 and k = -1. Each has magnitude A/2
    (the amplitude is split between the two) and they are complex
    conjugates, which is exactly what makes the sum real. The angle of a_1
    is the phase, so phase lives in the coefficient's angle.
  - Real signals require a_{-k} = conj(a_k). The synthesizer checks this
    before returning the real part.
  - Complex shapes are made by adding harmonics of the same fundamental.
    Square and triangle waves use only odd harmonics; the triangle's
    coefficients fall off like 1/k^2, so a few harmonics already look right,
    while the square's fall off like 1/k, so it needs many more. The sharper
    the corners, the slower the coefficients shrink.
  - Near the square wave's jumps the synthesis overshoots by about 9% of the
    jump and stays that way as K grows. The error over the whole waveform
    still shrinks; only the overshoot does not.
So a waveform and its coefficient list carry the same information: one is the
time-domain view, the other is the frequency-domain view we are building."""
    )
