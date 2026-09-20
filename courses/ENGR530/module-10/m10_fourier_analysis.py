"""Numerical Fourier analysis: estimate a_k from samples of x(t), rebuild the
signal with the synthesizer, and measure the RMSE. Analysis + synthesis."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module-8", "signal_toolkit"))
from fourier import (
    fourier_coefficients,
    fourier_synthesis,
    pulse_train_coeffs,
    sawtooth_wave_coeffs,
    square_wave_coeffs,
    triangle_wave_coeffs,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
F0 = 5.0
T0 = 1 / F0
N = 4000            # integration intervals per period
DUTY = 0.25
K_LIST = [1, 3, 9, 25, 51]


def cosine(t):
    return 3.0 * np.cos(2 * np.pi * F0 * t + np.pi / 4)


def square(t):
    return np.sign(np.sin(2 * np.pi * F0 * t))


def triangle(t):
    return (2 / np.pi) * np.arcsin(np.cos(2 * np.pi * F0 * t))


def sawtooth(t):
    return 2 * ((t * F0 + 0.5) % 1.0) - 1


def pulse(t):
    phase = (t * F0 + 0.5) % 1.0 - 0.5
    return (np.abs(phase) < DUTY / 2).astype(float)


def rmse(a, b):
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2)))


def analyze(signal, n=N, K=51):
    """Coefficients of `signal` (a function of t) from n intervals over one period."""
    t = np.linspace(0.0, T0, n + 1)
    return fourier_coefficients(t, signal(t), T0, K)


if __name__ == "__main__":
    # Evaluate reconstructions on midpoints so no sample sits exactly on a jump.
    t_eval = (np.arange(N) + 0.5) * T0 / N

    # 1. Cosine: exactly two nonzero coefficients.
    a = analyze(cosine, K=5)
    print("COSINE 3 cos(2 pi 5 t + pi/4): expect a_1 = 1.5 e^{j pi/4}, a_-1 = conjugate, all else 0")
    print(f"  a_1  numeric = {a[1]:.6f}   exact = {1.5 * np.exp(1j * np.pi / 4):.6f}")
    print(f"  a_-1 numeric = {a[-1]:.6f}   exact = {1.5 * np.exp(-1j * np.pi / 4):.6f}")
    print(f"  largest other |a_k| = {max(abs(a[k]) for k in a if abs(k) != 1):.2e}\n")

    # 2. Waveforms with known closed-form coefficients.
    waves = [
        ("square wave", square, square_wave_coeffs),
        ("triangle wave", triangle, triangle_wave_coeffs),
        ("sawtooth wave", sawtooth, sawtooth_wave_coeffs),
        (f"pulse train (duty {DUTY:g})", pulse, lambda K: pulse_train_coeffs(K, DUTY)),
    ]
    print("Numerical vs closed-form coefficients (|k| <= 51)")
    print(f"{'wave':<24}{'max |a_k numeric - exact|':>27}")
    analyzed = {}
    for name, sig, exact_fn in waves:
        num = analyze(sig)
        exact = exact_fn(51)
        analyzed[name] = num
        err = max(abs(num[k] - exact[k]) for k in num)
        print(f"{name:<24}{err:>27.2e}")

    print("\nRECONSTRUCTION: RMSE( x_original , x_reconstructed from numerical a_k )")
    print(f"{'wave':<24}" + "".join(f"{'K = ' + str(K):>11}" for K in K_LIST))
    for name, sig, _ in waves:
        num = analyzed[name]
        row = []
        for K in K_LIST:
            sub = {k: v for k, v in num.items() if abs(k) <= K}
            row.append(rmse(sig(t_eval), fourier_synthesis(sub, F0, t_eval)))
        print(f"{name:<24}" + "".join(f"{e:>11.4f}" for e in row))

    # 3. How does the numerical integration itself converge with the sample count?
    print("\nINTEGRATION ACCURACY: max |a_k numeric - exact| for |k| <= 7 vs. samples per period")
    counts = [50, 200, 1000, 5000]
    print(f"{'signal':<16}" + "".join(f"{'N = ' + str(c):>12}" for c in counts))
    for name, sig, exact_fn in [("cosine", cosine, None), ("triangle", triangle, triangle_wave_coeffs),
                                ("square", square, square_wave_coeffs)]:
        exact = ({1: 1.5 * np.exp(1j * np.pi / 4), -1: 1.5 * np.exp(-1j * np.pi / 4)} if exact_fn is None
                 else exact_fn(7))
        row = []
        for c in counts:
            num = analyze(sig, n=c, K=7)
            row.append(max(abs(num[k] - exact.get(k, 0.0)) for k in num))
        print(f"{name:<16}" + "".join(f"{e:>12.2e}" for e in row))

    # Figure 1: coefficients and reconstruction for each waveform.
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), gridspec_kw={"width_ratios": [1, 2]})
    tp = np.linspace(-0.5 * T0, 1.5 * T0, 2000, endpoint=False) + 0.5 * T0 / 2000
    for row, (name, sig, exact_fn) in enumerate(waves):
        num = analyzed[name]
        exact = exact_fn(25)
        ks = np.arange(0, 26)
        ax_c, ax_t = axes[row]
        ax_c.stem(ks, [abs(num[k]) for k in ks], label="numerical")
        ax_c.plot(ks, [abs(exact[k]) for k in ks], "rx", markersize=6, label="closed form")
        ax_c.set_title(f"{name}: |a_k|")
        ax_c.set_xlabel("k")
        ax_c.legend(fontsize=8)
        ax_c.grid(alpha=0.3)

        sub = {k: v for k, v in num.items() if abs(k) <= 25}
        ax_t.plot(tp, sig(tp), color="lightgray", linewidth=4, label="original")
        ax_t.plot(tp, fourier_synthesis(sub, F0, tp), color="steelblue", linewidth=1.5,
                  label="reconstructed from numerical a_k (K = 25)")
        ax_t.set_title(f"{name}: analysis -> synthesis")
        ax_t.set_xlabel("t (s)")
        ax_t.legend(loc="upper right", fontsize=8)
        ax_t.grid(alpha=0.3)
    fig.suptitle("Fourier analysis (numerical integration) followed by synthesis")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fourier_analysis.png"), dpi=150)

    # 4. A noisy record: the IMU-style signal, analyzed over its 2 s fundamental period.
    rng = np.random.default_rng(0)
    T_imu, f_imu, N_imu = 2.0, 0.5, 2000
    t_imu = np.linspace(0.0, T_imu, N_imu + 1)
    clean = 20 * np.sin(2 * np.pi * 0.5 * t_imu) + 3 * np.sin(2 * np.pi * 8 * t_imu)
    noisy = clean + rng.normal(0.0, 2.0, size=len(t_imu))
    c_imu = fourier_coefficients(t_imu, noisy, T_imu, 40)
    top = sorted(range(1, 41), key=lambda k: -abs(c_imu[k]))[:3]
    print("\nNOISY IMU-STYLE RECORD: 20 sin(2 pi 0.5 t) + 3 sin(2 pi 8 t) + noise, analyzed over 2 s")
    print(f"  expected: |a_1| = 10 (0.5 Hz), |a_16| = 1.5 (8 Hz)")
    print(f"  three largest |a_k|, k > 0: " + ", ".join(f"k={k} ({k * f_imu:g} Hz): {abs(c_imu[k]):.3f}" for k in top))
    floor = np.median([abs(c_imu[k]) for k in range(1, 41) if k not in (1, 16)])
    print(f"  typical |a_k| elsewhere (median, the noise floor): {floor:.3f}")
    keep = {k: v for k, v in c_imu.items() if abs(k) in (1, 16)}
    rebuilt = fourier_synthesis(keep, f_imu, t_imu)
    print(f"  RMSE(noisy, clean)               = {rmse(noisy, clean):.3f}")
    print(f"  RMSE(rebuilt from 4 coeffs, clean) = {rmse(rebuilt, clean):.3f}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    axes[0].plot(t_imu, noisy, color="gray", linewidth=0.8)
    axes[0].set_title("Measured record (time domain): motion, vibration and noise are mixed")
    axes[0].set_xlabel("t (s)")
    ks = np.arange(0, 41)
    axes[1].stem(ks, [abs(c_imu[k]) for k in ks])
    axes[1].set_title("|a_k| of the same record: two clear peaks at k = 1 (0.5 Hz) and k = 16 (8 Hz)")
    axes[1].set_xlabel("k  (frequency = 0.5 k Hz)")
    axes[2].plot(t_imu, clean, color="lightgray", linewidth=4, label="clean (motion + vibration)")
    axes[2].plot(t_imu, rebuilt, color="steelblue", linewidth=1.5, label="rebuilt from only k = +/-1, +/-16")
    axes[2].set_title("Keeping just four coefficients recovers the signal and discards the noise")
    axes[2].set_xlabel("t (s)")
    axes[2].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fourier_analysis_noisy_record.png"), dpi=150)
    print("\nSaved fourier_analysis.png and fourier_analysis_noisy_record.png")

    print(
        """
What this means
---------------
Analysis and synthesis are now inverses. fourier_coefficients() turns a
waveform into its list of a_k by integrating x(t) e^{-jk w0 t} over one period;
fourier_synthesis() turns that list back into the waveform. Feeding one into the
other reproduces the signal, and the leftover RMSE is only from truncating at K
harmonics, not from the analysis.
  - The numerical coefficients match the textbook formulas. Smooth signals
    (the cosine) are recovered to rounding error, kinked ones (triangle) improve
    quickly with more samples, and jumps (square) improve only slowly, which is
    the integration table above.
  - Reconstruction RMSE falls with K exactly as in the synthesis builds:
    triangle quickly, square/sawtooth/pulse slowly, since their sharp edges
    need many harmonics.
  - The noisy record shows why this matters. In the time domain motion,
    vibration, and noise are hard to separate by eye. In the coefficients the
    motion and the vibration stand out as two spikes far above a low, flat noise
    floor, so keeping four numbers recovers the signal. That is the
    frequency-domain answer to the IMU problem."""
    )
