"""DTFT calculator demo: impulse, delayed impulse, and the 5-point moving average."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module-8", "signal_toolkit"))
from transforms import dtft

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OMEGA = np.linspace(-np.pi, np.pi, 2001)
MASK = 1e-3            # hide phase where |X| is too small for it to mean anything


def dirichlet_amplitude(omega, M):
    """sin(M w / 2) / (M sin(w / 2)): the real amplitude of an M-point moving average."""
    return np.sinc(M * omega / (2 * np.pi)) / np.sinc(omega / (2 * np.pi))


def masked_phase(X):
    ph = np.angle(X)
    ph[np.abs(X) < MASK] = np.nan
    return ph


def pi_ticks(ax, axis="x"):
    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    labels = ["-$\\pi$", "-$\\pi$/2", "0", "$\\pi$/2", "$\\pi$"]
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


if __name__ == "__main__":
    M = 5
    cases = [
        ("impulse  d[n]", [1.0], [0], np.ones_like(OMEGA, dtype=complex)),
        ("delayed impulse  d[n - 5]", [1.0], [5], np.exp(-5j * OMEGA)),
        ("moving average  (1/5){1,1,1,1,1}", np.full(M, 1 / M), np.arange(M),
         np.exp(-2j * OMEGA) * dirichlet_amplitude(OMEGA, M)),
    ]

    print("DTFT X(e^{jw}) = sum x[n] e^{-jwn} on -pi <= w <= pi (2001 points)\n")
    print(f"{'signal':<36}{'max |X - closed form|':>24}")
    spectra = []
    for name, x, n, exact in cases:
        X = dtft(x, n, OMEGA)
        spectra.append(X)
        print(f"{name:<36}{np.max(np.abs(X - exact)):>24.2e}")
    print("  closed forms: 1,  e^{-j5w},  e^{-j2w} sin(5w/2) / (5 sin(w/2))")

    X_ma = spectra[2]
    per = np.max(np.abs(dtft(np.full(M, 1 / M), np.arange(M), OMEGA) - dtft(np.full(M, 1 / M), np.arange(M), OMEGA + 2 * np.pi)))
    print(f"\nPeriodicity: max |X(w) - X(w + 2 pi)| for the moving average = {per:.1e}")

    # Phase of the delayed impulse is a straight line of slope -5 (wrapped).
    slope = np.polyfit(OMEGA[1:-1], np.unwrap(np.angle(spectra[1]))[1:-1], 1)[0]
    print(f"Delayed impulse: unwrapped phase slope = {slope:.4f} rad per (rad/sample)  -> a delay of {-slope:.2f} samples")
    ma_slope = np.polyfit(OMEGA[np.abs(OMEGA) < 2 * np.pi / M * 0.95],
                          np.angle(X_ma[np.abs(OMEGA) < 2 * np.pi / M * 0.95]), 1)[0]
    print(f"Moving average: phase slope in the main lobe = {ma_slope:.4f} -> a delay of {-ma_slope:.2f} samples = (M - 1)/2")

    print("\nMoving average magnitude |X(e^{jw})| at selected frequencies")
    print(f"{'w (rad/sample)':>16}{'|X|':>9}   note")
    lookups = [
        (0.0, "DC: constants pass unchanged"),
        (2 * np.pi * 2 / 100, "2 Hz sinusoid sampled at 100 Hz  (Module 8 measured 0.984)"),
        (2 * np.pi * 10 / 100, "10 Hz sinusoid sampled at 100 Hz (Module 8 measured 0.647)"),
        (np.pi / 4, ""),
        (np.pi / 2, ""),
        (2 * np.pi / 5, "first null, w = 2 pi / M"),
        (np.pi, "highest possible frequency"),
    ]
    for w, note in lookups:
        print(f"{w:>16.4f}{abs(dtft(np.full(M, 1 / M), np.arange(M), [w])[0]):>9.4f}   {note}")
    w_dense = np.linspace(-np.pi, np.pi, 4000, endpoint=False)
    power = np.mean(np.abs(dtft(np.full(M, 1 / M), np.arange(M), w_dense)) ** 2)
    print(f"\nAverage of |X|^2 over all frequencies = {power:.4f}  (= sum h^2 = 1/M = {1 / M:.4f}: the fraction of white-noise power that gets through)")
    side = np.abs(X_ma)[(OMEGA > 2 * np.pi / M) & (OMEGA < 4 * np.pi / M)].max()
    print(f"Highest sidelobe: {side:.3f} ({20 * np.log10(side):.1f} dB)")

    # Figure 1: magnitude and phase for each case.
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for row, ((name, _, _, exact), X) in enumerate(zip(cases, spectra)):
        ax_m, ax_p = axes[row]
        ax_m.plot(OMEGA, np.abs(X), color="steelblue", linewidth=2, label="DTFT (direct sum)")
        ax_m.plot(OMEGA, np.abs(exact), "r--", linewidth=1, label="closed form")
        ax_m.set_title(f"|X(e^jw)|:  {name}")
        ax_m.set_ylim(-0.05, 1.15)
        ax_m.legend(fontsize=8, loc="upper right")
        ph = masked_phase(X)
        ax_p.plot(OMEGA, ph, ".", color="steelblue", markersize=2.5, label="DTFT (direct sum)")
        ax_p.plot(OMEGA, masked_phase(exact), "r-", linewidth=0.8, alpha=0.6, label="closed form")
        pi_ticks(ax_p, "y")
        ax_p.set_title(f"angle X(e^jw):  {name}")
        ax_p.legend(fontsize=8, loc="upper right")
        for ax in (ax_m, ax_p):
            pi_ticks(ax, "x")
            ax.set_xlabel("w (rad/sample)")
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dtft.png"), dpi=150)

    # Figure 2: why the moving average is a low-pass filter.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    A = dirichlet_amplitude(OMEGA, M)
    axes[0].plot(OMEGA, A, color="steelblue", linewidth=2)
    axes[0].axhline(0, color="gray", linewidth=0.8)
    for w0 in (2 * np.pi / M, 4 * np.pi / M):
        axes[0].axvline(w0, color="firebrick", linestyle=":")
        axes[0].axvline(-w0, color="firebrick", linestyle=":")
    axes[0].set_title("Signed amplitude A(w) = sin(5w/2) / (5 sin(w/2))", fontsize=10)
    axes[0].set_ylabel("A(w)")
    axes[1].plot(OMEGA, 20 * np.log10(np.abs(X_ma) + 1e-12), color="steelblue", linewidth=2)
    axes[1].set_ylim(-60, 5)
    axes[1].set_title("|X| in dB: 0 dB at DC, nulls at 2 pi k / 5, sidelobes -12 dB", fontsize=10)
    axes[1].set_ylabel("dB")
    for Ml, c in [(3, "firebrick"), (5, "darkorange"), (15, "steelblue"), (51, "darkgreen")]:
        Xl = dtft(np.full(Ml, 1 / Ml), np.arange(Ml), OMEGA)
        axes[2].plot(OMEGA, np.abs(Xl), color=c, linewidth=1.6, label=f"M = {Ml}")
    axes[2].set_title("Longer average -> narrower passband", fontsize=10)
    axes[2].set_ylabel("|X|")
    axes[2].legend()
    for ax in axes:
        pi_ticks(ax, "x")
        ax.set_xlabel("w (rad/sample)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dtft_moving_average.png"), dpi=150)
    print("\nSaved dtft.png and dtft_moving_average.png")

    print(
        """
What this means
---------------
  - Impulse: X = 1 at every frequency. A single spike contains all frequencies
    equally, which is why an impulse is the natural probe for a system.
  - Delayed impulse: the magnitude is still 1 everywhere, but the phase is the
    straight line -5w (drawn wrapped into -pi..pi, hence the sawtooth). A delay
    of n0 samples is exactly a phase slope of -n0: magnitude says which
    frequencies are present, phase says when.
  - Moving average: X = e^{-j2w} sin(5w/2) / (5 sin(w/2)). Read the magnitude:
    it equals 1 at w = 0 and falls to zero at w = 2 pi / 5, then stays small
    (never above 0.25) all the way to w = pi. Slow components pass, fast ones
    are suppressed: that is a low-pass filter, and it is why the 10 Hz sinusoid
    dropped to 0.647 and 2 Hz stayed at 0.984 in Module 8. The phase -2w is the
    2-sample delay (M - 1)/2 seen there too. Averaging |X|^2 over all frequencies
    gives 1/5: white noise, which is spread evenly over frequency, keeps only a
    fifth of its power, which is the noise reduction from Module 9.
  - X is 2 pi periodic, so w = pi is the highest frequency a sampled signal can
    have. Everything you need lives in -pi..pi."""
    )
