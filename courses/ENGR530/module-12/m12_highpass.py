"""High-pass filtering to remove slow drift, and what happens when the cutoff is
raised too far. RMSE vs. cutoff: filter selection is an optimization problem.
    x(t) = b(t) + s(t) + n(t)    (drift + faster desired signal + noise)
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 100                 # Hz
DURATION = 60.0          # s: every tone below fits a whole number of periods
NOISE_STD = 0.5
DRIFT = [(4.0, 0.05), (2.5, 0.10), (1.5, 0.20), (1.0, 0.30)]      # (amplitude, Hz): slow drift
DESIRED = [(2.0, 1.0), (1.5, 2.5)]                                  # (amplitude, Hz): the signal we want
CUTOFFS = np.logspace(np.log10(0.02), np.log10(10), 70)


def make_signal(seed=0):
    t = np.arange(int(FS * DURATION)) / FS
    b = sum(a * np.sin(2 * np.pi * f * t + 0.5 * k) for k, (a, f) in enumerate(DRIFT))
    s = sum(a * np.sin(2 * np.pi * f * t) for a, f in DESIRED)
    n = np.random.default_rng(seed).normal(0.0, NOISE_STD, size=len(t))
    return t, b, s, n


def ideal_highpass(x, cutoff):
    """Idealized: Y(f) = H(f) X(f), H = 0 for |f| < cutoff and 1 above."""
    f = np.fft.fftfreq(len(x), 1 / FS)
    return np.fft.ifft(np.where(np.abs(f) < cutoff, 0.0, 1.0) * np.fft.fft(x)).real


def butterworth_highpass_coeffs(cutoff):
    """2nd-order Butterworth high-pass biquad by the bilinear transform."""
    K = np.tan(np.pi * cutoff / FS)
    norm = 1 / (1 + np.sqrt(2) * K + K * K)
    b = np.array([norm, -2 * norm, norm])
    a = np.array([1.0, 2 * (K * K - 1) * norm, (1 - np.sqrt(2) * K + K * K) * norm])
    return b, a


def biquad(x, b, a):
    """Causal recursion y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]."""
    y = np.zeros(len(x))
    x1 = x2 = y1 = y2 = 0.0
    for i, xi in enumerate(x):
        yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        y[i] = yi
        x2, x1 = x1, xi
        y2, y1 = y1, yi
    return y


def realizable_highpass(x, cutoff):
    """Causal 2nd-order Butterworth high-pass, run to steady state.

    Filters three back-to-back copies of the (periodic) record and keeps the last,
    so the start-up transient does not contaminate the comparison.
    """
    b, a = butterworth_highpass_coeffs(cutoff)
    return biquad(np.tile(x, 3), b, a)[2 * len(x):]


def zero_phase_highpass(x, cutoff):
    """The same Butterworth run forward and then backward (offline only).

    The backward pass cancels the phase shift of the forward pass, leaving zero
    phase; the gain is |H|^2. Run on three copies, keeping the middle one, to
    avoid start-up transients at both ends.
    """
    b, a = butterworth_highpass_coeffs(cutoff)
    n = len(x)
    forward = biquad(np.tile(x, 3), b, a)
    backward = biquad(forward[::-1], b, a)[::-1]
    return backward[n:2 * n]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    t, b, s, n = make_signal()
    x = b + s + n
    print(f"x = drift b(t) + desired s(t) + noise, fs = {FS} Hz, {DURATION:g} s")
    print("  drift   : " + ", ".join(f"{a:g} @ {f:g} Hz" for a, f in DRIFT))
    print("  desired : " + ", ".join(f"{a:g} @ {f:g} Hz" for a, f in DESIRED))
    print(f"  noise   : std {NOISE_STD}\n")
    print(f"RMSE of the unfiltered record vs. the desired signal s(t): {rmse(x, s):.3f}  (drift RMS {np.sqrt(np.mean(b ** 2)):.3f}, noise {n.std():.3f})\n")

    rm_ideal = np.array([rmse(ideal_highpass(x, fc), s) for fc in CUTOFFS])
    rm_real = np.array([rmse(realizable_highpass(x, fc), s) for fc in CUTOFFS])
    rm_zp = np.array([rmse(zero_phase_highpass(x, fc), s) for fc in CUTOFFS])

    i_ideal = int(np.argmin(rm_ideal))
    i_real = int(np.argmin(rm_real))
    i_zp = int(np.argmin(rm_zp))
    print(f"{'cutoff (Hz)':>12}{'ideal':>9}{'zero-phase':>12}{'causal':>9}   (RMSE)   what is happening")
    notes = {
        0.03: "cutoff below the drift: drift is untouched",
        0.15: "cutoff inside the drift band: some drift left",
        0.6: "above the drift, below the signal: sweet spot",
        1.5: "cutoff passes the 1 Hz signal: desired signal is being removed",
        5.0: "far too high: nearly all of the desired signal is gone",
    }
    for fc, note in notes.items():
        i = int(np.argmin(np.abs(CUTOFFS - fc)))
        print(f"{CUTOFFS[i]:>12.3f}{rm_ideal[i]:>9.3f}{rm_zp[i]:>12.3f}{rm_real[i]:>9.3f}            {note}")
    print(f"\nBest cutoff, ideal brick-wall:        {CUTOFFS[i_ideal]:.3f} Hz  (RMSE {rm_ideal[i_ideal]:.3f}, vs {rmse(x, s):.3f} with no filter)")
    print(f"Best cutoff, zero-phase Butterworth:  {CUTOFFS[i_zp]:.3f} Hz  (RMSE {rm_zp[i_zp]:.3f})")
    print(f"Best cutoff, causal Butterworth:      {CUTOFFS[i_real]:.3f} Hz  (RMSE {rm_real[i_real]:.3f})")
    print(f"Highest cutoff tried ({CUTOFFS[-1]:g} Hz): ideal RMSE {rm_ideal[-1]:.3f} = {rm_ideal[-1] / rm_ideal[i_ideal]:.1f}x the best; "
          f"all of s is removed, so it approaches rms(s) = {np.sqrt(np.mean(s ** 2)):.3f} plus the noise")

    fc_low, fc_best, fc_high = 0.03, 0.6, 5.0
    y_low, y_best, y_high = (ideal_highpass(x, fc) for fc in (fc_low, fc_best, fc_high))
    y_causal = realizable_highpass(x, fc_best)

    # Figure 1
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    ax.plot(t, x, color="lightgray", linewidth=0.8, label="x = b + s + n")
    ax.plot(t, b, color="firebrick", linewidth=1.5, label="drift b(t)")
    ax.plot(t, s, color="steelblue", linewidth=1, label="desired s(t)")
    ax.set_xlim(0, 20)
    ax.set_title("Raw record: slow drift under a faster signal")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 1]
    ax.semilogx(CUTOFFS, rm_ideal, "o-", color="steelblue", markersize=3, label="ideal (brick-wall) high-pass")
    ax.semilogx(CUTOFFS, rm_zp, "^-", color="darkgreen", markersize=3, label="2nd-order Butterworth, zero phase (forward-backward)")
    ax.semilogx(CUTOFFS, rm_real, "s-", color="darkorange", markersize=3, label="2nd-order Butterworth, causal")
    ax.axhline(rmse(x, s), color="gray", linestyle=":", label=f"no filter ({rmse(x, s):.2f})")
    ax.axhline(NOISE_STD, color="green", linestyle=":", label=f"noise level ({NOISE_STD})")
    ax.axvspan(0.02, 0.3, color="firebrick", alpha=0.08)
    ax.axvspan(1.0, 10, color="steelblue", alpha=0.08)
    ax.plot(CUTOFFS[i_zp], rm_zp[i_zp], "k*", markersize=14, label=f"best zero-phase: {CUTOFFS[i_zp]:.2f} Hz")
    ax.text(0.024, 0.12, "drift left in", color="firebrick")
    ax.text(1.5, 0.12, "desired signal removed", color="steelblue")
    ax.set_title("RMSE vs. cutoff frequency")
    ax.set_xlabel("high-pass cutoff (Hz)")
    ax.set_ylabel("RMSE vs desired s(t)")
    ax.legend(fontsize=8, loc="upper center")

    ax = axes[1, 0]
    show = (t >= 20) & (t <= 26)
    ax.plot(t[show], s[show], "k--", linewidth=1.5, label="desired s(t)")
    ax.plot(t[show], y_low[show], color="firebrick", linewidth=1, label=f"cutoff {fc_low:g} Hz: drift remains ({rmse(y_low, s):.2f})")
    ax.plot(t[show], y_best[show], color="steelblue", linewidth=1.5, label=f"cutoff {fc_best:g} Hz: in the gap ({rmse(y_best, s):.2f})")
    ax.plot(t[show], y_high[show], color="darkgreen", linewidth=1.5, label=f"cutoff {fc_high:g} Hz: signal gone ({rmse(y_high, s):.2f})")
    ax.plot(t[show], y_causal[show], color="darkorange", linewidth=1.2, linestyle=":", label=f"causal Butterworth at {fc_best:g} Hz: phase-shifted ({rmse(y_causal, s):.2f})")
    ax.set_title("Filtered results at three cutoffs (ideal filter), plus a causal filter")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 1]
    f_full = np.fft.rfftfreq(len(x), 1 / FS)
    amp = 2 * np.abs(np.fft.rfft(x)) / len(x)
    ax.semilogx(f_full[1:], amp[1:], color="gray", linewidth=0.8, label="spectrum of x")
    for lo, hi, color, label in [(0.02, 0.35, "firebrick", "drift"), (0.9, 3.0, "steelblue", "desired signal")]:
        ax.axvspan(lo, hi, color=color, alpha=0.12, label=label)
    for fc, c in [(fc_low, "firebrick"), (fc_best, "steelblue"), (fc_high, "darkgreen")]:
        ax.axvline(fc, color=c, linestyle="--")
    ax.set_xlim(0.02, 10)
    ax.set_title("Spectrum: drift and signal are separated, with a gap for the cutoff", fontsize=11)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude")
    ax.legend(fontsize=8)
    for ax in axes.flatten():
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "highpass.png"), dpi=150)
    print("\nSaved highpass.png")

    print(
        f"""
What this means
---------------
RMSE against the known desired signal shows a cutoff is a choice with two
failure modes, not a 'more is better' dial:
  - Too low (below about 0.3 Hz): the drift still passes. The error starts at
    {rm_ideal[0]:.2f} (no better than unfiltered, {rmse(x, s):.2f}) and falls in steps as the cutoff crosses
    each drift component.
  - In the gap between drift and signal (about 0.3-1 Hz for the ideal filter): the
    drift is gone and the desired signal is untouched. The error bottoms out at
    {rm_ideal[i_ideal]:.2f}, essentially the noise level ({NOISE_STD}); a high-pass cannot remove noise that
    sits above its cutoff.
  - Too high (above 1 Hz): the filter removes the desired signal itself. The
    error jumps to {rm_ideal[np.argmin(np.abs(CUTOFFS - 1.5))]:.2f} once the 1 Hz component is cut and reaches {rm_ideal[-1]:.2f} at 10 Hz, where all of
    s is gone: {rm_ideal[-1] / rm_ideal[i_ideal]:.1f}x worse than the best cutoff.
The realizable filters show that the gap is not free. The zero-phase Butterworth
reaches {rm_zp[i_zp]:.2f} but over a narrower range of cutoffs, because its roll-off is gradual: a
cutoff high enough to reject the 0.3 Hz drift already clips the 1 Hz signal. The
causal Butterworth is far worse (best {rm_real[i_real]:.2f}), mostly because of phase: it advances
the 1 Hz signal in time, and RMSE against the true signal penalizes a shifted
copy heavily. (The forward-backward version cancels that phase, though it also
squares the gain, so the comparison is suggestive, not a pure phase test.) So the best filter depends on what you can afford: sharper
roll-off needs a higher order, and zero phase needs the whole record (offline).
Filter selection is an optimization against the signal's frequency structure and
your constraints, not simply 'more filtering'."""
    )
