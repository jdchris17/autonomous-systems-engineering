"""Stage 7: design and apply the digital filter, z[n] -> x_hat[n].

Design is read off the Stage 3/6 spectrum:
    bias (DC) and drift (0.05 Hz)  <  desired motion (0.5 and 2 Hz)  <  vibration (15 Hz)
so the filter is a BAND-PASS with its corners in the two gaps.

estimate_waveform() is reused by Stages 8 and 9.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import project_signal as P
import stage4_antialias as S4
import stage5_sample as S5

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = S5.FS

# --- design decisions -----------------------------------------------------------------------
HP_CUTOFF = 0.15       # Hz: between the drift (0.05) and the slowest wanted line (0.5); geometric mean is 0.16
LP_CUTOFF = 5.0        # Hz: between the fastest wanted line (2) and the vibration (15); geometric mean is 5.5
ORDER = 4              # per filter (each is two cascaded 2nd-order Butterworth sections)
EQ_BAND = 8.0          # Hz: front-end delay is equalized inside this band
PAD = 500              # samples of odd-reflection padding to keep filter start-up transients off the record


def butterworth_sections(order, fc, fs, kind):
    """Butterworth low/high-pass as cascaded biquads, by the bilinear transform.

    Section k of an even-order Butterworth has Q_k = 1 / (2 sin((2k - 1) pi / (2 order))).
    Returns a list of (b, a) with a[0] = 1.
    """
    K = np.tan(np.pi * fc / fs)
    sections = []
    for k in range(1, order // 2 + 1):
        Q = 1.0 / (2 * np.sin((2 * k - 1) * np.pi / (2 * order)))
        norm = 1.0 / (1 + K / Q + K * K)
        a = np.array([1.0, 2 * (K * K - 1) * norm, (1 - K / Q + K * K) * norm])
        b = (np.array([K * K, 2 * K * K, K * K]) if kind == "lowpass" else np.array([1.0, -2.0, 1.0])) * norm
        sections.append((b, a))
    return sections


def band_sections(hp=HP_CUTOFF, lp=LP_CUTOFF, order=ORDER, fs=FS, use_hp=True):
    secs = butterworth_sections(order, lp, fs, "lowpass")
    return (butterworth_sections(order, hp, fs, "highpass") + secs) if use_hp else secs


def sections_response(sections, f, fs=FS):
    """Frequency response H(e^{j 2 pi f / fs}) of a cascade of biquads."""
    z = np.exp(-1j * 2 * np.pi * np.asarray(f, dtype=float) / fs)
    H = np.ones_like(z, dtype=complex)
    for b, a in sections:
        H = H * (b[0] + b[1] * z + b[2] * z**2) / (1 + a[1] * z + a[2] * z**2)
    return H


def run_sections(x, sections):
    """Causal filtering by the cascade, one sample at a time (direct form I)."""
    y = np.asarray(x, dtype=float)
    for b, a in sections:
        out = np.zeros(len(y))
        x1 = x2 = y1 = y2 = 0.0
        for i, xi in enumerate(y):
            yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            out[i] = yi
            x2, x1 = x1, xi
            y2, y1 = y1, yi
        y = out
    return y


def odd_extend(x, n):
    """Extend a record at both ends by point-reflection about the end values (keeps level and slope)."""
    n = min(n, len(x) - 2)
    left = 2 * x[0] - x[1:n + 1][::-1]
    right = 2 * x[-1] - x[-n - 1:-1][::-1]
    return np.concatenate([left, x, right]), n


def zero_phase(x, sections, pad=PAD):
    """Forward-backward filtering: no phase shift, squared magnitude. Offline only."""
    xe, n = odd_extend(np.asarray(x, dtype=float), pad)
    y = run_sections(xe, sections)
    y = run_sections(y[::-1], sections)[::-1]
    return y[n:len(y) - n]


def equalize_front_end(y, pad=PAD, fs=FS, band=EQ_BAND):
    """Undo the KNOWN analog anti-alias filter's phase (delay) and droop inside `band`.

    The analog filter H_a(f) is known from Stage 4, and it is ~1 in magnitude in the wanted
    band, so dividing by it there is well conditioned. Everything is done on the padded record.
    """
    ye, n = odd_extend(np.asarray(y, dtype=float), pad)
    f = np.fft.fftfreq(len(ye), 1 / fs)
    Ha = S4.butter_response(f, S4.FC, S4.ORDER)
    eq = np.where(np.abs(f) <= band, 1.0 / Ha, 1.0)
    out = np.fft.ifft(np.fft.fft(ye) * eq).real
    return out[n:len(out) - n]


def estimate_waveform(z_n, sections=None, equalize=True):
    """z[n] -> x_hat[n]: zero-phase band-pass, then front-end equalization."""
    sections = sections or band_sections()
    x_hat = zero_phase(z_n, sections)
    return equalize_front_end(x_hat) if equalize else x_hat


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


if __name__ == "__main__":
    a = S5.acquire()
    z, x_true, t_n = a["z_n"], a["x_true_n"], a["t_n"]
    secs = band_sections()
    secs_lp = band_sections(use_hp=False)
    f_grid = np.logspace(-2, np.log10(FS / 2), 800)
    H = sections_response(secs, f_grid)
    Hzp = np.abs(H) ** 2
    f_lin = np.linspace(0, FS / 2, 5001)
    noise_frac = float(np.mean(np.abs(sections_response(secs, f_lin)) ** 4))

    lines = [("bias (DC)", 1e-4), ("drift", P.F_B), ("desired 0.5 Hz", P.F1), ("desired 2 Hz", P.F2), ("vibration", P.F_V)]
    print("DIGITAL FILTER DESIGN")
    print(f"  Type:            zero-phase Butterworth band-pass = high-pass ({HP_CUTOFF:g} Hz) + low-pass ({LP_CUTOFF:g} Hz), each order {ORDER}")
    print(f"                   built from {len(secs)} biquad sections, run forward then backward (effective order {2 * ORDER} each)")
    print(f"  Cutoffs:         high-pass {HP_CUTOFF:g} Hz, low-pass {LP_CUTOFF:g} Hz  (sampling rate {FS} Hz)")
    front_delay_ms = 1e3 / (2 * np.pi * S4.FC) / np.sin(np.pi / (2 * S4.ORDER))
    print(f"  Front-end fix:   the analog anti-alias filter delays the signal by about {front_delay_ms:.0f} ms; its known response is divided out over 0-{EQ_BAND:g} Hz")
    print("\nWHY THESE CHOICES (from the spectrum)")
    print(f"  * The spectrum has three groups: bias/drift at 0-0.05 Hz, the desired motion at 0.5-2 Hz, the vibration at 15 Hz.")
    print(f"  * High-pass at {HP_CUTOFF:g} Hz sits between 0.05 and 0.5 Hz (a factor of 3 from each), so the drift is removed and the motion is not.")
    print(f"  * Low-pass at {LP_CUTOFF:g} Hz sits between 2 and 15 Hz (a factor of 2.5 and 3), so the vibration is removed and the motion is not.")
    print(f"  * Butterworth: maximally flat in the passband, so the wanted lines keep their amplitude. Order {ORDER}: steep enough that a")
    print(f"    3x frequency ratio gives about {abs(20 * np.log10(abs(sections_response(butterworth_sections(ORDER, LP_CUTOFF, FS, 'lowpass'), [P.F_V])[0]))):.0f} dB per pass; forward-backward doubles that.")
    print(f"  * Zero phase, because x_hat is computed offline from a recorded stretch; a causal filter of this order would delay the 0.5 Hz motion.")
    print(f"  * A 5 Hz low-pass ALONE (the obvious choice) leaves the bias and drift in place: see the comparison below.")

    print("\nEXPECTED EFFECT ON EACH COMPONENT (zero-phase gain from the design)")
    print(f"  {'component':<20}{'frequency':>11}{'gain':>11}{'dB':>9}   verdict")
    for name, f0 in lines:
        g = float(np.abs(sections_response(secs, [f0])[0]) ** 2)
        keep = "RETAINED" if g > 0.99 else "ATTENUATED"
        print(f"  {name:<20}{(0 if f0 < 1e-3 else f0):>9g} Hz{g:>11.5f}{20 * np.log10(max(g, 1e-12)):>9.1f}   {keep}")
    print(f"  {'80 Hz interference':<20}{'80':>9} Hz   already removed before sampling by the analog filter")
    print(f"  noise: about {noise_frac * 100:.1f}% of the noise power that reaches the ADC survives (the band {HP_CUTOFF:g}-{LP_CUTOFF:g} Hz out of 0-{FS // 2} Hz)")

    x_hat = estimate_waveform(z)
    x_hat_noeq = estimate_waveform(z, equalize=False)
    x_lp = estimate_waveform(z, secs_lp)
    interior = slice(200, len(z) - 200)
    print("\nAPPLYING IT: z[n] -> x_hat[n]   (RMSE against the known x_true[n])")
    print(f"  {'raw z[n]':<52}{rmse(z, x_true):>8.3f}")
    print(f"  {'5 Hz low-pass only (zero phase, equalized)':<52}{rmse(x_lp, x_true):>8.3f}   <- bias and drift remain")
    print(f"  {'band-pass, NOT equalized for the analog delay':<52}{rmse(x_hat_noeq, x_true):>8.3f}   <- limited by the front-end delay")
    print(f"  {'band-pass + front-end equalization (chosen)':<52}{rmse(x_hat, x_true):>8.3f}")
    print(f"  {'   same, excluding 2 s at each end':<52}{rmse(x_hat[interior], x_true[interior]):>8.3f}   <- edges hold most of the remaining error")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    ax.semilogx(f_grid, 20 * np.log10(np.abs(H) + 1e-12), color="steelblue", linewidth=1.6, label="one pass")
    ax.semilogx(f_grid, 20 * np.log10(Hzp + 1e-12), color="firebrick", linewidth=2, label="zero-phase (forward-backward)")
    ax.semilogx(f_grid, 20 * np.log10(np.abs(sections_response(secs_lp, f_grid)) ** 2 + 1e-12), color="gray", linestyle="--", label="5 Hz low-pass only")
    for name, f0 in lines[1:]:
        ax.axvline(f0, color="black", linestyle=":", linewidth=0.8)
        ax.text(f0, -95, name.replace("desired ", ""), rotation=90, fontsize=7, va="bottom", ha="right")
    ax.set_ylim(-100, 5)
    ax.set_title("Frequency response of the digital filter (dotted: the known spectral lines)")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("gain (dB)")
    ax.legend(fontsize=8, loc="lower center")

    ax = axes[0, 1]
    fz, az = P.amplitude_spectrum(z, FS)
    fx, ax_ = P.amplitude_spectrum(x_hat, FS)
    ax.semilogy(fz[1:], az[1:] + 1e-9, color="lightgray", linewidth=1.2, label="|Z| of z[n]")
    ax.semilogy(fx[1:], ax_[1:] + 1e-9, color="firebrick", linewidth=0.9, label="|X_hat| of x_hat[n]")
    ax.set_ylim(1e-4, 40)
    ax.set_title("Spectrum before and after")
    ax.set_xlabel("frequency (Hz)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 0]
    sh = t_n <= 6
    ax.plot(t_n[sh], z[sh], color="lightgray", linewidth=1, label="raw z[n]")
    ax.plot(t_n[sh], x_hat[sh], color="firebrick", linewidth=1.6, label="x_hat[n]")
    ax.plot(t_n[sh], x_true[sh], "k--", linewidth=1, label="x_true[n]")
    ax.set_title("Filtered estimate against the truth (first 6 s)")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 1]
    names = ["raw", "5 Hz LP only", "band-pass\nno equalization", "band-pass\n+ equalization"]
    vals = [rmse(z, x_true), rmse(x_lp, x_true), rmse(x_hat_noeq, x_true), rmse(x_hat, x_true)]
    bars = ax.bar(names, vals, color=["gray", "darkorange", "steelblue", "seagreen"])
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.05, f"{v:.3f}", ha="center")
    ax.set_title("RMSE vs. the known truth")
    for ax in axes.flatten():
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stage7_filter.png"), dpi=150)
    print("\nSaved stage7_filter.png")

    print(
        f"""
What this means
---------------
The filter is justified line by line, not just named. The spectrum showed three
groups of content separated by wide gaps, so a band-pass with one corner in each gap
keeps the two motion lines (gain 1.000) and removes bias, drift and vibration by
tens of dB. The comparison shows why one corner is not enough: a low-pass at 5 Hz,
the first thing to try, still leaves the bias and drift, so its error is about
{rmse(x_lp, x_true):.2f}, dominated by them. The other lesson is the front end: the analog anti-alias
filter from Stage 4 shifts everything by about 21 ms, so an estimate that is
otherwise perfect still looks wrong by {rmse(x_hat_noeq, x_true):.2f} until that known delay is undone. What is left after
band-pass and equalization has two parts. One is the noise inside the passband
(about {noise_frac * 100:.0f}% of the noise power reaches x_hat), which no linear filter can remove
without also removing the signal. The other is edge error: the 0.15 Hz high-pass has
a memory of several seconds, so the first and last ~2 s of a 20 s record are estimated
worse ({rmse(x_hat, x_true):.2f} over the whole record, {rmse(x_hat[interior], x_true[interior]):.2f} away from the ends). That is a property of filtering a short
record, not a bug: a longer record or a higher high-pass corner (at the cost of
amplitude on the 0.5 Hz line) shrinks it."""
    )
