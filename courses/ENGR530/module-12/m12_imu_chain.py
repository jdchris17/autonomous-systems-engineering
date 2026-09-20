"""Engineering challenge: a complete IMU acquisition chain.

    physical angular rate -> analog anti-alias filter -> sampling -> digital filter -> estimated desired motion

Physical signal: w(t) = 20 sin(2 pi 0.5 t) + 3 sin(2 pi 8 t) + n(t), plus slow drift and a
broadband structural vibration that lives above the eventual Nyquist frequency.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- the physical world (simulated at a high "continuous-like" rate) -------------------------
FS_PHYS = 2000
DURATION = 60.0
DRIFT = [(6.0, 0.05, 0.7), (3.0, 0.10, 2.0)]        # (amplitude, Hz, phase): slow bias wander
STRUCT_BAND = (110.0, 450.0)                        # broadband structural vibration, above the ADC Nyquist
STRUCT_RMS = 1.0
MOTOR_TONE = (3.0, 201.0)                           # (amplitude, Hz): a motor/rotor tone just above the ADC rate
NOISE_STD = 1.0                                     # broadband sensor noise, white over 0-1000 Hz

# --- the design decisions (justified from the spectrum in the printout) ----------------------
FS_ADC = 200                                        # ADC sample rate (Hz)
AA_CUTOFF = 40.0                                    # analog anti-alias corner (Hz)
AA_ORDER = 4
HP_CUTOFF = 0.2                                     # digital high-pass corner (Hz): removes drift
LP_CUTOFF = 2.0                                     # digital low-pass corner (Hz): removes 8 Hz vibration and noise
FACTOR = FS_PHYS // FS_ADC


def make_physical(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS_PHYS * DURATION)) / FS_PHYS
    motion = 20 * np.sin(2 * np.pi * 0.5 * t)
    vib8 = 3 * np.sin(2 * np.pi * 8 * t)
    drift = sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in DRIFT)
    # Band-limited random structural vibration, built by shaping white noise in the frequency domain.
    W = np.fft.rfft(rng.standard_normal(len(t)))
    f = np.fft.rfftfreq(len(t), 1 / FS_PHYS)
    struct = np.fft.irfft(W * ((f >= STRUCT_BAND[0]) & (f <= STRUCT_BAND[1])), n=len(t))
    struct *= STRUCT_RMS / struct.std()
    struct = struct + MOTOR_TONE[0] * np.sin(2 * np.pi * MOTOR_TONE[1] * t + 1.1)
    noise = rng.normal(0.0, NOISE_STD, size=len(t))
    return t, motion, vib8, drift, struct, noise


def butterworth_lowpass(f, fc, order):
    """Complex response of an analog Butterworth low-pass (DC gain 1)."""
    s = 1j * 2 * np.pi * np.asarray(f, dtype=float)
    wc = 2 * np.pi * fc
    k = np.arange(1, order + 1)
    poles = wc * np.exp(1j * np.pi * (2 * k + order - 1) / (2 * order))
    H = np.ones_like(s, dtype=complex)
    for p in poles:
        H = H * (-p) / (s - p)
    return H


def analog_filter(x):
    """Analog-like anti-alias filter: multiply the high-rate spectrum by H(f)."""
    f = np.fft.fftfreq(len(x), 1 / FS_PHYS)
    return np.fft.ifft(butterworth_lowpass(f, AA_CUTOFF, AA_ORDER) * np.fft.fft(x)).real


def biquad_coeffs(cutoff, kind):
    """2nd-order Butterworth biquad by the bilinear transform at the ADC rate."""
    K = np.tan(np.pi * cutoff / FS_ADC)
    norm = 1 / (1 + np.sqrt(2) * K + K * K)
    a = np.array([1.0, 2 * (K * K - 1) * norm, (1 - np.sqrt(2) * K + K * K) * norm])
    if kind == "lowpass":
        b = np.array([K * K * norm, 2 * K * K * norm, K * K * norm])
    else:
        b = np.array([norm, -2 * norm, norm])
    return b, a


def biquad_run(x, b, a):
    y = np.zeros(len(x))
    x1 = x2 = y1 = y2 = 0.0
    for i, xi in enumerate(x):
        yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        y[i] = yi
        x2, x1 = x1, xi
        y2, y1 = y1, yi
    return y


def biquad_response(f, b, a):
    z = np.exp(-1j * 2 * np.pi * np.asarray(f) / FS_ADC)
    return (b[0] + b[1] * z + b[2] * z**2) / (1 + a[1] * z + a[2] * z**2)


def digital_filter(x, zero_phase):
    """Band-pass: 2nd-order Butterworth high-pass (drift) then low-pass (vibration, noise).

    The record is periodic, so it is run over three copies and the middle one kept to
    remove start-up transients. zero_phase=True adds a backward pass (offline only).
    """
    n = len(x)
    y = np.tile(x, 3)
    for kind, fc in [("highpass", HP_CUTOFF), ("lowpass", LP_CUTOFF)]:
        b, a = biquad_coeffs(fc, kind)
        y = biquad_run(y, b, a)
        if zero_phase:
            y = biquad_run(y[::-1], b, a)[::-1]
    return y[n:2 * n]


def digital_response(f, zero_phase):
    H = np.ones_like(np.asarray(f, dtype=complex))
    for kind, fc in [("highpass", HP_CUTOFF), ("lowpass", LP_CUTOFF)]:
        b, a = biquad_coeffs(fc, kind)
        h = biquad_response(f, b, a)
        H = H * (np.abs(h) ** 2 if zero_phase else h)
    return H


def amplitude_spectrum(x, fs):
    N = len(x)
    amp = 2 * np.abs(np.fft.rfft(x)) / N
    amp[0] /= 2
    return np.fft.rfftfreq(N, 1 / fs), amp


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_component(x, fs, f0):
    """Least-squares amplitude and phase of the f0 component of x."""
    tt = np.arange(len(x)) / fs
    A = np.column_stack([np.sin(2 * np.pi * f0 * tt), np.cos(2 * np.pi * f0 * tt)])
    (a, b), *_ = np.linalg.lstsq(A, x, rcond=None)
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


if __name__ == "__main__":
    t, motion, vib8, drift, struct, noise = make_physical()
    x_phys = motion + vib8 + drift + struct + noise
    t_adc = t[::FACTOR]
    truth = motion[::FACTOR]

    f_p, a_p = amplitude_spectrum(x_phys, FS_PHYS)

    def line(a, f, f0):
        return a[int(np.argmin(np.abs(f - f0)))]

    print("STEP 1: read the spectrum of the physical signal")
    print("  0.05 Hz, 0.10 Hz : drift            (amplitudes %.2f, %.2f)" % (line(a_p, f_p, 0.05), line(a_p, f_p, 0.10)))
    print("  0.50 Hz          : desired motion   (amplitude %.2f)" % line(a_p, f_p, 0.5))
    print("  8 Hz             : vibration        (amplitude %.2f)" % line(a_p, f_p, 8.0))
    print(f"  {STRUCT_BAND[0]:g}-{STRUCT_BAND[1]:g} Hz    : broadband structural vibration, RMS {STRUCT_RMS}")
    print(f"  {MOTOR_TONE[1]:g} Hz           : motor tone, amplitude {MOTOR_TONE[0]:g}   (at a {FS_ADC} Hz ADC this folds to |{MOTOR_TONE[1]:g} - {FS_ADC}| = {abs(MOTOR_TONE[1] - FS_ADC):g} Hz,")
    print(f"                       inside the wanted band, where no later filter could tell it from real motion)")
    print(f"  everywhere       : white noise, std {NOISE_STD} over 0-{FS_PHYS // 2} Hz")
    print("  => wanted content is below 1 Hz; the unwanted content sits at 0.05-0.1 Hz (below), at 8 Hz (above),")
    print("     and everywhere above 100 Hz.\n")

    print("STEP 2: choose the chain")
    print(f"  sample rate       {FS_ADC} Hz  (Nyquist {FS_ADC // 2} Hz)")
    print(f"      Everything we care about (0.05 - 8 Hz) is >12x below Nyquist, so the digital filter has room to work and the")
    print(f"      analog filter has a very wide transition band: images of the 0-10 Hz band come from {FS_ADC - 10}-{FS_ADC + 10} Hz,")
    print(f"      so the analog filter must pass 10 Hz and only reject above ~{FS_ADC - 10} Hz (a ratio of {(FS_ADC - 10) / 10:.0f}x).")
    print(f"  anti-alias cutoff {AA_CUTOFF:g} Hz, {AA_ORDER}th-order Butterworth: gain at 8 Hz = {abs(butterworth_lowpass(8.0, AA_CUTOFF, AA_ORDER)):.5f}, "
          f"at {FS_ADC - 10} Hz = {abs(butterworth_lowpass(FS_ADC - 10, AA_CUTOFF, AA_ORDER)):.5f} ({20 * np.log10(abs(butterworth_lowpass(FS_ADC - 10, AA_CUTOFF, AA_ORDER))):.0f} dB)")
    print(f"      (a modest analog filter: cheap, low delay; it needs only to protect the digital band from aliases)")
    print(f"  digital filter    band-pass = 2nd-order Butterworth high-pass {HP_CUTOFF:g} Hz + low-pass {LP_CUTOFF:g} Hz, run forward-backward (zero phase)")
    print(f"      high-pass {HP_CUTOFF:g} Hz: between the drift (0.1 Hz) and the motion (0.5 Hz); low-pass {LP_CUTOFF:g} Hz: between the motion (0.5 Hz) and the")
    print(f"      8 Hz vibration (geometric means are 0.22 and 2 Hz). Butterworth for a flat passband; forward-backward because the")
    print(f"      estimate is computed from a recorded stretch, so the phase delay can be cancelled.\n")

    # Build the pipeline variants.
    y_raw = x_phys[::FACTOR]                                  # no protection at all
    y_aa = analog_filter(x_phys)[::FACTOR]                    # analog filter + sampling only
    y_noaa = digital_filter(y_raw, zero_phase=True)           # digital filter, no anti-alias filter
    y_full = digital_filter(y_aa, zero_phase=True)            # the full chain
    y_causal = digital_filter(y_aa, zero_phase=False)         # full chain, causal digital filter

    def assess(y):
        """RMSE vs truth, plus its parts: 0.5 Hz amplitude, time shift, and contamination.

        Contamination is what is left after fitting away the 0.5 Hz component: aliased
        vibration, drift, noise. It isolates what the chain failed to reject, apart from
        the gain and delay that any filter imposes on the wanted motion.
        """
        amp, ph = fit_component(y, FS_ADC, 0.5)
        tt = np.arange(len(y)) / FS_ADC
        fit = amp * np.sin(2 * np.pi * 0.5 * tt + ph)
        shift_ms = -(((ph - ph_t) + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi * 0.5) * 1000
        return rmse(y, truth), amp, shift_ms, rmse(y, fit)

    amp_t, ph_t = fit_component(truth, FS_ADC, 0.5)
    print("STEP 3: check it, against the known true 0.5 Hz motion")
    print(f"{'pipeline':<58}{'RMSE':>7}{'0.5 Hz amp':>12}{'shift (ms)':>12}{'contamination':>15}")
    results = {}
    for name, y in [
        ("ADC only (no analog filter, no digital filter)", y_raw),
        ("analog filter + ADC (no digital filter)", y_aa),
        ("ADC + digital filter (NO analog filter)", y_noaa),
        ("analog filter + ADC + digital filter, zero phase (chosen)", y_full),
        ("analog filter + ADC + digital filter, causal", y_causal),
    ]:
        results[name] = assess(y)
        r = results[name]
        print(f"{name:<58}{r[0]:>7.3f}{r[1]:>12.3f}{r[2]:>12.1f}{r[3]:>15.3f}")
    print(f"  (true amplitude {amp_t:.3f}; shift: + = late, - = early; contamination = RMS left after fitting away the 0.5 Hz component)")
    r_full = results["analog filter + ADC + digital filter, zero phase (chosen)"]
    r_noaa = results["ADC + digital filter (NO analog filter)"]
    r_caus = results["analog filter + ADC + digital filter, causal"]
    delay = r_caus[2]
    print(f"\n  Without the analog filter the digital filter leaves contamination {r_noaa[3]:.2f}; with it, {r_full[3]:.2f}.")
    print(f"  The chosen chain reads the 0.5 Hz motion at {r_full[1]:.2f} (true {amp_t:.2f}): a {abs(1 - r_full[1] / amp_t) * 100:.1f}% low bias from the 0.2 Hz high-pass, "
          f"and {r_full[2]:.0f} ms late from the analog filter alone.")
    print(f"  A causal digital filter shifts the motion by {r_caus[2]:.0f} ms ({'early' if r_caus[2] < 0 else 'late'}); its RMSE ({r_caus[0]:.2f}) mostly measures that shift.")

    f_lo, a_noaa = amplitude_spectrum(y_raw, FS_ADC)
    _, a_aa = amplitude_spectrum(y_aa, FS_ADC)
    band = (f_lo >= 10) & (f_lo <= 90)
    print(f"\n  noise floor in 10-90 Hz after sampling (median amplitude): no analog filter {np.median(a_noaa[band]):.4f}, "
          f"with it {np.median(a_aa[band]):.5f}")
    print("  without the analog filter the structural vibration and all noise from 100-1000 Hz fold into 0-100 Hz;")
    print("  the part that lands on 0.2-2 Hz is in the digital passband and cannot be removed afterwards.")

    # Figure 1: spectra through the chain.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    ax.semilogy(f_p[f_p <= 1000], a_p[f_p <= 1000] + 1e-12, color="lightgray", linewidth=0.7, label="physical spectrum")
    ff = np.logspace(-2, np.log10(1000), 600)
    ax.semilogy(ff, np.abs(butterworth_lowpass(ff, AA_CUTOFF, AA_ORDER)), color="firebrick", linewidth=2, label=f"analog anti-alias |H| ({AA_ORDER}th order, {AA_CUTOFF:g} Hz)")
    ax.axvline(FS_ADC / 2, color="black", linestyle="--", label=f"ADC Nyquist {FS_ADC // 2} Hz")
    ax.axvspan(*STRUCT_BAND, color="darkorange", alpha=0.12, label="structural vibration")
    ax.set_xscale("log")
    ax.set_ylim(1e-4, 40)
    ax.set_xlim(0.03, 1000)
    ax.set_title("1. Physical signal and the analog anti-alias filter")
    ax.set_xlabel("frequency (Hz)")
    ax.legend(fontsize=7, loc="lower left")

    f_lo2, a_final = amplitude_spectrum(y_full, FS_ADC)
    for ax, a, title in [(axes[0, 1], a_noaa, "2. After sampling WITHOUT the analog filter: aliased vibration and noise"),
                         (axes[1, 0], a_aa, "3. After sampling WITH the analog filter")]:
        ax.semilogy(f_lo[1:], a[1:] + 1e-12, color="steelblue", linewidth=0.9)
        ax.set_ylim(1e-4, 40)
        ax.set_xlim(0, FS_ADC / 2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("amplitude")
    fd = np.logspace(-2, np.log10(FS_ADC / 2 - 1), 600)
    axes[1, 0].semilogy(fd, np.abs(digital_response(fd, True)), color="firebrick", linewidth=1.8, label="digital band-pass |H| (zero phase)")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlim(0.03, FS_ADC / 2)
    axes[1, 0].legend(fontsize=8, loc="lower left")
    axes[1, 0].annotate("drift", xy=(0.05, line(a_aa, f_lo, 0.05)), xytext=(0.04, 0.2), fontsize=8)
    axes[1, 0].annotate("motion 0.5", xy=(0.5, 20), xytext=(0.6, 25), fontsize=8)
    axes[1, 0].annotate("vibration 8", xy=(8, 3), xytext=(9, 5), fontsize=8)

    ax = axes[1, 1]
    m = f_lo2 <= 12
    ax.stem(f_lo2[m], a_final[m], basefmt=" ")
    ax.set_title("4. Final estimate: only the 0.5 Hz motion remains")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude")
    for ax in axes.flatten():
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "imu_chain_spectra.png"), dpi=150)

    # Figure 2: time domain and RMSE.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    show = t_adc <= 12
    for ax, y, title in [
        (axes[0, 0], y_raw, f"Raw ADC samples (RMSE {rmse(y_raw, truth):.2f})"),
        (axes[0, 1], y_full, f"Chosen chain, zero-phase digital filter (RMSE {rmse(y_full, truth):.2f})"),
        (axes[1, 0], y_causal, f"Same chain, causal digital filter (RMSE {rmse(y_causal, truth):.2f}, shifted {delay:.0f} ms)"),
    ]:
        ax.plot(t_adc[show], y[show], color="steelblue", linewidth=1.2, label="output")
        ax.plot(t_adc[show], truth[show], "k--", linewidth=1.2, label="true 0.5 Hz motion")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t (s)")
        ax.legend(fontsize=8, loc="upper right")
    labels = ["ADC only", "analog + ADC", "ADC + digital\n(no analog)", "full chain\nzero phase", "full chain\ncausal"]
    vals = [rmse(v, truth) for v in (y_raw, y_aa, y_noaa, y_full, y_causal)]
    axes[1, 1].bar(labels, vals, color=["gray", "gray", "firebrick", "seagreen", "darkorange"])
    for i, v in enumerate(vals):
        axes[1, 1].text(i, v + 0.15, f"{v:.2f}", ha="center")
    axes[1, 1].set_title("RMSE vs. the true desired motion")
    axes[1, 1].set_ylabel("RMSE")
    for ax in axes.flatten():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "imu_chain_time.png"), dpi=150)
    print("\nSaved imu_chain_spectra.png and imu_chain_time.png")

    print(
        f"""
What this means
---------------
Each decision is read off the spectrum:
  - The wanted motion (0.5 Hz) sits between the drift (0.05-0.1 Hz) and the
    vibration (8 Hz), so the digital band-pass corners go in those gaps, at
    {HP_CUTOFF:g} and {LP_CUTOFF:g} Hz. It works because the signal and its contaminants are separated in frequency.
    The price is visible: the high-pass droops the 0.5 Hz amplitude by about
    {abs(1 - r_full[1] / amp_t) * 100:.0f}%, and moving its corner up to reject more drift would droop it further.
  - Everything digital happens after sampling, so it can only work on what
    survived sampling. The 201 Hz motor tone folds to 1 Hz, inside the digital
    passband. No digital filter can remove it there, and the numbers show it: the
    same digital filter leaves contamination {r_noaa[3]:.2f} without the analog filter and
    {r_full[3]:.2f} with it, because the analog filter attenuates 201 Hz by a factor of about
    {1 / abs(butterworth_lowpass(201.0, AA_CUTOFF, AA_ORDER)):.0f} before the ADC ever sees it. (Broadband noise alone would matter much less:
    folded thinly into a narrow passband it adds little. Narrowband tones are the danger.)
  - The sample rate is chosen last, from the two filters: high enough that the
    analog filter can be gentle (the images of 0-10 Hz start at {FS_ADC - 10} Hz, so a 4th-order
    40 Hz filter is more than enough), low enough that the digital work is cheap.
  - Zero-phase digital filtering leaves only the analog filter's ~{r_full[2]:.0f} ms delay; a
    causal digital filter shifts the motion by {r_caus[2]:.0f} ms, acceptable only if the estimate
    feeds a real-time loop that can tolerate it.
Analog filter protects, sampling digitizes, digital filter selects: the order is
fixed by what can be undone after each step, and aliasing cannot be undone."""
    )
