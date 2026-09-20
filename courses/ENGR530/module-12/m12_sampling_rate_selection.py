"""Engineering challenge: choosing an ADC rate for a camera-stabilization gyro.

Motion to preserve: 0-25 Hz.  Structural vibration (unwanted): tones up to 300 Hz.
Candidate ADC rates: 100, 200, 500, 1000 Hz.  Not "25 x 2 = 50, so 100 works".
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
F_MOTION = 25.0
RATES = [100, 200, 500, 1000]
PASS_DROOP_DB = 1.0            # allowed gain loss at 25 Hz from the analog filter
STOP_ATTEN_DB = 60.0           # rejection needed at fs - 25 Hz: vibration is ~18 dB above the motion, plus ~40 dB (1%) of margin
ORDERS = [1, 2, 3, 4, 5, 6]    # analog anti-alias filter orders to try (each 2 orders ~ one op-amp biquad stage)
TOLERANCE = 0.01               # aliased error allowed, as a fraction of the motion RMS
VIB_BAND = (30.0, 300.0)       # structural vibration is broadband over this range
VIB_RMS = 10.0                 # about 18 dB above the motion (motion RMS is 1.22)
MOTION_TONES = [(1.0, 5.0), (1.0, 15.0), (1.0, 25.0)]                    # (amplitude, Hz)
EXAMPLE_TONES = [(1.0, 90.0), (1.0, 160.0), (1.0, 190.0), (1.0, 260.0), (1.0, 290.0)]
NOISE_STD = 0.3
FS_PHYS = 6000
DURATION = 5.0
DIGITAL_CUTOFF = 30.0          # ideal digital low-pass after the ADC, same for every rate


def min_order(fs, droop_db=PASS_DROOP_DB, atten_db=STOP_ATTEN_DB):
    """Smallest Butterworth order with <= droop_db at 25 Hz and >= atten_db at fs - 25 Hz."""
    f_stop = fs - F_MOTION
    num = np.log((10 ** (atten_db / 10) - 1) / (10 ** (droop_db / 10) - 1))
    return int(np.ceil(num / (2 * np.log(f_stop / F_MOTION))))


def corner_for_droop(order, droop_db=PASS_DROOP_DB):
    """Butterworth corner that gives exactly droop_db of loss at 25 Hz."""
    return F_MOTION / (10 ** (droop_db / 10) - 1) ** (1 / (2 * order))


def butter_gain(f, fc, order):
    return 1 / np.sqrt(1 + (np.abs(f) / fc) ** (2 * order))


def butter_response(f, fc, order):
    s = 1j * 2 * np.pi * np.asarray(f, dtype=float)
    wc = 2 * np.pi * fc
    k = np.arange(1, order + 1)
    poles = wc * np.exp(1j * np.pi * (2 * k + order - 1) / (2 * order))
    H = np.ones_like(s, dtype=complex)
    for p in poles:
        H = H * (-p) / (s - p)
    return H


def fold(f, fs):
    """Frequency a tone of frequency f appears at after sampling at fs."""
    return abs(f - fs * np.round(f / fs))


def make_signal(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS_PHYS * DURATION)) / FS_PHYS
    motion = sum(a * np.sin(2 * np.pi * f * t) for a, f in MOTION_TONES)
    W = np.fft.rfft(rng.standard_normal(len(t)))
    f = np.fft.rfftfreq(len(t), 1 / FS_PHYS)
    vib = np.fft.irfft(W * ((f >= VIB_BAND[0]) & (f <= VIB_BAND[1])), n=len(t))
    vib *= VIB_RMS / vib.std()
    noise = rng.normal(0.0, NOISE_STD, size=len(t))
    return t, motion, vib, noise


def acquire(x, fs, fc, order):
    """analog anti-alias filter -> ADC at fs -> ideal digital low-pass (zero phase)."""
    f = np.fft.fftfreq(len(x), 1 / FS_PHYS)
    y = np.fft.ifft(butter_response(f, fc, order) * np.fft.fft(x)).real[:: FS_PHYS // fs]
    fd = np.fft.fftfreq(len(y), 1 / fs)
    return np.fft.ifft(np.where(np.abs(fd) <= DIGITAL_CUTOFF, 1.0, 0.0) * np.fft.fft(y)).real


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def contamination(y, fs):
    """RMS left after fitting away the three motion tones: aliased vibration and noise only."""
    tt = np.arange(len(y)) / fs
    cols = []
    for _, f in MOTION_TONES:
        cols += [np.sin(2 * np.pi * f * tt), np.cos(2 * np.pi * f * tt)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return rmse(y, A @ coef)


if __name__ == "__main__":
    print("QUESTION 1: what does the Nyquist inequality alone tell us?")
    print(f"  25 Hz x 2 = 50 Hz, so 100 Hz 'works' only for a perfect brick-wall filter, which does not exist.\n")

    print(f"QUESTION 2: where do the unwanted vibration tones land after sampling? (0-{F_MOTION:g} Hz is the wanted band)")
    print(f"{'ADC rate':>9}{'Nyquist':>9}" + "".join(f"{f'{f:g} Hz ->':>10}" for _, f in EXAMPLE_TONES) + "   tones landing in the wanted band")
    for fs in RATES:
        landed = [fold(f, fs) for _, f in EXAMPLE_TONES]
        bad = [f for (_, f), l in zip(EXAMPLE_TONES, landed) if l <= F_MOTION]
        print(f"{fs:>9}{fs / 2:>9g}" + "".join(f"{l:>10.0f}" for l in landed) + f"   {', '.join(f'{b:g}' for b in bad) if bad else 'none'}")

    print(f"\nQUESTION 3: what analog anti-alias filter does each rate demand?")
    print(f"  Must keep 25 Hz within {PASS_DROOP_DB:g} dB and reject {STOP_ATTEN_DB:g} dB at fs - 25 Hz, the lowest frequency that folds into 0-25 Hz.")
    print(f"{'ADC rate':>9}{'transition band':>18}{'ratio':>7}{'Butterworth order needed':>26}{'digital samples/s':>19}   (filter order from the 60 dB spec)")
    orders = {}
    for fs in RATES:
        n = min_order(fs)
        orders[fs] = n
        print(f"{fs:>9}{f'25 - {fs - 25:g} Hz':>18}{(fs - 25) / 25:>7.1f}{n:>26d}{fs:>19}")

    t, motion, vib, noise = make_signal()
    x = motion + vib + noise
    motion_rms = rmse(motion, 0 * motion)
    print(f"\nQUESTION 4: measure it. Broadband vibration {VIB_BAND[0]:g}-{VIB_BAND[1]:g} Hz, RMS {VIB_RMS:g} ({20 * np.log10(VIB_RMS / motion_rms):.0f} dB above the motion), "
          f"noise {NOISE_STD}.")
    print(f"  For each rate and analog filter order (corner set for {PASS_DROOP_DB:g} dB loss at 25 Hz), residual = RMS left after fitting away the motion,")
    print(f"  then ALIASING = residual minus the same filter's residual at 1000 Hz (where nothing folds), as % of the motion RMS ({motion_rms:.3f}).")
    resid = {}
    for order in ORDERS:
        fc = corner_for_droop(order)
        for fs in RATES:
            resid[(fs, order)] = contamination(acquire(x, fs, fc, order), fs)
    alias = {(fs, o): (resid[(fs, o)] - resid[(1000, o)]) / motion_rms * 100 for fs in RATES for o in ORDERS}
    print(f"\n{'aliased error (% of motion RMS)':>34}" + "".join(f"{f'order {o}':>10}" for o in ORDERS))
    for fs in RATES:
        print(f"{f'ADC {fs} Hz':>34}" + "".join(f"{max(alias[(fs, o)], 0):>10.2f}" for o in ORDERS))

    delay_ms = {o: 1e3 / (2 * np.pi * corner_for_droop(o)) / np.sin(np.pi / (2 * o)) for o in ORDERS}
    need = {fs: next((o for o in ORDERS if alias[(fs, o)] <= TOLERANCE * 100), None) for fs in RATES}
    print(f"\nSmallest analog order meeting the {TOLERANCE * 100:g}% tolerance, and what it costs:")
    print(f"{'ADC rate':>9}{'order':>7}{'analog delay (ms)':>19}{'digital samples/s':>19}")
    for fs in RATES:
        o = need[fs]
        print(f"{fs:>9}{(str(o) if o else '>6'):>7}{(f'{delay_ms[o]:.1f}' if o else '-'):>19}{fs:>19}")
    print("  (delay = low-frequency group delay of the analog filter, which sits inside the stabilization loop)")

    fs_pick = 500
    print(f"\nCHOICE: {fs_pick} Hz")

    # Figure.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = ["firebrick", "darkorange", "seagreen", "steelblue"]
    ax = axes[0, 0]
    f_scan = np.linspace(0, 320, 3201)
    for fs, c in zip(RATES, colors):
        ax.plot(f_scan, [fold(f, fs) for f in f_scan], color=c, linewidth=1.3, label=f"fs = {fs} Hz")
    ax.axhspan(0, F_MOTION, color="gray", alpha=0.18, label="wanted band 0-25 Hz")
    ax.axvspan(*VIB_BAND, color="black", alpha=0.06, label="vibration band")
    ax.set_ylim(0, 130)
    ax.set_xlabel("true frequency (Hz)")
    ax.set_ylabel("apparent frequency after sampling (Hz)")
    ax.set_title("Where vibration lands: any curve inside the grey band is contamination")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 1]
    ff = np.logspace(0.5, 3.1, 600)
    for order, ls in [(2, "-"), (4, "--")]:
        ax.semilogx(ff, 20 * np.log10(butter_gain(ff, corner_for_droop(order), order)), color="black", linestyle=ls, linewidth=1.8,
                    label=f"order {order}, corner {corner_for_droop(order):.1f} Hz")
    for fs, c in zip(RATES, colors):
        ax.axvspan(fs - 25, fs + 25, color=c, alpha=0.18)
        ax.axvline(fs - 25, color=c, linestyle="--", linewidth=1)
        ax.text(fs - 25, -108, f"{fs}", color=c, ha="right", fontsize=9)
    ax.axvspan(1, 25, color="gray", alpha=0.18)
    ax.axhline(-STOP_ATTEN_DB, color="gray", linestyle=":", label=f"-{STOP_ATTEN_DB:g} dB target")
    ax.set_ylim(-115, 5)
    ax.set_xlim(3, 1300)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("gain (dB)")
    ax.set_title("Coloured bands fold into 0-25 Hz at that rate: the filter must be small there")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1, 0]
    im = np.array([[max(alias[(fs, o)], 1e-3) for o in ORDERS] for fs in RATES])
    img = ax.imshow(np.log10(im), cmap="RdYlGn_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(ORDERS)))
    ax.set_xticklabels([str(o) for o in ORDERS])
    ax.set_yticks(range(len(RATES)))
    ax.set_yticklabels([str(fs) for fs in RATES])
    for i, fs in enumerate(RATES):
        for j, o in enumerate(ORDERS):
            ax.text(j, i, f"{max(alias[(fs, o)], 0):.2f}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("analog anti-alias filter order")
    ax.set_ylabel("ADC rate (Hz)")
    ax.set_title(f"Aliased error, % of motion RMS (green = under {TOLERANCE * 100:g}%)")

    ax = axes[1, 1]
    for fs, c in zip(RATES, colors):
        os_ = [o for o in ORDERS]
        ax.plot(os_, [max(alias[(fs, o)], 1e-3) for o in os_], "o-", color=c, label=f"{fs} Hz")
    ax.axhline(TOLERANCE * 100, color="black", linestyle=":", label=f"{TOLERANCE * 100:g}% tolerance")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 300)
    ax.set_xlabel("analog filter order")
    ax.set_ylabel("aliased error (% of motion RMS)")
    ax2 = ax.twinx()
    ax2.plot(ORDERS, [delay_ms[o] for o in ORDERS], "k--", alpha=0.5)
    ax2.set_ylabel("analog filter delay (ms, dashed)")
    ax.set_title("More rate lets you use a lower order, which means less delay")
    ax.legend(fontsize=8, loc="upper right")
    for ax in axes.flatten():
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sampling_rate_selection.png"), dpi=150)
    print("\nSaved sampling_rate_selection.png")

    o100, o200, o500, o1000 = (need[fs] for fs in RATES)
    print(
        f"""
What this means
---------------
Nyquist says the rate must exceed 50 Hz. That is necessary and nowhere near
sufficient, because a real analog filter cannot pass 25 Hz and stop everything
above 50 Hz. The rate is a system decision, made from four things:
  - Desired bandwidth (0-25 Hz) sets the passband. What must be kept out is
    everything that folds INTO it, which is everything at or above fs - 25 Hz, not
    just above fs/2.
  - The anti-alias transition band is 25 Hz to fs - 25 Hz. At 100 Hz that is
    25-75 Hz (a ratio of 3); at 500 Hz it is 25-475 Hz (a ratio of 19). Rejecting
    60 dB takes order {orders[100]} at 100 Hz but only order {orders[500]} at 500 Hz.
  - The unwanted vibration decides how much is at stake. It is broadband up to
    300 Hz and about 18 dB above the motion, so at 100 Hz all of it folds into
    0-50 Hz, and at 200 Hz its 175-225 Hz part lands on the motion. At 500 Hz it
    folds only to 200-250 Hz, outside the wanted band, where the digital filter
    removes it. At 1000 Hz nothing folds.
  - Practical realizability is measured, not assumed. Aliased error (excess over
    the 1000 Hz reference) at a 1% tolerance needs analog order {o100} at 100 Hz, {o200} at
    200 Hz, {o500} at 500 Hz and {o1000} at 1000 Hz. A 2nd-order filter at 100 Hz leaves
    {max(alias[(100, 2)], 0):.0f}% error; at 500 Hz it leaves {max(alias[(500, 2)], 0):.2f}%.
The cost of a steep analog filter is not only parts count. It sits in the
stabilization loop, and its delay grows with order ({delay_ms[2]:.1f} ms at order 2,
{delay_ms[4]:.1f} ms at order 4, {delay_ms[6]:.1f} ms at order 6), on top of tighter component tolerances.
Choose {fs_pick} Hz. The vibration stops at 300 Hz and folds only to 200-250 Hz, so
even a 1st-order filter passes the 1% test here. I would still build order 2
({delay_ms[2]:.1f} ms): the model has no content above 300 Hz, but real electronics add
noise and interference there, and the worst-case spec (order {orders[500]} for 60 dB at 475
Hz) is what protects against it. Compare 100 Hz, which forces order {o100} and
{delay_ms[o100]:.0f} ms of delay; 200 Hz works at order {o200} ({delay_ms[o200]:.0f} ms) with little margin; and 1000 Hz
adds nothing measurable (identical error at the same order) while doubling the
data rate, processing and ADC power. That is the judgment the inequality alone
cannot give."""
    )
