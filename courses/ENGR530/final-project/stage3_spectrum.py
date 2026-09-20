"""Stage 3: inspect the spectrum BEFORE sampling.

Compute the spectra of x_true(t) and z_analog(t), identify what is in each, and let
the physical bandwidth (not a guess) drive the sampling decision.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import project_signal as P

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_THRESHOLD = 8.0          # a spectral line must stand this many times above the noise floor


def find_lines(f, amp, floor, k=LINE_THRESHOLD):
    """Bins that exceed k x the noise floor and are local maxima: the spectral lines."""
    idx = [i for i in range(1, len(amp) - 1)
           if amp[i] > k * floor and amp[i] >= amp[i - 1] and amp[i] >= amp[i + 1]]
    if amp[0] > k * floor:                 # DC is its own kind of line, so it need not beat its neighbor
        idx.insert(0, 0)
    return [(f[i], amp[i]) for i in idx]


def role_of(freq):
    """The truth table, used only to check what we identified from the spectrum."""
    for name, f0 in [("bias (DC)", 0.0), ("desired: primary motion", P.F1), ("desired: maneuvering", P.F2),
                     ("vibration", P.F_V), ("HF interference", P.F_H), ("drift", P.F_B)]:
        if abs(freq - f0) < (0.02 if f0 == 0.0 else 0.03):
            return name
    return "UNEXPECTED"


if __name__ == "__main__":
    c = P.build_analog()
    N = len(c["t"])
    f_x, a_x = P.amplitude_spectrum(c["x_true"], P.FS_ANALOG)
    f_z, a_z = P.amplitude_spectrum(c["z_analog"], P.FS_ANALOG)

    floor_band = (f_z > 200) & (f_z < 900)
    floor = float(np.median(a_z[floor_band]))
    floor_predicted = P.SIGMA_N * np.sqrt(2 * np.log(2)) * np.sqrt(2 / N)
    print(f"Analog record: {N} samples at {P.FS_ANALOG} Hz over {P.DURATION:g} s (bins {f_z[1]:.2f} Hz apart, 0 to {P.FS_ANALOG // 2} Hz)\n")

    print("SPECTRUM OF x_true(t): the desired motion alone")
    lines_x = find_lines(f_x, a_x, floor)
    for fr, am in lines_x:
        print(f"  {fr:>8.2f} Hz   amplitude {am:>7.3f}   {role_of(fr)}")
    print(f"  -> exactly two lines: everything else about the truth is empty. Highest desired frequency: {max(fr for fr, _ in lines_x):g} Hz\n")

    print("SPECTRUM OF z_analog(t): what the sensor actually produces")
    lines_z = find_lines(f_z, a_z, floor)
    print(f"  {'freq (Hz)':>10}{'amplitude':>11}   identified as")
    for fr, am in lines_z:
        print(f"  {fr:>10.2f}{am:>11.3f}   {role_of(fr)}")
    unexpected = [fr for fr, _ in lines_z if role_of(fr) == "UNEXPECTED"]
    print(f"  {'unexpected lines: ' + (str(unexpected) if unexpected else 'none')}")
    print(f"  noise floor (median bin amplitude, 200-900 Hz): {floor:.4f}   (predicted for sigma = {P.SIGMA_N:g}: {floor_predicted:.4f})")
    print(f"  the floor is flat: 0-5 Hz {np.median(a_z[(f_z > 0.5) & (f_z < 5) & (np.abs(f_z - 2) > 0.2)]):.4f}, "
          f"20-60 Hz {np.median(a_z[(f_z > 20) & (f_z < 60)]):.4f}, 400-900 Hz {floor:.4f}  -> white, extends to {P.FS_ANALOG // 2} Hz")

    total = np.mean(c["z_analog"] ** 2)
    parts = {"desired motion": np.mean(c["x_true"] ** 2), "vibration (15 Hz)": np.mean(c["vib"] ** 2),
             "HF interference (80 Hz)": np.mean(c["hf"] ** 2), "bias + drift": np.mean(c["bias"] ** 2),
             "noise (all 0-1000 Hz)": np.mean(c["noise"] ** 2)}
    print(f"\nWhere the power is (mean square, total {total:.1f}):")
    for name, v in parts.items():
        print(f"  {name:<26}{v:>8.2f}   {100 * v / total:>5.1f}%")
    print(f"  noise power inside 0-5 Hz alone: {P.SIGMA_N ** 2 * 5 / (P.FS_ANALOG / 2):.4f}  (the noise is spread thinly over {P.FS_ANALOG // 2} Hz)")

    f_use = 5.0
    print("\nWHAT THE SPECTRUM SAYS ABOUT THE ACQUISITION DESIGN")
    print(f"  useful content        : 0 to {max(fr for fr, _ in lines_x):g} Hz (motion), plus bias and drift below 0.1 Hz. Design useful band: 0-{f_use:g} Hz (margin above 2 Hz)")
    print(f"  in-band contamination : bias/drift at 0-0.05 Hz, vibration at {P.F_V:g} Hz. Both are below Nyquist for any rate above {2 * P.F_V:g} Hz, so a DIGITAL filter can deal with them")
    print(f"  out-of-band energy    : interference at {P.F_H:g} Hz and noise out to {P.FS_ANALOG // 2} Hz. Whatever lies above Nyquist folds down and cannot be removed after sampling")
    print(f"  So the sample rate must (1) exceed 2 x the vibration ({2 * P.F_V:g} Hz) to keep it identifiable and removable, and (2) put Nyquist far enough")
    print(f"  above {f_use:g} Hz that an analog filter can cut what lies above it. {100} Hz satisfies both (Nyquist 50 Hz, 10x the useful band): see Stage 4.")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    zoom = f_z <= 25
    for ax, (f, a, title, c_) in zip(axes[0], [(f_x, a_x, "x_true(t): two lines, nothing else", "steelblue"),
                                                  (f_z, a_z, "z_analog(t): the same view of the sensor output", "firebrick")]):
        ax.stem(f[f <= 25], a[f <= 25], basefmt=" ", linefmt=c_, markerfmt="o")
        ax.set_xlim(-0.5, 25)
        ax.set_ylim(0, 11)
        ax.set_title(title + "  (0-25 Hz)")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("amplitude")
    axes[0, 1].text(0.97, 0.95, "0 Hz: bias (1)\n0.05 Hz: drift (3)\n0.5 Hz: primary motion (10)\n2 Hz: maneuvering (4)\n15 Hz: vibration (3)",
                    transform=axes[0, 1].transAxes, ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="gray"))

    for ax, (f, a, title, c_) in zip(axes[1], [(f_x, a_x, "x_true: full 0-1000 Hz", "steelblue"), (f_z, a_z, "z_analog: full 0-1000 Hz", "firebrick")]):
        ax.semilogy(f[1:], a[1:] + 1e-9, color=c_, linewidth=0.7)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(0, P.FS_ANALOG / 2)
        ax.set_ylim(1e-4, 30)
        ax.set_title(title)
        ax.set_xlabel("frequency (Hz, log above 1 Hz)")
        ax.set_ylabel("amplitude")
    ax = axes[1, 1]
    ax.axhline(floor, color="black", linestyle="--", linewidth=1, label=f"noise floor {floor:.4f}")
    ax.annotate("80 Hz interference", xy=(P.F_H, P.A_H), xytext=(150, 3), arrowprops=dict(arrowstyle="->"))
    ax.annotate("15 Hz vibration", xy=(P.F_V, P.A_V), xytext=(22, 8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("broadband noise\nout to 1000 Hz", xy=(500, floor), xytext=(300, 0.05), arrowprops=dict(arrowstyle="->"))
    ax.axvline(50, color="gray", linestyle=":", label="Nyquist of a 100 Hz ADC")
    ax.legend(fontsize=8, loc="lower left")
    for ax in axes.flatten():
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stage3_spectrum.png"), dpi=150)
    print("\nSaved stage3_spectrum.png")

    print(
        """
What this means
---------------
The truth has two spectral lines and the measurement has six things in it. Read
straight from the spectrum, with no need to look at the time trace:
  - 0.5 Hz and 2 Hz (amplitudes 10 and 4): the desired motion, all of it below 5 Hz.
  - 15 Hz (3): the vibration. Far from the motion, so easily separated.
  - 80 Hz (2): high-frequency interference, well above anything a 100 Hz ADC can
    represent (Nyquist 50 Hz).
  - DC (1) and 0.05 Hz (3): bias and drift, at the very bottom of the spectrum.
  - a flat floor all the way to 1000 Hz: white noise. Its per-bin amplitude is tiny,
    but there are hundreds of bins of it, and everything above the ADC's Nyquist
    frequency will fold onto the ones that matter.
The order matters: this inspection comes BEFORE choosing the sample rate. The
physical bandwidth (motion below 5 Hz, contaminants at 0.05, 15 and 80 Hz, noise
to 1000 Hz) is what makes 100 Hz a reasoned choice rather than a guess, and it is
what tells us the 80 Hz line and the high-frequency noise need an analog filter
ahead of the ADC."""
    )
