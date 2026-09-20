"""Stage 5: sample.  z_analog(t) -> anti-alias filter -> ADC -> z[n].

acquire() is reused by later stages so every stage works from the same digital stream.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import project_signal as P
import stage4_antialias as S4

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = S4.FS                              # sampling rate (Hz), decided in Stage 4
TS = 1.0 / FS                           # sampling period (s)
F_DESIRED_BANDWIDTH = S4.F_USE          # design useful band (Hz)
F_HIGHEST_DISTURBANCE = P.F_H           # highest tonal analog disturbance (Hz)
F_ANTIALIAS = S4.FC                     # anti-alias corner (Hz)


def acquire(seed=0):
    """Run the analog front end and the ADC. Returns the digital stream and its metadata."""
    c = P.build_analog(seed=seed)
    factor = P.FS_ANALOG // FS
    z_filtered = S4.analog_filter(c["z_analog"])
    z_n = z_filtered[::factor]                       # the ADC: keep every `factor`-th instant
    z_raw = c["z_analog"][::factor]                  # the same instants WITHOUT the anti-alias filter
    return {
        "z_n": z_n, "z_raw_n": z_raw, "x_true_n": c["x_true"][::factor], "t_n": c["t"][::factor],
        "n": np.arange(len(z_n)), "fs": FS, "Ts": TS, "components": c,
    }


if __name__ == "__main__":
    a = acquire()
    z_n, t_n = a["z_n"], a["t_n"]
    nyquist = FS / 2

    print("ACQUISITION ASSUMPTIONS")
    print(f"  Sampling rate:                       {FS:g} Hz")
    print(f"  Sampling period T_s = 1/f_s:         {TS * 1e3:g} ms")
    print(f"  Nyquist frequency:                   {nyquist:g} Hz")
    print(f"  Desired signal bandwidth:            {F_DESIRED_BANDWIDTH:g} Hz (design); the motion itself tops out at {P.F2:g} Hz")
    print(f"  Highest analog disturbance frequency: {F_HIGHEST_DISTURBANCE:g} Hz (interference); broadband noise extends to {P.FS_ANALOG // 2} Hz")
    print(f"  Anti-alias cutoff:                   {F_ANTIALIAS:g} Hz ({S4.ORDER}th-order Butterworth, analog-like, applied before the ADC)")

    print("\nCONSISTENCY CHECKS")
    checks = [
        (f"Nyquist ({nyquist:g}) > 2 x desired bandwidth ({2 * F_DESIRED_BANDWIDTH:g})", nyquist > 2 * F_DESIRED_BANDWIDTH),
        (f"desired bandwidth ({F_DESIRED_BANDWIDTH:g}) < anti-alias cutoff ({F_ANTIALIAS:g}) < Nyquist ({nyquist:g})", F_DESIRED_BANDWIDTH < F_ANTIALIAS < nyquist),
        (f"disturbance ({F_HIGHEST_DISTURBANCE:g} Hz) is above Nyquist, so it needs the analog filter", F_HIGHEST_DISTURBANCE > nyquist),
        (f"anti-alias filter attenuates the disturbance by {abs(S4.db(float(S4.gain(F_HIGHEST_DISTURBANCE)))):.0f} dB (>= 40 dB)", S4.db(float(S4.gain(F_HIGHEST_DISTURBANCE))) <= -40),
        (f"anti-alias filter loses < 0.1 dB at the desired bandwidth ({abs(S4.db(float(S4.gain(F_DESIRED_BANDWIDTH)))):.5f} dB)", abs(S4.db(float(S4.gain(F_DESIRED_BANDWIDTH)))) < 0.1),
        (f"record holds a whole number of every tone's periods ({P.DURATION:g} s x each f is an integer)",
         all(abs(P.DURATION * f - round(P.DURATION * f)) < 1e-9 for f in (P.F1, P.F2, P.F_V, P.F_B, P.F_H))),
    ]
    for text, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {text}")

    print("\nTHE DIGITAL STREAM z[n]")
    print(f"  samples: {len(z_n)}   duration: {len(z_n) * TS:g} s   dtype {z_n.dtype}")
    print(f"  z[n] range: {z_n.min():.2f} to {z_n.max():.2f}   mean {z_n.mean():.3f} (true bias {P.B0:g})   std {z_n.std():.3f}")
    print(f"  first samples: {np.array2string(z_n[:6], precision=3)}")
    print(f"  samples per cycle: {FS / P.F1:g} at 0.5 Hz, {FS / P.F2:g} at 2 Hz, {FS / P.F_V:g} at 15 Hz")
    print(f"  data rate: {len(z_n) / P.DURATION:g} samples/s vs {P.FS_ANALOG} for the analog grid ({P.FS_ANALOG // FS}x fewer)")

    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    c = a["components"]
    show = c["t"] <= 1.0
    axes[0].plot(c["t"][show], c["z_analog"][show], color="lightgray", linewidth=0.8, label=f"z_analog(t) (on a {P.FS_ANALOG} Hz grid)")
    zf = S4.analog_filter(c["z_analog"])
    axes[0].plot(c["t"][show], zf[show], color="steelblue", linewidth=1.5, label="after the analog anti-alias filter")
    m = t_n <= 1.0
    axes[0].plot(t_n[m], z_n[m], "o", color="firebrick", markersize=5, label=f"samples z[n], T_s = {TS * 1e3:g} ms")
    axes[0].set_title("The ADC: picking instants off the filtered analog waveform (first second)")
    axes[0].set_xlabel("t (s)")
    axes[0].legend(fontsize=8, loc="upper right")

    m = t_n <= 4.0
    axes[1].plot(a["n"][m], z_n[m], "o-", color="firebrick", markersize=3, linewidth=0.8, label="z[n]")
    axes[1].plot(a["n"][m], a["x_true_n"][m], "k--", linewidth=1.2, label="x_true at the same instants")
    axes[1].set_title("The digital stream, indexed by n (first 4 s): the motion is visible under the vibration, drift and noise")
    axes[1].set_xlabel("sample index n")
    axes[1].legend(fontsize=8, loc="upper right")

    axes[2].plot(a["n"], z_n, color="firebrick", linewidth=0.7, label="z[n]")
    axes[2].plot(a["n"], a["x_true_n"], "k", linewidth=0.7, alpha=0.6, label="x_true")
    axes[2].set_title(f"All {len(z_n)} samples: the slow drift moves the whole record")
    axes[2].set_xlabel("sample index n")
    axes[2].legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stage5_sample.png"), dpi=150)
    print("\nSaved stage5_sample.png")

    print(
        f"""
What this means
---------------
z[n] is now all the rest of the project has to work with: {len(z_n)} numbers spaced
T_s = {TS * 1e3:g} ms apart. Everything the analog world contained above {nyquist:g} Hz was either
removed before this point (by the {F_ANTIALIAS:g} Hz filter) or would have been folded into it
irreversibly. The printed assumptions are the contract for everything downstream:
  - the sampling rate ({FS:g} Hz) and its Nyquist limit ({nyquist:g} Hz) set the largest
    frequency the stream can represent,
  - the desired bandwidth ({F_DESIRED_BANDWIDTH:g} Hz) and the anti-alias cutoff ({F_ANTIALIAS:g} Hz) say what
    was meant to survive the front end,
  - the highest disturbance ({F_HIGHEST_DISTURBANCE:g} Hz) is what the analog filter had to defeat.
The checks are not decoration: if any of them fails, the later spectral analysis
would be measuring an artifact of the acquisition, not the signal."""
    )
