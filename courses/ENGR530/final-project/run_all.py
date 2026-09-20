"""Run the whole final project in the recommended order, stopping at the first failure.

    python run_all.py            # everything (about 3 minutes)
    python run_all.py --quick    # smaller Monte Carlo counts (about 1 minute)
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = "--quick" in sys.argv

STEPS = [
    ("stage3_spectrum.py", [], "spectrum of x_true and z_analog: what is in the signal"),
    ("stage4_antialias.py", [], "choose fs = 100 Hz and the analog anti-alias filter"),
    ("stage5_sample.py", [], "sample: z_analog(t) -> z[n], print the acquisition assumptions"),
    ("stage6_analyze.py", [], "FFT of z[n]: find the peaks, compare with the truth"),
    ("stage7_filter.py", [], "design and apply the digital band-pass: z[n] -> x_hat[n]"),
    ("stage8_estimate.py", [], "estimate the waveform and A1, A2, f1, f2"),
    ("stage9_error.py", [], "error e[n]: bias, variance, RMSE, SNR, error budget"),
    ("stage30_monte_carlo.py", ["200"] if QUICK else [], "Monte Carlo the whole pipeline: distribution of RMSE"),
    ("stage34_correlated_noise.py", ["100"] if QUICK else [], "white vs. correlated noise through the same filters"),
]

if __name__ == "__main__":
    total = time.time()
    for name, args, what in STEPS:
        t0 = time.time()
        print(f"\n===== {name} {' '.join(args)} : {what}")
        proc = subprocess.run([sys.executable, os.path.join(HERE, name), *args], cwd=HERE, capture_output=True, text=True)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(proc.stdout[-1500:])
            print(proc.stderr[-2500:])
            print(f"FAILED after {dt:.0f} s: {name}")
            sys.exit(1)
        lines = [l for l in proc.stdout.splitlines() if l.startswith("Saved")]
        print(f"  ok in {dt:.0f} s   {'; '.join(lines)}")
    print(f"\nAll {len(STEPS)} stages finished in {time.time() - total:.0f} s")
