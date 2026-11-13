"""
motion_contamination_experiment.py - Space Systems Engineering, Module 5, Experiment C

When does the accelerometer stop being a reliable vertical sensor?

A perfectly calibrated IMU at a fixed, known orientation is given horizontal
accelerations a_N = [a_E, a_N, 0]. It measures f_I = C_IN (a_N - g_N) plus
noise. Local vertical is estimated with trailing averages of several
lengths, and each estimate's TRUE error is compared with the uncertainty
the estimator itself reports.

Scenarios
---------
A  stationary
B  constant 0.05 m/s^2 east for the whole record
C  one 0.5 m/s^2 pulse lasting 0.5 s
D  1 Hz oscillation, 0.2 m/s^2 amplitude
E  slow platform motion: 0.02 Hz, 0.1 m/s^2 amplitude

Then the quality indicator is run on every 1 s window, and scenario E is
compared with a genuine slow tilt that produces almost identical
accelerometer data.

Run:  python motion_contamination_experiment.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gravity_estimator import (
    QUALITY_CAVEAT,
    assess_measurement_quality,
    sliding_up_direction,
)
from imu_alignment import GRAVITY, imu_orientation, specific_force, true_up_imu

SEED = 55
SAMPLE_RATE_HZ = 100.0
DURATION_S = 60.0
NOISE_SIGMA = 0.005              # m/s^2 per axis per sample
GYRO_NOISE_SIGMA = 5e-4          # rad/s per axis per sample
WINDOWS_S = [0.1, 1.0, 10.0]
QUALITY_WINDOW_S = 1.0
ARCSEC_PER_RAD = np.degrees(1.0) * 3600.0

C_IN = imu_orientation(8.0, 30.0, 20.0)
UP_TRUE = true_up_imu(C_IN)
TIME = np.arange(0.0, DURATION_S, 1.0 / SAMPLE_RATE_HZ)


def east_acceleration(scenario):
    """Eastward acceleration a_E(t) in m/s^2 for each scenario."""
    t = TIME
    if scenario == "A":
        return np.zeros_like(t)
    if scenario == "B":
        return np.full_like(t, 0.05)
    if scenario == "C":
        return np.where((t >= 30.0) & (t < 30.5), 0.5, 0.0)
    if scenario == "D":
        return 0.2 * np.sin(2.0 * np.pi * 1.0 * t)
    if scenario == "E":
        return 0.1 * np.sin(2.0 * np.pi * 0.02 * t)
    raise ValueError(scenario)


SCENARIOS = {
    "A": "A - stationary",
    "B": "B - constant 0.05 m/s² east",
    "C": "C - 0.5 m/s² pulse for 0.5 s",
    "D": "D - 1 Hz oscillation, 0.2 m/s²",
    "E": "E - slow motion, 0.02 Hz, 0.1 m/s²",
}


def simulate(scenario, rng):
    """Accelerometer samples (T, 3) in IMU axes for a scenario."""
    acceleration_n = np.column_stack((east_acceleration(scenario),
                                      np.zeros_like(TIME), np.zeros_like(TIME)))
    ideal = specific_force(C_IN, acceleration_n)
    return ideal + rng.normal(0.0, NOISE_SIGMA, ideal.shape)


def angle_to(up_estimates, truth):
    """Angle [arcsec] between each row of up_estimates and truth (rows or one)."""
    truth = np.broadcast_to(truth, up_estimates.shape)
    cross = np.linalg.norm(np.cross(up_estimates, truth), axis=1)
    dot = np.einsum("ij,ij->i", up_estimates, truth)
    return np.arctan2(cross, dot) * ARCSEC_PER_RAD


def quality_over_record(samples, gyro=None):
    """Quality report for every consecutive 1 s window."""
    n = int(QUALITY_WINDOW_S * SAMPLE_RATE_HZ)
    reports, previous = [], None
    for start in range(0, len(samples) - n + 1, n):
        window = samples[start:start + n]
        reports.append(assess_measurement_quality(
            window, NOISE_SIGMA, previous_mean=previous,
            gyro_samples=None if gyro is None else gyro[start:start + n],
            gyro_noise_sigma=GYRO_NOISE_SIGMA))
        previous = window.mean(axis=0)
    return reports


def verdict_shares(reports):
    names = ["steady", "disturbed", "changing", "accelerating", "rotating"]
    return {name: 100.0 * sum(r.verdict == name for r in reports) / len(reports)
            for name in names}


def main():
    rng = np.random.default_rng(SEED)
    data = {key: simulate(key, rng) for key in SCENARIOS}
    print(f"{SAMPLE_RATE_HZ:g} Hz for {DURATION_S:g} s, noise {NOISE_SIGMA} "
          f"m/s^2 per axis, IMU tilted 8 deg, perfectly calibrated.")
    print("First-order rule: horizontal acceleration a tilts the apparent "
          "vertical by atan(a / g):")
    for a in (0.05, 0.1, 0.2, 0.5):
        print(f"   {a:>5} m/s^2 -> {np.degrees(np.arctan(a / GRAVITY)):.3f} deg "
              f"= {np.arctan(a / GRAVITY) * ARCSEC_PER_RAD:,.0f} arcsec")

    # ---- averaging-window study ---------------------------------------------
    results = {}
    print(f"\n  {'scenario':<36}{'window':>8}{'RMS true':>11}{'max true':>11}"
          f"{'reported 1s':>13}{'true/reported':>15}   [arcsec]")
    for key, label in SCENARIOS.items():
        for window_s in WINDOWS_S:
            window = int(round(window_s * SAMPLE_RATE_HZ))
            up, reported, _ = sliding_up_direction(data[key], window)
            true_error = angle_to(up, UP_TRUE)
            reported = reported * ARCSEC_PER_RAD
            results[(key, window_s)] = (TIME[window - 1:], true_error, reported)
            rms_true = np.sqrt(np.mean(true_error**2))
            rms_reported = np.sqrt(np.mean(reported**2))
            print(f"  {label:<36}{window_s:>6g} s{rms_true:>11.1f}"
                  f"{true_error.max():>11.1f}{rms_reported:>13.1f}"
                  f"{rms_true / rms_reported:>15.1f}")
        print()

    print("Does longer averaging help?")
    print("  A  yes: error falls as 1/sqrt(window) and matches the reported "
          "uncertainty.")
    print("  B  no: the error sits at atan(a/g) for every window. Only the "
          "reported uncertainty shrinks,\n     so a longer window is more "
          "confidently wrong.")
    print("  C  the pulse is diluted by the ratio pulse length / window, but "
          "it contaminates every\n     window that contains it, so a long "
          "window is wrong by less for longer.")
    print("  D  a window that spans whole 1 Hz cycles (1 s, 10 s) cancels "
          "the oscillation; the 0.1 s\n     window follows it. A window "
          "that is not a whole number of cycles leaves a residual.")
    print("  E  no window here is long compared with the 50 s period, so "
          "the estimate simply follows\n     the acceleration. Averaging "
          "cannot separate it from a real change in tilt.")

    # ---- validity assessment --------------------------------------------------
    print("\n" + "=" * 78)
    print(f"Quality indicator on consecutive {QUALITY_WINDOW_S:g} s windows "
          f"(accelerometer only)")
    print("=" * 78)
    print(f"  {'scenario':<36}{'steady':>8}{'disturbed':>11}{'changing':>10}"
          f"{'mean true error':>18}")
    n = int(QUALITY_WINDOW_S * SAMPLE_RATE_HZ)
    flags = {}
    for key, label in SCENARIOS.items():
        reports = quality_over_record(data[key])
        flags[key] = np.array([r.detectable_disturbance for r in reports])
        shares = verdict_shares(reports)
        means = data[key][:len(reports) * n].reshape(len(reports), n, 3).mean(axis=1)
        ups = means / np.linalg.norm(means, axis=1, keepdims=True)
        print(f"  {label:<36}{shares['steady']:>7.0f}%{shares['disturbed']:>10.0f}%"
              f"{shares['changing']:>9.0f}%"
              f"{np.mean(angle_to(ups, UP_TRUE)):>15.0f} as")
    print("""
  Detectable failures: C (the pulse raises the variance of its window and
  shifts the next mean) and D at 1 s (the oscillation is inside the window,
  so the variance check fires). E is flagged as "changing": the mean moves
  between windows, but an accelerometer alone cannot say why.
  NOT detectable: B. Its windows are all "steady" while the vertical is
  wrong by about 1050 arcsec. The magnitude check does not help: 0.05 m/s^2
  horizontal changes |f| by a^2/2g = 0.00013 m/s^2, far below any usable
  tolerance on g.""")
    print("  Caveat attached to every report:")
    print("   ", QUALITY_CAVEAT)

    # ---- slow acceleration versus genuine slow tilt ---------------------------
    print("\n" + "=" * 78)
    print("Ambiguity check: slow acceleration (E) versus a genuine slow tilt (F)")
    print("=" * 78)
    a_east = east_acceleration("E")
    ideal_e = specific_force(C_IN, np.column_stack((a_east, 0 * TIME, 0 * TIME)))
    # F: the instrument really tilts so that gravity alone has the direction
    # E's specific force had. Same direction, magnitude exactly g.
    ideal_f = GRAVITY * ideal_e / np.linalg.norm(ideal_e, axis=1, keepdims=True)
    noise = rng.normal(0.0, NOISE_SIGMA, ideal_e.shape)
    accel_e, accel_f = ideal_e + noise, ideal_f + noise
    # Gyro: E does not rotate. F rotates at d(theta)/dt about the horizontal
    # axis perpendicular to the acceleration (north, in IMU axes).
    theta_rate = np.gradient(np.arctan(a_east / GRAVITY), TIME)
    north_imu = C_IN.apply([0.0, 1.0, 0.0])
    gyro_noise = rng.normal(0.0, GYRO_NOISE_SIGMA, ideal_e.shape)
    gyro_e = gyro_noise
    gyro_f = theta_rate[:, None] * north_imu + gyro_noise

    print(f"  Largest difference between the two accelerometer signals: "
          f"{np.max(np.linalg.norm(ideal_e - ideal_f, axis=1)):.5f} m/s^2 "
          f"(noise is {NOISE_SIGMA})")
    up_e, _, _ = sliding_up_direction(accel_e, n)
    up_f, _, _ = sliding_up_direction(accel_f, n)
    # In F the true vertical really does move; compare each window's
    # estimate with the true vertical averaged over that same window.
    true_up_f, _, _ = sliding_up_direction(ideal_f, n)
    print(f"  True vertical error, 1 s windows:  E (acceleration) RMS "
          f"{np.sqrt(np.mean(angle_to(up_e, UP_TRUE) ** 2)):.0f} arcsec,   "
          f"F (real tilt) RMS {np.sqrt(np.mean(angle_to(up_f, true_up_f) ** 2)):.0f} arcsec")
    print(f"\n  {'case':<34}{'steady':>8}{'changing':>10}{'accelerating':>14}"
          f"{'rotating':>10}")
    for label, accel, gyro in (
            ("E, accelerometer only", accel_e, None),
            ("F, accelerometer only", accel_f, None),
            ("E, with gyro", accel_e, gyro_e),
            ("F, with gyro", accel_f, gyro_f)):
        s = verdict_shares(quality_over_record(accel, gyro))
        print(f"  {label:<34}{s['steady']:>7.0f}%{s['changing']:>9.0f}%"
              f"{s['accelerating']:>13.0f}%{s['rotating']:>9.0f}%")
    print("""
  Without a gyro the two cases get the same verdicts, yet in E the vertical
  estimate is wrong and in F it is right. With a gyro they separate: no
  rotation while the mean moves means translational acceleration. A gyro
  still cannot expose scenario B, where nothing changes at all; that needs
  an outside reference for the motion itself, such as a known-stationary
  mount or an independent velocity measurement.""")

    # ---- figure -----------------------------------------------------------------
    fig, axes = plt.subplots(len(SCENARIOS), len(WINDOWS_S), figsize=(16, 14),
                             sharex=True, sharey=True)
    for row, (key, label) in enumerate(SCENARIOS.items()):
        for col, window_s in enumerate(WINDOWS_S):
            ax = axes[row, col]
            t, true_error, reported = results[(key, window_s)]
            flagged_time = np.flatnonzero(flags[key]) * QUALITY_WINDOW_S
            for start in flagged_time:
                ax.axvspan(start, start + QUALITY_WINDOW_S, color="gold",
                           alpha=0.35, linewidth=0)
            ax.semilogy(t, np.maximum(true_error, 1e-2), color="tab:red",
                        linewidth=1.0, label="true vertical error")
            ax.semilogy(t, reported, color="tab:blue", linewidth=1.5,
                        linestyle="--", label="estimator's reported 1σ")
            ax.grid(True, which="major", alpha=0.3)
            if row == 0:
                ax.set_title(f"Averaging window {window_s:g} s")
            if col == 0:
                ax.set_ylabel(f"{label}\n\nerror [arcsec]", fontsize=9)
            if row == len(SCENARIOS) - 1:
                ax.set_xlabel("Time [s]")
    axes[0, 0].set_ylim(0.3, 3e4)
    axes[0, 0].set_xlim(0, DURATION_S)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color="gold", alpha=0.35))
    labels.append("quality indicator flags this 1 s window")
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Motion contamination: true vertical error versus the "
                 "uncertainty the estimator reports", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(__file__).with_name("motion_contamination_experiment.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()
