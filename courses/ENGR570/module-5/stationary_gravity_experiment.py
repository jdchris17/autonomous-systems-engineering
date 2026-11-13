"""
stationary_gravity_experiment.py - Space Systems Engineering, Module 5

Experiment A: simulated stationary IMU, vertical error before and after
              calibration, across several instrument orientations.
Experiment B: Monte Carlo check of the gravity estimator's predicted
              uncertainty against its actual error.

Run:  python stationary_gravity_experiment.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from accelerometer_calibration import (
    AccelerometerErrorModel,
    estimate_calibration,
    simulate_stationary,
    six_position_orientations,
)
from gravity_estimator import (
    calibrate_accelerometer,
    estimate_up_direction,
    vertical_angular_error,
)
from imu_alignment import GRAVITY, imu_orientation, specific_force, true_up_imu

SEED = 5
HERE = Path(__file__).parent
ARCSEC_PER_RAD = np.degrees(1.0) * 3600.0

ORIENTATIONS = {
    "level": imu_orientation(0.0),
    "tilt 10° about E": imu_orientation(10.0, 0.0),
    "tilt 10° about N": imu_orientation(10.0, 90.0),
    "tilt 30°, az 45°": imu_orientation(30.0, 45.0, 20.0),
    "tilt 60°, az 200°": imu_orientation(60.0, 200.0, -35.0),
    "on its side (90°)": imu_orientation(90.0, 0.0),
    "tilt 135°": imu_orientation(135.0, 120.0, 60.0),
    "inverted (180°)": imu_orientation(180.0, 0.0),
}

SAMPLE_COUNTS = [10, 100, 1000, 10000]
NOISE_LEVELS = [0.001, 0.005, 0.02]
HORIZONTAL_BIASES = [0.0, 0.001, 0.01]
TRIALS = 400
CHI2_95_2DOF = 5.991


# ------------------------------------------------------------------ Experiment A
def experiment_a(rng):
    print("=" * 78)
    print("Experiment A - stationary IMU, before and after calibration")
    print("=" * 78)
    model = AccelerometerErrorModel()
    n_samples = 1000
    print(f"f_N = [0, 0, {GRAVITY}] m/s^2 (specific force at rest points UP)")
    print(f"bias {model.bias_vector} m/s^2, scale-factor error 0.1 %, "
          f"cross-axis 0.05 %, noise {model.noise_sigma} m/s^2, "
          f"{n_samples} samples")

    # A calibration estimated from data, as a contrast to the known one:
    # six static poses, 1000 samples each, same sensor.
    poses = six_position_orientations()
    pose_truth = np.array([specific_force(p) for p in poses])
    pose_means = np.array([simulate_stationary(p, model, n_samples, rng).mean(axis=0)
                           for p in poses])
    est_bias, est_correction = estimate_calibration(pose_means, pose_truth)
    print(f"Six-position estimate of the bias: {np.round(est_bias, 5)} m/s^2")

    rows = []
    print(f"\n  {'orientation':<20}{'raw':>12}{'known cal':>12}"
          f"{'6-pos cal':>12}{'predicted 1s':>14}   [arcsec]")
    for label, c_in in ORIENTATIONS.items():
        truth = true_up_imu(c_in)
        raw = simulate_stationary(c_in, model, n_samples, rng)
        up_raw, _, _ = estimate_up_direction(raw)
        known = calibrate_accelerometer(raw, model.bias_vector,
                                        model.correction_matrix)
        up_known, uncertainty, _ = estimate_up_direction(known)
        estimated = calibrate_accelerometer(raw, est_bias, est_correction)
        up_estimated, _, _ = estimate_up_direction(estimated)
        errors = [vertical_angular_error(u, truth) * ARCSEC_PER_RAD
                  for u in (up_raw, up_known, up_estimated)]
        rows.append((label, *errors, uncertainty.angular_rms * ARCSEC_PER_RAD))
        print(f"  {label:<20}{errors[0]:>12.1f}{errors[1]:>12.2f}"
              f"{errors[2]:>12.2f}{rows[-1][4]:>14.2f}")

    raw_errors = np.array([r[1] for r in rows])
    cal_errors = np.array([r[2] for r in rows])
    print(f"\nUncalibrated error ranges from {raw_errors.min():.0f} to "
          f"{raw_errors.max():.0f} arcsec depending on orientation: only the "
          f"part of the bias\nand scale error PERPENDICULAR to the vertical "
          f"tilts it, and that part changes as the sensor turns.")
    print(f"After calibration with known parameters the error is "
          f"{cal_errors.mean():.1f} arcsec on average, the noise floor "
          f"sigma*sqrt(2)/(g*sqrt(N)) =\n"
          f"{model.noise_sigma * np.sqrt(2) / GRAVITY / np.sqrt(n_samples) * ARCSEC_PER_RAD:.1f} arcsec.")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rows))
    width = 0.27
    ax.bar(x - width, raw_errors, width, color="tab:red", label="uncalibrated")
    ax.bar(x, cal_errors, width, color="tab:blue",
           label="calibrated, known parameters")
    ax.bar(x + width, [r[3] for r in rows], width, color="tab:green",
           label="calibrated, six-position estimate")
    ax.plot(x, [r[4] for r in rows], "k_", markersize=22,
            label="predicted 1σ (noise only)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=25, ha="right")
    ax.set_ylabel("Angular vertical error [arcsec]")
    ax.set_title(f"Vertical error before and after calibration "
                 f"({n_samples} stationary samples)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "stationary_gravity_calibration.png", dpi=150)
    plt.close(fig)
    print("Saved stationary_gravity_calibration.png")


# ------------------------------------------------------------------ Experiment B
def experiment_b(rng):
    print("\n" + "=" * 78)
    print(f"Experiment B - estimator error versus predicted uncertainty "
          f"({TRIALS} trials each)")
    print("=" * 78)
    c_in = imu_orientation(20.0, 50.0, 15.0)
    truth = true_up_imu(c_in)
    ideal = specific_force(c_in)
    east_imu = c_in.apply([1.0, 0.0, 0.0])        # a horizontal direction
    print("Sensor otherwise perfectly calibrated; the bias is along a "
          "horizontal direction and is NOT known\nto the estimator. "
          "'cover' = share of trials whose true error lies inside the "
          "predicted 95 % ellipse.\n")
    print(f"  {'bias':>7}{'noise':>7}{'N':>7}{'RMS error':>11}{'predicted':>11}"
          f"{'ratio':>8}{'cover':>7}{'bias floor':>12}   [arcsec]")

    results = {}
    for bias in HORIZONTAL_BIASES:
        floor = np.arctan(bias / GRAVITY) * ARCSEC_PER_RAD
        for sigma in NOISE_LEVELS:
            for n in SAMPLE_COUNTS:
                errors, predicted, inside = [], [], 0
                for _ in range(TRIALS):
                    samples = ideal + bias * east_imu + rng.normal(0, sigma, (n, 3))
                    up, uncertainty, _ = estimate_up_direction(samples)
                    errors.append(vertical_angular_error(up, truth))
                    predicted.append(uncertainty.angular_rms)
                    offset = uncertainty.tangent_basis @ (up - truth)
                    distance = offset @ np.linalg.solve(
                        uncertainty.tangent_covariance, offset)
                    inside += distance < CHI2_95_2DOF
                rms = np.sqrt(np.mean(np.square(errors))) * ARCSEC_PER_RAD
                pred = np.sqrt(np.mean(np.square(predicted))) * ARCSEC_PER_RAD
                results[(bias, sigma, n)] = (rms, pred, floor)
                print(f"  {bias:>7g}{sigma:>7g}{n:>7d}{rms:>11.2f}{pred:>11.2f}"
                      f"{rms / pred:>8.2f}{100 * inside / TRIALS:>6.0f}%"
                      f"{floor:>12.2f}")
        print()

    print("With no bias the ratio is 1 and coverage is about 95 % (a little "
          "low at N = 10, where the\nsample covariance itself is poorly "
          "known): the uncertainty model is valid for random noise.")
    print("With a bias the error stops falling at atan(bias / g) while the "
          "predicted uncertainty keeps\nshrinking as 1/sqrt(N). More "
          "samples make the estimator more confident and no more correct:\n"
          "the reported uncertainty covers noise only.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = {0.0: "tab:blue", 0.001: "tab:orange", 0.01: "tab:red"}
    for ax, sigma in zip(axes, NOISE_LEVELS):
        for bias in HORIZONTAL_BIASES:
            rms = [results[(bias, sigma, n)][0] for n in SAMPLE_COUNTS]
            ax.loglog(SAMPLE_COUNTS, rms, "o-", color=colors[bias], linewidth=2,
                      label=f"actual RMS error, bias {bias:g} m/s²")
            if bias > 0:
                ax.axhline(results[(bias, sigma, 10)][2], color=colors[bias],
                           linestyle=":", linewidth=1)
        ax.loglog(SAMPLE_COUNTS,
                  [results[(0.0, sigma, n)][1] for n in SAMPLE_COUNTS], "k--",
                  linewidth=1.5, label="predicted from sample covariance")
        ax.set_title(f"Noise {sigma:g} m/s² per axis")
        ax.set_xlabel("Number of stationary samples")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Vertical direction error [arcsec]")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Gravity-direction error versus predicted uncertainty "
                 "(dotted lines: bias floor atan(b/g))")
    fig.tight_layout()
    fig.savefig(HERE / "stationary_gravity_uncertainty.png", dpi=150)
    plt.close(fig)
    print("Saved stationary_gravity_uncertainty.png")


if __name__ == "__main__":
    generator = np.random.default_rng(SEED)
    experiment_a(generator)
    experiment_b(generator)
