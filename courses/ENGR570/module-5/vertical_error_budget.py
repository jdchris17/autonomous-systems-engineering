"""
vertical_error_budget.py - Space Systems Engineering, Module 5, Experiment D

Carry the IMU's measured vertical into celestial coordinates,

    u_G = C_GC C_CI u_I

and find out which error source dominates the result.

Parts
-----
1. Ideal calibration: the chain reproduces the true celestial vertical.
2. Fixed horizontal accelerometer bias, perfect extrinsics.
3. Camera-to-IMU misalignment of 0.01, 0.1 and 0.5 degrees.
4. Combined errors: first-order budget versus Monte Carlo, in which the
   camera attitude comes from Module 4's actual star-based SVD estimator.
5. Correlated errors: a fixed extrinsic error and fixed bias shared by many
   observations do not average away.

All results are in celestial coordinates. No Earth-fixed transformation or
position is involved yet (Module 6).

Run:  python vertical_error_budget.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gravity_estimator import (
    estimate_up_direction,
    gravity_to_camera,
    vertical_angular_error,
)
from imu_alignment import (
    GRAVITY,
    Transform,
    angular_rms,
    orientation_with_up,
    perturb_calibration,
    rotation_vector_to_matrix,
    specific_force,
    true_up_imu,
    vertical_covariance,
    vertical_in_celestial,
    vertical_jacobians,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module-4"))
from star_field_simulation import (  # noqa: E402
    PinholeCamera,
    centroid_sigma_to_angle,
    observe_with_centroid_noise,
    stars_in_field,
    to_celestial,
    true_attitude_matrix,
)
from stellar_attitude import (  # noqa: E402
    attitude_covariance,
    attitude_information,
    wahba_svd,
)

SEED = 555
ARCSEC_PER_RAD = np.degrees(1.0) * 3600.0
HERE = Path(__file__).parent

# ---- modelled scenario --------------------------------------------------------
N_SAMPLES = 1000                       # stationary accelerometer samples
ACCEL_NOISE = 0.005                    # m/s^2 per axis per sample
ACCEL_BIAS_SIGMA = 0.005               # m/s^2 per axis, residual after calibration
CAL_SIGMA_DEG = 0.05                   # camera-to-IMU alignment, per axis
N_STARS = 20
CENTROID_NOISE_PX = 0.2
BORESIGHT_ZENITH_ANGLE_DEG = 30.0      # camera boresight is 30 deg off vertical
TRIALS = 2000

CAMERA = PinholeCamera()
C_GC_TRUE = Transform("G", "C", true_attitude_matrix().T)
C_CI_TRUE = Transform("C", "I", rotation_vector_to_matrix([0.02, -0.01, 0.03]))


def geometry(zenith_angle_deg, azimuth_deg=40.0):
    """True vertical in camera and IMU axes for a boresight zenith angle."""
    z, az = np.radians(zenith_angle_deg), np.radians(azimuth_deg)
    up_camera = np.array([np.sin(z) * np.cos(az), np.sin(z) * np.sin(az),
                          np.cos(z)])
    up_imu = C_CI_TRUE.inverse().apply(up_camera)
    return up_camera, up_imu, orientation_with_up(up_imu, heading_deg=25.0)


UP_C, UP_I, C_IN = geometry(BORESIGHT_ZENITH_ANGLE_DEG)
UP_G_TRUE = vertical_in_celestial(UP_I, C_CI_TRUE, C_GC_TRUE)
F_TRUE = specific_force(C_IN)


def arcsec(angle_rad):
    return angle_rad * ARCSEC_PER_RAD


def horizontal_direction_imu():
    """A unit vector perpendicular to the vertical, in IMU axes."""
    return C_IN.apply([1.0, 0.0, 0.0])


# ------------------------------------------------------------------ parts 1 - 3
def deterministic_parts():
    print("=" * 78)
    print("1. Ideal calibration")
    print("=" * 78)
    samples = np.tile(F_TRUE, (10, 1))
    up_imu, _, _ = estimate_up_direction(samples)
    reconstructed = vertical_in_celestial(up_imu, C_CI_TRUE, C_GC_TRUE)
    print(f"  true vertical, celestial:          {np.round(UP_G_TRUE, 9)}")
    print(f"  reconstructed C_GC C_CI u_I:       {np.round(reconstructed, 9)}")
    print(f"  error: {arcsec(vertical_angular_error(reconstructed, UP_G_TRUE)):.1e} arcsec")
    via_camera = C_GC_TRUE.apply(gravity_to_camera(up_imu, C_CI_TRUE))
    print(f"  same result via gravity_to_camera: "
          f"{np.allclose(via_camera, reconstructed, atol=1e-15)}")

    print("\n" + "=" * 78)
    print("2. Fixed horizontal accelerometer bias (perfect extrinsics, no noise)")
    print("=" * 78)
    print(f"  {'bias [m/s^2]':>14}{'vertical error':>18}{'atan(b/g)':>14}")
    bias_rows = []
    for bias in (0.001, 0.005, 0.01, 0.02):
        samples = np.tile(F_TRUE + bias * horizontal_direction_imu(), (10, 1))
        up_imu, _, _ = estimate_up_direction(samples)
        error = arcsec(vertical_angular_error(
            vertical_in_celestial(up_imu, C_CI_TRUE, C_GC_TRUE), UP_G_TRUE))
        bias_rows.append((bias, error))
        print(f"  {bias:>14g}{error:>15.1f} as"
              f"{arcsec(np.arctan(bias / GRAVITY)):>11.1f} as")
    print("  A rotation chain preserves angles, so the error in celestial "
          "coordinates equals the error in\n  IMU coordinates: about 21 "
          "arcsec per 0.001 m/s^2 (roughly 0.1 milli-g) of horizontal bias.")

    print("\n" + "=" * 78)
    print("3. Camera-to-IMU misalignment (perfect accelerometer)")
    print("=" * 78)
    perpendicular = np.cross(UP_C, [0.0, 0.0, 1.0])
    perpendicular /= np.linalg.norm(perpendicular)
    axes = {
        "perpendicular to vertical": perpendicular,
        "camera x axis": np.array([1.0, 0.0, 0.0]),
        "camera boresight (z)": np.array([0.0, 0.0, 1.0]),
        "along the vertical": UP_C,
    }
    print(f"  {'misalignment':>13}" + "".join(f"{name:>28}" for name in axes))
    misalignment_rows = {}
    for magnitude_deg in (0.01, 0.1, 0.5):
        line = f"  {magnitude_deg:>9g} deg"
        for name, axis in axes.items():
            used = perturb_calibration(C_CI_TRUE, np.radians(magnitude_deg) * axis)
            error = arcsec(vertical_angular_error(
                vertical_in_celestial(UP_I, used, C_GC_TRUE), UP_G_TRUE))
            misalignment_rows[(magnitude_deg, name)] = error
            line += f"{error:>25.2f} as"
        print(line)
    print(f"  ({0.01 * 3600:g}, {0.1 * 3600:g} and {0.5 * 3600:g} arcsec of "
          f"misalignment.) The error is |misalignment| x sin(angle between\n"
          f"  its axis and the vertical): the full amount when perpendicular, "
          f"nothing when the camera is\n  mis-rotated about the vertical "
          f"itself. With the boresight {BORESIGHT_ZENITH_ANGLE_DEG:g} deg from "
          f"vertical, a boresight-roll\n  misalignment passes "
          f"sin({BORESIGHT_ZENITH_ANGLE_DEG:g} deg) = "
          f"{np.sin(np.radians(BORESIGHT_ZENITH_ANGLE_DEG)):.2f} of itself.")
    return bias_rows, misalignment_rows


# -------------------------------------------------------------- part 4: budget
def star_scene(rng):
    camera_vectors = stars_in_field(CAMERA, N_STARS, rng)
    sigma = centroid_sigma_to_angle(CAMERA, camera_vectors, CENTROID_NOISE_PX)
    return camera_vectors, to_celestial(camera_vectors), sigma


def monte_carlo(rng, scene, noise=False, bias=False, camera=False,
                calibration=False, trials=TRIALS):
    """Angular error [arcsec] of the celestial vertical, per trial."""
    camera_vectors, reference, star_sigma = scene
    errors = np.empty(trials)
    for t in range(trials):
        samples = np.tile(F_TRUE, (N_SAMPLES, 1))
        if noise:
            samples = samples + rng.normal(0.0, ACCEL_NOISE, samples.shape)
        if bias:
            samples = samples + rng.normal(0.0, ACCEL_BIAS_SIGMA, 3)
        up_imu, _, _ = estimate_up_direction(samples)

        c_ci = C_CI_TRUE
        if calibration:
            c_ci = perturb_calibration(
                C_CI_TRUE, rng.normal(0.0, np.radians(CAL_SIGMA_DEG), 3))

        c_gc = C_GC_TRUE
        if camera:
            measured = observe_with_centroid_noise(CAMERA, camera_vectors,
                                                   CENTROID_NOISE_PX, rng)
            c_cg_estimated = wahba_svd(reference, measured, 1.0 / star_sigma**2)
            c_gc = Transform("G", "C", c_cg_estimated.T)

        errors[t] = vertical_angular_error(
            vertical_in_celestial(up_imu, c_ci, c_gc), UP_G_TRUE)
    return errors * ARCSEC_PER_RAD


def budget(rng):
    print("\n" + "=" * 78)
    print(f"4. Combined errors: first-order budget versus Monte Carlo "
          f"({TRIALS} trials per row)")
    print("=" * 78)
    scene = star_scene(rng)
    camera_covariance = attitude_covariance(
        attitude_information(scene[0], scene[2]))
    camera_sigmas = np.sqrt(np.diag(camera_covariance)) * ARCSEC_PER_RAD

    jacobians = vertical_jacobians(UP_I, C_CI_TRUE, C_GC_TRUE)
    covariances = {
        "noise": (ACCEL_NOISE**2 / N_SAMPLES) * np.eye(3),
        "bias": ACCEL_BIAS_SIGMA**2 * np.eye(3),
        "camera": camera_covariance,
        "calibration": np.radians(CAL_SIGMA_DEG) ** 2 * np.eye(3),
    }
    j = {"noise": jacobians["accelerometer"], "bias": jacobians["accelerometer"],
         "camera": jacobians["camera"], "calibration": jacobians["calibration"]}
    total, parts = vertical_covariance(j, covariances)

    sources = [
        ("Accelerometer random noise", "noise",
         f"{ACCEL_NOISE} m/s^2 per sample, {N_SAMPLES} samples"),
        ("Accelerometer bias", "bias",
         f"{ACCEL_BIAS_SIGMA} m/s^2 per axis (1 sigma)"),
        ("Camera attitude", "camera",
         f"Module 4: {N_STARS} stars, {CENTROID_NOISE_PX} px "
         f"({camera_sigmas[0]:.1f}, {camera_sigmas[1]:.1f}, {camera_sigmas[2]:.1f} as)"),
        ("Camera-to-IMU alignment", "calibration",
         f"{CAL_SIGMA_DEG} deg per axis (1 sigma)"),
    ]
    print(f"Boresight {BORESIGHT_ZENITH_ANGLE_DEG:g} deg from the vertical. "
          f"RMS = root-mean-square angular error of the celestial vertical.\n")
    print(f"  {'Error source':<28}{'Assumed uncertainty':<52}"
          f"{'predicted':>10}{'Monte Carlo':>13}{'share':>8}")
    rows = []
    for label, key, assumed in sources:
        predicted = arcsec(angular_rms(parts[key]))
        mc = monte_carlo(rng, scene, **{key: True})
        mc_rms = np.sqrt(np.mean(mc**2))
        share = 100.0 * np.trace(parts[key]) / np.trace(total)
        rows.append((label, predicted, mc_rms, share))
        print(f"  {label:<28}{assumed:<52}{predicted:>7.1f} as"
              f"{mc_rms:>10.1f} as{share:>7.1f}%")
    combined_predicted = arcsec(angular_rms(total))
    combined = monte_carlo(rng, scene, noise=True, bias=True, camera=True,
                           calibration=True)
    combined_rms = np.sqrt(np.mean(combined**2))
    rows.append(("Combined", combined_predicted, combined_rms, 100.0))
    print(f"  {'Combined':<28}{'full modelled scenario':<52}"
          f"{combined_predicted:>7.1f} as{combined_rms:>10.1f} as{100.0:>7.1f}%")
    print(f"  Combined Monte Carlo: median {np.median(combined):.1f} as, "
          f"95th percentile {np.percentile(combined, 95):.1f} as.")

    ranked = sorted(rows[:-1], key=lambda r: -r[3])
    print(f"""
  What dominates: {ranked[0][0]} ({ranked[0][3]:.0f} % of the variance), then
  {ranked[1][0]} ({ranked[1][3]:.0f} %). {ranked[2][0]} and {ranked[3][0]}
  together are {ranked[2][3] + ranked[3][3]:.1f} %. Averaging more samples or measuring more stars
  would change the total by almost nothing; the two fixed quantities,
  the alignment and the accelerometer bias, set the result. The sum of
  independent contributions agrees with the full nonlinear Monte Carlo, so
  the first-order model is adequate at these error sizes.""")

    print("  Camera-attitude contribution versus where the camera points "
          "(same star field):")
    print(f"  {'boresight zenith angle':>24}{'vertical error from camera attitude':>40}")
    for zenith in (0.0, 30.0, 60.0, 90.0):
        up_c, up_i, _ = geometry(zenith)
        j_cam = vertical_jacobians(up_i, C_CI_TRUE, C_GC_TRUE)["camera"]
        value = arcsec(angular_rms(j_cam @ camera_covariance @ j_cam.T))
        print(f"  {zenith:>20g} deg{value:>37.1f} as")
    print(f"  The star camera's weak axis is roll about the boresight "
          f"({camera_sigmas[2]:.0f} as here). Roll cannot move a\n  vertical "
          f"that lies along the boresight, so a zenith-pointing camera feeds "
          f"only its accurate\n  pointing axes into the vertical; a "
          f"horizon-pointing one feeds in the full roll error.")
    return rows, jacobians, covariances


# ------------------------------------------------------- part 5: correlated errors
def correlated_errors(rng, jacobians, covariances):
    print("\n" + "=" * 78)
    print("5. Correlated errors: averaging K observations")
    print("=" * 78)
    print("  Each observation has fresh accelerometer noise and a fresh star "
          "attitude solution, but all K share\n  one accelerometer bias and "
          "one camera-to-IMU alignment error (fixed properties of the "
          "instrument).\n  First-order model, 4000 instruments per K.\n")
    j_a, j_cal, j_cam = (jacobians["accelerometer"], jacobians["calibration"],
                         jacobians["camera"])
    independent = (np.trace(j_a @ covariances["noise"] @ j_a.T)
                   + np.trace(j_cam @ covariances["camera"] @ j_cam.T))
    shared = (np.trace(j_a @ covariances["bias"] @ j_a.T)
              + np.trace(j_cal @ covariances["calibration"] @ j_cal.T))
    camera_cholesky = np.linalg.cholesky(covariances["camera"])

    counts = [1, 2, 5, 10, 20, 50, 100, 200]
    instruments = 4000
    measured, naive, correct = [], [], []
    print(f"  {'K':>5}{'Monte Carlo':>14}{'with correlation':>19}"
          f"{'if all independent':>21}")
    for k in counts:
        bias = rng.normal(0.0, ACCEL_BIAS_SIGMA, (instruments, 1, 3))
        cal = rng.normal(0.0, np.radians(CAL_SIGMA_DEG), (instruments, 1, 3))
        noise = rng.normal(0.0, ACCEL_NOISE / np.sqrt(N_SAMPLES),
                           (instruments, k, 3))
        cam = rng.normal(size=(instruments, k, 3)) @ camera_cholesky.T
        delta = ((noise + bias) @ j_a.T + cal @ j_cal.T + cam @ j_cam.T)
        mean_error = delta.mean(axis=1)                 # average of K estimates
        rms = arcsec(np.sqrt(np.mean(np.sum(mean_error**2, axis=1))))
        measured.append(rms)
        correct.append(arcsec(np.sqrt(independent / k + shared)))
        naive.append(arcsec(np.sqrt((independent + shared) / k)))
        print(f"  {k:>5}{rms:>11.1f} as{correct[-1]:>16.1f} as{naive[-1]:>18.1f} as")
    floor = arcsec(np.sqrt(shared))
    print(f"\n  Averaging only removes the part that differs between "
          f"observations. The shared part leaves a\n  floor of {floor:.0f} "
          f"arcsec however many observations are taken, while treating every "
          f"error as\n  independent predicts {naive[-1]:.0f} arcsec at K = "
          f"{counts[-1]}: optimistic by a factor of {measured[-1] / naive[-1]:.0f}. "
          f"The cross-covariance\n  between observations is what the "
          f"independent-error formula leaves out.")
    return counts, measured, correct, naive, floor


def make_figure(rows, bias_rows, misalignment_rows, correlated):
    fig, (ax_budget, ax_sweep, ax_corr) = plt.subplots(1, 3, figsize=(18, 5.4))

    labels = ["Accel.\nnoise", "Accel.\nbias", "Camera\nattitude",
              "Camera-IMU\nalignment", "Combined"]
    x = np.arange(len(rows))
    ax_budget.bar(x - 0.2, [r[1] for r in rows], 0.4, color="tab:blue",
                  label="first-order prediction")
    ax_budget.bar(x + 0.2, [r[2] for r in rows], 0.4, color="tab:orange",
                  label="Monte Carlo")
    for xi, r in zip(x, rows):
        ax_budget.text(xi, max(r[1], r[2]) * 1.12, f"{r[3]:.0f}%", ha="center",
                       fontsize=9)
    ax_budget.set_yscale("log")
    ax_budget.set_xticks(x)
    ax_budget.set_xticklabels(labels)
    ax_budget.set_ylabel("Vertical angular RMS error [arcsec]")
    ax_budget.set_title("Error budget of the celestial vertical\n"
                        "(labels: share of total variance)")
    ax_budget.grid(True, axis="y", which="both", alpha=0.3)
    ax_budget.legend(fontsize=8, loc="upper left")

    magnitudes = [0.01, 0.1, 0.5]
    for name, style in (("perpendicular to vertical", "o-"),
                        ("camera boresight (z)", "s-"),
                        ("along the vertical", "x:")):
        values = [max(misalignment_rows[(m, name)], 1e-3) for m in magnitudes]
        ax_sweep.loglog(np.array(magnitudes) * 3600, values, style,
                        label=f"misalignment about: {name}")
    bias_equiv = [arcsec(np.arctan(b / GRAVITY)) for b, _ in bias_rows]
    for (b, error), xpos in zip(bias_rows, bias_equiv):
        ax_sweep.axhline(error, color="gray", linestyle="--", linewidth=0.8)
        ax_sweep.text(2100, error * 1.08, f"bias {b:g} m/s²", fontsize=7.5,
                      ha="right", color="gray")
    ax_sweep.set_xlabel("Camera-to-IMU misalignment [arcsec]")
    ax_sweep.set_ylabel("Vertical error [arcsec]")
    ax_sweep.set_title("Single error sources\n(dashed: horizontal accelerometer bias)")
    ax_sweep.set_ylim(1, 5000)
    ax_sweep.grid(True, which="both", alpha=0.3)
    ax_sweep.legend(fontsize=7.5, loc="upper left")

    counts, measured, correct, naive, floor = correlated
    ax_corr.loglog(counts, measured, "o", color="tab:orange", markersize=7,
                   label="Monte Carlo")
    ax_corr.loglog(counts, correct, "-", color="tab:blue",
                   label="prediction with shared errors")
    ax_corr.loglog(counts, naive, "--", color="tab:red",
                   label="prediction if all errors were independent")
    ax_corr.axhline(floor, color="gray", linestyle=":",
                    label=f"floor from shared errors ({floor:.0f} as)")
    ax_corr.set_xlabel("Number of observations averaged, K")
    ax_corr.set_ylabel("Vertical angular RMS error [arcsec]")
    ax_corr.set_title("Correlated errors do not average down")
    ax_corr.grid(True, which="both", alpha=0.3)
    ax_corr.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(HERE / "vertical_error_budget.png", dpi=150)
    print("\nSaved vertical_error_budget.png")


if __name__ == "__main__":
    generator = np.random.default_rng(SEED)
    assert np.allclose(true_up_imu(C_IN), UP_I)
    bias_table, misalignment_table = deterministic_parts()
    budget_rows, J, P = budget(generator)
    correlated = correlated_errors(generator, J, P)
    make_figure(budget_rows, bias_table, misalignment_table, correlated)
