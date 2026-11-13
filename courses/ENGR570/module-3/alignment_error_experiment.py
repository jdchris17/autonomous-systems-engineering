"""
alignment_error_experiment.py - Space Systems Engineering, Module 3, Experiment D

How much does an imperfect camera-to-IMU alignment corrupt a direction?

Setup
-----
* Nominal calibration: camera and IMU axes coincide, C_IC = I.
* Truth: the camera is actually mounted with a small fixed misalignment
  d_theta, a ROTATION VECTOR (axis * angle), not an Euler sequence.
* Convention (right-multiplication, error expressed in camera axes):

      C_IC,true = C_IC,nominal @ R(d_theta)

  i.e. the real camera axes are the nominal camera axes actively rotated by
  d_theta. Software that still uses C_IC,nominal makes the error studied
  here.
* A unit vector is measured in the IMU frame, s_I. Its true camera-frame
  direction is C_CI,true s_I; the software computes C_CI,nominal s_I. The
  angle between those two is the angular measurement error.

To first order the error is  -d_theta x s : only the part of d_theta
perpendicular to the measured vector shows up. A vector cannot sense a
rotation about itself.

Run:  python alignment_error_experiment.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rotation_engine import (
    quaternion_to_matrix,
    rotation_matrix_angle,
    rotation_vector_to_matrix,
    rotation_vector_to_quaternion,
)
from sensor_frames import Transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module-1"))
from celestial_geometry import angular_separation  # noqa: E402

MISALIGNMENT_DEG = np.array([0.1, -0.2, 0.05])
SWEEP_MAGNITUDES_DEG = np.array([1.0, 0.1, 0.01, 0.001])
EARTH_MEAN_RADIUS_KM = 6371.0088
KM_PER_DEG = np.radians(1.0) * EARTH_MEAN_RADIUS_KM       # 111.19 km

NOMINAL = Transform.identity("I", "C")                    # C_IC, nominal


def true_calibration(misalignment_deg, nominal=NOMINAL):
    """C_IC,true = C_IC,nominal @ R(d_theta)   (right-multiplication)."""
    perturbation = rotation_vector_to_matrix(np.radians(misalignment_deg))
    return Transform("I", "C", nominal.matrix @ perturbation)


def measurement_error_deg(imu_vector, calibration_used, calibration_true):
    """Angle between the camera-frame direction the software computes and
    the true one, for a vector measured in the IMU frame."""
    computed = calibration_used.inverse().apply(imu_vector)      # C_CI s_I
    truth = calibration_true.inverse().apply(imu_vector)
    return angular_separation(computed, truth)


def orientation_error_deg(calibration_used, calibration_true):
    """Total rotation angle between two calibrations (all three axes)."""
    residual = calibration_used.inverse() @ calibration_true     # C_CI C_IC'
    return np.degrees(rotation_matrix_angle(residual.matrix))


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def vector_at_angle_from(axis, angle_deg):
    """A unit vector making the given angle with `axis`."""
    axis = unit(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perpendicular = unit(np.cross(axis, helper))
    a = np.radians(angle_deg)
    return np.cos(a) * axis + np.sin(a) * perpendicular


def main():
    magnitude = np.linalg.norm(MISALIGNMENT_DEG)
    axis = unit(MISALIGNMENT_DEG)

    # ---- A: ideal alignment ------------------------------------------------
    print("A - Ideal alignment (C_IC = identity)")
    s_imu = unit([0.2, -0.1, 0.97])
    s_cam = NOMINAL.inverse().apply(s_imu)
    print(f"  IMU-frame vector    {np.round(s_imu, 6)}")
    print(f"  camera-frame vector {np.round(s_cam, 6)}")
    print(f"  difference from expected: "
          f"{angular_separation(s_cam, s_imu):.1e} deg")
    # The same machinery with a non-trivial mount, to show it is not only
    # passing vectors through: camera turned +90 deg about the IMU z axis,
    # so the camera x axis lies along the IMU y axis.
    rotated_mount = Transform("I", "C", rotation_vector_to_matrix(
        [0.0, 0.0, np.pi / 2]))
    seen = rotated_mount.inverse().apply([0.0, 1.0, 0.0])
    print(f"  check with a 90 deg mount about z: IMU +y -> camera "
          f"{np.round(seen, 12) + 0.0} (expected [1, 0, 0])")

    # ---- B: fixed misalignment ---------------------------------------------
    truth = true_calibration(MISALIGNMENT_DEG)
    print(f"\nB - Fixed misalignment d_theta = {MISALIGNMENT_DEG} deg "
          f"(rotation vector)")
    print(f"  magnitude {magnitude:.5f} deg about axis {np.round(axis, 4)}")
    print(f"  orientation error of the nominal calibration: "
          f"{orientation_error_deg(NOMINAL, truth):.5f} deg")

    vectors = {
        "camera boresight +z": np.array([0.0, 0.0, 1.0]),
        "camera +x": np.array([1.0, 0.0, 0.0]),
        "camera +y": np.array([0.0, 1.0, 0.0]),
        "along misalignment axis": axis,
        "45 deg from axis": vector_at_angle_from(axis, 45.0),
        "perpendicular to axis": vector_at_angle_from(axis, 90.0),
    }
    print(f"\n  {'measured vector':<26}{'angle to axis':>14}{'error [deg]':>13}"
          f"{'|d_theta x s|':>15}{'fraction seen':>15}{'surface equiv':>15}")
    for label, s in vectors.items():
        error = measurement_error_deg(s, NOMINAL, truth)
        first_order = np.linalg.norm(np.cross(MISALIGNMENT_DEG, s))
        print(f"  {label:<26}{angular_separation(s, axis):>10.2f} deg"
              f"{error:>13.6f}{first_order:>15.6f}{error / magnitude:>15.3f}"
              f"{error * KM_PER_DEG:>12.2f} km")
    print("  The boresight sees only the x and y components: "
          f"sqrt(0.1^2 + 0.2^2) = {np.hypot(0.1, 0.2):.4f} deg. The 0.05 deg "
          "about z (roll about the boresight) is invisible to it.")

    # ---- C: error sweep ----------------------------------------------------
    print(f"\nC - Error sweep (same axis, magnitude varied; "
          f"1 deg of arc = {KM_PER_DEG:.2f} km on a spherical Earth)")
    sweep_angles = [90.0, 45.0, 15.0, 0.0]
    header = "".join(f"{f'{a:g} deg from axis':>20}" for a in sweep_angles)
    print(f"  {'misalignment':>13}{'surface equiv':>15}{header}")
    for mag in SWEEP_MAGNITUDES_DEG:
        cal = true_calibration(mag * axis)
        errors = [measurement_error_deg(vector_at_angle_from(axis, a),
                                        NOMINAL, cal) for a in sweep_angles]
        surface = mag * KM_PER_DEG
        surface_text = (f"{surface:.1f} km" if surface >= 1
                        else f"{surface * 1000:.0f} m")
        print(f"  {mag:>9g} deg{surface_text:>15}"
              + "".join(f"{e:>16.3e} deg" for e in errors))
    print("  The surface figures are arc lengths for the full misalignment "
          "angle, not predicted navigation errors.")

    # ---- D: calibration correction -----------------------------------------
    print("\nD - Calibration correction")
    # The corrected calibration is built through the quaternion code path,
    # independently of the matrix Rodrigues formula that defined the truth.
    q_correction = rotation_vector_to_quaternion(np.radians(MISALIGNMENT_DEG))
    corrected = Transform("I", "C",
                          NOMINAL.matrix @ quaternion_to_matrix(q_correction))
    worst = max(measurement_error_deg(s, corrected, truth)
                for s in vectors.values())
    print(f"  before: orientation error "
          f"{orientation_error_deg(NOMINAL, truth):.5f} deg")
    print(f"  after:  orientation error "
          f"{orientation_error_deg(corrected, truth):.1e} deg, "
          f"worst vector error {worst:.1e} deg")
    wrong_side = Transform("I", "C", rotation_vector_to_matrix(
        np.radians(-MISALIGNMENT_DEG)) @ NOMINAL.matrix)
    print(f"  applying the correction with the wrong sign leaves "
          f"{orientation_error_deg(wrong_side, truth):.4f} deg "
          f"(twice the original), so the convention matters.")

    # ---- figure -------------------------------------------------------------
    fig, (ax_sweep, ax_angle) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    magnitudes = np.logspace(-3, 0, 31)
    for angle_deg, color in zip([90.0, 45.0, 15.0, 5.0],
                                ["tab:blue", "tab:orange", "tab:green",
                                 "tab:purple"]):
        s = vector_at_angle_from(axis, angle_deg)
        errors = [measurement_error_deg(s, NOMINAL, true_calibration(m * axis))
                  for m in magnitudes]
        ax_sweep.loglog(magnitudes, errors, color=color, linewidth=2,
                        label=f"vector {angle_deg:g}° from misalignment axis")
    boresight_errors = [measurement_error_deg([0, 0, 1], NOMINAL,
                                              true_calibration(m * axis))
                        for m in magnitudes]
    ax_sweep.loglog(magnitudes, boresight_errors, "k--", linewidth=1.5,
                    label="camera boresight")
    ax_sweep.loglog(SWEEP_MAGNITUDES_DEG, SWEEP_MAGNITUDES_DEG, "ko",
                    markersize=6, label="required sweep points (worst case)")
    ax_sweep.set_xlabel("Misalignment magnitude [deg]")
    ax_sweep.set_ylabel("Angular measurement error [deg]")
    ax_sweep.set_title("Error vs misalignment magnitude\n"
                       "(a vector along the axis has zero error: off this plot)")
    ax_sweep.grid(True, which="both", alpha=0.3)
    ax_sweep.legend(fontsize=8.5, loc="upper left")
    surface_axis = ax_sweep.secondary_yaxis(
        "right", functions=(lambda d: d * KM_PER_DEG, lambda k: k / KM_PER_DEG))
    surface_axis.set_ylabel("Equivalent arc on a spherical Earth [km]")

    angles = np.linspace(0.0, 180.0, 181)
    errors = [measurement_error_deg(vector_at_angle_from(axis, a), NOMINAL, truth)
              for a in angles]
    ax_angle.plot(angles, errors, linewidth=2, color="tab:blue",
                  label="exact")
    ax_angle.plot(angles, magnitude * np.sin(np.radians(angles)), "k:",
                  label="|δθ| sin(angle)")
    ax_angle.axhline(magnitude, color="gray", linestyle="--", linewidth=1,
                     label=f"full misalignment {magnitude:.4f}°")
    boresight_angle = angular_separation([0, 0, 1], axis)
    ax_angle.plot(boresight_angle,
                  measurement_error_deg([0, 0, 1], NOMINAL, truth), "o",
                  color="tab:red", markersize=8, label="camera boresight")
    ax_angle.set_xlabel("Angle between measured vector and misalignment axis [deg]")
    ax_angle.set_ylabel("Angular measurement error [deg]")
    ax_angle.set_title("δθ = [0.1°, −0.2°, 0.05°]: what one vector can see")
    ax_angle.set_xlim(0, 180)
    ax_angle.set_ylim(0, magnitude * 1.25)
    ax_angle.grid(True, alpha=0.3)
    ax_angle.legend(fontsize=8.5, loc="lower center")

    fig.tight_layout()
    out = Path(__file__).with_name("alignment_error_experiment.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()
