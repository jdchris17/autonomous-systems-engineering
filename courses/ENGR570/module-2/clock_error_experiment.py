"""
clock_error_experiment.py - Space Systems Engineering, Module 2, Experiment D

How wrong is an Earth-fixed direction if the timestamp is wrong?

Setup
-----
For each observer latitude (45 N, then the equator and 75 N) the fixed
celestial direction is the star at the observer's zenith at the correct
time, so its declination equals the observer's geodetic latitude. The
Earth-fixed direction of that star is computed at the correct time and at
timestamps offset by dt.

Two quantities are reported for each dt:
  * angular error: the angle between the correct and the mistimed
    Earth-fixed directions. For a pure rotation this is
    omega * |dt| * cos(declination).
  * equivalent east-west displacement: the mistimed model puts the star's
    sub-point at a longitude shifted by -omega * dt. A navigator who trusts
    the bad clock shifts their position fix by the same longitude, which at
    the observer's latitude is a ground distance of
    omega * dt * (distance from the rotation axis). Positive = east.

Simple model: rotation about z at the constant IAU rate only, isolating the
effect of an erroneous rotation angle.
Extension: the same experiment with the full IAU 2006/2000A + IERS
transformation from compare_reference_frames.py.

Run:  python clock_error_experiment.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module-1"))
from celestial_geometry import angular_separation, radec_to_unit  # noqa: E402

from astronomical_time import EARTH_ROTATION_RATE_UT1, time_scales  # noqa: E402
from compare_reference_frames import reference_matrix  # noqa: E402
from reference_frames import (  # noqa: E402
    celestial_to_earth_fixed,
    geodetic_to_ecef,
    rotation_z,
)

OBSERVATION_UTC = "2026-11-11T12:00:00"
OBSERVER_LONGITUDE_DEG = 0.0
LATITUDES_DEG = {"Equator (0°)": 0.0, "45°N": 45.0, "75°N": 75.0}
CLOCK_ERRORS_S = np.array([-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0])

ARCSEC_PER_DEG = 3600.0


def zenith_direction(latitude_deg, longitude_deg):
    """Earth-fixed unit vector of the geodetic vertical ("up")."""
    return radec_to_unit(longitude_deg, latitude_deg)


def axis_distance_m(latitude_deg):
    """Distance of a sea-level WGS84 observer from Earth's rotation axis."""
    x, y, _ = geodetic_to_ecef(latitude_deg, OBSERVER_LONGITUDE_DEG, 0.0)
    return float(np.hypot(x, y))


def longitude_shift_rad(correct, mistimed):
    """Signed change in Earth-fixed longitude of a direction, east positive."""
    shift = (np.arctan2(mistimed[1], mistimed[0])
             - np.arctan2(correct[1], correct[0]))
    return (shift + np.pi) % (2.0 * np.pi) - np.pi


def simple_experiment(latitude_deg, clock_errors_s):
    """Rotation-only model. Returns (angular error [arcsec], E-W shift [m])."""
    earth_angle = time_scales(OBSERVATION_UTC).era_rad
    correct = zenith_direction(latitude_deg, OBSERVER_LONGITUDE_DEG)
    # The fixed celestial direction: undo the rotation at the correct time.
    celestial = rotation_z(earth_angle) @ correct

    angles, shifts = [], []
    for dt in clock_errors_s:
        wrong_angle = earth_angle + EARTH_ROTATION_RATE_UT1 * dt
        mistimed = celestial_to_earth_fixed(celestial, wrong_angle)
        angles.append(angular_separation(correct, mistimed) * ARCSEC_PER_DEG)
        shifts.append(longitude_shift_rad(correct, mistimed)
                      * axis_distance_m(latitude_deg))
    return np.array(angles), np.array(shifts)


def reference_experiment(latitude_deg, clock_errors_s):
    """Same experiment with the full GCRS -> ITRS reference transformation."""
    t0 = Time(OBSERVATION_UTC, scale="utc")
    correct = zenith_direction(latitude_deg, OBSERVER_LONGITUDE_DEG)
    celestial = reference_matrix(t0).T @ correct

    angles, shifts = [], []
    for dt in clock_errors_s:
        mistimed = reference_matrix(t0 + TimeDelta(dt, format="sec")) @ celestial
        angles.append(angular_separation(correct, mistimed) * ARCSEC_PER_DEG)
        shifts.append(longitude_shift_rad(correct, mistimed)
                      * axis_distance_m(latitude_deg))
    return np.array(angles), np.array(shifts)


def main():
    rate_arcsec = np.degrees(EARTH_ROTATION_RATE_UT1) * ARCSEC_PER_DEG
    print(f"Earth rotation rate: {EARTH_ROTATION_RATE_UT1:.10e} rad/s "
          f"= {rate_arcsec:.4f} arcsec per second of time")
    print(f"Correct time: {OBSERVATION_UTC} UTC; star at each observer's "
          f"zenith (declination = latitude)\n")

    fig, (ax_angle, ax_shift) = plt.subplots(1, 2, figsize=(13, 5.2))
    worst_relative = 0.0

    for label, lat in LATITUDES_DEG.items():
        angle, shift = simple_experiment(lat, CLOCK_ERRORS_S)
        ref_angle, ref_shift = reference_experiment(lat, CLOCK_ERRORS_S)

        print(f"{label}: {axis_distance_m(lat) / 1000:.3f} km from the "
              f"rotation axis, {abs(shift[-2]):.2f} m of east-west error per "
              f"second of clock error")
        print(f"  {'dt [s]':>8}{'angle [arcsec]':>17}{'E-W shift [m]':>16}"
              f"{'ref angle':>14}{'ref shift [m]':>16}")
        for row in zip(CLOCK_ERRORS_S, angle, shift, ref_angle, ref_shift):
            print("  {:>8.1f}{:>17.5f}{:>16.4f}{:>14.5f}{:>16.4f}".format(*row))
        print()

        nonzero = CLOCK_ERRORS_S != 0.0
        worst_relative = max(worst_relative, np.max(
            np.abs(ref_shift[nonzero] / shift[nonzero] - 1.0)))

        ax_angle.plot(CLOCK_ERRORS_S, angle, "o-", linewidth=2, label=label)
        ax_shift.plot(CLOCK_ERRORS_S, shift, "o-", linewidth=2, label=label)
        ax_angle.plot(CLOCK_ERRORS_S, ref_angle, "k+", markersize=9)
        ax_shift.plot(CLOCK_ERRORS_S, ref_shift, "k+", markersize=9)

    print(f"Reference model versus simple rotation: east-west shifts differ "
          f"by at most {worst_relative:.1e} (relative).")
    print("The residual is polar motion: the true rotation axis is ~0.3 "
          "arcsec from the Earth-fixed z axis,\nso away from the equator a "
          "point's distance from the real axis differs slightly. Precession,"
          "\nnutation and UT1-UTC do not change over 10 s, so clock-error "
          "sensitivity is set by the rotation rate.")

    for ax in (ax_angle, ax_shift):
        # Symmetric-log axes: linear inside the smallest non-zero offset,
        # logarithmic outside, so 0.1 s and 10 s are both readable.
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xticks(CLOCK_ERRORS_S)
        ax.set_xticklabels([f"{v:g}" for v in CLOCK_ERRORS_S])
        ax.set_xlabel("Timestamp error Δt [s]")
        ax.grid(True, alpha=0.3)
    ax_angle.plot([], [], "k+", markersize=9, label="Reference model")
    ax_shift.plot([], [], "k+", markersize=9, label="Reference model")

    ax_angle.set_yscale("symlog", linthresh=1.0)
    ax_angle.set_yticks([0, 1, 10, 100])
    ax_angle.set_yticklabels(["0", "1", "10", "100"])
    ax_angle.set_ylabel("Angular error of Earth-fixed direction [arcsec]")
    ax_angle.set_title("Angular error (zenith star, dec = latitude)")
    ax_angle.legend(loc="upper center")

    ax_shift.set_yscale("symlog", linthresh=10.0)
    shift_ticks = [-1000, -100, -10, 0, 10, 100, 1000]
    ax_shift.set_yticks(shift_ticks)
    ax_shift.set_yticklabels([f"{v:g}" for v in shift_ticks])
    ax_shift.set_ylabel("Equivalent east-west displacement [m]  (east +)")
    ax_shift.set_title("East-west position error")
    ax_shift.legend(loc="upper right")

    fig.suptitle("Clock-error sensitivity: axes are symmetric-log, "
                 "both quantities are linear in Δt")
    fig.tight_layout()
    out = Path(__file__).with_name("clock_error_experiment.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out.name}")


if __name__ == "__main__":
    main()
