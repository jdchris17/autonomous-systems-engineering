"""
global_validation.py - Space Systems Engineering, Module 6

Experiment A: perfect position recovery at 45 N, 75 W, step by step.
Experiment B: the same noise-free pipeline over a global grid, with the
              boundary cases: longitude wrapping, the southern hemisphere,
              the poles, and different instrument orientations.

Run:  python global_validation.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import celestial_position as cp
import position_simulator as ps
from stellar_attitude import attitude_error

SEED = 6
HERE = Path(__file__).parent
UTC = ps.OBSERVATION_UTC

GRID_LATITUDES = [0.0, 30.0, -30.0, 60.0, -60.0, 80.0, -80.0, 89.9, -89.9]
GRID_LONGITUDES = [-180.0, -90.0, 0.0, 90.0, 179.0]
ORIENTATIONS = {"near zenith (2°)": 2.0, "30° zenith angle": 30.0,
                "near horizon (85°)": 85.0}


def experiment_a(rng, c_eg, eop):
    print("=" * 78)
    print("Experiment A - perfect terrestrial position recovery")
    print("=" * 78)
    lat_true, lon_true = 45.0, -75.0
    obs = ps.simulate_observation(lat_true, lon_true, UTC, C_EG=c_eg, rng=rng)
    t = obs.truth
    print(f"Truth: latitude {lat_true} deg, longitude {lon_true} deg, {UTC} UTC")
    print(f"Earth orientation: {eop['earth_orientation_source']}, "
          f"UT1-UTC = {eop['ut1_minus_utc_s']:+.4f} s")
    print(f"Step 1  true vertical, Earth-fixed  u_E = {np.round(t['up_ecef'], 9)}")
    print(f"Step 2  celestial  u_G = C_EG^T u_E     = {np.round(t['up_celestial'], 9)}")
    print(f"Step 3  {len(obs.reference_stars)} synthetic stars and "
          f"{len(obs.accel_samples)} accelerometer samples, generated from the "
          f"same attitude;\n        specific force (IMU axes) = "
          f"{np.round(t['specific_force_imu'], 6)} m/s^2")

    solution = ps.solve_observation(obs)          # measurements only
    recovered_attitude_error = attitude_error(
        solution.diagnostics["C_CG"], t["C_CG"])[1] * np.degrees(1.0) * 3600
    print("Step 4  estimator output (no true coordinates supplied):")
    print(f"        camera attitude recovered to {recovered_attitude_error:.1e} arcsec")
    print(f"        u_E estimated                    = {np.round(solution.up_ecef, 9)}")
    print(f"        latitude  {solution.latitude_deg:.12f} deg")
    print(f"        longitude {solution.longitude_deg:.12f} deg")
    print(f"        status '{solution.status}', height estimated: "
          f"{solution.height_estimated}")
    lat_error = solution.latitude_deg - lat_true
    lon_error = cp.longitude_difference(solution.longitude_deg, lon_true)
    distance = cp.horizontal_position_error(solution.lat_lon, (lat_true, lon_true))
    print(f"Step 5  latitude error  {lat_error:+.2e} deg")
    print(f"        longitude error {lon_error:+.2e} deg")
    print(f"        horizontal geodesic error {distance:.2e} m "
          f"({distance * 1e9:.0f} nanometres)")
    print("        Double precision resolves about 1e-16 rad per operation; "
          "after the SVD and three\n        rotations that is ~1e-14 rad, "
          "or ~1e-7 m on the ground. This is the numerical floor.")


def experiment_b(rng, c_eg):
    print("\n" + "=" * 78)
    print("Experiment B - global validation (noise-free)")
    print("=" * 78)
    records = []
    print(f"  {'latitude':>9}" + "".join(f"{f'lon {lon:g}':>14}"
                                         for lon in GRID_LONGITUDES)
          + "   horizontal error [m]")
    for lat in sorted(GRID_LATITUDES, reverse=True):
        line = f"  {lat:>9g}"
        for lon in GRID_LONGITUDES:
            obs = ps.simulate_observation(lat, lon, UTC, C_EG=c_eg, rng=rng)
            solution = ps.solve_observation(obs)
            error = cp.horizontal_position_error(solution.lat_lon, (lat, lon))
            records.append((lat, lon, error, solution))
            line += f"{error:>14.1e}"
        print(line)
    errors = np.array([r[2] for r in records])
    lon_errors = np.array([abs(cp.longitude_difference(r[3].longitude_deg, r[1]))
                           for r in records])
    worst = records[int(np.argmax(lon_errors))]
    print(f"\n  {len(records)} grid points: largest horizontal error "
          f"{errors.max():.1e} m, median {np.median(errors):.1e} m.")
    print(f"  Largest LONGITUDE error {lon_errors.max():.1e} deg at latitude "
          f"{worst[0]:g}: near a pole the same few\n  nanometres span more "
          f"longitude (dilution 1/cos(lat) = "
          f"{worst[3].diagnostics['longitude_dilution']:.0f}x), but the "
          f"position error does not grow.")

    # ---- boundary conditions ------------------------------------------------
    print("\n  Longitude wrapping")
    for lon_in in (179.999, -180.001, 180.0, -180.0):
        obs = ps.simulate_observation(10.0, lon_in, UTC, C_EG=c_eg, rng=rng)
        solution = ps.solve_observation(obs)
        print(f"    input {lon_in:>9.3f} -> reported {solution.longitude_deg:>16.9f}"
              f"   error {cp.horizontal_position_error(solution.lat_lon, (10.0, lon_in)):.1e} m")
    print("    179.999 and -180.001 are reported as the same longitude; the "
          "range is [-180, 180).")
    print("    Exactly on the date line the result sits within rounding of "
          "+/-180 on either side;")
    print("    both are the same meridian, and the geodesic error is "
          "unaffected.")

    print("\n  Southern hemisphere")
    obs = ps.simulate_observation(-33.9, 151.2, UTC, C_EG=c_eg, rng=rng)
    solution = ps.solve_observation(obs)
    print(f"    truth (-33.9, 151.2) -> ({solution.latitude_deg:.9f}, "
          f"{solution.longitude_deg:.9f}); u_E z-component "
          f"{solution.up_ecef[2]:+.4f} (negative: up points south of the equator plane)")

    print("\n  Poles")
    for lat in (90.0, -90.0):
        for lon in (0.0, 123.0):
            obs = ps.simulate_observation(lat, lon, UTC, C_EG=c_eg, rng=rng)
            solution = ps.solve_observation(obs)
            error = cp.horizontal_position_error(solution.lat_lon, (lat, lon))
            print(f"    truth ({lat:+.0f}, {lon:>5g}) -> latitude "
                  f"{solution.latitude_deg:+.9f}, longitude {solution.longitude_deg}, "
                  f"observable={solution.longitude_observable}, error {error:.1e} m")
    print(f"    status: '{solution.status}'")
    print("    The position is found (error is nanometres); it simply has no "
          "longitude. NaN is reported\n    instead of whatever atan2 would "
          "return for a vector of rounding noise.")

    print("\n  Instrument orientation (perfect calibration): horizontal error [m]")
    cases = [(45.0, -75.0), (0.0, 0.0), (-60.0, 90.0), (80.0, 179.0), (89.9, -90.0)]
    print(f"    {'position':<16}" + "".join(f"{name:>22}" for name in ORIENTATIONS))
    for lat, lon in cases:
        line = f"    ({lat:g}, {lon:g})".ljust(20)
        for zenith in ORIENTATIONS.values():
            worst_error = 0.0
            for azimuth, roll in ((0.0, 0.0), (135.0, 80.0), (290.0, -120.0)):
                obs = ps.simulate_observation(lat, lon, UTC, C_EG=c_eg, rng=rng,
                                              zenith_angle_deg=zenith,
                                              azimuth_deg=azimuth, roll_deg=roll)
                solution = ps.solve_observation(obs)
                worst_error = max(worst_error, cp.horizontal_position_error(
                    solution.lat_lon, (lat, lon)))
            line += f"{worst_error:>22.1e}"
        print(line)
    print("    Each entry is the worst of three azimuth/roll combinations. "
          "Rotating the instrument does\n    not move the solution.")

    # ---- map ------------------------------------------------------------------
    lats = np.array([r[0] for r in records])
    lons = np.array([r[1] for r in records])
    fig, ax = plt.subplots(figsize=(11, 5.6))
    scatter = ax.scatter(lons, lats, c=errors * 1e9, s=260, cmap="viridis",
                         edgecolors="black", linewidths=0.6)
    for lat, lon, error, _ in records:
        ax.annotate(f"{error * 1e9:.0f}", (lon, lat), ha="center", va="center",
                    fontsize=6.5, color="white")
    for lat in (90.0, -90.0):
        ax.scatter([45.0], [lat * 1.035], marker="*", s=260, color="tab:red",
                   edgecolors="black", zorder=3)
        ax.annotate("pole: longitude not observable", (45.0, lat * 1.035),
                    xytext=(10, -3), textcoords="offset points",
                    fontsize=8, color="tab:red")
    ax.set_xlim(-200, 200)
    ax.set_ylim(-100, 100)
    ax.set_xticks(GRID_LONGITUDES)
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Geodetic latitude [deg]")
    ax.set_title("Noise-free horizontal position error over the global grid "
                 "(numbers: nanometres)")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="Horizontal geodesic error [nm]")
    fig.tight_layout()
    fig.savefig(HERE / "global_validation.png", dpi=150)
    print("\nSaved global_validation.png")


if __name__ == "__main__":
    generator = np.random.default_rng(SEED)
    earth, earth_info = cp.earth_orientation(UTC)
    experiment_a(generator, earth, earth_info)
    experiment_b(generator, earth)
