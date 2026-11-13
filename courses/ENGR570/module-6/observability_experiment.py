"""
observability_experiment.py - Space Systems Engineering, Module 6, Experiment D

What the star-camera-plus-accelerometer measurement can and cannot
determine. Four reproducible demonstrations:

D1  Altitude ambiguity: two heights give the same measurements.
D2  Unknown clock offset: time and longitude cannot be separated.
D3  Polar singularity: longitude becomes unstable while position does not.
D4  Acceleration ambiguity: an unmodelled acceleration yields a plausible
    but wrong position, identical to a real observation somewhere else.

A demonstration succeeds here when it shows correctly that no unique answer
is supported by the measurements, not when the solver returns the truth.

Run:  python observability_experiment.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta

import celestial_position as cp
import position_simulator as ps
from accelerometer_calibration import AccelerometerErrorModel
from astronomical_time import EARTH_ROTATION_RATE_UT1, time_scales
from reference_frames import geodetic_to_ecef
from rotation_engine import rotation_vector_to_matrix
from sensor_frames import earth_from_celestial

SEED = 606
HERE = Path(__file__).parent
UTC = ps.OBSERVATION_UTC
T0 = Time(UTC, scale="utc")
ARCSEC = np.radians(1.0 / 3600.0)


def angle_arcsec(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)) * 3600


def heading(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ------------------------------------------------------------------------- D1
def d1_altitude():
    heading("D1 - Altitude ambiguity")
    lat, lon = 45.0, -75.0
    earth = cp.earth_orientation(UTC)
    results = []
    for height in (0.0, 5000.0):
        obs = ps.simulate_observation(lat, lon, UTC, height_m=height,
                                      C_EG=earth[0],
                                      rng=np.random.default_rng(SEED))
        solution = ps.solve_observation(obs, earth=earth)
        results.append((height, obs, solution))
        position = geodetic_to_ecef(lat, lon, height)
        print(f"  h = {height:>6g} m: ECEF position "
              f"[{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] m")
        print(f"      |specific force| = "
              f"{np.linalg.norm(obs.truth['specific_force_imu']):.5f} m/s^2, "
              f"normalized up (IMU) = {np.round(obs.truth['up_imu'], 12)}")
        print(f"      solution: lat {solution.latitude_deg:.10f}, lon "
              f"{solution.longitude_deg:.10f}, height {solution.height_m} "
              f"(estimated: {solution.height_estimated})")
    (_, low, sol_low), (_, high, sol_high) = results
    print(f"\n  Positions differ by "
          f"{np.linalg.norm(geodetic_to_ecef(lat, lon, 5000.0) - geodetic_to_ecef(lat, lon, 0.0)):.1f} m, "
          f"exactly along the ellipsoid normal.")
    print(f"  Largest difference between the two star measurement sets: "
          f"{np.max(np.abs(low.measured_stars - high.measured_stars)):.1e}")
    print(f"  Angle between the two measured vertical directions: "
          f"{angle_arcsec(low.truth['up_imu'], high.truth['up_imu']):.1e} arcsec")
    print(f"  Difference between the two solutions: "
          f"{cp.horizontal_position_error(sol_low.lat_lon, sol_high.lat_lon):.1e} m")
    print("""
  Every point on one ellipsoid normal has the same up direction, and stars
  are effectively at infinity, so their directions do not change with
  height either. The measurements are identical; height is a free parameter
  the estimator cannot fix, and the solution reports height = None rather
  than zero.
  The accelerometer MAGNITUDE does differ, by 0.0154 m/s^2 over 5000 m, but
  it is deliberately not used: gravity also varies by 0.05 m/s^2 with
  latitude and by more than that with local geology, and the accelerometer
  bias assumed in Experiment C is 0.005 m/s^2 per axis. Height from |f|
  would need an absolute gravity model and sensor accuracy this instrument
  does not have.""")


# ------------------------------------------------------------------------- D2
def d2_clock():
    heading("D2 - Unknown clock offset")
    lat, lon = 45.0, -75.0
    earth = cp.earth_orientation(UTC)
    obs = ps.simulate_observation(lat, lon, UTC, C_EG=earth[0],
                                  rng=np.random.default_rng(SEED))
    rate_deg = np.degrees(EARTH_ROTATION_RATE_UT1)
    print(f"  Measurement made at {UTC} UTC at ({lat}, {lon}), then solved with "
          f"wrong timestamps.")
    print(f"  Earth turns {rate_deg * 3600:.4f} arcsec per second of time.\n")
    print(f"  {'clock error':>12}{'longitude error':>19}{'-omega * dt':>16}"
          f"{'difference':>14}{'latitude error':>17}{'position error':>17}")
    offsets = [-3600.0, -60.0, -1.0, -0.01, 0.01, 1.0, 60.0, 3600.0]
    rows = []
    for dt in offsets:
        wrong = cp.earth_orientation(T0 + TimeDelta(dt, format="sec"))
        solution = ps.solve_observation(obs, earth=wrong)
        dlon = cp.longitude_difference(solution.longitude_deg, lon) * 3600
        dlat = (solution.latitude_deg - lat) * 3600
        predicted = -rate_deg * dt * 3600
        error = cp.horizontal_position_error(solution.lat_lon, (lat, lon))
        rows.append((dt, dlon, predicted, dlat, error))
        print(f"  {dt:>10g} s{dlon:>17.3f}\"{predicted:>15.3f}\"{dlon - predicted:>13.4f}\""
              f"{dlat:>16.5f}\"{error:>14.1f} m")
    print("""
  The longitude error equals Earth's rotation over the clock error to a
  small fraction of an arcsecond, and latitude is essentially unaffected.
  The tiny remainders at one hour come from the pole's slow precession,
  nutation and polar motion, which also depend on time.""")

    # Can a longitude change exactly compensate a time change?
    print("  Can longitude and time be told apart?")
    dt = 3600.0
    shift = rate_deg * dt
    era = time_scales(T0).era_rad

    def simple_c_eg(delta_t):                  # rotation about z only
        return earth_from_celestial(era + EARTH_ROTATION_RATE_UT1 * delta_t)

    up_here_now = simple_c_eg(0.0).inverse().apply(ps.true_vertical_ecef(lat, lon))
    up_west_later = simple_c_eg(dt).inverse().apply(
        ps.true_vertical_ecef(lat, lon - shift))
    print(f"    Simplified model (rotation about the pole only): observer at "
          f"lon {lon} at t, versus observer at\n    lon {lon - shift:.4f} at "
          f"t + {dt:g} s. Angle between their celestial verticals: "
          f"{angle_arcsec(up_here_now, up_west_later):.1e} arcsec.")
    full_now = earth[0].inverse().apply(ps.true_vertical_ecef(lat, lon))
    full_later = cp.earth_orientation(T0 + TimeDelta(dt, format="sec"))[0].inverse().apply(
        ps.true_vertical_ecef(lat, lon - shift))
    residual = angle_arcsec(full_now, full_later)
    print(f"    Full IAU model, same two observers: {residual:.2e} arcsec "
          f"({residual * ARCSEC * ps.EARTH_RADIUS_M:.3f} m on the ground).")
    print(f"""
  Under the simplified model the compensation is exact: the Earth-fixed
  frame is the celestial frame turned by an angle that grows linearly with
  time, and longitude enters as the same kind of turn. Only the sum
  "longitude + rotation rate x time" is measured. An observer {shift:.2f} deg
  further west, an hour later, has an identical celestial vertical and
  therefore identical measurements.
  The full model breaks the symmetry only by {residual * 1000:.0f} milliarcseconds per
  hour ({residual * ARCSEC * ps.EARTH_RADIUS_M:.1f} m on the ground), mostly because Earth turns about its
  rotation pole while longitude is measured about the Earth-fixed z axis,
  0.34 arcsec away (polar motion). That is far below the arcsecond-level
  measurement errors, so in practice time cannot be solved for from these
  observations. Longitude is only as good
  as the clock: {rate_deg * 3600 * ARCSEC * ps.EARTH_RADIUS_M * np.cos(np.radians(lat)):.0f} m per second of error at this latitude. Time must
  come from outside.""")
    return rows


# ------------------------------------------------------------------------- D3
def d3_pole(rng):
    heading("D3 - Polar singularity")
    error_arcsec = 10.0
    print(f"  A fixed {error_arcsec:g} arcsec error is applied to the "
          f"Earth-fixed vertical, toward local east (the worst\n  direction "
          f"for longitude), then in 2000 random directions.\n")
    print(f"  {'latitude':>12}{'dist. to pole':>15}{'longitude error':>18}"
          f"{'eps/cos(lat)':>15}{'displacement':>14}{'RMS lon error':>16}"
          f"{'RMS displ.':>12}")
    rows = []
    for lat in (0.0, 60.0, 80.0, 89.0, 89.9, 89.99, 89.999, 89.9999, 89.99999):
        lon = 20.0
        axes = ps.local_axes_ecef(lat, lon)
        east, north, up = axes[:, 0], axes[:, 1], axes[:, 2]
        tilted = up + np.tan(error_arcsec * ARCSEC) * east
        est = cp.earth_fixed_up_to_geodetic(tilted)
        dlon = abs(cp.longitude_difference(est[1], lon))
        displacement = cp.horizontal_position_error(est, (lat, lon))
        angles = rng.uniform(0.0, 2 * np.pi, 2000)
        lon_sq, disp_sq = [], []
        for a in angles:
            v = up + np.tan(error_arcsec * ARCSEC) * (np.cos(a) * east
                                                      + np.sin(a) * north)
            e = cp.earth_fixed_up_to_geodetic(v)
            lon_sq.append(cp.longitude_difference(e[1], lon) ** 2)
            disp_sq.append(cp.horizontal_position_error(e, (lat, lon)) ** 2)
        rms_lon, rms_disp = np.sqrt(np.mean(lon_sq)), np.sqrt(np.mean(disp_sq))
        pole_distance = cp.horizontal_position_error((90.0, 0.0), (lat, lon))
        predicted = error_arcsec / 3600 / np.cos(np.radians(lat))
        rows.append((90.0 - lat, dlon, displacement, rms_lon, pole_distance))
        distance_text = (f"{pole_distance / 1000:,.0f} km" if pole_distance >= 1e4
                         else f"{pole_distance:,.1f} m")
        predicted_text = (f"{predicted:.5f} deg" if predicted < 90.0
                          else "over 90 deg")
        print(f"  {lat:>12.5f}{distance_text:>15}{dlon:>14.5f} deg"
              f"{predicted_text:>15}{displacement:>12.1f} m"
              f"{rms_lon:>12.5f} deg{rms_disp:>10.1f} m")

    solution_lat, solution_lon = cp.earth_fixed_up_to_geodetic([0.0, 0.0, 1.0])
    axes = ps.local_axes_ecef(90.0, 20.0)
    tilted = axes[:, 2] + np.tan(error_arcsec * ARCSEC) * axes[:, 0]
    at_pole = cp.earth_fixed_up_to_geodetic(tilted)
    print(f"\n  Exactly at the pole, no error: latitude {solution_lat}, "
          f"longitude {solution_lon} (reported as not observable).")
    print(f"  At the pole with the {error_arcsec:g} arcsec error: latitude "
          f"{at_pole[0]:.6f}, longitude {at_pole[1]:.2f} deg, displacement "
          f"{cp.horizontal_position_error(at_pole, (90.0, 0.0)):.1f} m.")
    print(f"""
  The physical displacement stays at about {error_arcsec * ARCSEC * ps.EARTH_RADIUS_M:.0f} m at every latitude: the
  position estimate does not degrade. The longitude error grows as
  1/cos(latitude) and stops meaning anything once the observer is closer to
  the pole than the error itself ({error_arcsec * ARCSEC * ps.EARTH_RADIUS_M:.0f} m): there the reported longitude is
  simply the direction of the error, anywhere in +/-180 deg. This is a
  singularity of the latitude/longitude coordinates, not of the estimate.
  Near the poles, errors and covariances should be stated in metres north
  and east (or as a 3D vector), never in degrees of longitude.""")
    return rows


# ------------------------------------------------------------------------- D4
def d4_acceleration(rng):
    heading("D4 - Acceleration ambiguity")
    lat, lon = 45.0, -75.0
    a_east = 0.05                               # m/s^2
    earth = cp.earth_orientation(UTC)
    quiet = AccelerometerErrorModel(bias=(0, 0, 0), scale_factor_error=0.0,
                                    cross_axis=0.0, noise_sigma=0.0)
    star_seed = SEED + 4

    # Case 2: at the true position, accelerating east.
    case2 = ps.simulate_observation(lat, lon, UTC, C_EG=earth[0],
                                    rng=np.random.default_rng(star_seed),
                                    acceleration_enu=(a_east, 0.0, 0.0),
                                    accel_model=quiet, n_accel_samples=200)
    sol2 = ps.solve_observation(case2, earth=earth, accel_noise_sigma=0.005)

    # Case 1: stationary at the same position, but physically tilted so that
    # its accelerometer reads the same direction as case 2.
    up_true_g = case2.truth["up_celestial"]
    apparent_g = sol2.up_celestial
    axis = np.cross(up_true_g, apparent_g)
    tilt = np.arctan2(np.linalg.norm(axis), up_true_g @ apparent_g)
    rotation_g = rotation_vector_to_matrix(axis / np.linalg.norm(axis) * tilt)
    case1 = ps.simulate_observation(lat, lon, UTC, C_EG=earth[0],
                                    rng=np.random.default_rng(star_seed),
                                    C_CG=case2.truth["C_CG"] @ rotation_g,
                                    accel_model=quiet, n_accel_samples=200)
    sol1 = ps.solve_observation(case1, earth=earth, accel_noise_sigma=0.005)

    # Case 3: stationary at the position case 2 was mistaken for, with the
    # same celestial attitude as case 2.
    case3 = ps.simulate_observation(sol2.latitude_deg, sol2.longitude_deg, UTC,
                                    C_EG=earth[0],
                                    rng=np.random.default_rng(star_seed),
                                    C_CG=case2.truth["C_CG"],
                                    accel_model=quiet, n_accel_samples=200)
    sol3 = ps.solve_observation(case3, earth=earth, accel_noise_sigma=0.005)

    def direction(obs):
        mean = obs.accel_samples.mean(axis=0)
        return mean / np.linalg.norm(mean), np.linalg.norm(mean)

    print(f"  True position ({lat}, {lon}). Horizontal acceleration {a_east} "
          f"m/s^2 east tilts the apparent vertical by\n  atan(a/g) = "
          f"{np.degrees(tilt):.4f} deg = {np.degrees(tilt) * 3600:.0f} arcsec.\n")
    print(f"  {'case':<50}{'reported position':>26}{'own error':>11}{'status':>8}"
          f"{'IMU quality':>13}")
    for label, obs, sol, truth in (
            ("1  stationary, tilted by atan(a/g)", case1, sol1, (lat, lon)),
            ("2  untilted, accelerating 0.05 m/s^2 east", case2, sol2, (lat, lon)),
            ("3  stationary at the place case 2 reports", case3, sol3,
             (sol2.latitude_deg, sol2.longitude_deg))):
        error = cp.horizontal_position_error(sol.lat_lon, truth)
        print(f"  {label:<50}({sol.latitude_deg:>9.5f}, {sol.longitude_deg:>10.5f})"
              f"{error / 1000:>8.2f} km{sol.status:>8}{sol.quality.verdict:>13}")

    d1, m1 = direction(case1)
    d2, m2 = direction(case2)
    d3, m3 = direction(case3)
    print(f"\n  Accelerometer direction, case 1 versus case 2: "
          f"{angle_arcsec(d1, d2):.1e} arcsec apart (identical).")
    print(f"  Star measurements, case 1 versus case 2: boresights "
          f"{angle_arcsec(case1.truth['C_CG'].T[:, 2], case2.truth['C_CG'].T[:, 2]) / 3600:.4f} deg apart "
          f"(the instruments really are oriented differently).")
    print(f"  Case 2 versus case 3: accelerometer direction "
          f"{angle_arcsec(d2, d3):.1e} arcsec apart; largest star-measurement "
          f"difference {np.max(np.abs(case2.measured_stars - case3.measured_stars)):.1e};")
    print(f"  accelerometer magnitude {m2:.5f} versus {m3:.5f} m/s^2 "
          f"(difference {m2 - m3:.5f} = a^2/2g, far below any usable knowledge of g).")
    wrong_by = cp.horizontal_position_error(sol2.lat_lon, (lat, lon))
    print(f"""
  Cases 1 and 2 give the accelerometer the same direction, but the star
  camera sees different attitudes, so the complete estimator returns
  different positions. Case 1 is right. Case 2 is wrong by {wrong_by / 1000:.1f} km, with
  status "ok" and a "steady" accelerometer.
  Knowing the stellar attitude does not help, because it is not the
  attitude that is in doubt. The estimator assumes the specific force
  points up. Case 2's measurements are, to machine precision, the
  measurements of case 3: an unaccelerated instrument {wrong_by / 1000:.1f} km away. Nothing
  in a single camera-plus-accelerometer observation distinguishes them, so
  no unique position is supported unless the acceleration is known to be
  zero (a fixed mount) or measured independently.
  The sensitivity is severe: {np.radians(1 / 3600) * 1e6:.1f} micro-g of unmodelled horizontal
  acceleration per arcsecond, i.e. per {np.radians(1 / 3600) * ps.EARTH_RADIUS_M:.0f} m of position.""")
    return (lat, lon), sol1, sol2, sol3


def make_figure(clock_rows, pole_rows, d4):
    fig, (ax_clock, ax_pole, ax_map) = plt.subplots(1, 3, figsize=(18, 5.4))

    dts = np.array([abs(r[0]) for r in clock_rows])
    ax_clock.loglog(dts, [abs(r[1]) for r in clock_rows], "o", color="tab:purple",
                    markersize=8, label="longitude error from the solver")
    grid = np.logspace(-2, 3.6, 50)
    ax_clock.loglog(grid, np.degrees(EARTH_ROTATION_RATE_UT1) * 3600 * grid, "-",
                    color="gray", label="Earth rotation, ω·Δt")
    ax_clock.loglog(dts, [max(abs(r[3]), 1e-6) for r in clock_rows], "x",
                    color="tab:green", label="latitude error")
    ax_clock.set_xlabel("Clock error |Δt| [s]")
    ax_clock.set_ylabel("Coordinate error [arcsec]")
    ax_clock.set_title("D2: a clock error is a longitude error")
    ax_clock.grid(True, which="both", alpha=0.3)
    ax_clock.legend(fontsize=8)

    colat = [r[0] for r in pole_rows if r[0] > 0]
    ax_pole.loglog(colat, [r[1] for r in pole_rows if r[0] > 0], "o-",
                   color="tab:red", label="longitude error [deg], error toward east")
    ax_pole.loglog(colat, [r[3] for r in pole_rows if r[0] > 0], "s--",
                   color="tab:orange", label="RMS longitude error [deg], random direction")
    ax_pole.set_xlabel("Angular distance from the pole, 90° − latitude [deg]")
    ax_pole.set_ylabel("Longitude error [deg]")
    ax_pole.invert_xaxis()
    twin = ax_pole.twinx()
    twin.semilogx(colat, [r[2] for r in pole_rows if r[0] > 0], "^-",
                  color="tab:blue", label="physical displacement [m]")
    twin.set_ylim(0, 600)
    twin.set_ylabel("Horizontal displacement [m]")
    handles, labels = ax_pole.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax_pole.legend(handles + h2, labels + l2, fontsize=7.5, loc="upper left")
    ax_pole.set_title("D3: 10 arcsec vertical error approaching the pole")
    ax_pole.grid(True, which="both", alpha=0.3)

    (lat, lon), sol1, sol2, sol3 = d4
    ax_map.plot(lon, lat, "k*", markersize=18, label="true position (cases 1 and 2)")
    ax_map.plot(sol1.longitude_deg, sol1.latitude_deg, "o", color="tab:green",
                markersize=11, fillstyle="none", markeredgewidth=2,
                label="case 1 solution: stationary, tilted")
    ax_map.plot(sol2.longitude_deg, sol2.latitude_deg, "X", color="tab:red",
                markersize=12, label="case 2 solution: accelerating 0.05 m/s² east")
    ax_map.plot(sol3.longitude_deg, sol3.latitude_deg, "s", color="tab:blue",
                markersize=14, fillstyle="none", markeredgewidth=1.5,
                label="case 3: truly there, stationary")
    ax_map.annotate("", xy=(sol2.longitude_deg, sol2.latitude_deg), xytext=(lon, lat),
                    arrowprops=dict(arrowstyle="->", color="tab:red"))
    span = max(abs(sol2.longitude_deg - lon), 0.3) * 1.6
    ax_map.set_xlim(lon - span, lon + span)
    ax_map.set_ylim(lat - span * 0.7, lat + span * 0.7)
    ax_map.set_xlabel("Longitude [deg]")
    ax_map.set_ylabel("Latitude [deg]")
    ax_map.set_title("D4: unmodelled acceleration, plausible wrong position")
    ax_map.grid(True, alpha=0.3)
    ax_map.legend(fontsize=7.5, loc="lower left")

    fig.tight_layout()
    fig.savefig(HERE / "observability_experiment.png", dpi=140)
    print("\nSaved observability_experiment.png")


if __name__ == "__main__":
    generator = np.random.default_rng(SEED)
    d1_altitude()
    clock = d2_clock()
    pole = d3_pole(generator)
    acceleration = d4_acceleration(generator)
    make_figure(clock, pole, acceleration)
