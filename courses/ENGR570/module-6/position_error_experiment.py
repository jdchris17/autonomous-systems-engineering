"""
position_error_experiment.py - Space Systems Engineering, Module 6, Experiment C

How attitude, gravity, calibration and timing errors become position error.

The physical pipeline is Experiment A's. The measurements and models are
perturbed with the same tools as earlier modules:
  * camera: Module 4's SVD attitude estimator on stars with centroid noise;
  * IMU: Module 5's accelerometer error model and gravity estimator;
  * extrinsics: a rotation-vector error on the camera-IMU calibration;
  * clock: the timestamp used for Earth orientation is offset while the
    true observation time stays fixed.

How fixed errors are drawn (read this before interpreting any number)
---------------------------------------------------------------------
In every Monte Carlo below, the accelerometer bias and the camera-IMU
misalignment are drawn INDEPENDENTLY FOR EACH TRIAL. Each trial is a
different simulated instrument making one observation. The statistics are
therefore across a population of instruments, not the scatter one
instrument would show over time. A single instrument keeps its bias and
misalignment, so repeating or averaging its observations does not reduce
those errors; the "one instrument" section shows this directly.

Run:  python position_error_experiment.py
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
from astronomical_time import EARTH_ROTATION_RATE_UT1
from imu_alignment import perturb_calibration

SEED = 66
HERE = Path(__file__).parent
UTC = ps.OBSERVATION_UTC
TRUE_POSITION = (45.0, -75.0)
TRIALS = 500

# ---- modelled error sources (1 sigma) ------------------------------------------
N_STARS = 20
CENTROID_NOISE_PX = 0.2
ACCEL_NOISE = 0.005                # m/s^2 per axis per sample
ACCEL_SAMPLES = 1000
ACCEL_BIAS_SIGMA = 0.005           # m/s^2 per axis, residual after calibration
EXTRINSIC_SIGMA_DEG = 0.05         # camera-IMU misalignment per axis
CLOCK_SIGMA_S = 0.1                # timestamp error in the combined scenario
CLOCK_OFFSETS_S = [0.01, 0.1, 1.0, 10.0]

# ---- accuracy targets -------------------------------------------------------------
PHASE1_M = (11_100.0, 18_500.0)    # 6-10 nautical miles
PHASE2_M = (100.0, 200.0)

T0 = Time(UTC, scale="utc")
EARTH = cp.earth_orientation(UTC)


def one_trial(rng, position, *, camera=False, accel=False, extrinsic=False,
              clock_offset_s=0.0, zenith_angle_deg=30.0, fixed=None):
    """
    Simulate one observation and solve it. `fixed` optionally supplies
    {"bias": (3,), "misalignment": (3,)} to hold an instrument's fixed
    errors constant instead of drawing new ones.
    Returns (north_m, east_m, lat_error_deg, lon_error_deg).
    """
    if accel:
        bias = (fixed["bias"] if fixed else rng.normal(0.0, ACCEL_BIAS_SIGMA, 3))
        model = AccelerometerErrorModel(bias=tuple(bias), scale_factor_error=0.0,
                                        cross_axis=0.0, noise_sigma=ACCEL_NOISE)
        n_accel = ACCEL_SAMPLES
    else:
        model, n_accel = ps.IDEAL_SENSOR, 2

    obs = ps.simulate_observation(
        *position, UTC, C_EG=EARTH[0], rng=rng, n_stars=N_STARS,
        centroid_noise_px=CENTROID_NOISE_PX if camera else 0.0,
        accel_model=model, n_accel_samples=n_accel,
        zenith_angle_deg=zenith_angle_deg,
        azimuth_deg=rng.uniform(0.0, 360.0), roll_deg=rng.uniform(0.0, 360.0))

    believed_calibration = obs.C_CI
    if extrinsic:
        delta = (fixed["misalignment"] if fixed
                 else rng.normal(0.0, np.radians(EXTRINSIC_SIGMA_DEG), 3))
        believed_calibration = perturb_calibration(obs.C_CI, delta)

    if clock_offset_s:
        earth = cp.earth_orientation(T0 + TimeDelta(clock_offset_s, format="sec"))
    else:
        earth = EARTH
    solution = ps.solve_observation(obs, C_CI=believed_calibration, earth=earth)

    north, east = cp.north_east_error(solution.lat_lon, position)
    return (north, east, solution.latitude_deg - position[0],
            cp.longitude_difference(solution.longitude_deg, position[1]))


def run(rng, position=TRUE_POSITION, trials=TRIALS, clock_sigma_s=0.0, **flags):
    out = np.empty((trials, 4))
    for t in range(trials):
        offset = rng.normal(0.0, clock_sigma_s) if clock_sigma_s else 0.0
        out[t] = one_trial(rng, position, clock_offset_s=offset, **flags)
    return out


def summarize(results):
    horizontal = np.hypot(results[:, 0], results[:, 1])
    rms = np.sqrt(np.mean(results**2, axis=0))
    return {"horizontal": horizontal, "median": np.median(horizontal),
            "p95": np.percentile(horizontal, 95), "rms_north": rms[0],
            "rms_east": rms[1], "rms_lat_arcsec": rms[2] * 3600,
            "rms_lon_arcsec": rms[3] * 3600}


def metres(value):
    return f"{value / 1000:.2f} km" if value >= 1000 else f"{value:.1f} m"


def main():
    rng = np.random.default_rng(SEED)
    arcsec_m = np.radians(1 / 3600) * ps.EARTH_RADIUS_M
    print(f"Observer {TRUE_POSITION}, {UTC} UTC, boresight 30 deg from zenith, "
          f"{TRIALS} trials per scenario.")
    print(f"Scale: 1 arcsec of vertical error = {arcsec_m:.1f} m on the ground; "
          f"1 nautical mile = 1 arcminute.\n")

    scenarios = {
        "A - Ideal": dict(),
        "B - Camera attitude": dict(camera=True),
        "C - Accelerometer": dict(accel=True),
        "D - Extrinsic calibration": dict(extrinsic=True),
        "E - Clock": dict(clock_sigma_s=CLOCK_SIGMA_S),
        "F - Combined": dict(camera=True, accel=True, extrinsic=True,
                             clock_sigma_s=CLOCK_SIGMA_S),
    }
    assumed = {
        "A - Ideal": "no added error",
        "B - Camera attitude": f"{N_STARS} stars, {CENTROID_NOISE_PX} px centroid noise",
        "C - Accelerometer": f"bias {ACCEL_BIAS_SIGMA} m/s^2/axis, noise "
                             f"{ACCEL_NOISE} x {ACCEL_SAMPLES} samples",
        "D - Extrinsic calibration": f"{EXTRINSIC_SIGMA_DEG} deg per axis",
        "E - Clock": f"{CLOCK_SIGMA_S} s timestamp error",
        "F - Combined": "all of the above",
    }
    summaries, raw = {}, {}
    print(f"  {'scenario':<27}{'median':>10}{'95th pct':>11}{'RMS north':>11}"
          f"{'RMS east':>11}{'RMS lat':>10}{'RMS lon':>10}   assumed 1-sigma")
    for name, flags in scenarios.items():
        raw[name] = run(rng, **flags)
        s = summaries[name] = summarize(raw[name])
        print(f"  {name:<27}{metres(s['median']):>10}{metres(s['p95']):>11}"
              f"{metres(s['rms_north']):>11}{metres(s['rms_east']):>11}"
              f"{s['rms_lat_arcsec']:>8.1f}\"{s['rms_lon_arcsec']:>9.1f}\""
              f"   {assumed[name]}")
    print("  Bias and misalignment are drawn fresh for every trial "
          "(a population of instruments).")
    print("  North and east errors in metres are about equal, but the "
          "longitude error in arcsec is\n  1/cos(45 deg) = 1.41 times the "
          "latitude error: a degree of longitude is shorter there.")

    # ---- clock sweep -----------------------------------------------------------
    print("\n  Clock offsets (deterministic, everything else ideal):")
    print(f"  {'offset':>9}{'position error':>17}{'longitude error':>18}"
          f"{'omega * dt':>13}{'latitude error':>17}")
    clock_rows = []
    for dt in CLOCK_OFFSETS_S:
        north, east, dlat, dlon = one_trial(rng, TRUE_POSITION, clock_offset_s=dt)
        predicted = np.degrees(EARTH_ROTATION_RATE_UT1 * dt) * 3600
        clock_rows.append((dt, np.hypot(north, east)))
        print(f"  {dt:>7g} s{metres(np.hypot(north, east)):>17}"
              f"{dlon * 3600:>16.3f}\"{predicted:>12.3f}\"{dlat * 3600:>15.1e}\"")
    print(f"  A fast clock puts the observer west. {arcsec_m * 15.041 * np.cos(np.radians(45)):.0f} m "
          f"per second at 45 deg; {arcsec_m * 15.041:.0f} m/s at the equator.")

    # ---- targets -----------------------------------------------------------------
    combined = summaries["F - Combined"]["horizontal"]
    print("\n  Against the two accuracy targets (combined scenario):")
    print(f"    median {metres(np.median(combined))}, 95th percentile "
          f"{metres(np.percentile(combined, 95))}")
    for label, limit in (("Phase 1, 6 NM (11.1 km)", PHASE1_M[0]),
                         ("Phase 1, 10 NM (18.5 km)", PHASE1_M[1]),
                         ("Phase 2, 200 m", PHASE2_M[1]),
                         ("Phase 2, 100 m", PHASE2_M[0])):
        print(f"    within {label:<26}{100 * np.mean(combined < limit):>6.1f} % of trials")
    target_rms = 150.0
    per_source_arcsec = target_rms / arcsec_m / np.sqrt(2)     # two equal sources
    per_axis = per_source_arcsec / np.sqrt(2)
    print(f"""
    These error sizes are illustrative assumptions, so this is a statement
    about the model, not a prediction for any device. Under them the
    combined error is within 10 NM in {100 * np.mean(combined < PHASE1_M[1]):.0f} % of trials and within 6 NM in
    {100 * np.mean(combined < PHASE1_M[0]):.0f} %, and its median is about {np.median(combined) / 150:.0f} times too large for Phase 2.
    The two fixed errors account for
    nearly all of it. For a {target_rms:.0f} m RMS result split equally between them,
    each must contribute about {per_source_arcsec:.1f} arcsec: a camera-IMU alignment known to
    ~{per_axis:.1f} arcsec per axis ({per_axis / 3600:.5f} deg) and a horizontal accelerometer bias of
    ~{np.radians(per_axis / 3600) * 9.81:.1e} m/s^2 ({np.radians(per_axis / 3600) * 1e6:.0f} micro-g). The assumed values here are
    {EXTRINSIC_SIGMA_DEG * 3600 / per_axis:.0f}x and {ACCEL_BIAS_SIGMA / (np.radians(per_axis / 3600) * 9.81):.0f}x larger. The star camera alone is at the edge of Phase 2 with
    the boresight 30 deg from zenith (median {metres(summaries['B - Camera attitude']['median'])}, 95th percentile {metres(summaries['B - Camera attitude']['p95'])});
    the pointing table below shows it improving near the zenith. A {CLOCK_SIGMA_S} s
    clock contributes {metres(summaries['E - Clock']['median'])} median.""")

    # ---- one instrument versus many ------------------------------------------------
    print("  One instrument, repeated observations (fixed bias and misalignment):")
    fixed = {"bias": rng.normal(0.0, ACCEL_BIAS_SIGMA, 3),
             "misalignment": rng.normal(0.0, np.radians(EXTRINSIC_SIGMA_DEG), 3)}
    single = np.array([one_trial(rng, TRUE_POSITION, camera=True, accel=True,
                                 extrinsic=True, fixed=fixed, zenith_angle_deg=30.0)
                       for _ in range(TRIALS)])
    # Same instrument AND same pointing: hold azimuth/roll by reseeding.
    held = []
    for _ in range(TRIALS):
        obs = ps.simulate_observation(
            *TRUE_POSITION, UTC, C_EG=EARTH[0], rng=rng, n_stars=N_STARS,
            centroid_noise_px=CENTROID_NOISE_PX,
            accel_model=AccelerometerErrorModel(
                bias=tuple(fixed["bias"]), scale_factor_error=0.0,
                cross_axis=0.0, noise_sigma=ACCEL_NOISE),
            n_accel_samples=ACCEL_SAMPLES, zenith_angle_deg=30.0,
            azimuth_deg=40.0, roll_deg=15.0)
        solution = ps.solve_observation(
            obs, C_CI=perturb_calibration(obs.C_CI, fixed["misalignment"]),
            earth=EARTH)
        held.append(cp.north_east_error(solution.lat_lon, TRUE_POSITION))
    held = np.array(held)
    mean_offset = np.hypot(*held.mean(axis=0))
    scatter = np.sqrt(np.mean(np.sum((held - held.mean(axis=0)) ** 2, axis=1)))
    turning_mean = np.hypot(*single[:, :2].mean(axis=0))
    print(f"    Instrument held in one orientation, {TRIALS} observations: "
          f"mean offset {metres(mean_offset)}, scatter {metres(scatter)}.")
    print(f"    Averaging all {TRIALS} leaves {metres(mean_offset)}: the "
          f"scatter averages away, the offset does not.")
    print(f"    Same instrument turned to a random heading each time: mean "
          f"offset {metres(turning_mean)}.")
    print("    Turning the instrument about the vertical rotates its fixed "
          "errors with it, so over many\n    headings they largely cancel in "
          "the mean. That is a calibration opportunity, not something\n    "
          "a single static observation gets for free.")

    # ---- latitude dependence ----------------------------------------------------------
    print("\n  Error versus observer latitude (combined scenario, "
          f"{TRIALS} trials each):")
    latitudes = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 89.0]
    lat_rows = []
    print(f"  {'latitude':>9}{'median':>11}{'95th pct':>11}{'RMS lat':>11}"
          f"{'RMS lon':>12}{'1 s clock error':>18}")
    for lat in latitudes:
        s = summarize(run(rng, position=(lat, -75.0), camera=True, accel=True,
                          extrinsic=True, clock_sigma_s=CLOCK_SIGMA_S))
        clock = np.hypot(*one_trial(rng, (lat, -75.0), clock_offset_s=1.0)[:2])
        lat_rows.append((lat, s, clock))
        print(f"  {lat:>7g}  {metres(s['median']):>11}{metres(s['p95']):>11}"
              f"{s['rms_lat_arcsec']:>10.0f}\"{s['rms_lon_arcsec']:>11.0f}\""
              f"{metres(clock):>18}")
    print("    The horizontal error in metres does not depend on latitude: "
          "the vertical is wrong by the\n    same angle everywhere. The same "
          "distance is more degrees of longitude near the pole, so\n    "
          "longitude error grows as 1/cos(lat). Clock error is the one "
          "source that shrinks, as cos(lat).")

    # ---- camera pointing --------------------------------------------------------------
    print("\n  Error versus camera pointing direction:")
    pointing_rows = []
    print(f"  {'boresight zenith angle':>24}{'camera only, median':>22}"
          f"{'camera only, 95th':>20}{'combined, median':>19}")
    for zenith in (2.0, 30.0, 60.0, 85.0):
        cam = summarize(run(rng, camera=True, zenith_angle_deg=zenith))
        full = summarize(run(rng, camera=True, accel=True, extrinsic=True,
                             clock_sigma_s=CLOCK_SIGMA_S, zenith_angle_deg=zenith))
        pointing_rows.append((zenith, cam, full))
        print(f"  {zenith:>20g} deg{metres(cam['median']):>22}"
              f"{metres(cam['p95']):>20}{metres(full['median']):>19}")
    print("    Pointing near the zenith keeps the star camera's weak roll "
          "axis out of the vertical\n    (Module 5). It matters for the "
          "camera's own contribution and not at all for the combined\n    "
          "result, which the fixed errors dominate.")

    make_figure(summaries, raw, clock_rows, lat_rows, pointing_rows)


def make_figure(summaries, raw, clock_rows, lat_rows, pointing_rows):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    colors = {"A - Ideal": "gray", "B - Camera attitude": "tab:green",
              "C - Accelerometer": "tab:orange",
              "D - Extrinsic calibration": "tab:red", "E - Clock": "tab:purple",
              "F - Combined": "tab:blue"}

    def bands(ax, horizontal=True):
        span = ax.axhspan if horizontal else ax.axvspan
        span(*PHASE1_M, color="gold", alpha=0.25, label="Phase 1 target (6-10 NM)")
        span(*PHASE2_M, color="limegreen", alpha=0.25,
             label="Phase 2 target (100-200 m)")

    # 1. distribution
    ax = axes[0, 0]
    bins = np.logspace(0, 5, 60)
    for name in list(colors)[1:]:
        ax.hist(summaries[name]["horizontal"], bins=bins, histtype="step",
                linewidth=1.8, color=colors[name], label=name)
    bands(ax, horizontal=False)
    ax.set_xscale("log")
    ax.set_xlabel("Horizontal position error [m]")
    ax.set_ylabel(f"Trials (of {TRIALS})")
    ax.set_title("Position-error distribution by scenario")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    # 2. north/east scatter, combined
    ax = axes[0, 1]
    combined = raw["F - Combined"]
    ax.scatter(combined[:, 1] / 1000, combined[:, 0] / 1000, s=8, alpha=0.5,
               color="tab:blue")
    for radius, style in ((PHASE1_M[0] / 1000, "--"), (PHASE1_M[1] / 1000, "-")):
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="goldenrod",
                                linestyle=style, linewidth=1.5))
    ax.plot([], [], color="goldenrod", linestyle="--", label="6 NM")
    ax.plot([], [], color="goldenrod", label="10 NM")
    limit = max(20.0, np.abs(combined[:, :2]).max() / 1000 * 1.05)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("East error [km]")
    ax.set_ylabel("North error [km]")
    ax.set_title("Combined scenario: one point per simulated instrument")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. by source
    ax = axes[0, 2]
    names = list(colors)[1:]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [summaries[n]["median"] for n in names], 0.4,
           color=[colors[n] for n in names], label="median")
    ax.bar(x + 0.2, [summaries[n]["p95"] for n in names], 0.4,
           color=[colors[n] for n in names], alpha=0.45, label="95th percentile")
    bands(ax)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" - ")[1].replace(" ", "\n") for n in names],
                       fontsize=8.5)
    ax.set_ylabel("Horizontal position error [m]")
    ax.set_title("Error by source (solid: median, pale: 95th percentile)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")

    # 4. latitude
    ax = axes[1, 0]
    lats = [r[0] for r in lat_rows]
    ax.semilogy(lats, [r[1]["median"] for r in lat_rows], "o-", color="tab:blue",
                label="combined, median horizontal error [m]")
    ax.semilogy(lats, [r[1]["p95"] for r in lat_rows], "o--", color="tab:blue",
                alpha=0.6, label="combined, 95th percentile [m]")
    ax.semilogy(lats, [r[2] for r in lat_rows], "s-", color="tab:purple",
                label="1 s clock error alone [m]")
    ax.set_xlabel("Observer latitude [deg]")
    ax.set_ylabel("Horizontal position error [m]")
    twin = ax.twinx()
    twin.semilogy(lats, [r[1]["rms_lon_arcsec"] / 3600 for r in lat_rows], "^:",
                  color="tab:red", label="RMS longitude error [deg]")
    twin.semilogy(lats, [r[1]["rms_lat_arcsec"] / 3600 for r in lat_rows], "v:",
                  color="tab:brown", label="RMS latitude error [deg]")
    twin.set_ylabel("Coordinate error [deg]")
    handles = ax.get_legend_handles_labels()
    handles2 = twin.get_legend_handles_labels()
    ax.legend(handles[0] + handles2[0], handles[1] + handles2[1], fontsize=7.5,
              loc="center left")
    ax.set_title("Latitude dependence")
    ax.grid(True, which="both", alpha=0.3)

    # 5. pointing
    ax = axes[1, 1]
    zeniths = [r[0] for r in pointing_rows]
    ax.semilogy(zeniths, [r[1]["median"] for r in pointing_rows], "o-",
                color="tab:green", label="camera attitude only, median")
    ax.semilogy(zeniths, [r[1]["p95"] for r in pointing_rows], "o--",
                color="tab:green", alpha=0.6, label="camera attitude only, 95th pct")
    ax.semilogy(zeniths, [r[2]["median"] for r in pointing_rows], "s-",
                color="tab:blue", label="combined, median")
    bands(ax)
    ax.set_xlabel("Camera boresight zenith angle [deg]")
    ax.set_ylabel("Horizontal position error [m]")
    ax.set_title("Position error versus camera pointing direction")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)

    # 6. clock
    ax = axes[1, 2]
    ax.loglog([r[0] for r in clock_rows], [r[1] for r in clock_rows], "o-",
              color="tab:purple", label="position error at 45° latitude")
    bands(ax)
    ax.set_xlabel("Timestamp error [s]")
    ax.set_ylabel("Horizontal position error [m]")
    ax.set_title("Clock error (east-west displacement)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)

    fig.suptitle("Terrestrial position error budget (illustrative error "
                 "sizes; fixed errors drawn independently per simulated instrument)")
    fig.tight_layout()
    fig.savefig(HERE / "position_error_experiment.png", dpi=130)
    print("\nSaved position_error_experiment.png")


if __name__ == "__main__":
    main()
