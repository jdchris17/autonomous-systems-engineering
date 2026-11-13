"""
precision_sensitivity.py - Space Systems Engineering, Module 8

Experiment B  optical and mechanical calibration sensitivity
Experiment C  Earth orientation, time and gravity-model sensitivity

Method for B: star images are generated with one set of TRUE camera and
mounting parameters and processed with deliberately wrong ones. The Module 4
solver estimates attitude from the mis-calibrated rays and Module 6/7 turns
it into a position, so each calibration error propagates through the image
and navigation models instead of being added as attitude noise afterwards.

Every error here is a fixed, known perturbation with all other error
sources switched off, unless stated. The numbers are sensitivities, not
predictions for any device.

Run:  python precision_sensitivity.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import navigation_error_budget as nb
from astronomical_time import EARTH_ROTATION_RATE_UT1

SEED = 88
HERE = Path(__file__).parent
TRUE_POSITION = (45.0, -75.0)
EARTH = nb.EarthOrientation(nb.BASELINE.start_utc)
PERFECT = nb.perfect(nb.BASELINE)
ARCSEC = nb.ARCSEC
ARCSEC_M = ARCSEC * 6.371e6
INSTRUMENTS = 30
cp = nb.cp


def metres(value):
    return f"{value / 1000:.2f} km" if value >= 1000 else f"{value:.1f} m"


def position_errors(scenario, fixed_for, instruments=INSTRUMENTS, seed=SEED):
    """
    Horizontal error [m] for several instruments (different headings, rolls
    and star fields) that all carry the fixed error returned by
    fixed_for(rng).
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(instruments):
        fixed = fixed_for(rng) if callable(fixed_for) else fixed_for
        measurements, truth = nb.simulate_instrument(scenario, rng, EARTH, fixed)
        position, _, _ = nb.solve_instrument(scenario, measurements, truth)
        out.append(cp.horizontal_position_error(position, TRUE_POSITION))
    return np.array(out)


def random_direction(rng, magnitude):
    v = rng.standard_normal(3)
    return magnitude * v / np.linalg.norm(v)


def in_plane(rng, magnitude):
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return magnitude * np.array([np.cos(angle), np.sin(angle)])


# ------------------------------------------------------------------ Experiment B
def experiment_b():
    print("=" * 78)
    print("Experiment B - optical and mechanical calibration sensitivity")
    print("=" * 78)
    focal = nb.CameraModel().fx
    print(f"Nominal camera 1920 x 1080, 20 deg HFOV, focal length {focal:.1f} px "
          f"({np.degrees(1 / focal) * 3600:.1f} arcsec per pixel).")
    print(f"Boresight 30 deg from zenith, 20 stars, 10 observations, "
          f"{INSTRUMENTS} instruments per case (random heading and roll).\n")
    results = {}

    print("Focal length error")
    print(f"  {'error':>10}{'edge shift':>13}{'median position error':>24}{'maximum':>12}")
    rows = []
    for percent in (0.01, 0.1, 1.0):
        for sign in (+1, -1):
            e = position_errors(PERFECT.with_changes(focal_length_sigma_fraction=1.0),
                                {"focal_fraction": sign * percent / 100})
            rows.append((sign * percent, np.median(e), e.max()))
            print(f"  {sign * percent:>+9.2f}%{960 * percent / 100:>10.2f} px"
                  f"{metres(np.median(e)):>24}{metres(e.max()):>12}")
    results["focal"] = rows

    print("\nPrincipal point error (direction in the image plane random)")
    print(f"  {'error':>10}{'as an angle':>14}{'median position error':>24}{'maximum':>12}")
    rows = []
    for pixels in (0.1, 0.5, 1.0, 5.0):
        e = position_errors(PERFECT.with_changes(principal_point_sigma_px=1.0),
                            lambda rng, p=pixels: {"principal_point_px": in_plane(rng, p)})
        rows.append((pixels, np.median(e), e.max()))
        print(f"  {pixels:>7.1f} px{np.degrees(pixels / focal) * 3600:>12.1f}\""
              f"{metres(np.median(e)):>24}{metres(e.max()):>12}")
    results["principal_point"] = rows

    print("\nRadial distortion left uncorrected")
    print("  Convention: x_d = x (1 + k1 r^2) in normalized image coordinates "
          "(x = X/Z, r = tan of the off-axis angle); k1 dimensionless.")
    print(f"  {'k1':>10}{'edge shift':>13}{'corner shift':>14}"
          f"{'median position error':>24}{'maximum':>12}")
    rows = []
    r_edge, r_corner = np.tan(np.radians(10.0)), np.hypot(960, 540) / focal
    for k1 in (0.002, 0.02, 0.2, -0.2):
        e = position_errors(PERFECT.with_changes(lens_k1=k1), None)
        rows.append((k1, np.median(e), e.max()))
        print(f"  {k1:>+10.3f}{abs(k1) * r_edge**3 * focal:>10.2f} px"
              f"{abs(k1) * r_corner**3 * focal:>11.2f} px"
              f"{metres(np.median(e)):>24}{metres(e.max()):>12}")
    results["distortion"] = rows

    print("\nImage-to-IMU timestamp misalignment (image taken after the IMU "
          "sample while the platform turns about a horizontal axis)")
    print(f"  {'offset':>10}" + "".join(f"{f'{r:g} deg/s':>14}" for r in (0.0, 0.1, 1.0, 10.0)))
    rows = []
    for ms in (1.0, 10.0, 100.0):
        line = f"  {ms:>7g} ms"
        for rate in (0.0, 0.1, 1.0, 10.0):
            e = position_errors(PERFECT.with_changes(image_imu_offset_s=ms / 1000,
                                                     platform_rate_deg_s=rate),
                                None, instruments=8)
            rows.append((ms, rate, np.median(e)))
            line += f"{metres(np.median(e)):>14}"
        print(line)
    print("  A truly static instrument is immune (the first column): the error "
          "is rotation rate x time offset.\n  Earth's own rotation is a "
          "clock error, treated in Experiment C, not a camera-IMU timing error.")
    results["timing"] = rows

    print("\nCamera-IMU misalignment (rotation vector, random direction)")
    print(f"  {'error':>10}{'median position error':>24}{'maximum':>12}{'max / (angle x R)':>20}")
    rows = []
    for arcsec in (1.0, 10.0, 100.0, 500.0):
        e = position_errors(PERFECT.with_changes(misalignment_sigma_arcsec=1.0),
                            lambda rng, a=arcsec: {"misalignment_rad":
                                                   random_direction(rng, a * ARCSEC)})
        rows.append((arcsec, np.median(e), e.max()))
        print(f"  {arcsec:>8g}\"{metres(np.median(e)):>24}{metres(e.max()):>12}"
              f"{e.max() / (arcsec * ARCSEC_M):>20.2f}")
    results["misalignment"] = rows

    # ---- boresight zenith angle -------------------------------------------------
    print("\nPosition error versus boresight zenith angle (median over instruments)")
    cases = {
        "principal point 1 px": (dict(principal_point_sigma_px=1.0),
                                 lambda rng: {"principal_point_px": in_plane(rng, 1.0)}),
        "focal length +0.1 %": (dict(focal_length_sigma_fraction=1.0),
                                {"focal_fraction": 1e-3}),
        "distortion k1 = 0.02": (dict(lens_k1=0.02), None),
        "misalignment 100\"": (dict(misalignment_sigma_arcsec=1.0),
                               lambda rng: {"misalignment_rad":
                                            random_direction(rng, 100 * ARCSEC)}),
    }
    zeniths = [0.0, 15.0, 30.0, 45.0, 60.0]
    print(f"  {'error':<24}" + "".join(f"{f'{z:g} deg':>12}" for z in zeniths))
    pointing = {}
    for label, (changes, fixed) in cases.items():
        pointing[label] = []
        for z in zeniths:
            e = position_errors(PERFECT.with_changes(zenith_angle_deg=z, **changes),
                                fixed, instruments=20)
            pointing[label].append(np.median(e))
        print(f"  {label:<24}" + "".join(f"{metres(v):>12}" for v in pointing[label]))
    results["pointing"] = (zeniths, pointing)

    # ---- diverse-attitude calibration ---------------------------------------------
    print("\nThe same errors with 20 observations, with random noise "
          "(0.2 px, 0.005 m/s^2), median error:")
    noisy = PERFECT.with_changes(centroid_noise_px=0.2, accel_noise=0.005,
                                 n_observations=20)
    print(f"  {'error':<24}{'fixed attitude':>16}{'diverse attitudes':>19}"
          f"{'diverse + calibration states':>30}")
    calibration_rows = {}
    for label, (changes, fixed) in cases.items():
        still = position_errors(noisy.with_changes(**changes), fixed, instruments=16)
        turning = position_errors(
            noisy.with_changes(diverse_attitudes=True, **changes), fixed,
            instruments=16)
        solved = position_errors(
            noisy.with_changes(diverse_attitudes=True, estimate_calibration=True,
                               declare_uncertainties=False, **changes),
            fixed, instruments=16)
        calibration_rows[label] = (np.median(still), np.median(turning),
                                   np.median(solved))
        print(f"  {label:<24}{metres(np.median(still)):>16}"
              f"{metres(np.median(turning)):>19}{metres(np.median(solved)):>30}")
    floor = position_errors(noisy.with_changes(diverse_attitudes=True,
                                               estimate_calibration=True,
                                               declare_uncertainties=False),
                            None, instruments=16)
    print(f"  {'(no calibration error)':<24}{'':>16}{'':>19}{metres(np.median(floor)):>30}")
    results["calibration"] = calibration_rows
    print(f"""
Which parameters matter, and which a rotation reveals
  Most important for latitude/longitude: anything that tilts the measured
  vertical relative to the stars. Camera-IMU misalignment maps one to one
  into position (about {ARCSEC_M:.0f} m per arcsecond) and a principal-point error
  does the same through the focal length ({np.degrees(1 / focal) * 3600 * ARCSEC_M:.0f} m per pixel here).
  Much less important: focal length and symmetric radial distortion. They
  change the image scale about the boresight, which the attitude solver
  cannot mistake for a tilt; what remains comes from the uneven spread of
  the stars in each field.
  Image-to-IMU timing only matters when the platform is turning.
  Made observable by deliberately rotating the instrument: misalignment and
  principal-point error. Both are fixed rotations in the instrument frame,
  so turning the instrument sweeps their effect around the sky while the
  true position stays put. Varying the attitude alone already averages
  most of it away (the middle column), and Module 7's calibration states
  remove the remainder for misalignment, bringing both down to the
  random-noise level of the last row.
  Focal-length and distortion errors are neither averaged nor absorbed;
  they are already small and need an optical calibration to go lower.""")
    return results


# ------------------------------------------------------------------ Experiment C
def solve_with_matrix(c_eg_believed):
    """Position reported when the true celestial vertical is interpreted
    with a perturbed Earth-orientation matrix (complete Module 6 solver)."""
    up_celestial = EARTH.matrix().T @ nb.ps.true_vertical_ecef(*TRUE_POSITION)
    estimate = cp.celestial_vertical_to_position(up_celestial, c_eg_believed)
    north, east = cp.north_east_error(estimate, TRUE_POSITION)
    return float(np.hypot(north, east)), north, east


def experiment_c():
    print("\n" + "=" * 78)
    print("Experiment C - Earth orientation, time and gravity-model uncertainty")
    print("=" * 78)
    results = {}

    # ---- C1 ------------------------------------------------------------------------
    print("C1 - clock offset (complete pipeline, everything else ideal)")
    print(f"  {'offset':>10}{'position error':>17}{'east':>12}{'north':>12}")
    per_second = None
    for dt in (0.001, 0.01, 0.1, 1.0, 10.0):
        measurements, truth = nb.simulate_instrument(
            PERFECT.with_changes(clock_sigma_s=1.0), np.random.default_rng(SEED),
            EARTH, {"clock_s": dt})
        position, _, _ = nb.solve_instrument(PERFECT, measurements, truth)
        north, east = cp.north_east_error(position, TRUE_POSITION)
        error = np.hypot(north, east)
        per_second = error / dt
        print(f"  {dt:>8g} s{metres(error):>17}{east:>10.2f} m{north:>10.3f} m")
    print(f"  {per_second:.1f} m per second at 45 deg latitude (x 1/cos(45 deg) "
          f"= {per_second / np.cos(np.radians(45)):.0f} m/s at the equator); "
          f"a fast clock puts the fix west.")

    print("\n  Clock drift dt(t) = dt0 + eps * t, with dt0 = 0: position error "
          "after an offline period")
    periods = {"1 hour": 3600.0, "1 day": 86400.0, "7 days": 7 * 86400.0,
               "30 days": 30 * 86400.0}
    fractions = [1e-5, 1e-6, 1e-7, 1e-8]
    print(f"  {'fractional error':>18}" + "".join(f"{p:>22}" for p in periods))
    drift = {}
    for eps in fractions:
        line = f"  {eps:>18.0e}"
        drift[eps] = []
        for seconds in periods.values():
            dt = eps * seconds
            error = dt * per_second
            drift[eps].append(error)
            line += f"{f'{dt:.4g} s, {metres(error)}':>22}"
        print(line)
    print(f"  Check: 1e-6 over one day is {1e-6 * 86400:.4f} s = "
          f"{1e-6 * 86400 * per_second:.1f} m. These are illustrative "
          f"oscillator cases, not hardware specifications.")
    results["drift"] = (periods, drift, per_second)

    # ---- C2 ------------------------------------------------------------------------
    print("\nC2 - Earth-orientation parameter errors")
    record = EARTH.record()
    print(f"  Data in use: {record['source']}, {record['days_predicted']:.1f} days "
          f"past the last measured value;\n  UT1-UTC = {record['ut1_minus_utc_s']:+.4f} s "
          f"(sigma {record['ut1_sigma_s'] * 1000:.1f} ms), polar motion "
          f"({record['polar_motion_arcsec'][0]:+.3f}, {record['polar_motion_arcsec'][1]:+.3f}) arcsec "
          f"(sigma {record['polar_motion_sigma_arcsec'] * 1000:.1f} mas).")
    print(f"  {'perturbation':<38}{'position error':>16}{'east':>11}{'north':>11}")
    perturbations = [
        ("UT1-UTC in error by 1 ms", dict(dut1_error_s=0.001)),
        ("UT1-UTC in error by 10 ms", dict(dut1_error_s=0.010)),
        ("UT1-UTC in error by 100 ms", dict(dut1_error_s=0.100)),
        ("polar motion x in error by 0.01\"", dict(xp_error_rad=0.01 * ARCSEC)),
        ("polar motion x in error by 0.1\"", dict(xp_error_rad=0.1 * ARCSEC)),
        ("polar motion y in error by 0.1\"", dict(yp_error_rad=0.1 * ARCSEC)),
        ("celestial pole offset dX = 1 mas", dict(dx_rad=0.001 * ARCSEC)),
        ("celestial pole offset dY = 1 mas", dict(dy_rad=0.001 * ARCSEC)),
        ("UT1-UTC ignored (UT1 := UTC)", dict(dut1_error_s=-EARTH.dut1_s)),
        ("polar motion ignored (pole := 0, 0)", dict(xp_error_rad=-EARTH.xp,
                                                     yp_error_rad=-EARTH.yp)),
    ]
    for label, kwargs in perturbations:
        error, north, east = solve_with_matrix(EARTH.matrix(**kwargs))
        print(f"  {label:<38}{metres(error):>16}{east:>9.2f} m{north:>9.2f} m")
    print("  IAU 2006/2000A already models the celestial pole to about a "
          "milliarcsecond, so the observed\n  corrections dX, dY are worth "
          "centimetres here and can be left out.")

    print("\n  Predicted data of increasing age (uncertainty growth as "
          "recalled from IERS Bulletin A; confirm before relying on it):")
    print(f"  {'days past last measurement':>28}{'UT1 sigma':>12}{'pole sigma':>13}"
          f"{'position, 1 sigma':>20}")
    ages = [0.0, 7.0, 13.5, 30.0, 90.0, 365.0, 730.0]
    aged = []
    for days in ages:
        ut1, pole = nb.prediction_uncertainty(days) if days > 0 else (2e-5, 5e-5)
        east = solve_with_matrix(EARTH.matrix(dut1_error_s=ut1))[0]
        pole_error = solve_with_matrix(EARTH.matrix(xp_error_rad=pole * ARCSEC,
                                                    yp_error_rad=pole * ARCSEC))[0]
        total = np.hypot(east, pole_error)
        aged.append(total)
        label = "measured (final values)" if days == 0 else f"{days:g}"
        print(f"  {label:>28}{ut1 * 1000:>9.2f} ms{pole * 1000:>9.1f} mas{metres(total):>20}")
    print("  An offline system therefore needs to carry the EOP source, the "
          "date of the last measured value\n  and these uncertainties with "
          "every solution; EarthOrientation.record() does that. Even two-year-"
          "old\n  predictions cost about ten metres, far below the other "
          "terms, but they are not zero.")
    results["eop_age"] = (ages, aged)

    # ---- C3 ------------------------------------------------------------------------
    print("\nC3 - deflection of the vertical (synthetic values; direction "
          "random; everything else ideal)")
    print(f"  {'deflection':>12}{'uncorrected':>14}{'corrected exactly':>20}"
          f"{'corrected to 1 arcsec':>24}")
    deflection_rows = []
    for magnitude in (1.0, 3.0, 5.0, 10.0, 30.0):
        def fixed(rng, m=magnitude):
            return {"deflection_arcsec": in_plane(rng, m)}
        base = PERFECT.with_changes(deflection_sigma_arcsec=1.0)
        raw = position_errors(base, fixed, instruments=12)
        exact = position_errors(base.with_changes(deflection_model_residual_arcsec=0.0),
                                fixed, instruments=12)
        partial = position_errors(
            base.with_changes(deflection_model_residual_arcsec=1.0), fixed,
            instruments=200)
        deflection_rows.append((magnitude, np.median(raw), np.median(exact),
                                np.sqrt(np.mean(partial**2))))
        print(f"  {magnitude:>10g}\"{metres(np.median(raw)):>14}"
              f"{np.median(exact) * 1000:>17.2f} mm"
              f"{metres(np.sqrt(np.mean(partial**2))) + ' RMS':>24}")
    print(f"""
  The plumb line is what an accelerometer measures; latitude and longitude
  are defined by the ellipsoid normal. An uncorrected deflection d moves
  the fix by d x R, {ARCSEC_M:.1f} m per arcsecond, in the direction of the
  deflection, whatever the instrument does. It is the same for every
  observation from one place, so it does not average and no amount of
  rotating the instrument reveals it.
  Against the 100-200 m objective: {100 / ARCSEC_M:.1f} arcsec uses the whole of a 100 m
  allowance and {200 / ARCSEC_M:.1f} arcsec the whole of 200 m. If deflection is to take
  no more than a third of a 150 m RMS budget (in quadrature), it must be
  known to about {150 / np.sqrt(3) / ARCSEC_M / np.sqrt(2):.1f} arcsec per component. Deflections of several
  arcseconds are ordinary and tens of arcseconds occur in mountainous
  terrain, so for the dedicated-hardware objective a gravity-model
  correction is required, not optional. For the 6-10 nautical mile
  objective it is negligible.
  These deflection values are synthetic assumptions. A real system must
  distinguish measured values, gravity-model predictions and assumptions.""")
    results["deflection"] = deflection_rows
    return results


def make_figures(b, c):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    ax = axes[0, 0]
    pp = np.array(b["principal_point"])
    ax.loglog(pp[:, 0], pp[:, 1], "o-", color="tab:blue", label="median")
    ax.loglog(pp[:, 0], pp[:, 2], "o--", color="tab:blue", alpha=0.6, label="maximum")
    ax.set_xlabel("Principal-point calibration error [px]")
    ax.set_title("Position error versus principal-point error")

    ax = axes[0, 1]
    focal = np.array(b["focal"])
    positive = focal[focal[:, 0] > 0]
    negative = focal[focal[:, 0] < 0]
    ax.loglog(positive[:, 0], positive[:, 1], "o-", color="tab:green",
              label="median, focal length too long")
    ax.loglog(-negative[:, 0], negative[:, 1], "s--", color="tab:olive",
              label="median, too short")
    ax.loglog(positive[:, 0], positive[:, 2], "o:", color="tab:green", alpha=0.6,
              label="maximum")
    ax.set_xlabel("Focal-length error [%]")
    ax.set_title("Position error versus focal-length error")

    ax = axes[1, 0]
    mis = np.array(b["misalignment"])
    ax.loglog(mis[:, 0], mis[:, 1], "o-", color="tab:red", label="median")
    ax.loglog(mis[:, 0], mis[:, 2], "o--", color="tab:red", alpha=0.6, label="maximum")
    ax.loglog(mis[:, 0], mis[:, 0] * ARCSEC_M, "k:", label="angle × Earth radius")
    ax.set_xlabel("Camera-IMU misalignment [arcsec]")
    ax.set_title("Position error versus camera-IMU misalignment")

    ax = axes[1, 1]
    zeniths, pointing = b["pointing"]
    for label, values in pointing.items():
        ax.semilogy(zeniths, values, "o-", label=label)
    ax.set_xlabel("Boresight zenith angle [deg]")
    ax.set_title("Position error versus boresight zenith angle")

    for ax in axes.flat:
        ax.axhspan(100, 200, color="limegreen", alpha=0.2)
        ax.axhspan(11100, 18500, color="gold", alpha=0.2)
        ax.set_ylabel("Horizontal position error [m]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Calibration sensitivity (green band: 100-200 m objective; "
                 "yellow: 6-10 NM)")
    fig.tight_layout()
    fig.savefig(HERE / "precision_sensitivity_calibration.png", dpi=130)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    ax = axes[0]
    periods, drift, _ = c["drift"]
    hours = np.array(list(periods.values())) / 3600.0
    for eps, values in drift.items():
        ax.loglog(hours, values, "o-", label=f"fractional frequency error {eps:.0e}")
    ax.set_xticks(hours)
    ax.set_xticklabels(list(periods))
    ax.minorticks_off()
    ax.set_xlabel("Time since the clock was last set")
    ax.set_title("C1: clock drift at 45° latitude")

    ax = axes[1]
    ages, aged = c["eop_age"]
    ax.loglog(np.maximum(ages, 0.5), aged, "o-", color="tab:purple")
    ax.set_xlabel("Days past the last measured Earth-orientation value\n"
                  "(leftmost point: measured values)")
    ax.set_title("C2: age of Earth-orientation predictions")

    ax = axes[2]
    rows = np.array(c["deflection"])
    ax.loglog(rows[:, 0], rows[:, 1], "o-", color="tab:red", label="uncorrected")
    ax.loglog(rows[:, 0], rows[:, 3], "s-", color="tab:blue",
              label="corrected to 1 arcsec (RMS)")
    ax.set_xlabel("Deflection of the vertical [arcsec]")
    ax.set_title("C3: deflection of the vertical")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.axhspan(100, 200, color="limegreen", alpha=0.2, label="100-200 m objective")
        ax.set_ylabel("Horizontal position error [m]")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "precision_sensitivity_time_geodesy.png", dpi=130)
    print("\nSaved precision_sensitivity_calibration.png and "
          "precision_sensitivity_time_geodesy.png")


if __name__ == "__main__":
    calibration = experiment_b()
    time_geodesy = experiment_c()
    make_figures(calibration, time_geodesy)
