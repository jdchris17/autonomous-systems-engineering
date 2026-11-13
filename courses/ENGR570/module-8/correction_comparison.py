"""
correction_comparison.py - Space Systems Engineering, Module 8, Experiment A

Idealized catalog directions versus what an observer on the ground really
sees, and what each omitted correction does to the recovered attitude and
terrestrial position.

A1  catalog -> propagated -> topocentric -> observed, per-star differences,
    then attitude fits against each reference model
A2  atmospheric refraction versus altitude, pressure and temperature
A3  a star field with differential refraction: uncorrected and corrected
    attitude fits, and an iterative solution that starts with no knowledge
    of the observer's position

The true position is used to generate the measurements and to score the
results. It is not given to the estimator in A3.

Run:  python correction_comparison.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import astrometric_corrections as ac
import atmospheric_refraction as ar
import navigation_error_budget as nb
from stellar_attitude import angular_residuals, attitude_error, wahba_svd

SEED = 8
HERE = Path(__file__).parent
UTC = nb.BASELINE.start_utc
TRUE_POSITION = (45.0, -75.0)
LOCATION = ac.ObserverLocation(*TRUE_POSITION, 0.0)
ATMOSPHERE = ar.STANDARD_ATMOSPHERE
N_STARS = 30
ARCSEC_PER_RAD = np.degrees(1.0) * 3600.0
cp, ps = nb.cp, nb.ps


def separation_arcsec(a, b):
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.degrees(np.arctan2(cross, np.sum(a * b, axis=-1))) * 3600.0


def metres(value):
    return f"{value / 1000:.2f} km" if value >= 1000 else f"{value:.1f} m"


def scene(zenith_angle_deg, rng, azimuth_deg=100.0, roll_deg=20.0):
    """A camera pointing, a catalog of stars in its field, and the truth."""
    camera = nb.CameraModel()
    c_nc = ps.camera_axes_local(zenith_angle_deg, azimuth_deg, roll_deg)
    rays = camera.unproject(nb.unit_square_positions(N_STARS, rng)
                            * [camera.width, camera.height])
    stars = ac.synthetic_catalog(rays, c_nc, UTC, LOCATION, rng)
    c_eg, _ = cp.earth_orientation(UTC)
    c_ge = np.asarray(c_eg.matrix).T
    c_en = ps.local_axes_ecef(*TRUE_POSITION)
    c_cg_true = c_nc.T @ c_en.T @ c_ge.T                 # celestial -> camera
    observed_celestial, observed_enu = ac.observed_directions(
        stars, UTC, LOCATION, ATMOSPHERE)
    measured = observed_enu @ c_nc                       # what the camera sees
    c_ci = ps.DEFAULT_C_CI
    up_imu = np.asarray(c_ci.matrix).T @ c_nc.T @ np.array([0.0, 0.0, 1.0])
    return {"stars": stars, "measured": measured, "C_CG": c_cg_true,
            "C_EG": c_eg, "C_CI": c_ci, "up_imu": up_imu, "C_NC": c_nc,
            "observed": observed_celestial}


def fit(scene_data, reference):
    """Attitude fit against a reference model, and the resulting position."""
    c_cg = wahba_svd(reference, scene_data["measured"])
    phi, total = attitude_error(c_cg, scene_data["C_CG"])
    residual = angular_residuals(c_cg, reference, scene_data["measured"])
    solution = cp.solve_terrestrial_position(scene_data["up_imu"],
                                             scene_data["C_CI"], c_cg,
                                             scene_data["C_EG"])
    return {"phi_arcsec": phi * ARCSEC_PER_RAD, "total_arcsec": total * ARCSEC_PER_RAD,
            "residual_rms_arcsec": np.sqrt(np.mean(residual**2)) * ARCSEC_PER_RAD,
            "position": solution.lat_lon,
            "position_error_m": cp.horizontal_position_error(solution.lat_lon,
                                                             TRUE_POSITION)}


# ----------------------------------------------------------------------- A1
def experiment_a1(rng):
    print("=" * 78)
    print("A1 - catalog epoch and apparent place")
    print("=" * 78)
    data = scene(50.0, rng)
    stars = data["stars"]
    catalog = ac.catalog_directions(stars)
    propagated = ac.propagated_directions(stars, UTC)
    topocentric, _ = ac.topocentric_directions(stars, UTC, LOCATION)
    observed = data["observed"]
    print(f"{N_STARS} synthetic stars, catalog epoch J{stars[0].epoch_jyear:.1f} "
          f"(Gaia DR3 convention), observed {UTC} UTC from {TRUE_POSITION}.")
    print(f"Proper motions ~100 mas/yr per axis, parallaxes up to 80 mas "
          f"(illustrative of bright nearby stars).")
    print(f"Camera boresight at altitude 40 deg; stars between "
          f"{np.degrees(np.arcsin(ac.observed_directions(stars, UTC, LOCATION, ATMOSPHERE)[1][:, 2])).min():.0f} "
          f"and {np.degrees(np.arcsin(ac.observed_directions(stars, UTC, LOCATION, ATMOSPHERE)[1][:, 2])).max():.0f} deg.\n")

    print("Per-star angular difference between consecutive representations "
          "(all in GCRS-oriented axes):")
    steps = [
        ("1 catalog -> 2 propagated to date", catalog, propagated,
         "proper motion over 10.9 years"),
        ("2 propagated -> 3 topocentric apparent", propagated, topocentric,
         "aberration (+ parallax, light deflection)"),
        ("3 topocentric -> 4 observed", topocentric, observed,
         "atmospheric refraction"),
    ]
    print(f"  {'step':<42}{'mean':>10}{'min':>10}{'max':>10}   cause")
    for label, a, b, cause in steps:
        d = separation_arcsec(a, b)
        print(f"  {label:<42}{d.mean():>9.2f}\"{d.min():>9.2f}\"{d.max():>9.2f}\"   {cause}")
    spread = separation_arcsec(propagated, topocentric)
    print(f"  Aberration is nearly the same for every star in the field "
          f"(spread {spread.max() - spread.min():.1f} arcsec), so it acts\n  "
          f"almost as a pure attitude offset. Proper motion is random star to "
          f"star, so it mostly averages.")

    print("\nAttitude and position when the measured (observed) rays are "
          "fitted with each reference model:")
    models = [
        ("4 observed (complete model)", observed),
        ("3 topocentric: refraction omitted", topocentric),
        ("2 propagated: + apparent place omitted", propagated),
        ("1 catalog: + proper motion omitted", catalog),
    ]
    rows = []
    print(f"  {'reference model':<42}{'attitude error':>16}{'x, y, z [arcsec]':>26}"
          f"{'residual RMS':>14}{'position error':>16}")
    for label, reference in models:
        r = fit(data, reference)
        rows.append((label, r))
        phi = r["phi_arcsec"]
        print(f"  {label:<42}{r['total_arcsec']:>14.2f}\""
              f"{phi[0]:>10.1f},{phi[1]:>7.1f},{phi[2]:>7.1f}"
              f"{r['residual_rms_arcsec']:>13.2f}\"{metres(r['position_error_m']):>16}")

    print("\nEach correction omitted on its own (everything else modelled):")
    singles = [
        ("refraction", observed - (observed - topocentric)),
        ("apparent place (aberration etc.)", observed - (topocentric - propagated)),
        ("proper motion", observed - (propagated - catalog)),
    ]
    single_rows = []
    print(f"  {'omitted correction':<42}{'attitude error':>16}{'residual RMS':>14}"
          f"{'position error':>16}")
    for label, reference in singles:
        reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
        r = fit(data, reference)
        single_rows.append((label, r))
        print(f"  {label:<42}{r['total_arcsec']:>14.2f}\""
              f"{r['residual_rms_arcsec']:>13.2f}\"{metres(r['position_error_m']):>16}")
    print("""
  Refraction is the largest omitted effect at this pointing and aberration
  the second; both shift the whole field in one direction, so the attitude
  solver absorbs them as a rotation and the residuals stay small. Proper
  motion is smaller and incoherent: it shows up mainly as residual scatter.
  Precession and nutation do not appear here at all. They are frame
  rotations and are already inside the Earth-orientation matrix; adding
  them to the star directions as well would apply them twice.""")
    return single_rows


# ----------------------------------------------------------------------- A2
def experiment_a2():
    print("\n" + "=" * 78)
    print("A2 - atmospheric refraction sweep")
    print("=" * 78)
    altitudes = np.array([10.0, 20.0, 30.0, 45.0, 60.0, 80.0])
    conditions = {
        "1013 hPa, 15 C (reference)": ar.Atmosphere(),
        "700 hPa, 15 C (about 3 km up)": ar.Atmosphere(pressure_hpa=700.0),
        "1040 hPa, 15 C": ar.Atmosphere(pressure_hpa=1040.0),
        "1013 hPa, -20 C": ar.Atmosphere(temperature_c=-20.0),
        "1013 hPa, +40 C": ar.Atmosphere(temperature_c=40.0),
    }
    print("Refractive displacement [arcsec] at 50 % humidity, 550 nm:")
    print(f"  {'conditions':<32}" + "".join(f"{f'{a:g} deg':>10}" for a in altitudes))
    for label, atmosphere in conditions.items():
        values = ar.refraction_arcsec(altitudes, atmosphere)
        print(f"  {label:<32}" + "".join(f"{v:>10.1f}" for v in values))
    reference = ar.refraction_arcsec(altitudes)
    simple = ar.simple_tan_refraction_arcsec(altitudes)
    empirical = ar.saemundsson_refraction_arcsec(altitudes)
    print(f"  {'k tan(z) approximation':<32}" + "".join(f"{v:>10.1f}" for v in simple))
    print(f"  {'  its error':<32}" + "".join(f"{v:>+10.1f}" for v in simple - reference))
    print(f"  {'Saemundsson (independent)':<32}" + "".join(f"{v:>10.1f}" for v in empirical))
    print(f"  {'ground distance of reference':<32}"
          + "".join(f"{metres(v * 30.9):>10}" for v in reference))
    print("""
  Refraction is roughly 1 arcminute at 45 deg altitude and 5 arcminutes at
  10 deg. It scales with pressure and inversely with absolute temperature:
  a 40 hPa or 15 C error in the assumed conditions is a few percent of the
  correction, i.e. arcseconds at low altitude.
  The one-term k tan(z) form is within 0.1 arcsec above 45 deg but is 1.7
  arcsec out at 20 deg and 14 at 10 deg. The two-term ERFA model used here
  is itself only specified to about 0.1 arcsec down to z = 70 deg and
  degrades below 20 deg altitude, which is why this module refuses zenith
  distances beyond 87 deg. Its few-percent disagreement with the
  independent empirical formula is within that formula's own accuracy.""")
    return altitudes, conditions


# ----------------------------------------------------------------------- A3
def iterative_solution(data, max_iterations=8):
    """
    Position with the full apparent-place and refraction model, starting
    from no knowledge of where the observer is.

    Iteration 0 fits the stars with position-independent directions
    (catalog propagated to the date). Each later pass computes the observed
    directions for the CURRENT position estimate, refits the attitude and
    re-solves the position. Only measurements, the catalog, the time and
    the atmosphere are used.
    """
    stars = data["stars"]
    history = []
    reference = ac.propagated_directions(stars, UTC)
    position = None
    for iteration in range(max_iterations):
        if position is not None:
            reference, _ = ac.observed_directions(
                stars, UTC, ac.ObserverLocation(*position), ATMOSPHERE)
        c_cg = wahba_svd(reference, data["measured"])
        new = cp.solve_terrestrial_position(data["up_imu"], data["C_CI"], c_cg,
                                            data["C_EG"]).lat_lon
        step = (np.inf if position is None
                else cp.horizontal_position_error(new, position))
        position = new
        history.append((iteration, position,
                        cp.horizontal_position_error(position, TRUE_POSITION), step))
        if step < 1e-4:
            break
    return history


def experiment_a3(rng):
    print("\n" + "=" * 78)
    print("A3 - differential refraction and attitude estimation")
    print("=" * 78)
    print(f"{N_STARS} stars across the 20 x 11.3 deg field, noise-free, "
          f"standard atmosphere. The uncorrected fit\nuses airless "
          f"topocentric directions; the corrected fit uses observed "
          f"directions.\n")
    print(f"  {'boresight':>10}{'star altitudes':>17}{'refraction':>13}"
          f"{'attitude bias':>15}{'x, y, z [arcsec]':>24}{'residual':>10}"
          f"{'position error':>16}{'corrected':>11}")
    rows = []
    data_for_iteration = None
    for zenith in (0.0, 15.0, 30.0, 50.0, 65.0):
        data = scene(zenith, rng)
        stars = data["stars"]
        topocentric, enu = ac.topocentric_directions(stars, UTC, LOCATION)
        altitude = np.degrees(np.arcsin(enu[:, 2]))
        refraction = ar.refraction_arcsec(altitude)
        uncorrected = fit(data, topocentric)
        corrected = fit(data, data["observed"])
        rows.append((zenith, uncorrected, corrected, refraction))
        phi = uncorrected["phi_arcsec"]
        print(f"  {f'{zenith:g} deg':>10}{f'{altitude.min():.0f}-{altitude.max():.0f} deg':>17}"
              f"{f'{refraction.min():.0f}-{refraction.max():.0f}':>12}\""
              f"{uncorrected['total_arcsec']:>14.1f}\""
              f"{phi[0]:>9.1f},{phi[1]:>6.1f},{phi[2]:>6.1f}"
              f"{uncorrected['residual_rms_arcsec']:>9.2f}\""
              f"{metres(uncorrected['position_error_m']):>16}"
              f"{corrected['position_error_m'] * 1000:>9.3f} mm")
        if zenith == 50.0:
            data_for_iteration = data
    print("""
  The fit absorbs the mean refraction across the field as a tilt of the
  camera toward the horizon, so the attitude bias is close to the
  refraction at the boresight and the position moves by the same angle.
  Only the DIFFERENTIAL part, the change of refraction across the field,
  is left in the residuals, and it is a small fraction of the bias:
  residuals would not reveal the problem. Looking straight up, refraction
  is symmetric about the boresight, the bias vanishes and only a slight
  change of image scale remains.""")

    print("  Iterative solution at 50 deg zenith angle (no position supplied):")
    print(f"  {'pass':>6}{'estimated latitude':>21}{'longitude':>14}"
          f"{'error vs truth':>17}{'change from last pass':>24}")
    history = iterative_solution(data_for_iteration)
    for iteration, position, error, step in history:
        step_text = "-" if np.isinf(step) else f"{step:.2e} m"
        print(f"  {iteration:>6}{position[0]:>21.9f}{position[1]:>14.9f}"
              f"{metres(error) if error >= 1 else f'{error * 1000:.3f} mm':>17}"
              f"{step_text:>24}")
    print("""
  The corrections depend on the position being solved for, but only
  weakly: a kilometre of position error changes the direction of the local
  vertical, and hence of the refraction, by half an arcminute, which
  changes a one-arcminute correction by a negligible amount. Starting from
  the uncorrected answer, each pass shrinks the error by about a factor of
  a thousand: kilometres, then metres, then millimetres. The circular dependence is handled
  by iteration, and the true position never enters the estimator.""")
    return rows, history


def make_figure(single_rows, a2, a3_rows, history):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))

    ax = axes[0, 0]
    altitudes = np.linspace(8.0, 89.0, 200)
    _, conditions = a2
    styles = ["k-", "b--", "c--", "r:", "m:"]
    for (label, atmosphere), style in zip(conditions.items(), styles):
        ax.semilogy(altitudes, ar.refraction_arcsec(altitudes, atmosphere), style,
                    label=label)
    ax.semilogy(altitudes, ar.simple_tan_refraction_arcsec(altitudes), color="gray",
                linestyle="-.", label="k tan(z) approximation")
    ax.plot(a2[0], ar.refraction_arcsec(a2[0]), "ko", markersize=5)
    ax.set_xlabel("Geometric stellar altitude [deg]")
    ax.set_ylabel("Refractive displacement [arcsec]")
    ax.set_title("A2: refraction versus altitude")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    labels = [row[0].replace(" (", "\n(") for row in single_rows]
    attitude = [row[1]["total_arcsec"] for row in single_rows]
    position = [row[1]["position_error_m"] for row in single_rows]
    x = np.arange(len(labels))
    bars = ax.bar(x, attitude, color=["tab:blue", "tab:orange", "tab:green"])
    for bar, value, pos in zip(bars, attitude, position):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.1,
                f'{value:.1f}"\n{metres(pos)}', ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.05, max(attitude) * 6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Attitude error [arcsec]")
    ax.set_title("A1: attitude and position error from each omitted correction\n"
                 "(boresight at 40° altitude)")
    ax.grid(True, axis="y", which="both", alpha=0.3)

    ax = axes[1, 0]
    zenith = [row[0] for row in a3_rows]
    ax.semilogy(zenith, [row[1]["position_error_m"] for row in a3_rows], "o-",
                color="tab:red", label="position error, refraction ignored")
    ax.semilogy(zenith, [max(row[2]["position_error_m"], 1e-7) for row in a3_rows],
                "s-", color="tab:blue", label="position error, refraction modelled")
    ax.semilogy(zenith, [row[1]["residual_rms_arcsec"] * 30.9 for row in a3_rows],
                "^:", color="gray",
                label="fit residual RMS (as metres), refraction ignored")
    ax.axhspan(100, 200, color="limegreen", alpha=0.25, label="100-200 m objective")
    ax.set_xlabel("Camera boresight zenith angle [deg]")
    ax.set_ylabel("Horizontal position error [m]")
    ax.set_title("A3: unmodelled refraction versus pointing direction")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.semilogy([h[0] for h in history], [max(h[2], 1e-7) for h in history], "o-",
                color="tab:purple")
    ax.set_xticks([h[0] for h in history])
    ax.set_xlabel("Iteration (0 = position-independent star model)")
    ax.set_ylabel("Position error versus truth [m]")
    ax.set_title("A3: iterating the position-dependent corrections")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(HERE / "correction_comparison.png", dpi=140)
    print("\nSaved correction_comparison.png")


if __name__ == "__main__":
    generator = np.random.default_rng(SEED)
    singles = experiment_a1(generator)
    sweep = experiment_a2()
    a3, iteration_history = experiment_a3(generator)
    make_figure(singles, sweep, a3, iteration_history)
