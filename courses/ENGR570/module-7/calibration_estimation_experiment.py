"""
calibration_estimation_experiment.py - Space Systems Engineering, Module 7

Companion to Experiment C3: actually estimating the camera-IMU calibration
error together with position, for the four attitude sequences of
navigation_observability.py.

An instrument has a fixed, unknown misalignment. Three estimators process
the same observations:
  ignore   position only, misalignment not modelled
  solve    position + three calibration states, no prior
  prior    position + calibration states with the calibration uncertainty
           as a Gaussian prior (the "uncertain shared state" alternative to
           Experiment B's cross-observation covariance)

Absolute time is known and there is no accelerometer bias here.

Run:  python calibration_estimation_experiment.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import navigation_estimator as ne
import navigation_scenarios as ns
from navigation_observability import ATTITUDE_CONFIGURATIONS

SEED = 770
HERE = Path(__file__).parent
TRUE_MISALIGNMENT = np.radians([0.05, -0.03, 0.04])
INSTRUMENTS = 200
cp = ne.cp


def metres(value):
    return f"{value / 1000:.2f} km" if value >= 1000 else f"{value:.0f} m"


def fit_all(measurements):
    start = ns.single_fix(measurements[0])
    return {
        "ignore": ne.batch_position_estimate(measurements, start),
        "solve": ne.batch_position_estimate(measurements, start,
                                            estimate_calibration=True),
        "prior": ne.batch_position_estimate(
            measurements, start, estimate_calibration=True,
            calibration_prior_sigma_rad=ns.MISALIGNMENT_SIGMA),
    }


def main():
    rng = np.random.default_rng(SEED)
    arcsec = np.degrees(TRUE_MISALIGNMENT) * 3600
    print(f"True misalignment (rotation vector, camera axes): "
          f"[{arcsec[0]:.0f}, {arcsec[1]:.0f}, {arcsec[2]:.0f}] arcsec; "
          f"prior sigma {np.degrees(ns.MISALIGNMENT_SIGMA) * 3600:.0f} arcsec per axis.\n")

    for label, attitudes in ATTITUDE_CONFIGURATIONS.items():
        measurements, _ = ns.simulate_batch(
            rng, len(attitudes), attitudes=attitudes,
            misalignment_rad=TRUE_MISALIGNMENT, run_quality_check=False)
        fits = fit_all(measurements)
        print(f"{label}")
        print(f"  {'estimator':<10}{'position error':>16}{'reported sigma':>17}"
              f"{'reduced chi2':>14}  observability / calibration estimate [arcsec]")
        for name, fit in fits.items():
            error = cp.horizontal_position_error(fit.lat_lon, ns.TRUE_POSITION)
            sigma = ("withheld" if fit.position_covariance_ne is None else
                     metres(np.sqrt(np.trace(fit.position_covariance_ne))))
            text = fit.observability
            if fit.calibration_rad is not None and fit.covariance is not None:
                estimate = np.degrees(fit.calibration_rad) * 3600
                cal_sigma = np.degrees(np.sqrt(np.diag(fit.covariance)[2:])) * 3600
                text += "; " + ", ".join(f"{e:+.0f} ± {s:.0f}"
                                         for e, s in zip(estimate, cal_sigma))
            print(f"  {name:<10}{metres(error):>16}{sigma:>17}"
                  f"{fit.reduced_chi2:>14.2f}  {text}")
        print()

    print("""Reading the table
  ignore  A fixed attitude hides the misalignment completely: the fit is
          clean (reduced chi2 near 1) and the position is kilometres off.
          Once the attitude varies, the same unmodelled error makes the
          observations disagree and the reduced chi2 rises far above 1,
          so varied attitudes also make the problem DETECTABLE.
  solve   Refuses to report where the geometry cannot support five states
          (A, and B's roll-axis component). With diverse attitudes it
          recovers the misalignment to its stated uncertainty and the
          position error falls from kilometres to the random-noise level.
  prior   Always solvable. From a fixed attitude the data cannot improve
          on the prior, so the position sigma honestly stays at kilometres;
          as attitudes diversify, the data take over from the prior.
""")

    # ---- many instruments, diverse attitudes -----------------------------------
    print(f"{INSTRUMENTS} instruments, each with its own random misalignment, "
          f"20 diverse attitudes (configuration D):")
    attitudes = ATTITUDE_CONFIGURATIONS["D: 20 deliberately diverse"]
    records = {name: {"error": [], "inside": [], "cal_error": [], "chi2": []}
               for name in ("ignore", "solve")}
    for _ in range(INSTRUMENTS):
        truth = rng.normal(0.0, ns.MISALIGNMENT_SIGMA, 3)
        measurements, _ = ns.simulate_batch(rng, 20, attitudes=attitudes,
                                            misalignment_rad=truth,
                                            run_quality_check=False)
        start = ns.single_fix(measurements[0])
        fits = {"ignore": ne.batch_position_estimate(measurements, start),
                "solve": ne.batch_position_estimate(measurements, start,
                                                    estimate_calibration=True)}
        for name, fit in fits.items():
            error = ns.north_east_error(fit.lat_lon)
            records[name]["error"].append(np.linalg.norm(error))
            records[name]["chi2"].append(fit.reduced_chi2)
            records[name]["inside"].append(
                ne.inside_confidence_ellipse(error, fit.position_covariance_ne))
            if fit.calibration_rad is not None:
                normalized = (fit.calibration_rad - truth) / np.sqrt(
                    np.diag(fit.covariance)[2:])
                records[name]["cal_error"].append(normalized)
    print(f"  {'estimator':<10}{'RMS position error':>20}{'95% coverage':>15}"
          f"{'median reduced chi2':>22}")
    for name, r in records.items():
        print(f"  {name:<10}{metres(np.sqrt(np.mean(np.square(r['error'])))):>20}"
              f"{100 * np.mean(r['inside']):>14.1f}%{np.median(r['chi2']):>22.2f}")
    normalized = np.array(records["solve"]["cal_error"])
    print(f"  Calibration estimates: normalized error (estimate - truth)/sigma "
          f"has standard deviation {normalized.std(axis=0).round(2)} per axis "
          f"(1 expected).")
    print("  Averaging over varied attitudes already reduces the 'ignore' "
          "error well below the fixed-attitude\n  case, because the "
          "misalignment's effect points in different directions; "
          "solving for it removes\n  the rest and restores a trustworthy "
          "covariance.")

    fig, (ax_error, ax_cal) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    bins = np.logspace(0.5, 4.3, 45)
    ax_error.hist(records["ignore"]["error"], bins=bins, alpha=0.6,
                  color="tab:red", label="misalignment ignored")
    ax_error.hist(records["solve"]["error"], bins=bins, alpha=0.6,
                  color="tab:blue", label="misalignment estimated")
    ax_error.set_xscale("log")
    ax_error.set_xlabel("Horizontal position error [m]")
    ax_error.set_ylabel(f"Instruments (of {INSTRUMENTS})")
    ax_error.set_title("Position error with 20 diverse attitudes")
    ax_error.grid(True, which="both", alpha=0.3)
    ax_error.legend()

    grid = np.linspace(-4, 4, 200)
    for axis, color in zip(range(3), ("tab:blue", "tab:green", "tab:purple")):
        ax_cal.hist(normalized[:, axis], bins=np.linspace(-4, 4, 33), density=True,
                    histtype="step", linewidth=1.6, color=color,
                    label=f"δθ {'xyz'[axis]}")
    ax_cal.plot(grid, np.exp(-grid**2 / 2) / np.sqrt(2 * np.pi), "k-",
                label="unit normal")
    ax_cal.set_xlabel("(estimated − true misalignment) / reported sigma")
    ax_cal.set_ylabel("Density")
    ax_cal.set_title("Calibration estimates are consistent with their covariance")
    ax_cal.grid(True, alpha=0.3)
    ax_cal.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "calibration_estimation_experiment.png", dpi=140)
    print("\nSaved calibration_estimation_experiment.png")


if __name__ == "__main__":
    main()
