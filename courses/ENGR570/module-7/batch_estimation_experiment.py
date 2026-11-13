"""
batch_estimation_experiment.py - Space Systems Engineering, Module 7, Experiment A

Latitude and longitude from 20 noisy celestial-vertical observations of one
stationary instrument, by weighted nonlinear least squares.

Step 1  generate measurements through the real sensor chain (Modules 4-6)
Step 2  batch estimator with whitened tangent-plane residuals
Step 3  compare: one direct Module 6 fix, the average of 20 fixes, and the
        weighted batch
Step 4  make five observations much noisier and see what weighting does

Calibration is exact and there are no fixed biases in this experiment.

Run:  python batch_estimation_experiment.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

import navigation_estimator as ne
import navigation_scenarios as ns

SEED = 7
HERE = Path(__file__).parent
N_OBS = 20
BATCHES = 300
NOISY_FACTOR = 10.0
cp = ne.cp


def average_of_fixes(measurements):
    """Mean of the separately inverted positions and its reported covariance."""
    fixes = np.array([ns.single_fix(m) for m in measurements])
    mean = fixes.mean(axis=0)
    T = ne.radians_to_metres(np.radians(mean[0]))
    covariance = T @ (np.cov(np.radians(fixes).T) / len(fixes)) @ T.T
    return tuple(mean), covariance


def three_methods(measurements):
    """(estimate, reported NE covariance) for each of the three approaches."""
    first = ns.single_fix(measurements[0])
    single = ne.batch_position_estimate(measurements[:1], first)
    batch = ne.batch_position_estimate(measurements, first)
    return {
        "one direct fix": (single.lat_lon, single.position_covariance_ne),
        "average of 20 fixes": average_of_fixes(measurements),
        "weighted batch": (batch.lat_lon, batch.position_covariance_ne),
    }, batch


def monte_carlo(rng, noise_multipliers=None):
    errors = {}
    coverage = {}
    reported = {}
    for _ in range(BATCHES):
        measurements, _ = ns.simulate_batch(rng, N_OBS,
                                            noise_multipliers=noise_multipliers,
                                            run_quality_check=False)
        methods, _ = three_methods(measurements)
        for name, (estimate, covariance) in methods.items():
            error = ns.north_east_error(estimate)
            errors.setdefault(name, []).append(error)
            coverage.setdefault(name, []).append(
                ne.inside_confidence_ellipse(error, covariance))
            reported.setdefault(name, []).append(np.sqrt(np.trace(covariance)))
    return ({k: np.array(v) for k, v in errors.items()},
            {k: 100 * np.mean(v) for k, v in coverage.items()},
            {k: np.sqrt(np.mean(np.square(v))) for k, v in reported.items()})


def print_table(errors, coverage, reported):
    print(f"  {'method':<22}{'RMS horizontal':>16}{'RMS north':>11}{'RMS east':>10}"
          f"{'reported RMS':>14}{'95% coverage':>14}")
    for name in errors:
        e = errors[name]
        rms = np.sqrt(np.mean(e**2, axis=0))
        print(f"  {name:<22}{np.sqrt(np.mean(np.sum(e**2, axis=1))):>14.1f} m"
              f"{rms[0]:>9.1f} m{rms[1]:>8.1f} m{reported[name]:>12.1f} m"
              f"{coverage[name]:>13.1f}%")


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("Experiment A - batch terrestrial position estimation")
    print("=" * 78)
    print(f"Truth {ns.TRUE_POSITION} (never given to the estimator), "
          f"{N_OBS} observations {ns.SPACING_S:g} s apart from {ns.START_UTC} UTC.")
    print(f"Per observation: {ns.N_STARS} stars at {ns.CENTROID_NOISE_PX} px, "
          f"{ns.ACCEL_SAMPLES} accelerometer samples at {ns.ACCEL_NOISE} m/s^2.")

    # ---- one batch in detail -------------------------------------------------
    measurements, _ = ns.simulate_batch(rng, N_OBS)
    sigma = np.sqrt([np.trace(m.covariance) for m in measurements]) / ne.ARCSEC
    print(f"\nSteps 1-2. Supplied random-error sigma per measurement: "
          f"{sigma.mean():.1f} arcsec total ({sigma.min():.1f} to {sigma.max():.1f}), "
          f"about {sigma.mean() * ne.ARCSEC * 6.37e6:.0f} m on the ground.")
    methods, batch = three_methods(measurements)
    print(f"Batch fit: converged in {batch.diagnostics['function_evaluations']} "
          f"evaluations, chi2 = {batch.chi2:.1f} for {batch.dof} degrees of "
          f"freedom (reduced {batch.reduced_chi2:.2f}).")
    print(f"\nStep 3. One batch of {N_OBS}:")
    print(f"  {'method':<22}{'latitude':>13}{'longitude':>14}{'error':>10}"
          f"{'reported 1-sigma N, E':>26}")
    for name, (estimate, covariance) in methods.items():
        error = cp.horizontal_position_error(estimate, ns.TRUE_POSITION)
        s = np.sqrt(np.diag(covariance))
        print(f"  {name:<22}{estimate[0]:>13.6f}{estimate[1]:>14.6f}{error:>8.1f} m"
              f"{s[0]:>14.1f}, {s[1]:.1f} m")

    # ---- many batches ----------------------------------------------------------
    print(f"\n  Over {BATCHES} independent batches:")
    errors, coverage, reported = monte_carlo(rng)
    print_table(errors, coverage, reported)
    gain = (np.sqrt(np.mean(np.sum(errors["one direct fix"] ** 2, axis=1)))
            / np.sqrt(np.mean(np.sum(errors["weighted batch"] ** 2, axis=1))))
    print(f"""
  With equal-quality independent observations the batch improves on a single
  fix by {gain:.2f}x, against sqrt({N_OBS}) = {np.sqrt(N_OBS):.2f}. Averaging the 20 fixes does
  equally well here, because the measurements are nearly equally weighted
  and the problem is nearly linear over a few hundred metres. The batch's
  advantages are a covariance derived from the measurement model (the
  average's comes from the scatter of only 20 points, so its ellipse is
  noisier) and correct handling of unequal weights, time and frames.""")

    # ---- unequal quality -----------------------------------------------------------
    multipliers = np.ones(N_OBS)
    multipliers[-5:] = NOISY_FACTOR
    print(f"Step 4. Five of the {N_OBS} observations made {NOISY_FACTOR:g}x noisier:")
    measurements, _ = ns.simulate_batch(rng, N_OBS, noise_multipliers=multipliers)
    _, batch = three_methods(measurements)
    fraction = batch.diagnostics["information_fraction"]
    print(f"  Share of the batch's position information carried by the five "
          f"noisy observations: {100 * fraction[-5:].sum():.2f} %")
    print(f"  (each good observation {100 * fraction[:15].mean():.2f} %, each "
          f"noisy one {100 * fraction[-5:].mean():.3f} %; "
          f"1/{NOISY_FACTOR:g}^2 of the weight).")
    print(f"  Their whitened residuals stay ordinary: chi2 per observation "
          f"{batch.diagnostics['measurement_chi2'][-5:].mean():.2f} for the noisy "
          f"five, {batch.diagnostics['measurement_chi2'][:15].mean():.2f} for the "
          f"rest (expected about 2).\n")
    errors_u, coverage_u, reported_u = monte_carlo(rng, multipliers)
    print_table(errors_u, coverage_u, reported_u)
    rms = {k: np.sqrt(np.mean(np.sum(v**2, axis=1))) for k, v in errors_u.items()}
    print(f"""
  The plain average treats all 20 alike and is dragged by the noisy five:
  {rms['average of 20 fixes']:.0f} m against {rms['weighted batch']:.0f} m for the weighted batch, {rms['average of 20 fixes'] / rms['weighted batch']:.1f}x worse. The
  weighted estimator is close to what the fifteen good observations alone
  would give (sqrt(20/15) = 1.15x the all-good result), so the noisy
  observations are neither trusted nor allowed to do harm.""")

    make_figure(errors, errors_u, reported, reported_u)


def make_figure(errors, errors_unequal, reported, reported_unequal):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    colors = {"one direct fix": "tab:gray", "average of 20 fixes": "tab:orange",
              "weighted batch": "tab:blue"}

    for ax, data, title in ((axes[0], errors, "Equal-quality observations"),
                            (axes[1], errors_unequal,
                             f"Five observations {NOISY_FACTOR:g}× noisier")):
        for name in ("one direct fix", "average of 20 fixes", "weighted batch"):
            e = data[name]
            ax.scatter(e[:, 1], e[:, 0], s=7, alpha=0.45, color=colors[name],
                       label=name)
        batch = data["weighted batch"]
        covariance = np.cov(batch.T)
        values, vectors = np.linalg.eigh(covariance[::-1, ::-1])   # (east, north)
        angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
        scale = 2 * np.sqrt(5.991 * values)
        ax.add_patch(Ellipse((0, 0), scale[1], scale[0], angle=angle, fill=False,
                             color="tab:blue", linewidth=1.5,
                             label="batch: empirical 95% ellipse"))
        limit = np.percentile(np.abs(data["one direct fix"]), 99.5)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_xlabel("East error [m]")
        ax.set_ylabel("North error [m]")
        ax.set_title(f"{title}\n({BATCHES} batches of {N_OBS})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    names = list(colors)
    x = np.arange(len(names))
    for offset, data, rep, hatch, label in (
            (-0.2, errors, reported, None, "equal quality"),
            (0.2, errors_unequal, reported_unequal, "//", "five noisy")):
        actual = [np.sqrt(np.mean(np.sum(data[n] ** 2, axis=1))) for n in names]
        ax.bar(x + offset, actual, 0.4, color=[colors[n] for n in names],
               hatch=hatch, edgecolor="black", linewidth=0.5,
               label=f"actual RMS error, {label}")
        ax.plot(x + offset, [rep[n] for n in names], "k_", markersize=26,
                markeredgewidth=2)
    ax.plot([], [], "k_", markersize=14, markeredgewidth=2,
            label="reported RMS (from covariance)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" of ", "\nof ").replace("direct ", "direct\n")
                        for n in names])
    ax.set_ylabel("RMS horizontal error [m]")
    ax.set_title("Actual error versus reported uncertainty")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "batch_estimation_experiment.png", dpi=140)
    print("\nSaved batch_estimation_experiment.png")


if __name__ == "__main__":
    main()
