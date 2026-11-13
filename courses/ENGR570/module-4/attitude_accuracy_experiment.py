"""
attitude_accuracy_experiment.py - Space Systems Engineering, Module 4

Experiment A: TRIAD versus SVD, noise-free and with 0.2 px centroid noise.
Experiment B: attitude accuracy versus star count and centroid noise.

All errors are attitude-error rotation vectors in camera axes: x and y are
pointing (boresight tilt), z is roll about the optical axis.

Run:  python attitude_accuracy_experiment.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from star_field_simulation import (
    ARCSEC_PER_RAD,
    PinholeCamera,
    centroid_sigma_to_angle,
    observe_with_centroid_noise,
    stars_in_field,
    to_celestial,
    true_attitude_matrix,
)
from stellar_attitude import (
    attitude_covariance,
    attitude_error,
    attitude_information,
    triad,
    wahba_svd,
)

SEED = 4
HERE = Path(__file__).parent
CAMERA = PinholeCamera()
C_TRUE = true_attitude_matrix()

STAR_COUNTS = [2, 3, 5, 10, 20, 40]
NOISE_LEVELS_PX = [0.05, 0.1, 0.2, 0.5, 1.0]
TRIALS_B = 200


def separation_deg(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b))


def widest_pair(vectors):
    cosines = vectors @ vectors.T
    i, j = np.unravel_index(np.argmin(cosines), cosines.shape)
    return int(i), int(j)


def closest_pair(vectors):
    cosines = vectors @ vectors.T
    np.fill_diagonal(cosines, -1.0)
    i, j = np.unravel_index(np.argmax(cosines), cosines.shape)
    return int(i), int(j)


def predicted_sigmas_arcsec(camera_vectors, sigma_px, camera=CAMERA):
    """1-sigma attitude error about camera x, y, z from the information matrix."""
    sigma = centroid_sigma_to_angle(camera, camera_vectors, sigma_px)
    covariance = attitude_covariance(attitude_information(camera_vectors, sigma))
    return np.sqrt(np.diag(covariance)) * ARCSEC_PER_RAD


def monte_carlo(reference, camera_vectors, sigma_px, trials, rng, solver="svd"):
    """Error rotation vectors [arcsec], shape (trials, 3)."""
    weights = 1.0 / centroid_sigma_to_angle(CAMERA, camera_vectors, sigma_px) ** 2
    errors = np.empty((trials, 3))
    for t in range(trials):
        measured = observe_with_centroid_noise(CAMERA, camera_vectors,
                                               sigma_px, rng)
        if solver == "triad":
            C = triad(reference, measured)
        else:
            C = wahba_svd(reference, measured, weights)
        errors[t] = attitude_error(C, C_TRUE)[0]
    return errors * ARCSEC_PER_RAD


def summarize(errors):
    total = np.linalg.norm(errors, axis=1)
    return {
        "median": np.median(total),
        "p95": np.percentile(total, 95),
        "rms_xyz": np.sqrt(np.mean(errors**2, axis=0)),
    }


# ------------------------------------------------------------------ Experiment A
def experiment_a(rng):
    print("=" * 78)
    print("Experiment A - TRIAD versus SVD")
    print("=" * 78)
    camera_vectors = stars_in_field(CAMERA, 20, rng)
    reference = to_celestial(camera_vectors)
    i, j = widest_pair(camera_vectors)
    pair = [i, j]
    print(f"20 stars in a {CAMERA.horizontal_fov_deg:g} x "
          f"{CAMERA.vertical_fov_deg:.1f} deg field. TRIAD uses the widest "
          f"pair, stars {i} and {j}, {separation_deg(camera_vectors[i], camera_vectors[j]):.2f} deg apart.")

    print("\nNoise-free observations:")
    theta_triad = attitude_error(triad(reference[pair], camera_vectors[pair]),
                                 C_TRUE)[1]
    theta_svd = attitude_error(wahba_svd(reference, camera_vectors), C_TRUE)[1]
    print(f"  theta_TRIAD = {theta_triad * ARCSEC_PER_RAD:.2e} arcsec")
    print(f"  theta_SVD   = {theta_svd * ARCSEC_PER_RAD:.2e} arcsec")

    sigma_px, trials = 0.2, 100
    print(f"\n{sigma_px} px centroid noise "
          f"({sigma_px / CAMERA.fx * ARCSEC_PER_RAD:.2f} arcsec per axis on "
          f"the boresight), {trials} trials:")
    cases = {
        "TRIAD, 2 widest stars": monte_carlo(reference[pair], camera_vectors[pair],
                                             sigma_px, trials, rng, "triad"),
        "SVD, same 2 stars": monte_carlo(reference[pair], camera_vectors[pair],
                                         sigma_px, trials, rng),
        "SVD, all 20 stars": monte_carlo(reference, camera_vectors, sigma_px,
                                         trials, rng),
    }
    predicted = {
        "TRIAD, 2 widest stars": predicted_sigmas_arcsec(camera_vectors[pair], sigma_px),
        "SVD, same 2 stars": predicted_sigmas_arcsec(camera_vectors[pair], sigma_px),
        "SVD, all 20 stars": predicted_sigmas_arcsec(camera_vectors, sigma_px),
    }
    print(f"  {'estimator':<24}{'median':>9}{'95th pct':>10}"
          f"{'RMS x':>9}{'RMS y':>9}{'RMS z':>9}   predicted 1-sigma x, y, z")
    for label, errors in cases.items():
        s = summarize(errors)
        p = predicted[label]
        print(f"  {label:<24}{s['median']:>9.2f}{s['p95']:>10.2f}"
              f"{s['rms_xyz'][0]:>9.2f}{s['rms_xyz'][1]:>9.2f}"
              f"{s['rms_xyz'][2]:>9.2f}   {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}")
    print("  (arcsec; the information-matrix prediction for two stars is the "
          "optimal two-star\n   estimator's, which SVD attains and TRIAD "
          "approaches)")

    fig, (ax_hist, ax_axes) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["tab:red", "tab:orange", "tab:blue"]
    all_totals = np.concatenate([np.linalg.norm(e, axis=1) for e in cases.values()])
    bins = np.logspace(np.log10(all_totals.min() * 0.8),
                       np.log10(all_totals.max() * 1.2), 28)
    for (label, errors), color in zip(cases.items(), colors):
        ax_hist.hist(np.linalg.norm(errors, axis=1), bins=bins, alpha=0.55,
                     color=color, label=label)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("Total attitude error [arcsec]")
    ax_hist.set_ylabel(f"Trials (of {trials})")
    ax_hist.set_title(f"Total attitude error, {sigma_px} px centroid noise")
    ax_hist.grid(True, which="both", alpha=0.3)
    ax_hist.legend()

    x = np.arange(3)
    for k, ((label, errors), color) in enumerate(zip(cases.items(), colors)):
        ax_axes.bar(x + (k - 1) * 0.27, summarize(errors)["rms_xyz"], 0.27,
                    color=color, label=label)
    ax_axes.set_yscale("log")
    ax_axes.set_xticks(x)
    ax_axes.set_xticklabels(["about x\n(pointing)", "about y\n(pointing)",
                             "about z\n(roll)"])
    ax_axes.set_ylabel("RMS attitude error [arcsec]")
    ax_axes.set_title("Per-axis RMS error")
    ax_axes.grid(True, axis="y", which="both", alpha=0.3)
    ax_axes.legend()
    fig.tight_layout()
    fig.savefig(HERE / "attitude_accuracy_triad_vs_svd.png", dpi=150)
    plt.close(fig)
    print("Saved attitude_accuracy_triad_vs_svd.png")


# ------------------------------------------------------------------ Experiment B
def experiment_b(rng):
    print("\n" + "=" * 78)
    print("Experiment B - star count and centroid noise (SVD, "
          f"{TRIALS_B} trials each)")
    print("=" * 78)
    # One master field of 40 stars. A run with N stars uses the first N, so
    # every larger set contains the smaller ones and the attitude is fixed.
    master = stars_in_field(CAMERA, max(STAR_COUNTS), rng)
    reference = to_celestial(master)
    print(f"Star selection: first N of one fixed {max(STAR_COUNTS)}-star field. "
          f"The two-star case is {separation_deg(master[0], master[1]):.2f} deg apart.")

    results = {}
    print(f"\n  {'N':>3}{'noise':>7}{'median':>9}{'95th':>9}"
          f"{'RMS x':>9}{'RMS y':>9}{'RMS z':>10}"
          f"{'pred x':>9}{'pred y':>9}{'pred z':>10}  [arcsec]")
    for sigma_px in NOISE_LEVELS_PX:
        for n in STAR_COUNTS:
            errors = monte_carlo(reference[:n], master[:n], sigma_px,
                                 TRIALS_B, rng)
            s = summarize(errors)
            p = predicted_sigmas_arcsec(master[:n], sigma_px)
            results[(n, sigma_px)] = (s, p)
            r = s["rms_xyz"]
            print(f"  {n:>3}{sigma_px:>5g}px{s['median']:>9.2f}{s['p95']:>9.2f}"
                  f"{r[0]:>9.2f}{r[1]:>9.2f}{r[2]:>10.2f}"
                  f"{p[0]:>9.2f}{p[1]:>9.2f}{p[2]:>10.2f}")
        print()

    # Not every two-star configuration is equally well conditioned.
    print("Two stars are not all alike (0.2 px noise, same field):")
    print(f"  {'pair':<22}{'separation':>12}{'RMS x':>9}{'RMS y':>9}"
          f"{'RMS z':>10}{'pred z':>10}{'cond(J)':>11}")
    for label, (i, j) in (("closest pair", closest_pair(master)),
                          ("first two (used above)", (0, 1)),
                          ("widest pair", widest_pair(master))):
        pair = [i, j]
        errors = monte_carlo(reference[pair], master[pair], 0.2, TRIALS_B, rng)
        r = summarize(errors)["rms_xyz"]
        sigma = centroid_sigma_to_angle(CAMERA, master[pair], 0.2)
        J = attitude_information(master[pair], sigma)
        p = np.sqrt(np.diag(attitude_covariance(J))) * ARCSEC_PER_RAD
        print(f"  {label:<22}{separation_deg(master[i], master[j]):>8.2f} deg"
              f"{r[0]:>9.2f}{r[1]:>9.2f}{r[2]:>10.1f}{p[2]:>10.1f}"
              f"{np.linalg.cond(J):>11.0f}")

    ratio_20 = results[(20, 0.2)][0]["rms_xyz"]
    print(f"\nWith 20 stars at 0.2 px: pointing {ratio_20[:2].mean():.2f} arcsec, "
          f"roll {ratio_20[2]:.1f} arcsec, ratio {ratio_20[2] / ratio_20[:2].mean():.0f}x.")
    print("Both improve as 1/sqrt(N) once the field is filled. Roll stays "
          "worse by roughly 1 / (RMS\nangular distance of the stars from the "
          "boresight), because a star's lever arm for roll is its\noff-axis "
          "angle while its lever arm for pointing is the full unit vector.")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    colors = {"x": "tab:blue", "y": "tab:green", "z": "tab:red"}
    for ax, sigma_px in zip(axes.flat, NOISE_LEVELS_PX):
        for k, axis_name in enumerate("xyz"):
            rms = [results[(n, sigma_px)][0]["rms_xyz"][k] for n in STAR_COUNTS]
            pred = [results[(n, sigma_px)][1][k] for n in STAR_COUNTS]
            role = "roll" if axis_name == "z" else "pointing"
            ax.loglog(STAR_COUNTS, rms, "o-", color=colors[axis_name],
                      linewidth=2, label=f"about {axis_name} ({role}), Monte Carlo")
            ax.loglog(STAR_COUNTS, pred, "--", color=colors[axis_name],
                      linewidth=1, label=f"about {axis_name}, predicted")
        ax.set_title(f"Centroid noise {sigma_px:g} px")
        ax.set_xticks(STAR_COUNTS)
        ax.set_xticklabels([str(n) for n in STAR_COUNTS])
        ax.minorticks_off()
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylabel("RMS attitude error [arcsec]")
    axes[0, 0].legend(fontsize=7.5)

    ax = axes[1, 2]
    for sigma_px in NOISE_LEVELS_PX:
        ratio = [results[(n, sigma_px)][0]["rms_xyz"][2]
                 / results[(n, sigma_px)][0]["rms_xyz"][:2].mean()
                 for n in STAR_COUNTS]
        ax.semilogx(STAR_COUNTS, ratio, "o-", label=f"{sigma_px:g} px")
    ax.set_title("Roll error / mean pointing error")
    ax.set_xticks(STAR_COUNTS)
    ax.set_xticklabels([str(n) for n in STAR_COUNTS])
    ax.minorticks_off()
    ax.set_ylabel("Ratio")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Centroid noise", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Number of matched stars")
    fig.suptitle(f"SVD attitude error versus star count "
                 f"({CAMERA.horizontal_fov_deg:g}° HFOV, {TRIALS_B} trials per point)")
    fig.tight_layout()
    fig.savefig(HERE / "attitude_accuracy_star_count.png", dpi=150)
    plt.close(fig)
    print("Saved attitude_accuracy_star_count.png")


if __name__ == "__main__":
    generator = np.random.default_rng(SEED)
    experiment_a(generator)
    experiment_b(generator)
