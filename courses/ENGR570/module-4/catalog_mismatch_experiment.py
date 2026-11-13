"""
catalog_mismatch_experiment.py - Space Systems Engineering, Module 4, Experiment D

What a wrong star identification does to the attitude, and how well two
robust estimators cope.

Part 1 - one wrong match among 20: plain SVD, the residuals, and iterative
         outlier rejection.
Part 2 - increasing numbers of wrong matches: plain SVD, iterative
         rejection and RANSAC (stretch challenge), including how often each
         method fails and whether it knows it has failed.

A "wrong match" keeps the measured camera direction but pairs it with the
catalog direction of a different star.

Run:  python catalog_mismatch_experiment.py
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
    angular_residuals,
    attitude_covariance,
    attitude_error,
    attitude_information,
    ransac_attitude,
    residual_threshold_factor,
    robust_wahba,
    wahba_svd,
)

SEED = 404
N_STARS = 20
N_DECOYS = 40                    # other catalog stars in the same sky area
CENTROID_NOISE_PX = 0.2
FALSE_ALARM = 1e-3
MISMATCH_COUNTS = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
TRIALS = 200
SUCCESS_SIGMAS = 5.0             # "correct" = within 5x the predicted 1-sigma

CAMERA = PinholeCamera()
C_TRUE = true_attitude_matrix()
HERE = Path(__file__).parent


def make_scene(rng):
    camera_vectors = stars_in_field(CAMERA, N_STARS, rng)
    reference = to_celestial(camera_vectors)
    measured = observe_with_centroid_noise(CAMERA, camera_vectors,
                                           CENTROID_NOISE_PX, rng)
    sigma = centroid_sigma_to_angle(CAMERA, camera_vectors, CENTROID_NOISE_PX)
    return camera_vectors, reference, measured, sigma


def error_arcsec(C):
    return attitude_error(C, C_TRUE)[1] * ARCSEC_PER_RAD


def part_one(rng):
    print("=" * 78)
    print("Part 1 - one wrong catalog association among 20 stars")
    print("=" * 78)
    camera_vectors, reference, measured, sigma = make_scene(rng)
    weights = 1.0 / sigma**2
    k = residual_threshold_factor(FALSE_ALARM)
    print(f"Centroid noise {CENTROID_NOISE_PX} px = about "
          f"{np.mean(sigma) * ARCSEC_PER_RAD:.1f} arcsec per axis per star.")
    print(f"Outlier limit: residual > {k:.2f} sigma. A correct match exceeds "
          f"that with probability {FALSE_ALARM:g}\n(Rayleigh-distributed "
          f"residual), so the limit is {k * np.mean(sigma) * ARCSEC_PER_RAD:.1f} "
          f"arcsec here, not an arbitrary angle.")

    # Star `bad` is given the catalog direction of its nearest neighbour.
    bad = 7
    cosines = camera_vectors @ camera_vectors[bad]
    cosines[bad] = -1.0
    neighbour = int(np.argmax(cosines))
    offset = np.degrees(np.arccos(cosines[neighbour]))
    corrupted = reference.copy()
    corrupted[bad] = reference[neighbour]
    print(f"\nStar {bad} is paired with the catalog entry of star {neighbour}, "
          f"{offset:.2f} deg away ({offset * 3600:.0f} arcsec).")

    clean = wahba_svd(reference, measured, weights)
    plain = wahba_svd(corrupted, measured, weights)
    residuals_plain = angular_residuals(plain, corrupted, measured)
    robust = robust_wahba(corrupted, measured, sigma, FALSE_ALARM)
    phi_plain = attitude_error(plain, C_TRUE)[0] * ARCSEC_PER_RAD
    phi_robust = attitude_error(robust.C, C_TRUE)[0] * ARCSEC_PER_RAD

    print(f"\n  {'estimate':<34}{'total':>10}{'about x':>10}{'about y':>10}"
          f"{'about z':>10}  [arcsec]")
    phi_clean = attitude_error(clean, C_TRUE)[0] * ARCSEC_PER_RAD
    for label, phi in (("SVD, all matches correct", phi_clean),
                       ("SVD, one wrong match", phi_plain),
                       ("iterative rejection", phi_robust)):
        print(f"  {label:<34}{np.linalg.norm(phi):>10.2f}{phi[0]:>10.2f}"
              f"{phi[1]:>10.2f}{phi[2]:>10.2f}")

    order = np.argsort(residuals_plain)[::-1]
    limits = robust.thresholds
    print("\n  Residuals after the uncorrected SVD fit (largest five):")
    print(f"  {'star':>6}{'residual':>12}{'limit':>10}{'ratio':>9}")
    for i in order[:5]:
        flag = "  <- wrong match" if i == bad else ""
        print(f"  {i:>6}{residuals_plain[i] * ARCSEC_PER_RAD:>9.1f} as"
              f"{limits[i] * ARCSEC_PER_RAD:>8.1f} as"
              f"{residuals_plain[i] / limits[i]:>9.1f}{flag}")
    over = int(np.sum(residuals_plain > limits))
    print(f"  Largest residual: star {order[0]}. Stars over their limit: "
          f"{over} of {N_STARS}; the bad match drags the fit, so correctly "
          f"matched stars are pushed over the limit too.\n  That is why the "
          f"rejection loop removes only the single worst star per pass.")
    print(f"\n  Iterative rejection: {robust.status}; removed star(s) "
          f"{robust.rejected} in {robust.iterations} fits, "
          f"{int(robust.inliers.sum())} stars kept.")
    print(f"  Error reduced from {np.linalg.norm(phi_plain):.1f} to "
          f"{np.linalg.norm(phi_robust):.2f} arcsec "
          f"(all-correct SVD: {np.linalg.norm(phi_clean):.2f}).")
    return residuals_plain, robust, bad


def part_two(rng):
    print("\n" + "=" * 78)
    print(f"Part 2 - increasing mismatch rate ({N_STARS} stars, {TRIALS} "
          f"trials per rate)")
    print("=" * 78)
    methods = ["Plain SVD", "Iterative rejection", "RANSAC"]
    stats = {m: {"success": [], "median": [], "silent": [], "flagged": []}
             for m in methods}

    for wrong in MISMATCH_COUNTS:
        errors = {m: [] for m in methods}
        good = {m: 0 for m in methods}
        silent = {m: 0 for m in methods}
        flagged = {m: 0 for m in methods}
        for _ in range(TRIALS):
            camera_vectors, reference, measured, sigma = make_scene(rng)
            decoys = to_celestial(stars_in_field(CAMERA, N_DECOYS, rng))
            catalog = np.vstack((reference, decoys))
            corrupted = reference.copy()
            wrong_stars = rng.choice(N_STARS, size=wrong, replace=False)
            for star in wrong_stars:
                choices = np.delete(np.arange(len(catalog)), star)
                corrupted[star] = catalog[rng.choice(choices)]

            # Judge each solution against what the CORRECTLY matched stars
            # could deliver: with few of them left, a right answer is noisier.
            correct_stars = np.ones(N_STARS, dtype=bool)
            correct_stars[wrong_stars] = False
            limit = SUCCESS_SIGMAS * np.sqrt(np.trace(attitude_covariance(
                attitude_information(camera_vectors[correct_stars],
                                     sigma[correct_stars])))) * ARCSEC_PER_RAD

            outcomes = {}
            try:
                outcomes["Plain SVD"] = (
                    wahba_svd(corrupted, measured, 1.0 / sigma**2), True)
            except ValueError:
                outcomes["Plain SVD"] = (np.eye(3), False)
            result = robust_wahba(corrupted, measured, sigma, FALSE_ALARM)
            outcomes["Iterative rejection"] = (result.C, result.converged)
            result = ransac_attitude(corrupted, measured, sigma, rng=rng,
                                     false_alarm_probability=FALSE_ALARM)
            outcomes["RANSAC"] = (result.C, result.converged)

            for method, (C, converged) in outcomes.items():
                err = error_arcsec(C)
                errors[method].append(err)
                correct = err < limit
                good[method] += correct and converged
                silent[method] += (not correct) and converged
                flagged[method] += not converged

        for method in methods:
            stats[method]["success"].append(100.0 * good[method] / TRIALS)
            stats[method]["median"].append(np.median(errors[method]))
            stats[method]["silent"].append(silent[method])
            stats[method]["flagged"].append(flagged[method])

    print(f"Success = method reports a solution AND it is within "
          f"{SUCCESS_SIGMAS:g}x the 1-sigma total error predicted")
    print("for the correctly matched stars alone.")
    print("Silent = reported a solution that is wrong. Flagged = the method "
          "itself reported failure.\n")
    print(f"  {'wrong':>6}" + "".join(f"{m:>36}" for m in methods))
    print(f"  {'of ' + str(N_STARS):>6}"
          + f"{'success  median err  silent flagged':>36}" * 3)
    for row, wrong in enumerate(MISMATCH_COUNTS):
        line = f"  {wrong:>6}"
        for method in methods:
            s = stats[method]
            line += (f"{s['success'][row]:>9.0f}%"
                     f"{s['median'][row]:>10.1f} as"
                     f"{s['silent'][row]:>7d}{s['flagged'][row]:>8d}")
        print(line)

    def last_reliable(method):
        ok = []
        for wrong, success in zip(MISMATCH_COUNTS, stats[method]["success"]):
            if success < 99.0:
                break
            ok.append(wrong)
        return ok[-1] if ok else None

    def totals(method):
        return (sum(stats[method]["silent"][1:]),
                sum(stats[method]["flagged"][1:]))

    iter_silent, iter_flagged = totals("Iterative rejection")
    ransac_silent, ransac_flagged = totals("RANSAC")
    print(f"""
Reading the result
------------------
* Plain SVD has no defence: a single wrong match already ruins the answer,
  and it always reports a solution.
* Iterative rejection starts from a fit to ALL matches. It holds at 99% or
  better up to {last_reliable('Iterative rejection')} wrong of {N_STARS}. It survives that far because these wrong
  matches are random: their pulls on the first fit partly cancel, and each
  one still has a far larger residual than any correct star. Beyond that
  the first fit is corrupted enough that correct stars are discarded.
  Over all rates it failed {iter_silent + iter_flagged} times: {iter_flagged} flagged by the method itself,
  {iter_silent} silent (a self-consistent but wrong subset).
* RANSAC never fits to the contaminated set. Each hypothesis comes from two
  stars, and it only needs one correct pair plus enough supporters. It holds
  at 99% or better up to {last_reliable('RANSAC')} wrong of {N_STARS}, and failed {ransac_silent + ransac_flagged} times in total:
  {ransac_flagged} flagged, {ransac_silent} silent.
* With {N_STARS - MISMATCH_COUNTS[-1]} correct stars left ({MISMATCH_COUNTS[-1]} wrong) neither method can succeed by design:
  both require at least 3 mutually consistent stars before reporting a
  solution.
* Silent failures (a reported solution that is wrong) are the dangerous
  kind; the counts above show how many occurred in this run. A converged
  robust fit means "self-consistent", not "correct", and the minimum of 3
  agreeing stars is what turns most failures here into flagged ones.
* These mismatches are random, so wrong matches do not agree with each
  other. Systematic errors, such as a whole pattern identified as the wrong
  part of the sky, form their own consistent set and can defeat both
  methods.""")
    return stats


def make_figure(residuals_plain, robust, bad, stats):
    fig, (ax_res, ax_success, ax_err) = plt.subplots(1, 3, figsize=(17, 5.2))

    stars = np.arange(N_STARS)
    width = 0.4
    before = residuals_plain * ARCSEC_PER_RAD
    after = robust.residuals * ARCSEC_PER_RAD
    ax_res.bar(stars - width / 2, before, width, color="tab:red",
               label="after plain SVD")
    ax_res.bar(stars + width / 2, after, width, color="tab:blue",
               label="after iterative rejection")
    ax_res.plot(stars, robust.thresholds * ARCSEC_PER_RAD, "k_", markersize=14,
                label="outlier limit")
    ax_res.set_yscale("log")
    ax_res.set_xticks(stars)
    ax_res.tick_params(axis="x", labelsize=7)
    ax_res.set_xlabel("Star index")
    ax_res.set_ylabel("Angular residual [arcsec]")
    ax_res.set_title(f"One wrong match (star {bad}): residuals")
    ax_res.grid(True, axis="y", which="both", alpha=0.3)
    ax_res.legend(fontsize=8)

    colors = {"Plain SVD": "tab:red", "Iterative rejection": "tab:orange",
              "RANSAC": "tab:blue"}
    for method, color in colors.items():
        ax_success.plot(MISMATCH_COUNTS, stats[method]["success"], "o-",
                        color=color, linewidth=2, label=method)
        ax_err.semilogy(MISMATCH_COUNTS, stats[method]["median"], "o-",
                        color=color, linewidth=2, label=method)
    ax_success.set_ylabel("Correct attitude recovered [% of trials]")
    ax_success.set_title("Success rate")
    ax_success.set_ylim(-3, 103)
    ax_err.set_ylabel("Median attitude error [arcsec]")
    ax_err.set_title("Median error")
    for ax in (ax_success, ax_err):
        ax.set_xlabel(f"Wrong matches out of {N_STARS}")
        ax.set_xticks(MISMATCH_COUNTS)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "catalog_mismatch_experiment.png", dpi=150)
    print("\nSaved catalog_mismatch_experiment.png")


if __name__ == "__main__":
    generator = np.random.default_rng(SEED)
    residuals, robust_result, bad_star = part_one(generator)
    statistics = part_two(generator)
    make_figure(residuals, robust_result, bad_star, statistics)
