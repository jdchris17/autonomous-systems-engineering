"""
geometry_experiment.py - Space Systems Engineering, Module 4, Experiment C

How does the field of view affect three-axis attitude precision?

The same 1920 x 1080 sensor is used at horizontal fields of view of 5, 10,
20 and 40 degrees. The 20 stars sit at the SAME pixel positions in every
case, so the image looks identical and only the angular scale changes.

Case 1 - fixed centroid noise (0.2 px). A narrower field has finer pixels,
         so the same pixel noise is a smaller angle.
Case 2 - fixed angular noise (the angle that 0.2 px subtends at 20 deg
         HFOV). This removes the resolution effect and leaves only the
         geometry of the star field.

Limitation: no star density, limiting magnitude, aperture, SNR or
distortion model. A real narrow field also sees fewer stars; here the count
is held at 20 by construction.

Run:  python geometry_experiment.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from star_field_simulation import (
    ARCSEC_PER_RAD,
    PinholeCamera,
    camera_vectors_from_positions,
    centroid_sigma_to_angle,
    observe_with_angular_noise,
    observe_with_centroid_noise,
    to_celestial,
    true_attitude_matrix,
    unit_square_positions,
)
from stellar_attitude import (
    attitude_covariance,
    attitude_error,
    attitude_information,
    wahba_svd,
)

SEED = 44
FIELDS_OF_VIEW_DEG = [5.0, 10.0, 20.0, 40.0]
N_STARS = 20
CENTROID_NOISE_PX = 0.2
REFERENCE_HFOV_DEG = 20.0
TRIALS = 500
C_TRUE = true_attitude_matrix()


def run_case(camera, camera_vectors, reference, rng, sigma_px=None,
             sigma_rad=None):
    """Monte Carlo and predicted statistics for one field of view."""
    if sigma_px is not None:
        sigma = centroid_sigma_to_angle(camera, camera_vectors, sigma_px)
    else:
        sigma = np.full(len(camera_vectors), sigma_rad)
    information = attitude_information(camera_vectors, sigma)
    covariance = attitude_covariance(information)
    weights = 1.0 / sigma**2

    errors = np.empty((TRIALS, 3))
    for t in range(TRIALS):
        if sigma_px is not None:
            measured = observe_with_centroid_noise(camera, camera_vectors,
                                                   sigma_px, rng)
        else:
            measured = observe_with_angular_noise(camera_vectors, sigma_rad, rng)
        errors[t] = attitude_error(wahba_svd(reference, measured, weights),
                                   C_TRUE)[0]
    errors *= ARCSEC_PER_RAD
    return {
        "star_noise": np.mean(sigma) * ARCSEC_PER_RAD,
        "pointing_mc": np.sqrt(np.mean(errors[:, 0] ** 2 + errors[:, 1] ** 2)),
        "roll_mc": np.sqrt(np.mean(errors[:, 2] ** 2)),
        "pointing_pred": np.sqrt(covariance[0, 0] + covariance[1, 1]) * ARCSEC_PER_RAD,
        "roll_pred": np.sqrt(covariance[2, 2]) * ARCSEC_PER_RAD,
        "total_pred": np.sqrt(np.trace(covariance)) * ARCSEC_PER_RAD,
        "condition": np.linalg.cond(information),
    }


def main():
    rng = np.random.default_rng(SEED)
    positions = unit_square_positions(N_STARS, rng)       # same image layout
    fixed_sigma_rad = CENTROID_NOISE_PX / PinholeCamera(
        horizontal_fov_deg=REFERENCE_HFOV_DEG).fx

    print(f"{N_STARS} stars at identical pixel positions, {TRIALS} trials "
          f"per case, 1920 x 1080 sensor")
    print(f"Case 1: centroid noise {CENTROID_NOISE_PX} px")
    print(f"Case 2: angular noise {fixed_sigma_rad * ARCSEC_PER_RAD:.2f} arcsec "
          f"per axis (= {CENTROID_NOISE_PX} px at {REFERENCE_HFOV_DEG:g} deg HFOV)")

    case1, case2 = [], []
    for hfov in FIELDS_OF_VIEW_DEG:
        camera = PinholeCamera(horizontal_fov_deg=hfov)
        camera_vectors = camera_vectors_from_positions(camera, positions)
        reference = to_celestial(camera_vectors)
        case1.append(run_case(camera, camera_vectors, reference, rng,
                              sigma_px=CENTROID_NOISE_PX))
        case2.append(run_case(camera, camera_vectors, reference, rng,
                              sigma_rad=fixed_sigma_rad))

    for title, results in (("Case 1 - fixed centroid noise", case1),
                           ("Case 2 - fixed angular noise", case2)):
        print(f"\n{title}")
        print(f"  {'HFOV':>6}{'arcsec/px':>11}{'star noise':>12}"
              f"{'pointing MC':>13}{'predicted':>11}{'roll MC':>10}"
              f"{'predicted':>11}{'total pred':>12}{'cond(J)':>10}")
        for hfov, r in zip(FIELDS_OF_VIEW_DEG, results):
            scale = ARCSEC_PER_RAD / PinholeCamera(horizontal_fov_deg=hfov).fx
            print(f"  {hfov:>4g} deg{scale:>11.2f}{r['star_noise']:>12.2f}"
                  f"{r['pointing_mc']:>13.3f}{r['pointing_pred']:>11.3f}"
                  f"{r['roll_mc']:>10.2f}{r['roll_pred']:>11.2f}"
                  f"{r['total_pred']:>12.2f}{r['condition']:>10.0f}")
    print("  (arcsec; pointing = RMS of the combined x and y error, "
          "roll = RMS about z)")

    best1 = FIELDS_OF_VIEW_DEG[int(np.argmin([r["total_pred"] for r in case1]))]
    best2 = FIELDS_OF_VIEW_DEG[int(np.argmin([r["total_pred"] for r in case2]))]
    print(f"""
Reading the result
------------------
Pointing uses each star's full direction, so its precision is about
(angular noise per star) / sqrt(N). Roll uses each star's distance from the
boresight as a lever arm, so its precision is about
(angular noise per star) / (RMS off-axis angle) / sqrt(N).

Case 1: halving the field halves the angle per pixel, so pointing error
halves. For roll the two effects cancel: finer pixels, but a lever arm
shorter by the same factor. In pixel terms roll error is simply
(pixel noise) / (star's pixel distance from the centre), which does not
depend on the field of view at all. Roll goes from {case1[0]['roll_pred']:.1f} arcsec at 5 deg to
{case1[-1]['roll_pred']:.1f} at 40 deg, a change of only {abs(case1[-1]['roll_pred'] / case1[0]['roll_pred'] - 1) * 100:.0f}%
while pointing changes by {case1[-1]['pointing_pred'] / case1[0]['pointing_pred']:.1f}x.

Case 2: with the same angular noise everywhere, pointing is the same at
every field of view and roll gets worse in proportion as the field narrows:
{case2[0]['roll_pred']:.0f} arcsec at 5 deg against {case2[-1]['roll_pred']:.1f} at 40 deg.

The condition number of the information matrix is the ratio of pointing
information to roll information. It grows roughly as 1 / FOV^2:
{case1[0]['condition']:.0f} at 5 deg, {case1[-1]['condition']:.0f} at 40 deg, and it is nearly the same in both
cases because it is a property of the geometry, not of the noise level.

Smallest total (three-axis) predicted error: {best1:g} deg in Case 1, {best2:g} deg
in Case 2. In Case 1 that "win" is nominal: the total three-axis error is
{case1[0]['total_pred']:.2f} arcsec at 5 deg and {case1[-1]['total_pred']:.2f} at 40 deg, because the total is
dominated by roll and roll does not improve. The narrow field buys
pointing precision only, and only if pixel noise stays fixed and the star
count is somehow held at {N_STARS}. With fixed angular noise the widest field is
clearly best overall.""")

    # ---- figure -------------------------------------------------------------
    fig, (ax_point, ax_roll, ax_cond) = plt.subplots(1, 3, figsize=(16, 5))
    styles = (("Case 1: fixed 0.2 px centroid noise", case1, "tab:blue"),
              ("Case 2: fixed angular noise", case2, "tab:orange"))
    for ax, key, title in ((ax_point, "pointing", "Pointing error (x and y combined)"),
                           (ax_roll, "roll", "Roll error (about boresight)")):
        for label, results, color in styles:
            ax.loglog(FIELDS_OF_VIEW_DEG, [r[f"{key}_mc"] for r in results],
                      "o", color=color, markersize=8, label=f"{label}, Monte Carlo")
            ax.loglog(FIELDS_OF_VIEW_DEG, [r[f"{key}_pred"] for r in results],
                      "-", color=color, linewidth=1.5,
                      label=f"{label}, predicted")
        ax.set_title(title)
        ax.set_ylabel("RMS error [arcsec]")
        ax.legend(fontsize=8)
    for label, results, color in styles:
        ax_cond.loglog(FIELDS_OF_VIEW_DEG, [r["condition"] for r in results],
                       "o-", color=color, linewidth=1.5, label=label)
    ax_cond.set_title("Condition number of the attitude information matrix")
    ax_cond.set_ylabel("cond(J)")
    ax_cond.legend(fontsize=8)
    for ax in (ax_point, ax_roll, ax_cond):
        ax.set_xlabel("Horizontal field of view [deg]")
        ax.set_xticks(FIELDS_OF_VIEW_DEG)
        ax.set_xticklabels([f"{f:g}" for f in FIELDS_OF_VIEW_DEG])
        ax.minorticks_off()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Field of view and attitude precision: {N_STARS} stars, "
                 f"same sensor, same image layout")
    fig.tight_layout()
    out = Path(__file__).with_name("geometry_experiment.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()
