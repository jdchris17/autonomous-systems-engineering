"""
Stage 6: verification and acceptance of the best hypothesis (all thresholds in AcceptanceConfig).

Criteria are evaluated in a fixed order; the first failure determines the FailureCode, and every
criterion's value / threshold / pass is reported in diagnostics["criteria"] regardless.

  1 independent matches     n_verify (matches beyond the 3 hypothesis stars) >= min_verify_matches
  2 catalogue consistency   matched / catalogue stars predicted inside the frame >= min_catalog_fraction
  3 camera-model validity   fitted focal length not pinned at the permitted bound (f0 (1 +- tol))
  4 residuals vs uncertainty d_i^2 = r_i^T Sigma_i^-1 r_i with Sigma_i = sigma_i^2 I (per-axis calibrated
                            centroid sigma); reduced chi2 = sum d_i^2 / (2N - 4)  (3 rotation + 1 focal
                            parameter). Used as a plausibility bound, NOT an exact chi-square test: the
                            matching radius truncates large residuals and sigma_i are estimates.
                            Also an absolute RMS bound.
  5 attitude geometry       Fisher covariance P = J^-1 * max(1, reduced chi2), J = sum_i w_i (I - b_i b_i^T),
                            w_i = 1 / sigma_theta_i^2; reject if roll or cross-boresight 1-sigma too large
  6 competing hypotheses    best distinct hypothesis (rotation > distinct_hypothesis_deg away) with
                            >= min_verify_matches support and >= ambiguity_ratio x the best's matches
"""

import numpy as np

from .geometry import ARCSEC, rotation_angle
from .result import FailureCode


def attitude_covariance(camera, u, v, sigma_px, f, scale=1.0):
    b = camera.pixel_to_vec(u, v, f)
    w = 1 / camera.angular_sigma(u, v, sigma_px, f) ** 2
    J = np.einsum("i,ijk->jk", w, np.eye(3)[None] - b[:, :, None] * b[:, None, :])
    return np.linalg.inv(J) * scale


def verify(best, hypotheses, src, camera, acc):
    """Returns (passed, failure_code, reason, info) for the solver's best hypothesis."""
    di = best["det_idx"]
    u, v = src["x"][di], src["y"][di]
    sig = src["sigma_px"][di]
    pu, pv = best["pred_u"], best["pred_v"]
    r = np.column_stack([u - pu, v - pv])
    d2 = np.sum(r**2, axis=1) / sig**2
    n = len(di)
    dof = max(2 * n - 4, 1)
    chi2_red = float(np.sum(d2) / dof)
    rms = float(np.sqrt(np.mean(np.sum(r**2, axis=1)))) if n else np.inf
    frac = n / max(best["n_pred"], 1)
    P = attitude_covariance(camera, u, v, sig, best["f"], max(1.0, chi2_red)) if n >= 3 else np.full((3, 3), np.inf)
    sd = np.sqrt(np.clip(np.diag(P), 0, None)) / ARCSEC
    f_lo, f_hi = camera.focal_length_px * (1 - camera.focal_length_tolerance), camera.focal_length_px * (1 + camera.focal_length_tolerance)
    f_margin = min(best["f"] - f_lo, f_hi - best["f"]) / camera.focal_length_px
    distinct = [h for h in hypotheses if rotation_angle(h["R"], best["R"]) > np.radians(acc.distinct_hypothesis_deg)]
    runner = max(distinct, key=lambda h: (h["n_match"], -h["rms_px"]), default=None)
    runner_ratio = runner["n_match"] / max(best["n_match"], 1) if runner else 0.0
    runner_ok = runner is not None and runner["n_verify"] >= acc.min_verify_matches
    crit = [
        ("independent_matches", best["n_verify"], acc.min_verify_matches, best["n_verify"] >= acc.min_verify_matches,
         FailureCode.INSUFFICIENT_MATCHES),
        ("catalog_fraction", round(frac, 4), acc.min_catalog_fraction, frac >= acc.min_catalog_fraction,
         FailureCode.LOW_CATALOG_FRACTION),
        ("focal_length_margin", round(f_margin, 5), acc.focal_bound_margin, f_margin > acc.focal_bound_margin,
         FailureCode.CAMERA_MODEL_INVALID),
        ("reduced_chi2", round(chi2_red, 3), acc.max_reduced_chi2, chi2_red <= acc.max_reduced_chi2,
         FailureCode.RESIDUALS_INCONSISTENT),
        ("reprojection_rms_px", round(rms, 4), acc.max_rms_px, rms <= acc.max_rms_px, FailureCode.RESIDUALS_INCONSISTENT),
        ("roll_sigma_arcsec", round(float(sd[2]), 2), acc.max_roll_sigma_arcsec, sd[2] <= acc.max_roll_sigma_arcsec,
         FailureCode.WEAK_GEOMETRY),
        ("cross_sigma_arcsec", round(float(max(sd[0], sd[1])), 3), acc.max_cross_sigma_arcsec,
         max(sd[0], sd[1]) <= acc.max_cross_sigma_arcsec, FailureCode.WEAK_GEOMETRY),
        ("runner_up_match_ratio", round(runner_ratio, 3), acc.ambiguity_ratio,
         not (runner_ok and runner_ratio >= acc.ambiguity_ratio), FailureCode.AMBIGUOUS),
    ]
    criteria = [{"name": c[0], "value": c[1], "threshold": c[2], "passed": bool(c[3])} for c in crit]
    fail = next((c for c in crit if not c[3]), None)
    info = {"criteria": criteria, "residuals": r, "d2": d2, "chi2_red": chi2_red, "rms": rms, "frac": frac,
            "covariance": P, "sigma_arcsec": {"cross_x": float(sd[0]), "cross_y": float(sd[1]), "roll": float(sd[2])},
            "runner_up": None if runner is None else {"n_match": runner["n_match"], "n_verify": runner["n_verify"],
                                                      "rms_px": runner["rms_px"],
                                                      "angle_from_best_deg": float(np.degrees(rotation_angle(runner["R"], best["R"])))},
            "n_distinct_hypotheses": len(distinct) + 1,
            "frac_d2_over_13.8": float(np.mean(d2 > 13.82)) if n else np.nan}
    if fail is None:
        return True, None, None, info
    reason = {
        FailureCode.INSUFFICIENT_MATCHES: f"best hypothesis has only {best['n_verify']} matches beyond its 3 generating"
                                          f" stars (need {acc.min_verify_matches})",
        FailureCode.LOW_CATALOG_FRACTION: f"only {frac:.0%} of the {best['n_pred']} catalogue stars predicted in the frame"
                                          f" were detected (need {acc.min_catalog_fraction:.0%})",
        FailureCode.CAMERA_MODEL_INVALID: f"solution needs focal length {best['f']:.1f} px, at the permitted bound"
                                          f" ({f_lo:.1f}-{f_hi:.1f}); camera calibration outside tolerance",
        FailureCode.RESIDUALS_INCONSISTENT: f"residuals implausible for the stated uncertainties (reduced chi2 {chi2_red:.1f},"
                                            f" RMS {rms:.2f} px)",
        FailureCode.WEAK_GEOMETRY: f"attitude poorly observable: roll sigma {sd[2]:.0f}\", cross-boresight sigma"
                                   f" {max(sd[0], sd[1]):.1f}\"",
        FailureCode.AMBIGUOUS: (f"competing hypothesis {0 if runner is None else np.degrees(rotation_angle(runner['R'], best['R'])):.2f}"
                                f" deg away matches {0 if runner is None else runner['n_match']} stars vs {best['n_match']}"),
    }[fail[4]]
    return False, fail[4], reason, info
