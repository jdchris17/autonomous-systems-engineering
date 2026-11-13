"""
Experiment B - verified solutions and explicit failure.

Every scenario in tests/scenarios.py runs through both modes. Required: positives verified (Outcome A,
with associations, frame, uncertainty, diagnostics); negatives rejected with a specific reason (Outcome B).
Also shown: the full criteria table for one accepted and one rejected case, and the anisotropic attitude
uncertainty (roll vs cross-boresight).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
OUT = ROOT / "outputs"

import scenarios as S  # noqa: E402
from stellar_vision import StellarVisionPipeline, VisionConfig  # noqa: E402
from stellar_vision.geometry import ARCSEC, rotation_angle  # noqa: E402


def run(sc, mode):
    pipe = StellarVisionPipeline.from_config(VisionConfig(detection_mode=mode), catalog=sc.catalog)
    return pipe.solve(sc.image, camera=sc.camera) if sc.image is not None else pipe.solve_sources(sc.sources, sc.camera)


def main():
    print("Experiment B: verified solutions and explicit failure (default acceptance thresholds)")
    rows, keep = [], {}
    for b in S.ALL_POSITIVE + S.ALL_NEGATIVE:
        sc = b()
        for mode in ("classical", "hybrid"):
            r = run(sc, mode)
            correct = (r.success == sc.expect_solution) and (not r.success or rotation_angle(r.attitude, sc.R_true) < 60 * ARCSEC) \
                and (r.success or r.failure_code in sc.acceptable_codes)
            rows.append((sc, mode, r, correct))
            keep[(b.__name__, mode)] = r
    lines = ["| Scenario | Purpose | Mode | Outcome | Failure code | Reason / evidence | Correct |", "|---|---|---|---|---|---|---|"]
    print(f"\n  {'scenario':<42}{'mode':<10}{'outcome':<11}{'code':<24}{'evidence'}")
    for sc, mode, r, ok in rows:
        if r.success:
            s = r.attitude_sigma_arcsec
            ev = (f"{r.n_matched} matched ({r.n_verify} independent), RMS {r.reprojection_rms_px:.3f} px, chi2r "
                  f"{r.diagnostics['reduced_chi2']:.2f}, err {rotation_angle(r.attitude, sc.R_true) / ARCSEC:.1f}\", "
                  f"sigma roll/cross {s['roll']:.1f}\"/{max(s['cross_x'], s['cross_y']):.2f}\"")
            outcome, code = "VERIFIED", "-"
        else:
            ev, outcome, code = r.failure_reason, "REJECTED", r.failure_code.value
        print(f"  {sc.name:<42}{mode:<10}{outcome:<11}{code:<24}{ev}{'' if ok else '   <-- UNEXPECTED'}")
        lines.append(f"| {sc.name} | {sc.purpose} | {mode} | {outcome} | {code} | {ev} | {'yes' if ok else 'NO'} |")
    n_ok = sum(r[3] for r in rows)
    print(f"\n  {n_ok}/{len(rows)} scenario runs ended in the expected outcome; false verified solutions: "
          f"{sum(1 for sc, m, r, ok in rows if r.success and not sc.expect_solution)}")
    (OUT / "exp_b_table.md").write_text("\n".join(lines) + "\n")

    for key, title in ((("nominal", "hybrid"), "accepted (nominal field, hybrid)"),
                       (("ambiguous_twin", "hybrid"), "rejected (duplicated pattern, hybrid)"),
                       (("weak_geometry", "hybrid"), "rejected (tight faint cluster, hybrid)")):
        r = keep[key]
        print(f"\n  criteria - {title}:")
        for c in r.diagnostics.get("criteria", []):
            print(f"    {c['name']:<24} value {str(c['value']):>10}   threshold {str(c['threshold']):>8}   {'pass' if c['passed'] else 'FAIL'}")
        if r.diagnostics.get("runner_up"):
            ru = r.diagnostics["runner_up"]
            print(f"    runner-up hypothesis: {ru['n_match']} matches, {ru['angle_from_best_deg']:.2f} deg from the best")
    r = keep[("nominal", "hybrid")]
    P = r.attitude_covariance
    sd = np.sqrt(np.diag(P)) / ARCSEC
    corr = P / np.outer(np.sqrt(np.diag(P)), np.sqrt(np.diag(P)))
    print(f"\n  attitude covariance (nominal, camera frame): sigma = ({sd[0]:.3f}\", {sd[1]:.3f}\", {sd[2]:.2f}\") for"
          f" (cross-x, cross-y, roll); roll / cross ratio {sd[2] / max(sd[0], sd[1]):.0f}x; correlations"
          f" xy {corr[0, 1]:+.2f}, x-roll {corr[0, 2]:+.2f}, y-roll {corr[1, 2]:+.2f}")
    print(f"  normalised residuals d^2 (nominal): mean {np.mean(r.normalized_residuals_sq):.2f} (2 expected per star before"
          f" fitting), fraction above the 99.9% point (13.8): {np.mean(r.normalized_residuals_sq > 13.8):.2f}")
    print("  table ->", OUT / "exp_b_table.md")


if __name__ == "__main__":
    main()
