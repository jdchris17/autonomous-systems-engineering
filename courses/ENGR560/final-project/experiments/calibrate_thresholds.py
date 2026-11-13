"""Statistics of CORRECT solutions on calibration fields (seeds 300-311; never used for evaluation).
Used to set AcceptanceConfig.max_reduced_chi2 / max_rms_px / max_roll_sigma_arcsec / max_cross_sigma_arcsec."""

import numpy as np

from conditions import CALIBRATION_SEEDS, CONDITIONS, StellarVisionPipeline, config_for, make_field
from stellar_vision.geometry import ARCSEC, rotation_angle

PERMISSIVE = dict(max_reduced_chi2=1e9, max_rms_px=1e9, max_roll_sigma_arcsec=1e9, max_cross_sigma_arcsec=1e9)


def main():
    print("Threshold calibration: correct solutions on calibration seeds 300-311 (permissive acceptance)")
    print(f"  {'condition':<24}{'mode':<10}{'solved':>7}{'chi2 med':>9}{'chi2 max':>9}{'rms max':>8}{'roll sd max':>12}"
          f"{'cross sd max':>13}{'z^2 mean':>9}")
    allmax = {"chi2": 0, "rms": 0, "roll": 0, "cross": 0}
    for cname in ("nominal", "low SNR", "strong defocus", "hot pixels & streaks"):
        for mode in ("classical", "hybrid"):
            stats = []
            for sd in CALIBRATION_SEEDS:
                image, truth, R, cat, _ = make_field(cname, sd)
                pipe = StellarVisionPipeline.from_config(config_for(mode, cname, **PERMISSIVE), catalog=cat)
                r = pipe.solve(image)
                if not r.success or rotation_angle(r.attitude, R) > 600 * ARCSEC:
                    continue
                phi = 0.5 * np.array([(r.attitude @ R.T)[2, 1] - (r.attitude @ R.T)[1, 2],
                                      (r.attitude @ R.T)[0, 2] - (r.attitude @ R.T)[2, 0],
                                      (r.attitude @ R.T)[1, 0] - (r.attitude @ R.T)[0, 1]])
                z2 = float(phi @ np.linalg.solve(r.attitude_covariance, phi))
                s = r.attitude_sigma_arcsec
                stats.append((r.diagnostics["reduced_chi2"], r.reprojection_rms_px, s["roll"], max(s["cross_x"], s["cross_y"]), z2))
            a = np.array(stats)
            print(f"  {cname:<24}{mode:<10}{len(a):>4}/{len(CALIBRATION_SEEDS):<2}{np.median(a[:, 0]):>9.2f}{a[:, 0].max():>9.2f}"
                  f"{a[:, 1].max():>8.3f}{a[:, 2].max():>12.1f}{a[:, 3].max():>13.2f}{a[:, 4].mean():>9.2f}")
            for k, j in (("chi2", 0), ("rms", 1), ("roll", 2), ("cross", 3)):
                allmax[k] = max(allmax[k], a[:, j].max())
    print("  maxima over all conditions:", {k: round(v, 3) for k, v in allmax.items()})
    print("  (z^2 = phi^T P^-1 phi for the true attitude error; ~3 if the reported covariance is calibrated)")


if __name__ == "__main__":
    main()
