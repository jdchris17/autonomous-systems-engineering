"""
Experiment A: known field, known camera geometry, CORRECT associations -> weighted SVD (Wahba).

Catalogue: 180 synthetic stars in a 7 deg radius patch around RA 150, Dec 20 (magnitudes 3-8).
Each trial: a random true attitude (boresight within 2 deg of the patch centre, random roll), visible
stars projected with the true pinhole camera, Gaussian centroid noise added, directions back-projected,
R_CI estimated with the weighted SVD solution, error angle = arccos((tr(R_est R_true^T) - 1) / 2).

Noise models (sigma = noise level on the x axis)
  homoscedastic    every star sigma                           -> weighting cannot help (all w equal)
  heteroscedastic  sigma_i = sigma * g(m_i), g log-linear from 0.5 (m = 3) to 2 (m = 8)
                   (fainter stars centroid worse; at sigma = 0.1 px the stars span 0.05-0.2 px)
  strong hetero    g from 0.25 to 4 (a 16x spread) - table only
Weighted solution: w_i = 1 / sigma_theta_i^2 with sigma_theta_i from the pixel sigma and the camera scale.
Prediction: Fisher information J = sum w_i (I - b_i b_i^T); error vector ~ N(0, J^-1).
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sky13 as s

OUT = Path(__file__).resolve().parent / "outputs"
SIGMAS = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
TRIALS = 1000


def g_of_mag(mag, lo, hi):
    t = (mag - 3.0) / 5.0
    return lo * (hi / lo) ** t


def trial(cat, rng, sig_star):
    R, _ = s.random_pointing(rng)
    o = s.observe(cat, R, rng, sig_star)
    b = s.pixel_to_vec(o["u"], o["v"])
    r = cat["vec"][o["cat_idx"]]
    w = 1 / s.angular_sigma(o["u"], o["v"], o["sigma_px"]) ** 2
    Ru, Rw = s.wahba_svd(b, r), s.wahba_svd(b, r, w)
    J = sum(wi * (np.eye(3) - np.outer(bi, bi)) for wi, bi in zip(w, b))
    phi = rng.multivariate_normal(np.zeros(3), np.linalg.inv(J))
    return (s.rotation_angle(Ru, R), s.rotation_angle(Rw, R), float(np.linalg.norm(phi)),
            s.error_components(Ru, R), s.error_components(Rw, R), len(b))


def run(cat, rng, sigma, lo=None, hi=None, n=TRIALS):
    res = []
    for _ in range(n):
        sig = sigma if lo is None else sigma * g_of_mag(cat["mag"], lo, hi)
        res.append(trial(cat, rng, sig))
    unw = np.array([r[0] for r in res]) / s.ARCSEC
    wtd = np.array([r[1] for r in res]) / s.ARCSEC
    pred = np.array([r[2] for r in res]) / s.ARCSEC
    cu = np.array([r[3] for r in res]) / s.ARCSEC
    cw = np.array([r[4] for r in res]) / s.ARCSEC
    return {"unw": unw, "wtd": wtd, "pred": pred, "comp_u": cu, "comp_w": cw, "nstars": np.array([r[5] for r in res])}


def main():
    OUT.mkdir(exist_ok=True)
    cat = s.make_catalog()
    rng = np.random.default_rng(1)
    print("Experiment A: weighted-SVD attitude with correct associations")
    print(f"  catalogue {len(cat['id'])} stars, patch radius {s.PATCH_RADIUS} deg; camera 512 x 512, {s.PIXSCALE} arcsec/px"
          f" (f = {s.F_TRUE:.1f} px), field {2 * np.degrees(np.arctan((s.CX + 0.5) / s.F_TRUE)):.2f} deg; {TRIALS} random attitudes per point")
    for i in range(3):
        print(f"    star {cat['id'][i]:03d}  RA {cat['ra'][i]:8.4f}  Dec {cat['dec'][i]:+8.4f}  s = ({cat['vec'][i][0]:+.4f},"
              f" {cat['vec'][i][1]:+.4f}, {cat['vec'][i][2]:+.4f})  mag {cat['mag'][i]:.2f}")
    print("    ...")
    curves = {"homo": [], "het": []}
    for sg in SIGMAS:
        curves["homo"].append(run(cat, rng, sg))
        curves["het"].append(run(cat, rng, sg, 0.5, 2.0))
    print(f"  visible stars per field: median {np.median(curves['homo'][0]['nstars']):.0f}"
          f" (range {curves['homo'][0]['nstars'].min()}-{curves['homo'][0]['nstars'].max()})")
    print(f"\n  median attitude error (arcsec) vs centroid noise sigma (px)")
    print(f"    {'sigma':>6}{'homo':>9}{'homo pred':>11}{'het unw':>10}{'het wtd':>10}{'het pred':>10}{'p90 het w':>11}")
    for k, sg in enumerate(SIGMAS):
        h, t = curves["homo"][k], curves["het"][k]
        print(f"    {sg:>6.3f}{np.median(h['wtd']):>9.2f}{np.median(h['pred']):>11.2f}{np.median(t['unw']):>10.2f}"
              f"{np.median(t['wtd']):>10.2f}{np.median(t['pred']):>10.2f}{np.percentile(t['wtd'], 90):>11.2f}")
    k = SIGMAS.index(0.1)
    slope = np.polyfit(np.log(SIGMAS), np.log([np.median(c["wtd"]) for c in curves["homo"]]), 1)[0]
    print(f"  log-log slope of error vs sigma (homoscedastic): {slope:.3f}  (1 = linear, as small-error theory predicts)")
    same = np.max(np.abs(curves["homo"][k]["unw"] - curves["homo"][k]["wtd"]))
    print(f"  homoscedastic: max |weighted - unweighted| = {same:.2e} arcsec (weights nearly equal; only the cos^1.5 factor differs)")

    strong = run(cat, rng, 0.1, 0.25, 4.0)
    print("\n  compact weighted vs unweighted comparison at sigma = 0.1 px (arcsec; components in the camera frame)")
    print(f"    {'case':<34}{'median':>8}{'p90':>8}{'cross-bore':>12}{'roll':>8}{'predicted':>11}")
    for name, d in (("homoscedastic (uniform = weighted)", curves["homo"][k]), ("hetero 0.05-0.2 px, unweighted", curves["het"][k]),
                    ("hetero 0.05-0.2 px, weighted", curves["het"][k]), ("hetero 0.025-0.4 px, unweighted", strong),
                    ("hetero 0.025-0.4 px, weighted", strong)):
        key, comp = ("unw", "comp_u") if "unweighted" in name else ("wtd", "comp_w")
        print(f"    {name:<34}{np.median(d[key]):>8.2f}{np.percentile(d[key], 90):>8.2f}"
              f"{np.median(d[comp][:, 0]):>12.2f}{np.median(d[comp][:, 1]):>8.2f}{np.median(d['pred']):>11.2f}")
    for name, d in (("0.05-0.2 px", curves["het"][k]), ("0.025-0.4 px", strong)):
        better = np.mean(d["wtd"] < d["unw"])
        print(f"    weighting gain {name}: median error x{np.median(d['unw']) / np.median(d['wtd']):.2f} smaller;"
              f" weighted better in {better:.0%} of trials")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.loglog(SIGMAS, [np.median(c["wtd"]) for c in curves["homo"]], "o-", label="uniform noise (weighted = unweighted)")
    ax.loglog(SIGMAS, [np.median(c["unw"]) for c in curves["het"]], "s--", label="star-dependent noise, unweighted")
    ax.loglog(SIGMAS, [np.median(c["wtd"]) for c in curves["het"]], "s-", label="star-dependent noise, weighted 1/sigma^2")
    ax.loglog(SIGMAS, [np.median(c["pred"]) for c in curves["het"]], "k:", label="Fisher prediction (weighted)")
    lo = [np.percentile(c["wtd"], 10) for c in curves["het"]]
    hi = [np.percentile(c["wtd"], 90) for c in curves["het"]]
    ax.fill_between(SIGMAS, lo, hi, alpha=0.15, color="C2", label="10-90% (weighted)")
    ax.axvspan(0.05, 0.2, color="0.9", zorder=0)
    ax.set_xlabel("centroid noise sigma (px)   [star-dependent: noise scale; stars span 0.5-2x]")
    ax.set_ylabel("attitude error angle (arcsec, median of 1000)")
    ax.set_title(f"Experiment A: attitude error vs centroid noise ({s.PIXSCALE:.0f} arcsec/px)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_attitude_error.png", dpi=130)
    plt.close(fig)
    print("  saved", OUT / "exp_a_attitude_error.png")


if __name__ == "__main__":
    main()
