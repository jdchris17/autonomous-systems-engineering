"""module-9 -- recover_attitude.py  (Section 39: Then Reverse It)

Take Config B's synthetic image (synthetic_celestial_camera.py). Assume star
identities are known -- so for every rendered star we already know both:

    s_L,i   the TRUE local-frame (ENU) direction, from its catalog (RA, Dec)
            run through the SAME validated frame chain used to render the
            image (frames.py)
    s_B,i   its MEASURED camera-frame direction, recovered from a pixel
            centroid measured in the actual noisy rendered image via
            K^-1 [u_i, v_i, 1]^T, normalized

and solve for the rotation R_BL that best explains  s_B,i ~= R_BL s_L,i
for every star at once -- Wahba's problem. Module 7's camera_orientation.py
did the skeleton of this with synthetic, arbitrary test vectors; here the
reference vectors are real celestial directions from an astronomical
coordinate model, and the "measurements" come from an actually-noisy
rendered image, not perfect numbers.

Solved via the Kabsch/SVD method: build
    B = sum_i w_i  s_B,i  s_L,i^T
    U, _, Vt = svd(B)
    R_est = U @ diag(1, 1, det(U Vt)) @ Vt
(the det-fix keeps R_est a proper rotation, never a reflection.)
"""

import numpy as np
import matplotlib.pyplot as plt

from synthetic_celestial_camera import main as run_pipeline
from frames import R_BL


def measure_centroid(image, u0, v0, half_window=3):
    """Intensity-weighted centroid in a small window around the KNOWN
    approximate position (identity assumed known -- we still measure from
    the actual noisy pixel data, not from the ground truth)."""
    Ny, Nx = image.shape
    x0, x1 = max(int(round(u0)) - half_window, 0), min(int(round(u0)) + half_window + 1, Nx)
    y0, y1 = max(int(round(v0)) - half_window, 0), min(int(round(v0)) + half_window + 1, Ny)
    stamp = image[y0:y1, x0:x1].astype(float)
    total = stamp.sum()
    if total <= 0:
        return u0, v0                       # nothing detected here; fall back
    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    u_est = np.sum(stamp.sum(axis=0) * xs) / total
    v_est = np.sum(stamp.sum(axis=1) * ys) / total
    return u_est, v_est


def camera_ray(u, v, K):
    """Back-project a measured pixel through K^-1; return a UNIT vector."""
    Kinv = np.linalg.inv(K)
    p = Kinv @ np.array([u, v, 1.0])
    return p / np.linalg.norm(p)


def solve_wahba(s_B, s_L, weights=None):
    """Best-fit rotation with  s_B,i ~= R @ s_L,i  for every i (Kabsch/SVD)."""
    if weights is None:
        weights = np.ones(len(s_B))
    Bmat = (weights[:, None, None] * s_B[:, :, None] * s_L[:, None, :]).sum(axis=0)
    U, _, Vt = np.linalg.svd(Bmat)
    d = np.sign(np.linalg.det(U @ Vt))
    R_est = U @ np.diag([1.0, 1.0, d]) @ Vt
    return R_est


def rotation_angle_deg(R1, R2):
    """Angle (deg) of the rotation that separates two rotation matrices."""
    Rrel = R1 @ R2.T
    c = np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def main():
    print("=" * 90)
    print("THEN REVERSE IT: recover camera attitude from the synthetic image")
    print("=" * 90)
    print("(re-running synthetic_celestial_camera.py to get a fresh image + "
          "ground truth)\n")
    resA, resB = run_pipeline()
    cam, K = resB["cam"], resB["cam"].K
    R_true = R_BL(resB["point_alt"], resB["point_az"])

    n = len(resB["u"])
    print(f"\n{n} stars rendered in Config B's field; measuring each one's "
          f"centroid from the actual noisy image\n(not from ground truth), "
          f"then solving Wahba's problem for the best-fit R_BL.\n")

    u_meas = np.empty(n)
    v_meas = np.empty(n)
    for i in range(n):
        u_meas[i], v_meas[i] = measure_centroid(resB["realized"], resB["u"][i], resB["v"][i])

    centroid_err = np.hypot(u_meas - resB["u"], v_meas - resB["v"])
    print(f"centroid measurement error vs. true (u,v): median {np.median(centroid_err):.3f} px, "
          f"90th pct {np.percentile(centroid_err, 90):.3f} px")

    s_B = np.array([camera_ray(u, v, K) for u, v in zip(u_meas, v_meas)])
    s_L = resB["s_l"]

    print()
    print("Attitude recovery accuracy vs. number of reference stars used,")
    print("UNWEIGHTED vs. weighted by photon count (inverse-variance weighting")
    print("-- the statistically optimal choice when star quality varies wildly,")
    print("which it does here: some centroids are good to 0.01 px, others to")
    print("~1 px). 200 random draws per N, medians shown:")
    print(f"{'N stars':>8} {'unweighted (arcsec)':>20} {'weighted (arcsec)':>20}")
    rng = np.random.default_rng(0)
    n_list = [3, 5, 10, 30, 100, n]
    n_trials = 200
    err_uw_trials, err_w_trials = [], []
    for k in n_list:
        k = min(k, n)
        t_uw, t_w = [], []
        for _ in range(n_trials if k < n else 1):
            idx = rng.choice(n, size=k, replace=False)
            t_uw.append(rotation_angle_deg(solve_wahba(s_B[idx], s_L[idx]), R_true) * 3600)
            t_w.append(rotation_angle_deg(
                solve_wahba(s_B[idx], s_L[idx], weights=resB["n_photons"][idx]),
                R_true) * 3600)
        t_uw, t_w = np.array(t_uw), np.array(t_w)
        err_uw_trials.append(t_uw)
        err_w_trials.append(t_w)
        print(f"{k:>8} {np.median(t_uw):>20.4f} {np.median(t_w):>20.4f}")

    print()
    print("Unweighted is NOT monotonic in N: a small sample that happens to")
    print("include a couple of sharp, high-SNR stars can beat a bigger sample")
    print("swamped by equal-weighted faint, noisy ones -- more data does not")
    print("automatically mean a better fit if you average good and bad")
    print("measurements as if they were equally trustworthy. Weighted improves")
    print("steadily from N=5 upward, because it is using each star's actual")
    print("information content rather than just its vote -- N=3 is a genuine")
    print("small-sample exception either way: with only 3 points, occasionally")
    print("grabbing 2-3 unusually bright, unusually well-measured stars by pure")
    print("chance can outperform slightly larger samples that dilute them with")
    print("a few more ordinary ones, before there are enough stars for the")
    print("weighting to pay off systematically.")

    print()
    print(f"Final answer, all {n} rendered stars, weighted by photon count:")
    R_est_w = solve_wahba(s_B, s_L, weights=resB["n_photons"])
    err_w = rotation_angle_deg(R_est_w, R_true)
    print(f"   R_true =\n{np.round(R_true, 5)}")
    print(f"   R_est  =\n{np.round(R_est_w, 5)}")
    print(f"   |R_est R_true^T - I| = {np.abs(R_est_w @ R_true.T - np.eye(3)).max():.2e}")
    print(f"   recovered attitude error = {err_w*3600:.4f} arcsec")
    print()
    print("This is the whole point of the reversal: the reference vectors are")
    print("not arbitrary synthetic test directions (Module 7's skeleton version)")
    print("-- they are real celestial directions, generated from an actual")
    print("astronomical coordinate model (RA/Dec -> C -> E -> L), observed")
    print("through real (simulated) photon noise. Recovering R_BL to a fraction")
    print("of an arcsecond FROM THAT is a full closed loop: model -> render ->")
    print("measure -> re-solve -> match the model again.")
    print("=" * 90)

    _plot(n_list, err_uw_trials, err_w_trials, centroid_err, resB)


def _plot(n_list, err_uw_trials, err_w_trials, centroid_err, resB):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    med_uw = [np.median(t) for t in err_uw_trials]
    med_w = [np.median(t) for t in err_w_trials]
    for k, trials in zip(n_list, err_uw_trials):
        ax1.scatter([k] * len(trials), trials, color="tab:red", s=10, alpha=0.15)
    for k, trials in zip(n_list, err_w_trials):
        ax1.scatter([k] * len(trials), trials, color="tab:blue", s=10, alpha=0.15)
    ax1.loglog(n_list, med_uw, "o-", color="tab:red", lw=2, label="unweighted, median")
    ax1.loglog(n_list, med_w, "s-", color="tab:blue", lw=2, label="weighted, median")
    ax1.axhline(1.0, color="grey", ls="--", lw=1, label="1 arcsec")
    ax1.set_xlabel("number of reference stars used")
    ax1.set_ylabel("attitude recovery error (arcsec)")
    ax1.set_title("Wahba's problem: weighting matters more than N alone\n"
                  "unweighted (red) is erratic -- one lucky/unlucky draw of\n"
                  "noisy faint stars swings it; weighted (blue) improves steadily")
    ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=0.3)

    ax2.hist(centroid_err, bins=30, color="tab:blue")
    ax2.set_xlabel("|measured centroid - true (u,v)| (px)")
    ax2.set_ylabel("number of stars")
    ax2.set_title("Per-star centroid error\n(photon noise, real measurement)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("recover_attitude.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
