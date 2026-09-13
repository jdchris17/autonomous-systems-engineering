"""camera/attitude_determination.py  (Reverse the Problem)

Forward problem (Phases II-VI):   R_CI, s_hat_I  ->  (u, v)
Inverse problem (this file):      known s_hat_I,i and measured (u_i, v_i)
                                   for the SAME stars  ->  R_CI

Given known-star image measurements, K^-1 turns each one back into a
camera-frame direction:

    s~_C,i  propto  K^-1 [u_i, v_i, 1]^T,   then normalise

(the demo confirms this literally: K^-1[u,v,1]^T = (1/Zc)[Xc,Yc,Zc], the true
camera-frame ray direction up to a positive scale -- normalising removes the
scale.) With paired unit vectors {s_hat_I,i} and {s~_C,i}, the rotation

    s~_C,i  ~=  R_CI s_hat_I,i        for all i

is found by least squares -- this is Wahba's problem, solved here with its
classic closed-form SVD solution (Markley 1988 / the same algorithm as
Kabsch/orthogonal-Procrustes point-cloud registration).

This assumes star IDENTIFICATION is already solved -- i.e. we already know
WHICH catalog star produced each image dot. Matching unlabelled dots to
catalog entries (the "lost in space" problem) is a separate, harder problem
this sets up but does not solve.
"""

import numpy as np
import matplotlib.pyplot as plt

from camera_model import Camera
from star_field import make_catalog, look_at, render

N_CATALOG = 400            # a denser sky than star_field.py, so more stars
SIGMAS_PX = [0.0, 0.1, 0.5, 1.0]
N_TRIALS = 150               # Monte Carlo repeats per noise level


# ---------------------------------------------------------------------------
# K^-1: pixel measurement -> camera-frame unit direction
# ---------------------------------------------------------------------------
def pixels_to_directions(uv, K):
    K_inv = np.linalg.inv(K)
    homog = np.column_stack([uv, np.ones(len(uv))])
    d = homog @ K_inv.T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Wahba's problem: optimal R with  s_C,i ~= R @ s_I,i  (least squares)
# ---------------------------------------------------------------------------
def solve_wahba(s_I, s_C, weights=None):
    if weights is None:
        weights = np.ones(len(s_I))
    H = (s_I * weights[:, None]).T @ s_C          # 3x3
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T))
    R = V @ np.diag([1.0, 1.0, d]) @ U.T
    return R                                       # s_C,i ~= R @ s_I,i


def rotation_error_deg(R_est, R_true):
    dR = R_est @ R_true.T
    cos_ang = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_ang))


# ---------------------------------------------------------------------------
def main():
    cam = Camera(f=8.0, Ws=10.0, Hs=8.0, Nx=1024, Ny=820, name="star camera")
    s_hat, mag = make_catalog(n=N_CATALOG, seed=11)
    R_true = look_at(s_hat[0])

    uv_true, valid, _ = render(s_hat, R_true, cam)
    s_I_all = s_hat[valid]
    uv_true = uv_true[valid]
    n_visible = len(s_I_all)

    print("=" * 82)
    print("ATTITUDE DETERMINATION -- reversing the forward model")
    print("=" * 82)
    print(f"catalog: {N_CATALOG} stars, {n_visible} visible in this frame")
    print()

    # ---- noiseless sanity check: must recover R_true to machine precision -
    s_C_perfect = pixels_to_directions(uv_true, cam.K)
    R_recovered = solve_wahba(s_I_all, s_C_perfect)
    err0 = rotation_error_deg(R_recovered, R_true)
    print(f"VALIDATION -- zero measurement noise:")
    print(f"   attitude recovery error = {err0:.2e} deg  "
          f"({err0*3600:.2e} arcsec) -- should be ~machine precision")
    print()

    # ---- sweep 1: attitude error vs pixel noise, all visible stars used --
    rng = np.random.default_rng(42)
    print(f"SWEEP 1 -- attitude error vs centroid noise "
          f"(all {n_visible} visible stars, {N_TRIALS} Monte Carlo trials)")
    print(f"{'sigma_px':>10} {'sigma_arcsec/star':>18} {'mean err (arcsec)':>19} "
          f"{'std err (arcsec)':>18}")
    sweep1 = []
    for s_px in SIGMAS_PX:
        errs = []
        for _ in range(N_TRIALS):
            uv_noisy = uv_true + rng.normal(0.0, s_px, uv_true.shape) if s_px > 0 else uv_true
            s_C = pixels_to_directions(uv_noisy, cam.K)
            R_est = solve_wahba(s_I_all, s_C)
            errs.append(rotation_error_deg(R_est, R_true) * 3600.0)
        errs = np.array(errs)
        sweep1.append((s_px, errs.mean(), errs.std()))
        s_arcsec = s_px * cam.angular_scale_x
        print(f"{s_px:>10.2f} {s_arcsec:>18.3f} {errs.mean():>19.4f} "
              f"{errs.std():>18.4f}")

    # ---- sweep 2: attitude error vs number of stars used, fixed noise -----
    print()
    s_fixed = 0.5
    n_list = [2, 3, 5, 8, 12, 20, 30, n_visible]
    n_list = sorted(set(n for n in n_list if n <= n_visible))
    print(f"SWEEP 2 -- attitude error vs number of stars used "
          f"(sigma = {s_fixed} px, {N_TRIALS} trials)")
    print(f"{'N stars':>8} {'mean err (arcsec)':>19} {'predicted ~ 1/sqrt(N)':>22}")
    sweep2 = []
    for n in n_list:
        errs = []
        for _ in range(N_TRIALS):
            idx = rng.choice(n_visible, size=n, replace=False)
            uv_noisy = uv_true[idx] + rng.normal(0.0, s_fixed, (n, 2))
            s_C = pixels_to_directions(uv_noisy, cam.K)
            R_est = solve_wahba(s_I_all[idx], s_C)
            errs.append(rotation_error_deg(R_est, R_true) * 3600.0)
        errs = np.array(errs)
        sweep2.append((n, errs.mean(), errs.std()))
    # anchor the 1/sqrt(N) reference at the LARGEST N (most stable estimate,
    # least Monte Carlo noise), not the smallest (noisiest, few stars)
    ref_err = sweep2[-1][1] * np.sqrt(sweep2[-1][0])
    for n, mean_e, std_e in sweep2:
        predicted = ref_err / np.sqrt(n)
        print(f"{n:>8d} {mean_e:>19.4f} {predicted:>22.4f}")

    print()
    print("Both trends are the point: attitude error grows LINEARLY with")
    print("centroid noise (Sweep 1 -- garbage in, garbage out, in a completely")
    print("predictable way), and shrinks as ~1/sqrt(N) with more stars")
    print("(Sweep 2 -- exactly like averaging down any other measurement")
    print("noise). Neither trend was assumed; both fall out of solving Wahba's")
    print("problem the same way every time and just changing the inputs.")
    print("=" * 82)

    _plot(sweep1, sweep2, n_visible)


def _plot(sweep1, sweep2, n_visible):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    s_px = np.array([r[0] for r in sweep1])
    mean1 = np.array([r[1] for r in sweep1])
    std1 = np.array([r[2] for r in sweep1])
    ax1.errorbar(s_px, np.maximum(mean1, 1e-6), yerr=std1, fmt="o-", capsize=3)
    ax1.set_xlabel("centroid noise sigma (px)")
    ax1.set_ylabel("attitude recovery error (arcsec)")
    ax1.set_title(f"Sweep 1: error vs centroid noise\n"
                  f"({n_visible} stars used, error bars = Monte Carlo std)")
    ax1.grid(True, alpha=0.3)

    n_stars = np.array([r[0] for r in sweep2])
    mean2 = np.array([r[1] for r in sweep2])
    std2 = np.array([r[2] for r in sweep2])
    ax2.errorbar(n_stars, mean2, yerr=std2, fmt="o", capsize=3, label="measured")
    nn = np.geomspace(n_stars.min(), n_stars.max(), 50)
    ref = mean2[-1] * np.sqrt(n_stars[-1])
    ax2.plot(nn, ref / np.sqrt(nn), "k--", label="1/sqrt(N) reference")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("number of stars used")
    ax2.set_ylabel("attitude recovery error (arcsec)")
    ax2.set_title("Sweep 2: error vs star count\n(sigma = 0.5 px, fixed)")
    ax2.legend(fontsize=8); ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("attitude_determination.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
