"""
Module 9, Part 5 checkpoint: plant stability vs estimator stability.

Three different questions, three different matrices:
    1. Does the uncontrolled plant stay near equilibrium?      -> A, or its discrete F
    2. Does feedback drive the physical state to the target?   -> A - B K_c (here A_cl)
    3. Does the estimator forget an initial estimation error?  -> the error dynamics
       e^+_k = (I - K_f H) F e^+_{k-1}   (noise-free, constant gain K_f)
       With time-varying gains: Phi_k = prod_j (I - K_j H) F_j, and we ask whether ||Phi_k|| -> 0.

Experiment: the "negative damping" closed loop (kp = 1, kv = -0.4) is the plant. It is unstable
(|mu| = 1.004 per 0.02 s step, amplitude x20 over 15 s). A Kalman filter with the same model gets
position readings (SD 0.5 m) every 0.02 s, and starts from a wrong initial estimate [0.5, 0] while the
truth starts at [1, 0.2]. Process noise: random acceleration, SD 0.1 m/s^2 held over each interval.
For comparison, an open-loop predictor runs the same model from the same wrong start with no
readings. The uncontrolled double integrator is checked the same way (table only).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from kalman import correct

OUT_DIR = Path(__file__).resolve().parent
H_STEP, T_END = 0.02, 15.0
N = int(round(T_END / H_STEP))
SIGMA_W, SIGMA_P = 0.1, 0.5
X0_TRUE = np.array([1.0, 0.2])
X0_EST, P0 = np.array([0.5, 0.0]), np.diag([1.0, 1.0])
H = np.array([[1.0, 0.0]])
R = np.array([[SIGMA_P**2]])
SEED = 2
PLANTS = {"neg. damping (kp=1, kv=-0.4)": (1.0, -0.4), "uncontrolled (kp=kv=0)": (0.0, 0.0)}


def discretize(kp, kv, h=H_STEP):
    M = np.zeros((3, 3))
    M[:2, :2] = [[0.0, 1.0], [-kp, -kv]]
    M[1, 2] = 1.0
    E = expm(h * M)
    return E[:2, :2], E[:2, 2]


def steady_gain(F, Q, iters=20000):
    P = P0.copy()
    for _ in range(iters):
        P_pred = F @ P @ F.T + Q
        _, P, _, _, K = correct(np.zeros(2), P_pred, np.zeros(1), H, R)
    return K, P_pred


def run(F, G, rng):
    Q = SIGMA_W**2 * np.outer(G, G)
    x = np.zeros((N + 1, 2)); x[0] = X0_TRUE
    xh = np.zeros((N + 1, 2)); xh[0] = X0_EST
    xo = np.zeros((N + 1, 2)); xo[0] = X0_EST
    Ps = np.zeros((N + 1, 2, 2)); Ps[0] = P0
    phi_norm = np.zeros(N + 1); phi_norm[0] = 1.0
    Phi = np.eye(2)
    P = P0.copy()
    w = rng.normal(0.0, SIGMA_W, N)
    v = rng.normal(0.0, SIGMA_P, N + 1)
    for k in range(N):
        x[k + 1] = F @ x[k] + G * w[k]
        xo[k + 1] = F @ xo[k]
        x_pred, P_pred = F @ xh[k], F @ P @ F.T + Q
        xh[k + 1], P, _, _, K = correct(x_pred, P_pred, np.array([x[k + 1, 0] + v[k + 1]]), H, R)
        Ps[k + 1] = P
        Phi = (np.eye(2) - K @ H) @ F @ Phi          # error transition with time-varying gain
        phi_norm[k + 1] = np.linalg.norm(Phi, 2)
    return x, xh, xo, Ps, phi_norm


def main():
    rng = np.random.default_rng(SEED)
    t = np.arange(N + 1) * H_STEP
    print("Spectral radius (largest |eigenvalue|) per 0.02 s step")
    print(f"  {'plant':<30}{'plant F':>9}{'(I - K_f H) F':>15}   error-dynamics eigenvalues")
    results = {}
    for name, (kp, kv) in PLANTS.items():
        F, G = discretize(kp, kv)
        K_f, _ = steady_gain(F, SIGMA_W**2 * np.outer(G, G))
        E = (np.eye(2) - K_f @ H) @ F
        rho_F = np.max(np.abs(np.linalg.eigvals(F)))
        eig_E = np.linalg.eigvals(E)
        print(f"  {name:<30}{rho_F:9.5f}{np.max(np.abs(eig_E)):15.5f}   "
              f"{', '.join(f'{z.real:.4f}{z.imag:+.4f}j' for z in eig_E)}   K_f = {K_f.ravel().round(4)}")
        results[name] = run(F, G, rng)

    name = list(PLANTS)[0]
    x, xh, xo, Ps, phi_norm = results[name]
    e, eo = x - xh, x - xo
    late = t >= 5.0
    print(f"\n{name}, seed {SEED}:")
    print(f"  truth position amplitude: max |p| for t < 3 s = {np.max(np.abs(x[t < 3, 0])):.2f} m, "
          f"for t > 12 s = {np.max(np.abs(x[t > 12, 0])):.2f} m")
    print(f"  Kalman filter position error RMSE (t >= 5 s) = {np.sqrt(np.mean(e[late, 0]**2)):.3f} m, "
          f"mean predicted SD = {np.mean(np.sqrt(Ps[late, 0, 0])):.3f} m")
    print(f"  open-loop predictor position error at 15 s = {eo[-1, 0]:+.2f} m (max {np.max(np.abs(eo[:, 0])):.2f} m)")
    for tt in (1, 2, 5, 15):
        print(f"  ||Phi(t = {tt:2d} s)|| = {phi_norm[int(round(tt / H_STEP))]:.2e}   "
              f"(plant ||F^k|| = {np.linalg.norm(np.linalg.matrix_power(discretize(*PLANTS[name])[0], int(round(tt / H_STEP))), 2):.2f})")

    # figure
    F = discretize(*PLANTS[name])[0]
    fk = [np.linalg.norm(np.linalg.matrix_power(F, k), 2) for k in range(N + 1)]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 10))
    ax = axes[0]
    ax.plot(t, x[:, 0], color="k", lw=1.5, label="true position (unstable plant)")
    ax.plot(t, xh[:, 0], color="tab:blue", lw=1.0, ls="--", label="Kalman estimate")
    ax.plot(t, xo[:, 0], color="tab:red", lw=1.0, label="open-loop predictor (no readings)")
    ax.set_ylabel("position (m)")
    ax.set_title("(a) the plant diverges; the estimate follows it")
    ax.legend(loc="lower left", fontsize=9)
    ax = axes[1]
    sd = np.sqrt(Ps[:, 0, 0])
    ax.fill_between(t, -2 * sd, 2 * sd, color="tab:blue", alpha=0.15, lw=0, label="filter ±2σ")
    ax.plot(t, e[:, 0], color="tab:blue", lw=1.0, label="Kalman position error")
    ax.plot(t, eo[:, 0], color="tab:red", lw=1.0, label="open-loop predictor error")
    ax.set_ylim(-3, 3)
    ax.set_ylabel("error (m)")
    ax.set_title("(b) estimation error stays bounded; without readings it grows with the plant")
    ax.legend(loc="lower left", fontsize=9)
    ax = axes[2]
    ax.semilogy(t, fk, color="k", label="plant ||F^k||")
    ax.semilogy(t, phi_norm, color="tab:blue", label="error transition ||Π (I − K_j H) F||")
    ax.set_ylabel("matrix 2-norm")
    ax.set_xlabel("time (s)")
    ax.set_title("(c) how an initial error is amplified (plant) or forgotten (estimator)")
    ax.legend(loc="lower left", fontsize=9)
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle("Module 9: estimating an unstable plant (negative damping), position readings every 0.02 s")
    fig.tight_layout()
    path = OUT_DIR / "estimator_vs_plant.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nFigure saved: {path.name}")


if __name__ == "__main__":
    main()
