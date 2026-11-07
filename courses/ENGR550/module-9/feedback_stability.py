"""
Module 9, Parts 1-3 and the deliverable table: state feedback on the double integrator.

Part 1: closed loop and characteristic equation
    Plant  x' = A x + B u,  A = [[0, 1], [0, 0]],  B = [0, 1]^T,  x = [p, v].
    Feedback u = -kp p - kv v  (with disturbance acceleration d added):
        p' = v,   v' = -kp p - kv v + d   ->   A_cl = A - B [kp, kv] = [[0, 1], [-kp, -kv]].
    det(lambda I - A_cl) = lambda (lambda + kv) + kp   ->   lambda^2 + kv lambda + kp = 0,
        lambda = -kv/2 +- sqrt(kv^2/4 - kp).
    kp acts as a spring (sets the natural frequency sqrt(kp)), kv as a damper.

    Predictions before simulating:
      damped      kp=1, kv=1     -0.5 +- 0.866j  decaying oscillation, period 2pi/0.866 = 7.3 s,
                                                 envelope e^{-0.5 t}: x0 is ~gone by t = 8 s
      undamped    kp=1, kv=0     +- j            steady oscillation, period 2pi = 6.3 s, never decays
      neg. damp.  kp=1, kv=-0.4  0.2 +- 0.980j   oscillation growing as e^{0.2 t}: x20 over 15 s
      uncontrolled kp=kv=0       0, 0            p(t) = p0 + v0 t: position drifts linearly

    Why a repeated zero eigenvalue needs the matrix structure:
      Eigenvalues alone say "no growth and no decay", which is true for A = 0 (every state stays
      put) and also for A = [[0, 1], [0, 0]]. The difference is the eigenvectors. Here A - 0 I has
      rank 1, so lambda = 0 has algebraic multiplicity 2 but only ONE eigenvector [1, 0]^T: a
      Jordan block. Then exp(A t) = I + A t = [[1, t], [0, 1]] (A^2 = 0), which contains a t term,
      so any nonzero initial velocity makes p grow linearly. For eigenvalues on the boundary
      (Re = 0, or |mu| = 1), it is the Jordan block that turns "marginal" into "unstable".

Part 2: exact sampled propagation, feedback applied continuously and d held over each interval:
    expm(h [[A_cl, B], [0, 0]]) = [[F_cl, G_d], [0, 1]],   x_{k+1} = F_cl x_k + G_d d_k.

Part 3: mu_i = exp(lambda_i h). Stable: all |mu| < 1. Unstable: any |mu| > 1. Boundary: |mu| = 1.
    Euler F_E = I + h A_cl has eigenvalues 1 + h lambda instead. For the damped case
    |1 + h lambda|^2 = (1 - h/2)^2 + (0.866 h)^2 = 1 - h + h^2, which is < 1 only for h < 1.
    Exact exp(lambda h) always has |mu| = e^{-0.5 h} < 1. Euler replaces the exponential by its
    first-order Taylor term, a straight-line step that overshoots the curve; with a coarse step the
    overshoot adds energy faster than the damping removes it, so the stable system looks unstable.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

OUT_DIR = Path(__file__).resolve().parent
A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([0.0, 1.0])
T_END, H_STEP = 15.0, 0.02
N = int(round(T_END / H_STEP))           # 750 intervals
X0 = np.array([1.0, 0.2])
D_SIZE, D_TICKS = 0.5, (250, 300)          # 0.5 m/s^2 for 5 <= t < 6 s (by integer tick)
EULER_STEPS = (0.02, 0.5, 1.2)

CASES = {
    "damped":       {"kp": 1.0, "kv": 1.0,  "color": "tab:blue"},
    "undamped":     {"kp": 1.0, "kv": 0.0,  "color": "tab:purple"},
    "neg. damping": {"kp": 1.0, "kv": -0.4, "color": "tab:red"},
    "uncontrolled": {"kp": 0.0, "kv": 0.0,  "color": "tab:gray"},
}


def closed_loop(kp, kv):
    return A - np.outer(B, [kp, kv])


def discretize(A_cl, h=H_STEP):
    """Block exponential: exact F_cl and disturbance input G_d for d held over [t_k, t_k + h)."""
    M = np.zeros((3, 3))
    M[:2, :2] = A_cl
    M[:2, 2] = B
    E = expm(h * M)
    return E[:2, :2], E[:2, 2]


def disturbance(on):
    d = np.zeros(N)
    if on:
        d[D_TICKS[0]:D_TICKS[1]] = D_SIZE
    return d


def simulate(F_cl, G_d, d):
    x = np.zeros((N + 1, 2))
    x[0] = X0
    for k in range(N):
        x[k + 1] = F_cl @ x[k] + G_d * d[k]
    return x


def classify(lams, A_cl, tol=1e-9):
    re = lams.real
    if np.all(re < -tol):
        return "asymptotically stable"
    if np.any(re > tol):
        return "unstable"
    # boundary: check for a Jordan block on any repeated boundary eigenvalue
    for lam in lams[np.abs(re) <= tol]:
        alg = np.sum(np.abs(lams - lam) < 1e-6)
        geo = 2 - np.linalg.matrix_rank(A_cl - lam * np.eye(2), tol=1e-9)
        if alg > geo:
            return "unstable (Jordan block, polynomial growth)"
    return "marginally stable (bounded, no decay)"


def fmt_c(z):
    if abs(z.imag) < 1e-12:
        return f"{z.real:+.4f}"
    return f"{z.real:+.4f}{z.imag:+.4f}j"


def eig_table():
    print("Part 1 / deliverable: continuous eigenvalues, exact discrete eigenvalues (h = 0.02 s)")
    print(f"  {'case':<13}{'kp':>4}{'kv':>6}   {'lambda':<20}{'mu = eig(F_cl)':<22}{'|mu|':>8}"
          f"{'max|mu - e^(lambda h)|':>24}   classification")
    for name, c in CASES.items():
        A_cl = closed_loop(c["kp"], c["kv"])
        lams = np.linalg.eigvals(A_cl)
        lams = lams[np.argsort(-lams.imag)]
        F_cl, _ = discretize(A_cl)
        mus = np.linalg.eigvals(F_cl)
        mus = mus[np.argsort(-mus.imag)]
        err = np.max(np.abs(mus - np.exp(lams * H_STEP)))
        print(f"  {name:<13}{c['kp']:4.0f}{c['kv']:6.1f}   {fmt_c(lams[0]):<20}{fmt_c(mus[0]):<22}"
              f"{abs(mus[0]):8.5f}{err:24.1e}   {classify(lams, A_cl)}")
    A0 = closed_loop(0, 0)
    F0, _ = discretize(A0)
    print(f"  uncontrolled: rank(A - 0 I) = {np.linalg.matrix_rank(A0)} -> one eigenvector for a double "
          f"eigenvalue (Jordan block);\n    F = expm(A h) = {F0.round(4).tolist()}, so F^k = [[1, k h], [0, 1]] "
          f"grows linearly.\n")


def disturbance_effect(t, dx):
    """Compare the peak |dx| just after the pulse with the peak at the end of the run."""
    early = (t >= 6.0 - 1e-9) & (t < 10.5 - 1e-9)
    late = t >= 10.5 - 1e-9
    out = []
    for i in range(2):
        a, b = np.max(np.abs(dx[early, i])), np.max(np.abs(dx[late, i]))
        ratio = b / a
        verdict = "decays" if ratio < 0.7 else ("grows" if ratio > 1.3 else "persists")
        out.append((a, b, verdict))
    return out


def plot_responses(t, runs):
    fig, axes = plt.subplots(4, 2, sharex=True, figsize=(12, 12))
    for row, (name, c) in enumerate(CASES.items()):
        x_u, x_d = runs[name]
        for col, (lab, unit) in enumerate((("position", "m"), ("velocity", "m/s"))):
            ax = axes[row, col]
            ax.axvspan(5, 6, color="0.88", label="disturbance pulse")
            ax.plot(t, x_u[:, col], color=c["color"], lw=1.6, label="no disturbance")
            ax.plot(t, x_d[:, col], color="k", lw=1.0, ls="--", label="with disturbance")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylabel(f"{lab} ({unit})")
            ax.set_title(f"{name} (kp = {c['kp']:g}, kv = {c['kv']:g}): {lab}", fontsize=10)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Module 9: closed-loop responses, x0 = [1 m, 0.2 m/s], d = 0.5 m/s² for 5 ≤ t < 6 s")
    fig.tight_layout()
    path = OUT_DIR / "closed_loop_responses.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_disturbance(t, runs):
    fig, axes = plt.subplots(4, 2, sharex=True, figsize=(12, 10))
    for row, (name, c) in enumerate(CASES.items()):
        x_u, x_d = runs[name]
        dx = x_d - x_u
        for col, (lab, unit) in enumerate((("δp", "m"), ("δv", "m/s"))):
            ax = axes[row, col]
            ax.axvspan(5, 6, color="0.88")
            ax.plot(t, dx[:, col], color=c["color"], lw=1.6)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylabel(f"{lab} ({unit})")
            ax.set_title(f"{name}: {lab} = disturbed − undisturbed", fontsize=10)
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Module 9: effect of the disturbance pulse alone (shaded: 5 ≤ t < 6 s)")
    fig.tight_layout()
    path = OUT_DIR / "disturbance_effect.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def euler_check():
    A_cl = closed_loop(1.0, 1.0)
    lam = np.linalg.eigvals(A_cl)
    print("Part 3: Euler F_E = I + h A_cl vs exact expm(A_cl h), damped case")
    print(f"  {'h (s)':>6}  {'Euler mu':<20}{'|mu_E|':>8}   {'exact mu':<20}{'|mu|':>8}   Euler verdict")
    for h in EULER_STEPS:
        mu_e = np.linalg.eigvals(np.eye(2) + h * A_cl)
        mu_x = np.linalg.eigvals(expm(h * A_cl))
        mu_e, mu_x = mu_e[np.argmax(mu_e.imag)], mu_x[np.argmax(mu_x.imag)]
        verdict = "stable" if abs(mu_e) < 1 else "UNSTABLE"
        print(f"  {h:6.2f}  {fmt_c(mu_e):<20}{abs(mu_e):8.4f}   {fmt_c(mu_x):<20}{abs(mu_x):8.4f}   {verdict}")
    print(f"  |1 + h lambda|^2 = 1 - h + h^2 -> Euler stable only for h < 1 s; "
          f"exact |mu| = e^(-0.5 h) < 1 for every h.\n")


def main():
    eig_table()
    t = np.arange(N + 1) * H_STEP
    runs = {}
    print("Part 2: disturbance effect dx = x_disturbed - x_undisturbed "
          "(peak |dx| for 6-10.5 s vs 10.5-15 s)")
    for name, c in CASES.items():
        F_cl, G_d = discretize(closed_loop(c["kp"], c["kv"]))
        x_u = simulate(F_cl, G_d, disturbance(False))
        x_d = simulate(F_cl, G_d, disturbance(True))
        runs[name] = (x_u, x_d)
        (pa, pb, pv), (va, vb, vv) = disturbance_effect(t, x_d - x_u)
        print(f"  {name:<13} dp: {pa:6.3f} -> {pb:6.3f} m ({pv:<8})  dv: {va:6.3f} -> {vb:6.3f} m/s ({vv})"
              f"   final x = [{x_d[-1, 0]:+8.3f}, {x_d[-1, 1]:+7.3f}]")
    print()
    euler_check()
    print(f"Figures saved: {plot_responses(t, runs).name}, {plot_disturbance(t, runs).name}")


if __name__ == "__main__":
    main()
