"""
Module 4, Part 2: add uncertainty to the motion and propagate mean and covariance.

Motion with an unknown acceleration error held over each interval:
    x_{k+1} = F x_k + G u_k + G a_k,   a_k ~ N(0, sigma_a^2), independent across k and of x_0

Prediction:
    mu_{k+1} = F mu_k + G u_k
    P_{k+1}  = F P_k F^T + Q,   Q = G sigma_a^2 G^T = sigma_a^2 [[h^4/4, h^3/2], [h^3/2, h^2]]

Why position and velocity errors are correlated:
    One acceleration error a_k moves velocity by h a_k and position by (h^2/2) a_k, with the
    same sign. Q's off-diagonal h^3/2 sigma_a^2 is that shared cause. Afterwards, F keeps
    adding h * (velocity error) to the position error each step, so a velocity error that is
    too high today becomes a position error that is too high later. Both effects push P_pv
    positive. Dropping the off-diagonal would claim that knowing the velocity error tells you
    nothing about the position error, which is false under this model.

The prediction is checked against an ensemble of 5,000 simulated trajectories.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from observability import F, G

DURATION = 10.0    # s
H_STEP = 0.1       # s
U = 0.0            # m/s^2, known input
MU0 = np.array([0.0, 1.0])
P0 = np.diag([0.5**2, 0.2**2])
SIGMA_A = 0.3      # m/s^2
N_TRIALS = 5000
SEED = 2           # fixed, recorded seed
OUT_DIR = Path(__file__).resolve().parent


def process_noise(h, sigma_a):
    return sigma_a**2 * np.outer(G(h), G(h))


def predict(n_steps, h=H_STEP, mu0=MU0, P0=P0, sigma_a=SIGMA_A, u=U):
    """Propagate mean and covariance; returns arrays shaped (n+1, 2) and (n+1, 2, 2)."""
    Fh, Gh, Q = F(h), G(h), process_noise(h, sigma_a)
    mu = np.zeros((n_steps + 1, 2))
    P = np.zeros((n_steps + 1, 2, 2))
    mu[0], P[0] = mu0, P0
    for k in range(n_steps):
        mu[k + 1] = Fh @ mu[k] + Gh * u
        P[k + 1] = Fh @ P[k] @ Fh.T + Q
    return mu, P


def simulate_ensemble(n_steps, h=H_STEP, n_trials=N_TRIALS, seed=SEED, u=U):
    """Monte Carlo: x_0 ~ N(mu0, P0), fresh a_k per trajectory per step. Returns (n+1, n_trials, 2)."""
    rng = np.random.default_rng(seed)
    Fh, Gh = F(h), G(h)
    x = np.zeros((n_steps + 1, n_trials, 2))
    x[0] = rng.multivariate_normal(MU0, P0, size=n_trials)
    for k in range(n_steps):
        a = rng.normal(0.0, SIGMA_A, n_trials)
        # Rows are trajectories, so x F^T applies F to each one.
        x[k + 1] = x[k] @ Fh.T + np.outer(u + a, Gh)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = parser.parse_args()

    n = int(round(DURATION / H_STEP))
    t = np.arange(n + 1) * H_STEP

    # Q from the formula vs the boxed closed form.
    h = H_STEP
    Q_closed = SIGMA_A**2 * np.array([[h**4 / 4, h**3 / 2], [h**3 / 2, h**2]])
    print(f"Q = G sigma_a^2 G^T =\n{process_noise(h, SIGMA_A)}")
    print(f"matches closed form: {np.allclose(process_noise(h, SIGMA_A), Q_closed)}\n")

    # Quick check: Q = 0.
    _, P_noQ = predict(n, sigma_a=0.0)
    print("Quick check with Q = 0, P(T) (expected [[4.25, 0.40], [0.40, 0.04]]):")
    print(np.round(P_noQ[-1], 6), "\n")

    mu, P = predict(n)
    x = simulate_ensemble(n)
    mu_emp = x.mean(axis=1)
    P_emp = np.array([np.cov(x[k], rowvar=False) for k in range(n + 1)])

    print(f"Predicted vs ensemble ({N_TRIALS} trajectories, seed {SEED})")
    print(f"{'t [s]':>5} {'mu_p':>7} {'emp':>7} {'mu_v':>6} {'emp':>6} | {'sd_p':>6} {'emp':>6} "
          f"{'sd_v':>6} {'emp':>6} {'P_pv':>6} {'emp':>6}")
    for tk in (0, 2.5, 5, 7.5, 10):
        k = int(round(tk / H_STEP))
        print(f"{tk:5.1f} {mu[k, 0]:7.3f} {mu_emp[k, 0]:7.3f} {mu[k, 1]:6.3f} {mu_emp[k, 1]:6.3f} | "
              f"{np.sqrt(P[k, 0, 0]):6.3f} {np.sqrt(P_emp[k, 0, 0]):6.3f} "
              f"{np.sqrt(P[k, 1, 1]):6.3f} {np.sqrt(P_emp[k, 1, 1]):6.3f} "
              f"{P[k, 0, 1]:6.3f} {P_emp[k, 0, 1]:6.3f}")

    rel = np.abs(P_emp - P) / np.sqrt(np.einsum("kii,kjj->kij", P, P))
    print(f"\nLargest |empirical - predicted| covariance entry over all times, scaled by sqrt(P_ii P_jj): "
          f"{rel.max():.3f}")
    print(f"  (sampling error of a covariance estimate from {N_TRIALS} draws is ~sqrt(2/N) = "
          f"{np.sqrt(2 / N_TRIALS):.3f} per standard error)")
    print(f"P(T) predicted:\n{np.round(P[-1], 4)}\nP(T) ensemble:\n{np.round(P_emp[-1], 4)}")
    corr = P[-1, 0, 1] / np.sqrt(P[-1, 0, 0] * P[-1, 1, 1])
    print(f"Predicted position-velocity correlation at T: {corr:.3f}")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
    panels = [
        (np.sqrt(P[:, 0, 0]), np.sqrt(P_emp[:, 0, 0]), np.sqrt(P_noQ[:, 0, 0]), "position SD $\\sqrt{P_{pp}}$ [m]"),
        (np.sqrt(P[:, 1, 1]), np.sqrt(P_emp[:, 1, 1]), np.sqrt(P_noQ[:, 1, 1]), "velocity SD $\\sqrt{P_{vv}}$ [m/s]"),
        (P[:, 0, 1], P_emp[:, 0, 1], P_noQ[:, 0, 1], "covariance $P_{pv}$ [m$^2$/s]"),
    ]
    for ax, (pred, emp, noq, label) in zip(axes, panels):
        ax.plot(t, pred, lw=2, color="tab:blue", label="predicted ($FPF^T + Q$)")
        ax.plot(t[::2], emp[::2], "o", ms=3, color="tab:red", label=f"ensemble ({N_TRIALS} runs)")
        ax.plot(t, noq, "--", lw=1.2, color="gray", label="predicted with $Q = 0$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Covariance propagation, $\\sigma_a$ = {SIGMA_A} m/s$^2$, h = {H_STEP} s, seed {SEED}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "covariance_propagation.png", dpi=130)
    print("\nSaved covariance_propagation.png")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
