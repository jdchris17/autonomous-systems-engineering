"""
Module 2, Part 2: fusion lab.

Two sensors read the true position from Module 1 at every time step:
    z_A = p + n_A (+ bias),   sigma_A = 0.5 m
    z_B = p + n_B,            sigma_B = 1.0 m

Fusion rules compared at each step:
    equal average:              (z_A + z_B) / 2
    inverse-variance weighted:  w* z_A + (1 - w*) z_B,   w* = 0.8

Cases: both sensors unbiased, then +0.6 m added to sensor A (same noise draws).
Scored with error = estimate - p: mean (bias), SD (scatter), RMSE (total).
The table pools 200 noise realizations; the plot uses the first one.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fusion_weights import SIGMA_A, SIGMA_B, inverse_variance_weight

# ---------------------------------------------------------------- settings (Module 1 trajectory)
DURATION = 10.0    # s
DT = 0.1           # s
P0 = 0.0           # m
V0 = 0.0           # m/s
BIAS_A = 0.6       # m, added to sensor A in the biased case
SEED = 2           # fixed, recorded seed (same as Module 1)
N_REALIZATIONS = 200
OUT_DIR = Path(__file__).resolve().parent

W_STAR = inverse_variance_weight(SIGMA_A, SIGMA_B)
METHODS = {
    "Equal average": 0.5,
    "Inverse-variance weighted": W_STAR,
}
CASES = {
    "Both unbiased": 0.0,
    f"A biased by +{BIAS_A} m": BIAS_A,
}


def accel_command(t):
    """Piecewise-constant acceleration schedule u(t) in m/s^2."""
    # Small tolerance so floating-point times such as 2.9999999 land in the right interval.
    t = t + 1e-9
    if t < 3.0:
        return 1.0
    if t < 6.0:
        return 0.0
    return -1.0


def simulate(duration=DURATION, dt=DT, p0=P0, v0=V0):
    """Propagate the true state with the exact discrete update (same as Module 1)."""
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    p = np.zeros(n)
    v = np.zeros(n)
    u = np.zeros(n)

    p[0], v[0] = p0, v0
    for k in range(n):
        u[k] = accel_command(t[k])
        if k + 1 < n:
            p[k + 1] = p[k] + v[k] * dt + 0.5 * u[k] * dt**2
            v[k + 1] = v[k] + u[k] * dt
    return t, p, v, u


def draw_noise(rng, n):
    """Successive draws from one generator: A's noise array, then B's."""
    n_a = rng.normal(0.0, SIGMA_A, n)
    n_b = rng.normal(0.0, SIGMA_B, n)
    return n_a, n_b


def fuse(z_a, z_b, w):
    return w * z_a + (1 - w) * z_b


def predicted(w, bias_a):
    """Theory: bias = w * bias_A, SD = sqrt(w^2 sA^2 + (1-w)^2 sB^2), RMSE^2 = SD^2 + bias^2."""
    bias = w * bias_a
    sd = np.sqrt(w**2 * SIGMA_A**2 + (1 - w) ** 2 * SIGMA_B**2)
    return bias, sd, np.sqrt(sd**2 + bias**2)


def scores(err):
    """Mean, SD and RMSE of an error array (population SD, so RMSE^2 = mean^2 + SD^2 exactly)."""
    return err.mean(), err.std(), np.sqrt(np.mean(err**2))


def print_predictions():
    print("Predictions (written down before looking at the simulation)")
    print("  Unbiased: both rules are unbiased, so RMSE = SD and the weighted rule wins")
    print(f"            ({predicted(W_STAR, 0)[1]:.3f} m vs {predicted(0.5, 0)[1]:.3f} m).")
    print(f"  A biased: the weighted rule trusts A more ({W_STAR:.0%} vs 50%), so it inherits more of")
    print(f"            A's bias (+{W_STAR * BIAS_A:.2f} m vs +{0.5 * BIAS_A:.2f} m). Its lower scatter no longer")
    print("            makes up for that, so the equal average should have the lower RMSE.\n")


def run_pooled(p):
    """Errors for every case and method, pooled over N_REALIZATIONS noise draws."""
    rng = np.random.default_rng(SEED)
    errors = {(c, m): [] for c in CASES for m in METHODS}
    for _ in range(N_REALIZATIONS):
        n_a, n_b = draw_noise(rng, p.size)
        for case, bias_a in CASES.items():
            z_a = p + n_a + bias_a           # same noise draws in both cases
            z_b = p + n_b
            for method, w in METHODS.items():
                errors[(case, method)].append(fuse(z_a, z_b, w) - p)
    return {key: np.concatenate(errs) for key, errs in errors.items()}


def print_table(errors):
    n_samples = next(iter(errors.values())).size
    print(f"Results pooled over {N_REALIZATIONS} realizations x {n_samples // N_REALIZATIONS} steps "
          f"= {n_samples} errors per row (seed {SEED})")
    header = (f"{'Case':<20} {'Fusion rule':<27} {'bias':>7} {'pred':>6}  "
              f"{'SD':>6} {'pred':>6}  {'RMSE':>6} {'pred':>6}")
    print(header)
    print("-" * len(header))
    for case, bias_a in CASES.items():
        for method, w in METHODS.items():
            mean, sd, rmse = scores(errors[(case, method)])
            pb, psd, prmse = predicted(w, bias_a)
            print(f"{case:<20} {method:<27} {mean:+7.3f} {pb:+6.2f}  "
                  f"{sd:6.3f} {psd:6.3f}  {rmse:6.3f} {prmse:6.3f}")
    print()


def plot_errors(t, p):
    """Fusion errors over time for one realization (the first one drawn from SEED)."""
    rng = np.random.default_rng(SEED)
    n_a, n_b = draw_noise(rng, p.size)

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))
    colors = {"Equal average": "tab:blue", "Inverse-variance weighted": "tab:red"}
    for ax, (case, bias_a) in zip(axes, CASES.items()):
        z_a = p + n_a + bias_a
        z_b = p + n_b
        for method, w in METHODS.items():
            err = fuse(z_a, z_b, w) - p
            mean, sd, rmse = scores(err)
            ax.plot(t, err, lw=1.2, color=colors[method],
                    label=f"{method}: mean {mean:+.2f}, SD {sd:.2f}, RMSE {rmse:.2f} m")
            ax.axhline(mean, color=colors[method], ls="--", lw=1)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(case)
        ax.set_ylabel("error = estimate - p [m]")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Fusion errors on the Module 1 trajectory (seed {SEED}; dashed lines = mean error)")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = parser.parse_args()

    t, p, _, _ = simulate()
    print(f"w* = {W_STAR:.2f} on sensor A, {1 - W_STAR:.2f} on sensor B\n")
    print_predictions()
    print_table(run_pooled(p))

    fig = plot_errors(t, p)
    fig.savefig(OUT_DIR / "fusion_errors.png", dpi=130)
    print("Saved fusion_errors.png")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
