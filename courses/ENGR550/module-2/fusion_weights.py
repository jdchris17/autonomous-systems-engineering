"""
Module 2, Part 1: derive the fusion weight for two position sensors.

    z_A = p + n_A,   n_A ~ N(0, sigma_A^2)
    z_B = p + n_B,   n_B ~ N(0, sigma_B^2),   n_A, n_B independent

Estimator:  p_hat = w z_A + (1 - w) z_B   ->   p_hat - p = w n_A + (1 - w) n_B

Derivation (minimum variance):
    V(w)   = w^2 sA^2 + (1 - w)^2 sB^2
    dV/dw  = 2 w sA^2 - 2 (1 - w) sB^2 = 0
    w*     = sB^2 / (sA^2 + sB^2)
    V(w*)  = sA^2 sB^2 / (sA^2 + sB^2)  =  1 / (1/sA^2 + 1/sB^2)
    d2V/dw2 = 2 (sA^2 + sB^2) > 0, so this is a minimum.

Derivation (weighted least squares):
    J(q)   = (z_A - q)^2 / sA^2 + (z_B - q)^2 / sB^2
    dJ/dq  = -2 (z_A - q) / sA^2 - 2 (z_B - q) / sB^2 = 0
    q*     = (z_A / sA^2 + z_B / sB^2) / (1/sA^2 + 1/sB^2)
           = w* z_A + (1 - w*) z_B          (same weight, inverse-variance form)

This script checks both results numerically and does the 10.0 m / 12.0 m hand check.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SIGMA_A = 0.5      # m (sensor from Module 1)
SIGMA_B = 1.0      # m
OUT_DIR = Path(__file__).resolve().parent


def inverse_variance_weight(sigma_a=SIGMA_A, sigma_b=SIGMA_B):
    """Minimum-variance weight on sensor A."""
    return sigma_b**2 / (sigma_a**2 + sigma_b**2)


def error_variance(w, sigma_a=SIGMA_A, sigma_b=SIGMA_B):
    """Var(p_hat - p) for p_hat = w z_A + (1 - w) z_B."""
    return w**2 * sigma_a**2 + (1 - w) ** 2 * sigma_b**2


def wls_cost(q, z_a, z_b, sigma_a=SIGMA_A, sigma_b=SIGMA_B):
    return (z_a - q) ** 2 / sigma_a**2 + (z_b - q) ** 2 / sigma_b**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = parser.parse_args()

    w_star = inverse_variance_weight()
    v_star = error_variance(w_star)
    print(f"sigma_A = {SIGMA_A} m, sigma_B = {SIGMA_B} m")
    print(f"w*            = sB^2/(sA^2+sB^2) = {w_star:.4f}   (weight on B: {1 - w_star:.4f})")
    print(f"Var at w*     = {v_star:.4f} m^2  ->  SD {np.sqrt(v_star):.4f} m")
    print(f"Var at w=0.5  = {error_variance(0.5):.4f} m^2  ->  SD {np.sqrt(error_variance(0.5)):.4f} m")
    print(f"SD of A alone = {SIGMA_A:.4f} m  (fused estimate beats the better sensor)\n")

    # Check 1: derivative of the variance is zero at w*, and w* is the sweep minimum.
    h = 1e-6
    dv = (error_variance(w_star + h) - error_variance(w_star - h)) / (2 * h)
    w_grid = np.linspace(0, 1, 100001)
    w_sweep = w_grid[np.argmin(error_variance(w_grid))]
    print("Check 1: minimum-variance weight")
    print(f"  dVar/dw at w* (finite difference) = {dv:.2e}")
    print(f"  argmin of Var over a grid of w    = {w_sweep:.4f}")

    # Check 2: minimizing J(q) gives the same weighted average, for random reading pairs.
    rng = np.random.default_rng(0)
    q_grid = np.linspace(-20, 40, 600001)
    worst = 0.0
    for z_a, z_b in rng.uniform(0, 20, size=(5, 2)):
        q_min = q_grid[np.argmin(wls_cost(q_grid, z_a, z_b))]
        worst = max(worst, abs(q_min - (w_star * z_a + (1 - w_star) * z_b)))
    print("Check 2: argmin of J(q) vs w* z_A + (1 - w*) z_B")
    print(f"  largest difference over 5 random reading pairs = {worst:.1e} m (grid spacing 1e-4 m)")

    # Hand check from the problem.
    z_a, z_b = 10.0, 12.0
    p_hat = w_star * z_a + (1 - w_star) * z_b
    q_min = q_grid[np.argmin(wls_cost(q_grid, z_a, z_b))]
    print("Hand check: z_A = 10.0 m, z_B = 12.0 m")
    print(f"  weighted estimate = {w_star:.1f}*10 + {1 - w_star:.1f}*12 = {p_hat:.4f} m   (expected 10.4)")
    print(f"  argmin of J(q)    = {q_min:.4f} m")

    # Check 3: Monte Carlo variance of the fused error matches V(w*).
    n = 1_000_000
    n_a = rng.normal(0, SIGMA_A, n)
    n_b = rng.normal(0, SIGMA_B, n)
    err = w_star * n_a + (1 - w_star) * n_b
    print("Check 3: Monte Carlo (1e6 draws)")
    print(f"  sample variance of fused error = {err.var():.4f} m^2  (theory {v_star:.4f})")

    # Figure: variance vs weight, and J(q) for the hand-check readings.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(w_grid, error_variance(w_grid), lw=2)
    ax.plot(w_star, v_star, "o", color="tab:red", label=f"$w^*$ = {w_star:.2f}, Var = {v_star:.2f} m$^2$")
    ax.plot(0.5, error_variance(0.5), "s", color="tab:gray", label=f"equal average, Var = {error_variance(0.5):.4f} m$^2$")
    ax.set_xlabel("weight on sensor A, $w$")
    ax.set_ylabel("Var($\\hat p - p$) [m$^2$]")
    ax.set_title("Error variance vs weight")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    q_plot = np.linspace(8, 14, 400)
    ax.plot(q_plot, wls_cost(q_plot, z_a, z_b), lw=2)
    ax.axvline(z_a, color="tab:blue", ls=":", label="$z_A$ = 10.0 m")
    ax.axvline(z_b, color="tab:orange", ls=":", label="$z_B$ = 12.0 m")
    ax.plot(p_hat, wls_cost(p_hat, z_a, z_b), "o", color="tab:red", label=f"minimum at q = {p_hat:.1f} m")
    ax.set_xlabel("candidate position $q$ [m]")
    ax.set_ylabel("$J(q)$")
    ax.set_title("Weighted least-squares cost (hand check)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fusion_weights.png", dpi=130)
    print("\nSaved fusion_weights.png")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
