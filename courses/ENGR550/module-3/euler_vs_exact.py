"""
Module 3, Part 3: forward Euler vs exact propagation of the two-state model.

    exact:  x_{k+1} = F x_k + G u_k,     F = I + Ah,  G   = [h^2/2, h]^T
    Euler:  x_{k+1} = F x_k + G_E u_k,   F = I + Ah,  G_E = [0, h]^T

Verification case: p0 = v0 = 0, u = 1 m/s^2 held for T = 10 s, h = 0.1, 0.5, 1.0 s.
Truth: p(t) = t^2 / 2, v(t) = t.

Predicted Euler error (derived before running):
    Velocity rows are identical (both add h*u), so Euler's v_k is exact.
    Each step the Euler position update omits the (1/2) u h^2 term, and the velocity
    it does use is exact, so the misses simply add up:
        position error at t_k = -k * (1/2) u h^2 = -(1/2) u h t_k
        final error at T       = -(T/h) * (1/2) u h^2 = -(1/2) u h T
    For u = 1, T = 10: -0.5 m, -2.5 m, -5.0 m. The error is linear in t and in h
    (first-order method), and halving h halves the error.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from state_space import F, G, G_euler, propagate

T = 10.0           # s
U = 1.0            # m/s^2
STEPS = (0.1, 0.5, 1.0)
OUT_DIR = Path(__file__).resolve().parent


def run(h):
    n = int(round(T / h))
    t = np.arange(n + 1) * h
    u = np.full(n, U)
    x0 = np.zeros(2)
    x_exact = propagate(F(h), G(h), x0, u)
    x_euler = propagate(F(h), G_euler(h), x0, u)
    p_true = U * t**2 / 2
    v_true = U * t
    return t, x_exact, x_euler, p_true, v_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = parser.parse_args()

    print("Predicted Euler position error at T: -(1/2) u h T")
    for h in STEPS:
        print(f"  h = {h}: {-0.5 * U * h * T:+.2f} m")
    print()

    header = (f"{'h [s]':>5} {'true p':>8} {'exact p':>9} {'Euler p':>9} {'Euler err':>10} "
              f"{'pred err':>9} {'exact err':>10} {'exact v':>8} {'Euler v':>8}")
    print(header)
    print("-" * len(header))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = dict(zip(STEPS, ("tab:blue", "tab:orange", "tab:red")))
    for h in STEPS:
        t, x_exact, x_euler, p_true, v_true = run(h)
        err_euler = x_euler[:, 0] - p_true
        err_exact = x_exact[:, 0] - p_true
        print(f"{h:5.1f} {p_true[-1]:8.2f} {x_exact[-1, 0]:9.4f} {x_euler[-1, 0]:9.4f} "
              f"{err_euler[-1]:+10.4f} {-0.5 * U * h * T:+9.2f} {err_exact[-1]:+10.1e} "
              f"{x_exact[-1, 1]:8.4f} {x_euler[-1, 1]:8.4f}")

        ax.plot(t, err_euler, "o-", ms=3, color=colors[h], label=f"Euler, h = {h} s")
        ax.plot(t, -0.5 * U * h * t, ":", color=colors[h], lw=1.5)
        assert np.allclose(x_euler[:, 1], v_true) and np.allclose(err_exact, 0, atol=1e-9)
    print("\nBoth methods give v = 10 m/s at every step size; the exact method's position error is roundoff.")

    ax.axhline(0, color="k", lw=1, label="exact propagation (all h)")
    ax.plot([], [], ":", color="gray", label="predicted $-\\frac{1}{2}uht$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error = estimate - $t^2/2$ [m]")
    ax.set_title("Forward Euler position error, $u$ = 1 m/s$^2$")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "euler_position_error.png", dpi=130)
    print("Saved euler_position_error.png")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
