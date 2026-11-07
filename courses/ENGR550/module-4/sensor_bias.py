"""
Module 4, Part 3: a position sensor with an unknown constant bias.

Augmented state x_b = [p, v, b]^T:
    F_b = [[1, h, 0], [0, 1, 0], [0, 0, 1]],   G_b = [h^2/2, h, 0]^T
    z = p + b + n,   H_b = [1, 0, 1]

O_b = [H_b; H_b F_b; H_b F_b^2] = [[1, 0, 1], [1, h, 1], [1, 2h, 1]]: rank 2.
Columns 1 and 3 are identical, so the null space is [1, 0, -1]^T: moving the initial
position up by d and the bias down by d changes nothing the sensor can see.

Why x_A = [0, 1, 0.6] and x_B = [0.6, 1, 0] give identical outputs:
    Their difference d = x_A - x_B = [-0.6, 0, 0.6] is in that null space.
    The system is linear and both get the same known inputs, so the input terms cancel
    and the difference evolves on its own: F_b d = d (equal velocities, so the 0.6 m
    position gap never grows or shrinks, and the bias is constant). The sensor sees
    H_b d = -0.6 + 0.6 = 0 at every step. Only the sum p + b reaches the output, and
    the model provides nothing that separates the two parts of it.

Fixes:
    + unbiased velocity sensor: still rank 2. Velocity tells you nothing about absolute
      position (Part 1), so it cannot split p from b.
    + independent unbiased position sensor: rank 3. It measures p directly, and then
      the biased sensor's reading minus p gives b.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from observability import normalize_direction, null_space, observability_matrix, rank_table

H_STEP = 0.1       # s
DURATION = 10.0    # s
OUT_DIR = Path(__file__).resolve().parent

X0_A = np.array([0.0, 1.0, 0.6])
X0_B = np.array([0.6, 1.0, 0.0])

H_B = np.array([[1.0, 0.0, 1.0]])
H_B_VEL = np.array([[1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0]])
H_B_POS = np.array([[1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0]])


def F_b(h):
    return np.array([[1.0, h, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])


def G_b(h):
    return np.array([h**2 / 2, h, 0.0])


def accel_command(t):
    """Module 1 acceleration schedule, used as the shared known input."""
    t = t + 1e-9
    if t < 3.0:
        return 1.0
    if t < 6.0:
        return 0.0
    return -1.0


def propagate(x0, u, h=H_STEP):
    x = np.zeros((len(u) + 1, 3))
    x[0] = x0
    for k, uk in enumerate(u):
        x[k + 1] = F_b(h) @ x[k] + G_b(h) * uk
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = parser.parse_args()

    h = H_STEP
    O = observability_matrix(F_b(h), H_B)
    print(f"O_b for the biased position sensor (h = {h}):\n{O}")
    print(f"null space: {normalize_direction(null_space(O).ravel())}  "
          f"(x_A - x_B = {X0_A - X0_B})\n")

    ok = rank_table(F_b(h), [
        ("Biased position          H = [1 0 1]", H_B, 2),
        ("+ unbiased velocity      H = [[1 0 1],[0 1 0]]", H_B_VEL, 2),
        ("+ unbiased position      H = [[1 0 1],[1 0 0]]", H_B_POS, 3),
    ], "Rank checks, three-state bias model")

    n = int(round(DURATION / h))
    t = np.arange(n + 1) * h
    u = np.array([accel_command(tk) for tk in t[:-1]])
    xa, xb = propagate(X0_A, u), propagate(X0_B, u)

    print("Propagating x_A = [0, 1, 0.6] and x_B = [0.6, 1, 0] with the Module 1 input schedule")
    for name, Hm in (("biased position", H_B), ("+ unbiased velocity", H_B_VEL), ("+ unbiased position", H_B_POS)):
        za, zb = xa @ Hm.T, xb @ Hm.T
        diff = np.max(np.abs(za - zb))
        print(f"  {name:<20} max output difference = {diff:.2e}  -> "
              f"{'indistinguishable' if diff < 1e-9 else 'distinguishable'}")
    gap = xa[:, 0] - xb[:, 0]
    print(f"  true position gap p_A - p_B: min {gap.min():+.4f} m, max {gap.max():+.4f} m")
    print("\nAll rank checks match." if ok else "\nRANK MISMATCH.")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    axes[0].plot(t, xa[:, 0], lw=2, color="tab:blue", label="$x^{(A)}$: true position (b = 0.6 m)")
    axes[0].plot(t, xb[:, 0], lw=2, color="tab:orange", label="$x^{(B)}$: true position (b = 0)")
    axes[0].set_ylabel("true position p [m]")
    axes[0].set_title("Different trajectories (0.6 m apart throughout)...")
    axes[0].legend(loc="upper left")

    axes[1].plot(t, xa @ H_B[0], lw=4, color="tab:blue", alpha=0.5, label="$x^{(A)}$: z = p + b")
    axes[1].plot(t, xb @ H_B[0], "--", lw=2, color="tab:orange", label="$x^{(B)}$: z = p + b")
    axes[1].set_ylabel("noise-free reading z [m]")
    axes[1].set_xlabel("time [s]")
    axes[1].set_title("...produce identical biased-sensor readings")
    axes[1].legend(loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bias_ambiguity.png", dpi=130)
    print("Saved bias_ambiguity.png")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
