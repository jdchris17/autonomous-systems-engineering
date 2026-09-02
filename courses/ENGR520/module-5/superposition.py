"""module-5 -- superposition.py  (Superposition: Multiple Charges)

Fields add:

    V(r) = sum_i  k q_i / |r - r_i|          E(r) = sum_i E_i(r)

Three configurations:

  A  two positive charges          -> a field null (saddle) on the midline
  B  electric dipole (+q, -q)      -> field lines run from + to -
  C  asymmetric 3-4 charge cluster -> the geometry you could never read off
                                      a single-point calculation

Each panel: filled equipotentials (diverging colour, + red / - blue), black
contour lines, and unit E vectors. E crosses the equipotentials at 90 deg and
runs from high V to low V.
"""

import numpy as np
import matplotlib.pyplot as plt

from fields import make_grid, potential, field_coulomb, magnitude, K_E

NC = 1e-9
N = 301

CASES = {
    "A: two positive charges": [(+2 * NC, -0.35, 0.0), (+2 * NC, 0.35, 0.0)],
    "B: electric dipole": [(+2 * NC, -0.35, 0.0), (-2 * NC, 0.35, 0.0)],
    "C: asymmetric cluster": [(+3 * NC, -0.45, 0.30), (-1 * NC, 0.50, 0.15),
                              (+1 * NC, 0.05, -0.55), (-2 * NC, -0.25, -0.30)],
}


def field_null_between(cA, cB):
    """For two like charges q_a at a, q_b at b, the on-axis null position."""
    qa, xa, ya = cA
    qb, xb, yb = cB
    if qa * qb <= 0:
        return None
    L = np.hypot(xb - xa, yb - ya)
    s = L / (1 + np.sqrt(qb / qa))          # distance from a toward b
    return (xa + s * (xb - xa) / L, ya + s * (yb - ya) / L)


def main():
    x, y, X, Y = make_grid((-1, 1), (-1, 1), N)

    print("=" * 74)
    print("SUPERPOSITION OF POINT CHARGES")
    print("=" * 74)
    for name, charges in CASES.items():
        qtot = sum(q for q, _, _ in charges)
        # dipole moment about the centroid of |q|
        w = np.array([abs(q) for q, _, _ in charges])
        pos = np.array([(xx, yy) for _, xx, yy in charges])
        centre = (w[:, None] * pos).sum(0) / w.sum()
        p = np.array([q * (np.array([xx, yy]) - centre)
                      for q, xx, yy in charges]).sum(0)
        print(f"\n{name}")
        print(f"   charges (q[nC], x, y): "
              + ", ".join(f"({q/NC:+.0f}, {xx:+.2f}, {yy:+.2f})"
                          for q, xx, yy in charges))
        print(f"   total charge   = {qtot/NC:+.1f} nC")
        print(f"   dipole moment  = ({p[0]/NC:+.3f}, {p[1]/NC:+.3f}) nC*m  "
              f"(|p| = {np.hypot(*p)/NC:.3f})")
        if name.startswith("A"):
            null = field_null_between(charges[0], charges[1])
            Ex0, Ey0 = field_coulomb(np.array([[null[0]]]), np.array([[null[1]]]),
                                     charges, x=x, y=y)
            print(f"   predicted field null at ({null[0]:+.3f}, {null[1]:+.3f}); "
                  f"|E| there = {magnitude(Ex0, Ey0)[0, 0]:.2e} V/m (~0)")
        elif name.startswith("B"):
            print("   no finite null: like a single dipole, |E| -> 0 only at "
                  "infinity")
    print()
    print("The point: with one charge the field is trivially radial. Add a")
    print("second and a null appears (A) or a through-field forms (B); with a")
    print("cluster (C) the equipotentials and field lines take a shape you")
    print("simply cannot see from a value at one point.")
    print("=" * 74)

    _plot(x, y, X, Y)


def _plot(x, y, X, Y):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    for ax, (name, charges) in zip(axes, CASES.items()):
        V = potential(X, Y, charges, x=x, y=y)
        Ex, Ey = field_coulomb(X, Y, charges, x=x, y=y)
        mag = magnitude(Ex, Ey)

        cap = np.percentile(np.abs(V), 97)
        levels = np.linspace(-cap, cap, 21)
        ax.contourf(X, Y, V, levels=levels, cmap="RdBu_r", extend="both")
        ax.contour(X, Y, V, levels=levels, colors="k", linewidths=0.3)

        s = slice(None, None, 15)
        U = Ex[s, s] / np.maximum(mag[s, s], 1e-30)
        W = Ey[s, s] / np.maximum(mag[s, s], 1e-30)
        ax.quiver(X[s, s], Y[s, s], U, W, pivot="mid", scale=32,
                  width=0.003, color="0.15")

        for q, xx, yy in charges:
            ax.plot(xx, yy, "o", ms=9 + 3 * (abs(q) > 2e-9),
                    color="red" if q > 0 else "blue",
                    markeredgecolor="k")
            ax.annotate(f"{q/NC:+.0f}", (xx, yy), color="w", ha="center",
                        va="center", fontsize=7, fontweight="bold")

        if name.startswith("A"):
            nb = field_null_between(charges[0], charges[1])
            ax.plot(*nb, "gx", ms=12, mew=2)
            ax.annotate("field null", nb, textcoords="offset points",
                        xytext=(8, 8), fontsize=8)

        ax.set_aspect("equal")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(name)

    fig.suptitle("Superposition: equipotentials (red = +V, blue = -V) with "
                 "unit E vectors\n"
                 "E is always perpendicular to the contours and points from "
                 "red toward blue", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("superposition.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
