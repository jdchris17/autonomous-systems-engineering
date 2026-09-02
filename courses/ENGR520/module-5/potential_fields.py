"""module-5 -- potential_field.py  (Computational Exercise: Potential Field)

Build a 2-D grid, drop one point charge on it, evaluate

    V(x, y) = (1 / 4 pi eps0) * q / sqrt((x - xq)^2 + (y - yq)^2)

everywhere, deal with the 1/r singularity, and visualise it as a contour plot
(kept for the next exercise) and a 3-D surface.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter

from fields import make_grid, potential, K_E

Q = 1e-9                       # 1 nC
CHARGE_XY = (0.23, -0.17)      # deliberately off the grid nodes
N = 241


def main():
    x, y, X, Y = make_grid((-1, 1), (-1, 1), N)
    charges = [(Q, *CHARGE_XY)]
    V = potential(X, Y, charges, x=x, y=y)
    dx = x[1] - x[0]

    print("=" * 70)
    print("POTENTIAL FIELD OF A POINT CHARGE")
    print("=" * 70)
    print(f"grid: {N} x {N} on [-1, 1]^2,  spacing dx = {dx:.5f} m")
    print(f"charge: q = {Q:.1e} C at {CHARGE_XY}")
    print(f"k_e = 1/(4 pi eps0) = {K_E:.4e} N m^2 / C^2")
    print()
    print("singularity handling: distance clipped at r_min = dx/2 = "
          f"{dx/2:.5f} m, and the charge sits between nodes, so no sample is")
    print("exactly on it. Result stays finite:")
    print(f"    V range on grid : {V.min():.3f} .. {V.max():.3f} V")
    print(f"    V at r = 0.5 m   : {K_E*Q/0.5:.4f} V   (matches k q / r)")
    print(f"    V at r = 0.1 m   : {K_E*Q/0.1:.4f} V")
    print()
    print("V is a single smooth hill centred on the charge, falling off as 1/r.")
    print("Equipotentials are circles around the charge -- that is what the")
    print("contour plot shows, and what the field lines will cross at 90 deg.")
    print("=" * 70)

    _plot(x, y, X, Y, V)


def _plot(x, y, X, Y, V):
    fig = plt.figure(figsize=(13, 5.5))

    ax1 = fig.add_subplot(1, 2, 1)
    # log color scale + log-spaced levels, so the 1/r falloff reads as evenly
    # spaced rings instead of one dark plane with a bright dot
    levels = np.geomspace(V.min(), V.max() * 0.98, 22)
    cf = ax1.contourf(X, Y, V, levels=levels, cmap="viridis",
                      norm=LogNorm(vmin=V.min(), vmax=V.max()))
    ax1.contour(X, Y, V, levels=levels, colors="k", linewidths=0.4)
    ax1.plot(*CHARGE_XY, "r*", ms=12)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.set_title("Equipotential contours V(x, y)\n"
                  "concentric circles; log color scale")
    ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000, 2000)
             if V.min() <= t <= V.max()]
    fig.colorbar(cf, ax=ax1, label="V (volts, log scale)",
                 ticks=ticks, format=LogFormatter(labelOnlyBase=False))

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    Vc = np.clip(V, None, np.percentile(V, 99.5))     # cap the spike for viewing
    ax2.plot_surface(X, Y, Vc, cmap="viridis", linewidth=0, antialiased=True)
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)"); ax2.set_zlabel("V (V)")
    ax2.set_title("Same V as a surface\n(peak clipped at 99.5th percentile)")

    fig.tight_layout()
    fig.savefig("potential_field.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
