"""module-5 -- recover_field.py  (Computational Exercise: Recover the Electric Field)

Don't use Coulomb's law for E. Take the numerical potential from
potential_field.py and differentiate it:

    E_x = -dV/dx      E_y = -dV/dy        (np.gradient, centred differences)

Overlay the field on the equipotential contours -- it should cross them at
90 degrees and point toward lower V. Then compute E from Coulomb's law
independently and compare: E_gradient vs E_Coulomb is a numerical validation
of the whole pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt

from fields import (make_grid, potential, field_from_gradient, field_coulomb,
                    magnitude)

Q = 1e-9
CHARGE_XY = (0.23, -0.17)
N = 241


def main():
    x, y, X, Y = make_grid((-1, 1), (-1, 1), N)
    charges = [(Q, *CHARGE_XY)]
    dx = x[1] - x[0]

    V = potential(X, Y, charges, x=x, y=y)
    Ex_g, Ey_g = field_from_gradient(V, x, y)          # from the potential
    Ex_c, Ey_c = field_coulomb(X, Y, charges, x=x, y=y)  # from Coulomb, independently

    mag_g = magnitude(Ex_g, Ey_g)
    mag_c = magnitude(Ex_c, Ey_c)

    # distance from the charge, in grid cells
    R = np.hypot(X - CHARGE_XY[0], Y - CHARGE_XY[1])
    rel_err = magnitude(Ex_g - Ex_c, Ey_g - Ey_c) / np.maximum(mag_c, 1e-12)

    far = R > 5 * dx
    near = (R > dx) & (R <= 5 * dx)

    # perpendicularity: angle between E and the local contour tangent.
    # grad V is normal to the contour, and E = -grad V, so E should be exactly
    # normal -> cos(angle between E_gradient and grad V) ~ -1.
    gV_y, gV_x = np.gradient(V, y, x)
    dot = (Ex_g * gV_x + Ey_g * gV_y)
    cos_align = dot / (magnitude(Ex_g, Ey_g) * magnitude(gV_x, gV_y) + 1e-30)

    print("=" * 74)
    print("RECOVER E FROM V, THEN VALIDATE AGAINST COULOMB")
    print("=" * 74)
    print(f"grid {N}x{N}, dx = {dx:.5f} m, charge q = {Q:.1e} C at {CHARGE_XY}")
    print()
    print("1. Direction check  (E = -grad V should be exactly anti-parallel to grad V)")
    print(f"   mean cos(angle(E, grad V)) = {cos_align.mean():.6f}   "
          f"(-1 => E is normal to every equipotential)")
    print(f"   E points toward lower V:  E . (-grad V) > 0 everywhere? "
          f"{np.all(dot <= 1e-12)}")
    print()
    print("2. Magnitude check  (gradient of V  vs  Coulomb's law)")
    print(f"   relative error, r > 5 cells : median {np.median(rel_err[far]):.2e}, "
          f"max {rel_err[far].max():.2e}")
    print(f"   relative error, 1-5 cells   : median {np.median(rel_err[near]):.2e}, "
          f"max {rel_err[near].max():.2e}")
    i05 = np.argmin(np.abs(R.ravel() - 0.5))
    print(f"   at r ~ 0.5 m: |E_grad| = {mag_g.ravel()[i05]:.3f}, "
          f"|E_coulomb| = {K_from_r(0.5):.3f} V/m")
    print()
    print("The centred difference has a relative error ~ (dx / r)^2: it is order")
    print("1 within a cell of the cusp, ~1e-3 at r = 10 dx, and ~1e-5 in the far")
    print("field. Refine the grid (smaller dx) and it drops quadratically. The")
    print("only other blemish is the domain edge, where np.gradient falls back")
    print("to one-sided differences.")
    print("=" * 74)

    _plot(x, y, X, Y, V, Ex_g, Ey_g, mag_g, mag_c, rel_err, R, dx)


def K_from_r(r):
    from fields import K_E
    return K_E * Q / r**2


def _plot(x, y, X, Y, V, Ex, Ey, mag_g, mag_c, rel_err, R, dx):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    levels = np.geomspace(max(V.min(), V.max() / 200), V.max() * 0.98, 16)

    # (0,0) field vectors on equipotentials
    ax = axes[0, 0]
    ax.contour(X, Y, V, levels=levels, colors="0.5", linewidths=0.6)
    s = slice(None, None, 12)
    U = Ex[s, s] / np.maximum(mag_g[s, s], 1e-30)      # unit vectors: direction only
    Vv = Ey[s, s] / np.maximum(mag_g[s, s], 1e-30)
    ax.quiver(X[s, s], Y[s, s], U, Vv, np.log10(mag_g[s, s]),
              cmap="plasma", pivot="mid", scale=30)
    ax.plot(*CHARGE_XY, "r*", ms=12)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("E (unit arrows) on equipotentials\n"
                 "cross the contours at 90 deg, point down-potential; "
                 "colour = log10|E|")

    # (0,1) streamlines colored by |E|
    ax = axes[0, 1]
    ax.contour(X, Y, V, levels=levels, colors="0.7", linewidths=0.5)
    strm = ax.streamplot(x, y, Ex, Ey, color=np.log10(mag_g), cmap="plasma",
                         density=1.3, linewidth=0.8, arrowsize=0.8)
    ax.plot(*CHARGE_XY, "r*", ms=12)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Field lines (streamplot of E)\n"
                 "radial from the + charge, densest where |E| is largest")
    fig.colorbar(strm.lines, ax=ax, label="log10 |E|  (V/m)")

    # (1,0) relative error map
    ax = axes[1, 0]
    err = np.log10(np.clip(rel_err, 1e-8, None))
    err_masked = np.where(R > dx, err, np.nan)
    im = ax.pcolormesh(X, Y, err_masked, cmap="inferno", shading="auto",
                       vmin=-6, vmax=-1)
    ax.plot(*CHARGE_XY, "c*", ms=12)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("log10  relative error  |E_grad - E_coulomb| / |E_coulomb|\n"
                 "grows toward the cusp (~1/r^2); faint spokes = stencil "
                 "anisotropy; bright edge = one-sided differences")
    fig.colorbar(im, ax=ax, label="log10 relative error")

    # (1,1) error vs distance
    ax = axes[1, 1]
    rr = R.ravel()
    ee = rel_err.ravel()
    keep = rr > dx
    ax.loglog(rr[keep], ee[keep], ".", ms=1.5, alpha=0.3)
    rk, ek = rr[keep], ee[keep]
    rs = np.geomspace(dx, R.max(), 40)
    centres, med = [], []
    for a, b in zip(rs[:-1], rs[1:]):
        sel = (rk >= a) & (rk < b)
        if sel.any():
            centres.append(np.sqrt(a * b))
            med.append(np.median(ek[sel]))
    ax.loglog(centres, med, "r-", lw=2, label="median error")
    rref = np.array([2 * dx, R.max()])
    ax.loglog(rref, 0.9 * (dx / rref) ** 2, "k--", lw=1,
              label=r"$(dx/r)^2$ reference")
    ax.axvline(5 * dx, color="grey", ls=":", lw=1, label="5 cells")
    ax.set_xlabel("distance from charge  r (m)")
    ax.set_ylabel("relative error in E")
    ax.set_title("Relative error ~ (dx/r)^2 : order-1 at the cusp,\n"
                 "~1e-5 in the far field (upturn at r>1 = domain corners)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout(pad=2.0)
    fig.savefig("recover_field.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
