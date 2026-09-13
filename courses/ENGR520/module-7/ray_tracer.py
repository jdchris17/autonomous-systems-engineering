"""module-7 -- ray_tracer.py  (Computational Exercise: ABCD Ray Tracer)

Two systems, both built from nothing but propagation_matrix and
thin_lens_matrix (abcd_optics.py):

  1. free space -> lens -> free space.
     Launch several parallel rays (theta = 0) at different heights and check
     that they converge at a distance f past the lens -- exactly, within this
     linear (paraxial, ideal-lens) model. Real lenses only do this
     approximately, because real lenses have aberrations this model omits.

  2. lens -> space -> lens : a two-lens AFOCAL system (space = f1 + f2).
     Parallel rays in should come out parallel (theta_out = 0) and scaled in
     height by -f2/f1 -- predicted from the system matrix before a single ray
     is traced, then confirmed ray by ray.
"""

import numpy as np
import matplotlib.pyplot as plt

from abcd_optics import (propagation_matrix, thin_lens_matrix, system_matrix,
                         propagate, trace_ray, lens_positions)


def find_focus(elements, y0s, n=4000):
    """Grid-search the z of minimum ray-height spread (no knowledge of f used)."""
    traces = [trace_ray(y0, 0.0, elements, samples_per_space=400) for y0 in y0s]
    zmax = max(t[0][-1] for t in traces)
    zg = np.linspace(0.0, zmax, n)
    Y = np.array([np.interp(zg, z, y) for z, y in traces])
    spread = Y.max(axis=0) - Y.min(axis=0)
    i = np.argmin(spread)
    return zg[i], spread[i]


def main():
    print("=" * 78)
    print("SYSTEM 1 -- free space -> lens -> free space")
    print("=" * 78)
    D1, F, D2 = 30.0, 100.0, 250.0          # mm
    elements1 = [("space", D1), ("lens", F), ("space", D2)]
    y0s = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])

    print(f"lens f = {F} mm at z = {D1} mm;  parallel rays (theta=0) at "
          f"heights {[float(y) for y in y0s]} mm")
    z_pred = D1 + F
    traces1 = [trace_ray(y0, 0.0, elements1) for y0 in y0s]
    y_at_pred = [np.interp(z_pred, z, y) for z, y in traces1]
    print(f"\npredicted focus at z = d1 + f = {z_pred:.1f} mm")
    print(f"   ray heights there: {[f'{y:.2e}' for y in y_at_pred]} mm  "
          f"(all ~0 -- exact in this ideal-lens model)")

    z_found, spread_found = find_focus(elements1, y0s)
    print(f"\ngrid search (no f used): minimum spread {spread_found:.2e} mm "
          f"found at z = {z_found:.2f} mm")
    print(f"   matches d1 + f = {z_pred:.1f} mm to within the search grid "
          f"resolution")
    print()
    print("Within THIS model the convergence is exact -- it's a linear map, so")
    print("y(z) = y0 (1 - (z-d1)/f) is exactly zero at z = d1+f for every y0.")
    print("Real lenses only converge parallel rays APPROXIMATELY at f, because")
    print("real lenses are not perfectly described by this ideal 2x2 matrix")
    print("(spherical/chromatic aberration live outside the paraxial model).")

    # ---- system 2: two-lens afocal system ------------------------------
    print()
    print("=" * 78)
    print("SYSTEM 2 -- lens -> space (f1+f2) -> lens  (afocal / telescope)")
    print("=" * 78)
    F1, F2 = 50.0, 150.0
    D = F1 + F2
    elements2 = [("space", 20.0), ("lens", F1), ("space", D),
                ("lens", F2), ("space", 50.0)]
    M2 = system_matrix(elements2)
    A, B, C, Dd = M2[0, 0], M2[0, 1], M2[1, 0], M2[1, 1]
    print(f"f1 = {F1} mm, f2 = {F2} mm, lens spacing = f1+f2 = {D} mm")
    print(f"system matrix M = [[{A:.6f}, {B:.4f}], [{C:.2e}, {Dd:.6f}]]")
    print(f"   predicted (by hand): A = -f2/f1 = {-F2/F1:.6f},  C = 0")
    print(f"   -> parallel rays in (theta=0) should stay parallel (theta_out=0)")
    print(f"      and scale in height by A = {A:.4f}  (a factor of "
          f"{F2/F1:.1f}x, inverted)")
    print()

    y0s2 = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    print(f"{'y_in (mm)':>10} {'y_out predicted':>17} {'y_out traced':>14} "
          f"{'theta_out traced':>18}")
    for y0 in y0s2:
        y_out, th_out = propagate([y0, 0.0], elements2)
        print(f"{y0:>10.2f} {A*y0:>17.4f} {y_out:>14.4f} {th_out:>18.2e}")
    print()
    print("Every ray comes out parallel (theta_out ~ 0) and at height A*y_in --")
    print("predicted from the matrix alone, then matched ray by ray. This")
    print("is the working principle of a Keplerian beam expander / telescope:")
    print(f"collimated light in, collimated light out, beam size x{F2/F1:.0f}, "
          "image inverted.")
    print("=" * 78)

    _plot(elements1, y0s, z_pred, elements2, y0s2)


def _plot(elements1, y0s, z_pred, elements2, y0s2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for ax, elements, y0s_, title in [
        (ax1, elements1, y0s, "System 1: parallel rays converge at f"),
        (ax2, elements2, y0s2, "System 2: afocal two-lens system"),
    ]:
        for y0 in y0s_:
            z, y = trace_ray(y0, 0.0, elements)
            ax.plot(z, y, lw=1.3)
        for zl in lens_positions(elements):
            ymax = max(abs(y0_) for y0_ in y0s_) * 1.3
            ax.plot([zl, zl], [-ymax, ymax], color="k", lw=2)
            ax.annotate("lens", (zl, ymax), ha="center", fontsize=8)
        ax.axhline(0, color="0.7", lw=0.7)
        ax.set_xlabel("z (mm)"); ax.set_ylabel("ray height y (mm)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    ax1.axvline(z_pred, color="r", ls="--", lw=1)
    ax1.annotate(f"focus, z = d1+f = {z_pred:.0f} mm", (z_pred, ax1.get_ylim()[0]),
                 xytext=(5, 5), textcoords="offset points", color="r", fontsize=8)

    fig.tight_layout()
    fig.savefig("ray_tracer.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
