"""module-8 -- interference_revisited.py  (Computational Exercise: Interference Revisited)

Module 6's double_slit.py computed intensity along a distant screen, using the
FAR-FIELD path-difference approximation dL = d sin(theta) from the start. This
file builds the general two-wave model with no far-field assumption baked in,
then shows the far-field result falls out of it as a special case.

Two coherent point sources, complex amplitude nowhere in sight -- just

    E1 = A1 cos(k r1 - w t)          E2 = A2 cos(k r2 - w t)

r1, r2 the distances from each source to the field point (x, y): a genuine
2-D spatial field, valid arbitrarily close to the sources, not just on a
distant screen.

We never animate the w t oscillation. Time-averaging cos^2 over a cycle is
what turns a field into an INTENSITY, and it only needs the phase difference

    dphi(x, y) = k ( r2(x, y) - r1(x, y) )

    I = < (E1 + E2)^2 >_t = A1^2 + A2^2 + 2 A1 A2 cos(dphi)      (dropping the
                                                overall factor of 1/2 from
                                                <cos^2> = 1/2 -- a constant
                                                scale, not a physical effect)
"""

import numpy as np
import matplotlib.pyplot as plt

WAVELENGTH0 = 0.5     # um (500 nm, green light)
D0 = 5.0              # um, source separation
A1_0, A2_0 = 1.0, 1.0


# ---------------------------------------------------------------------------
# the general two-wave function -- no geometry baked in
# ---------------------------------------------------------------------------
def interference_pattern(A1, A2, phase_difference):
    """Time-averaged two-wave intensity (up to the overall 1/2 from <cos^2>)."""
    return A1**2 + A2**2 + 2.0 * A1 * A2 * np.cos(phase_difference)


# ---------------------------------------------------------------------------
# expand spatially: two point sources, r1(x,y), r2(x,y), dphi(x,y)
# ---------------------------------------------------------------------------
def two_source_field(X, Y, src1, src2, A1, A2, wavelength):
    r1 = np.hypot(X - src1[0], Y - src1[1])
    r2 = np.hypot(X - src2[0], Y - src2[1])
    k = 2.0 * np.pi / wavelength
    dphi = k * (r2 - r1)
    I = interference_pattern(A1, A2, dphi)
    return I, r1, r2, dphi


def make_grid(half_width, y_range, n=500):
    x = np.linspace(-half_width, half_width, n)
    y = np.linspace(*y_range, n)
    return x, y, *np.meshgrid(x, y)


# ---------------------------------------------------------------------------
# far-field closed form (Module 6's double_slit.py formula) -- for validation
# ---------------------------------------------------------------------------
def far_field_intensity(x, L, d, A1, A2, wavelength):
    """I(x) on a screen at distance L, using dL ~= d sin(theta) (Module 6)."""
    k = 2.0 * np.pi / wavelength
    sin_theta = x / np.hypot(x, L)
    return interference_pattern(A1, A2, k * d * sin_theta)


def main():
    src1 = (-D0 / 2, 0.0)
    src2 = (D0 / 2, 0.0)

    print("=" * 78)
    print("INTERFERENCE REVISITED -- the general 2-D two-source field")
    print("=" * 78)
    print(f"sources at x = +/-{D0/2} um (separation {D0} um), wavelength "
          f"{WAVELENGTH0} um, A1=A2={A1_0}")
    print()

    # -- validation: does the general model reduce to Module 6's far-field --
    L_far = 5000.0    # um; d^2/wavelength = 50 um, so L_far is ~100x that
    x_line = np.linspace(-200.0, 200.0, 2000)
    I_general, r1, r2, dphi = two_source_field(x_line, np.full_like(x_line, L_far),
                                               src1, src2, A1_0, A2_0, WAVELENGTH0)
    I_far = far_field_intensity(x_line, L_far, D0, A1_0, A2_0, WAVELENGTH0)
    rel_err = np.abs(I_general - I_far) / (I_general.max())
    print(f"VALIDATION against Module 6's far-field formula, at L = {L_far} um "
          f"(d^2/wavelength = {D0**2/WAVELENGTH0:.0f} um):")
    print(f"   max |I_general - I_farfield| / I_max = {rel_err.max():.2e}")
    print("   The exact two-source field and the d*sin(theta) approximation")
    print("   agree once you are far enough away -- Module 6 was the special")
    print("   case all along.")
    print()

    # -- fringe visibility with amplitude imbalance --------------------
    # evaluate interference_pattern directly at its true extrema (dphi=0,pi)
    # rather than searching a spatial line -- immune to fringe-sampling issues
    for A1, A2 in [(1.0, 1.0), (1.0, 0.5), (1.0, 0.2)]:
        I_max = interference_pattern(A1, A2, 0.0)
        I_min = interference_pattern(A1, A2, np.pi)
        V_measured = (I_max - I_min) / (I_max + I_min)
        V_formula = 2 * A1 * A2 / (A1**2 + A2**2)
        print(f"   A1={A1}, A2={A2}: visibility (Imax-Imin)/(Imax+Imin) = "
              f"{V_measured:.4f}  (formula 2 A1 A2 /(A1^2+A2^2) = {V_formula:.4f})")
    print()
    print("Unequal source strengths never fully cancel: the dark fringes rise")
    print("off zero, and visibility caps out below 1 -- exactly 2 A1 A2/(A1^2+A2^2).")
    print("=" * 78)

    _plot(src1, src2)


def _plot(src1, src2):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def show(ax, x, y, X, Y, I, title, vmax=None):
        im = ax.pcolormesh(X, Y, I, cmap="inferno", shading="auto", vmax=vmax)
        for s in (src1, src2):
            ax.plot(*s, "c+", ms=10, mew=1.5)
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        return im

    # (0,0) baseline 2-D map: near-field curvature -> far-field straight fringes
    x, y, X, Y = make_grid(40, (1.0, 90.0))
    I, *_ = two_source_field(X, Y, src1, src2, A1_0, A2_0, WAVELENGTH0)
    show(axes[0, 0], x, y, X, Y, I,
        f"baseline: wavelength={WAVELENGTH0} um, d={D0} um\n"
        f"hyperbolic fringes near the sources, straightening with distance")

    # (0,1) wavelength doubled
    wl2 = 2 * WAVELENGTH0
    I2, *_ = two_source_field(X, Y, src1, src2, A1_0, A2_0, wl2)
    show(axes[0, 1], x, y, X, Y, I2, f"wavelength doubled ({wl2} um)\nfringe spacing widens")

    # (0,2) separation doubled
    src1b, src2b = (-D0, 0.0), (D0, 0.0)
    I3, *_ = two_source_field(X, Y, src1b, src2b, A1_0, A2_0, WAVELENGTH0)
    im3 = show(axes[0, 2], x, y, X, Y, I3, f"separation doubled (d={2*D0} um)\nmore, narrower fringes")
    for s in (src1b, src2b):
        axes[0, 2].plot(*s, "c+", ms=10, mew=1.5)

    # (1,0) amplitude imbalance
    I4, *_ = two_source_field(X, Y, src1, src2, 1.0, 0.25, WAVELENGTH0)
    show(axes[1, 0], x, y, X, Y, I4,
        "amplitude imbalance A1=1.0, A2=0.25\ndark fringes no longer reach zero")

    # (1,1) zoomed way out: the far-field regime, same angular framing
    xf, yf, Xf, Yf = make_grid(400, (200.0, 900.0))
    I5, *_ = two_source_field(Xf, Yf, src1, src2, A1_0, A2_0, WAVELENGTH0)
    show(axes[1, 1], xf, yf, Xf, Yf, I5,
        "zoomed far out (y up to 900 um)\nfringes now essentially straight & parallel")

    # (1,2) validation line-out vs Module 6 far-field formula
    ax = axes[1, 2]
    L_far = 5000.0
    x_line = np.linspace(-150, 150, 1000)
    I_gen, *_ = two_source_field(x_line, np.full_like(x_line, L_far), src1, src2,
                                 A1_0, A2_0, WAVELENGTH0)
    I_ff = far_field_intensity(x_line, L_far, D0, A1_0, A2_0, WAVELENGTH0)
    ax.plot(x_line, I_gen, lw=2.5, label="general 2-D model")
    ax.plot(x_line, I_ff, "--", lw=1.2, label="Module 6 far-field formula")
    ax.set_xlabel("x (um)"); ax.set_ylabel("I")
    ax.set_title(f"Line-out at L={L_far} um: general model\nvs Module 6's "
                 f"d sin(theta) formula", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Two-source interference: the full 2-D field, and Module 6 "
                 "recovered as its far-field limit", fontsize=13)
    fig.tight_layout()
    fig.savefig("interference_revisited.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
