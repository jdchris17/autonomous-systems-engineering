"""optics_sim/aperture.py  (Optics Simulator, Phase I: Aperture)

A pupil / aperture function P(x, y): 1 where light gets through, 0 where a
stop blocks it. Everything downstream -- Fraunhofer diffraction, the PSF,
image formation -- starts from this one 2-D array, so getting its sampling
convention right here matters for every later phase.

CONVENTION: the pupil grid is n x n, pixel spacing dx (metres), centred on
array index n//2 -- i.e. `make_grid` returns physical coordinates, not pixel
indices. Phase II's FFT (`fftshift(fft2(ifftshift(P)))`) assumes exactly
this centring, so always build P through `make_grid` rather than indexing
the array by hand.
"""

import numpy as np
import matplotlib.pyplot as plt


def make_grid(n, dx):
    """n x n physical-coordinate grid (metres), pixel spacing dx, centred
    on index n//2."""
    coords = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(coords, coords)
    return X, Y


def circular_aperture(X, Y, D):
    """Filled circle, diameter D."""
    r = np.hypot(X, Y)
    return (r <= D / 2.0).astype(float)


def rectangular_aperture(X, Y, width, height=None):
    """Rectangle (or square if height is None), full width/height."""
    height = width if height is None else height
    return ((np.abs(X) <= width / 2.0) & (np.abs(Y) <= height / 2.0)).astype(float)


def annular_aperture(X, Y, D_outer, D_inner):
    """Annulus (ring): the classic model of a telescope pupil with a
    secondary-mirror obstruction of diameter D_inner."""
    r = np.hypot(X, Y)
    return ((r <= D_outer / 2.0) & (r >= D_inner / 2.0)).astype(float)


def main():
    n = 512
    D = 10.0e-3            # 10 mm aperture
    dx = D / 64.0           # 64 pixels across the aperture
    X, Y = make_grid(n, dx)

    apertures = {
        "circular": circular_aperture(X, Y, D),
        "rectangular": rectangular_aperture(X, Y, D, D * 0.6),
        "annular (30% obstruction)": annular_aperture(X, Y, D, 0.3 * D),
    }

    print("=" * 70)
    print("OPTICS SIMULATOR -- PHASE I: APERTURE")
    print("=" * 70)
    print(f"grid: {n} x {n}, pixel spacing dx = {dx*1e6:.2f} um, "
          f"physical extent = {n*dx*1e3:.2f} mm")
    print(f"nominal aperture diameter D = {D*1e3:.1f} mm "
          f"({D/dx:.0f} pixels across)")
    print()
    for name, P in apertures.items():
        area_px = P.sum()
        print(f"{name:>28}: {area_px:>7.0f} open pixels "
              f"({100*area_px/P.size:.2f}% of the array), "
              f"values in {{{P.min():.0f}, {P.max():.0f}}}")
    print()
    print("P(x,y) is a plain amplitude transmission mask for now (0 or 1,")
    print("real-valued) -- Phase II Fourier-transforms it directly. Phase")
    print("errors (aberrations) would make P complex; not needed yet.")
    print("=" * 70)

    _plot(X, Y, apertures, dx)


def _plot(X, Y, apertures, dx):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    extent = [X.min() * 1e3, X.max() * 1e3, Y.min() * 1e3, Y.max() * 1e3]
    for ax, (name, P) in zip(axes, apertures.items()):
        ax.imshow(P, cmap="gray", origin="lower", extent=extent)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        lim = 6.0
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.suptitle("Phase I: aperture functions P(x, y)", fontsize=13)
    fig.tight_layout()
    fig.savefig("aperture.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
