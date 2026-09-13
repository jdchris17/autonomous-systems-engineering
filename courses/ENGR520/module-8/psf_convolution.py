"""module-8 -- psf_convolution.py  (Computational Exercise: PSF Convolution)

Diffraction is a formula right up until you point it at a picture. A
circular aperture D at wavelength lambda doesn't just have a "resolution
limit" in the abstract -- every point in the true scene gets smeared into an
Airy disk, and the image you actually record is the convolution of the ideal
scene with that point-spread function (PSF):

    I_formed = I_ideal * PSF

Airy PSF (circular aperture, intensity):

    I(theta) = I0 [ 2 J1(v) / v ]^2          v = (pi D / lambda) theta

(theta the angle from the optical axis; v -> 0 gives 2 J1(v)/v -> 1, the
peak). This is the same physics as single_slit.py's (sin(beta)/beta)^2 --
same diffraction integral, a disc-shaped opening instead of a slit-shaped
one, so a Bessel function instead of a sine.

Build a synthetic scene (points, lines, shapes), convolve it with Airy PSFs
of different D and lambda, and watch resolution stop being a number and
start being visible blur.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.signal import fftconvolve

N = 256                     # ideal-image size, pixels
PIXEL_SCALE = 3.0e-6        # rad/pixel (a fixed detector -- only D, lambda change)
D0 = 50.0e-3                # m, baseline aperture (50 mm)
WAVELENGTH0 = 0.5e-6         # m, baseline wavelength (500 nm, green)


# ---------------------------------------------------------------------------
# the Airy PSF
# ---------------------------------------------------------------------------
def airy_psf(npix, pixel_scale, D, wavelength):
    half = npix // 2
    coords = (np.arange(npix) - half) * pixel_scale
    X, Y = np.meshgrid(coords, coords)
    theta = np.hypot(X, Y)
    v = (np.pi * D / wavelength) * theta
    v_safe = np.where(np.abs(v) < 1e-8, 1.0, v)
    amp = np.where(np.abs(v) < 1e-8, 1.0, 2.0 * j1(v_safe) / v_safe)
    psf = amp ** 2
    return psf / psf.sum()


def first_null_pixels(D, wavelength, pixel_scale=PIXEL_SCALE):
    return 1.22 * wavelength / D / pixel_scale


# ---------------------------------------------------------------------------
# a synthetic "ideal" scene: points, lines, shapes
# ---------------------------------------------------------------------------
def make_ideal_image(n=N):
    img = np.zeros((n, n))

    for px, py in [(40, 40), (215, 50), (60, 220), (200, 210)]:   # point sources
        img[py, px] = 1.0

    img[128, 15:95] = 0.7                          # horizontal line
    img[15:95, 175] = 0.7                           # vertical line
    for i in range(70):                             # diagonal line
        img[140 + i, 20 + i] = 0.7

    x0, x1, y0, y1 = 150, 225, 140, 215             # hollow square
    img[y0:y1, x0] = 1.0; img[y0:y1, x1] = 1.0
    img[y0, x0:x1] = 1.0; img[y1, x0:x1] = 1.0

    yy, xx = np.mgrid[0:n, 0:n]                      # filled disk ("a planet")
    disk = (xx - 95) ** 2 + (yy - 165) ** 2 <= 16 ** 2
    img[disk] = 0.9

    return img


def point_source_fwhm_px(img, center, window=25):
    """Crude FWHM proxy: count pixels above half the local peak, in a small
    window around one isolated point source, and report an equivalent
    diameter sqrt(4*count/pi)."""
    cx, cy = center
    sub = img[cy - window:cy + window, cx - window:cx + window]
    half_max = sub.max() / 2.0
    n_above = np.sum(sub >= half_max)
    return 2.0 * np.sqrt(n_above / np.pi)


def main():
    ideal = make_ideal_image()

    cases = [
        ("2. small aperture", D0 / 4.0, WAVELENGTH0),
        ("3. large aperture", D0 * 4.0, WAVELENGTH0),
        ("4. short wavelength", D0, WAVELENGTH0 / 2.0),
        ("5. long wavelength", D0, WAVELENGTH0 * 2.0),
    ]

    print("=" * 78)
    print("PSF CONVOLUTION -- I_formed = I_ideal * PSF (Airy disk)")
    print("=" * 78)
    print(f"baseline: D0 = {D0*1e3:.1f} mm, wavelength0 = {WAVELENGTH0*1e9:.0f} nm, "
          f"pixel scale = {PIXEL_SCALE*206265:.3f} arcsec/px")
    print()
    print(f"{'case':>22} {'D (mm)':>9} {'wavelength (nm)':>16} "
          f"{'first null (px)':>16} {'point-source FWHM (px)':>24}")
    print("-" * 92)

    ideal_fwhm = point_source_fwhm_px(ideal, (40, 40))
    print(f"{'1. ideal (no blur)':>22} {D0*1e3:>9.1f} {WAVELENGTH0*1e9:>16.0f} "
          f"{0.0:>16.2f} {ideal_fwhm:>24.2f}")

    formed = {"1. ideal (no blur)": ideal}
    psfs = {}
    for name, D, wl in cases:
        n_null = first_null_pixels(D, wl)
        npix = int(np.clip(8 * n_null, 33, 257)) | 1     # odd kernel, big enough
        psf = airy_psf(npix, PIXEL_SCALE, D, wl)
        img = fftconvolve(ideal, psf, mode="same")
        formed[name] = img
        psfs[name] = psf
        fwhm = point_source_fwhm_px(img, (40, 40))
        print(f"{name:>22} {D*1e3:>9.1f} {wl*1e9:>16.0f} {n_null:>16.2f} "
              f"{fwhm:>24.2f}")

    print()
    print("First null in pixels = 1.22 wavelength / D / pixel_scale -- the")
    print("SAME formula from Module 6's resolution.py and Module 8's")
    print("single_slit.py, now literally the size of the blur blob on a")
    print("picture. A small aperture or a long wavelength doesn't just raise")
    print("an abstract 'resolution limit' number -- it visibly merges point")
    print("sources, thickens lines, and rounds off sharp edges.")
    print("=" * 78)

    _plot(formed, psfs)


def _plot(formed, psfs):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.2))
    order = ["1. ideal (no blur)", "2. small aperture", "3. large aperture",
            "4. short wavelength", "5. long wavelength"]
    for ax, name in zip(axes.ravel(), order):
        ax.imshow(formed[name], cmap="inferno", origin="upper")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes.ravel()[5]
    for name in order[1:]:
        psf = psfs[name]
        c = psf.shape[0] // 2
        profile = psf[c, c:]
        profile = profile / profile.max()
        ax.plot(np.arange(len(profile)), profile, label=name.split(". ")[1])
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlabel("radius (px)"); ax.set_ylabel("PSF intensity (normalised)")
    ax.set_title("Airy PSF radial profiles\n(log scale -- rings visible)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("I_formed = I_ideal * Airy PSF: diffraction as an imaging "
                 "limitation, not just a formula", fontsize=13)
    fig.tight_layout()
    fig.savefig("psf_convolution.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
