"""optics_sim/image_formation.py  (Optics Simulator, Phase III: Image Formation)

    I_optical = I_ideal * PSF

with PSF now the general Phase II `fraunhofer_psf` -- not the closed-form
Airy/sinc formulas `psf_convolution.py` (module-8, one level up) was stuck
with. Same convolution idea, but the PSF can now come from ANY pupil shape,
including ones with no analytical formula at all (an obstructed/annular
telescope pupil, tried at the end of this file).

Start with point sources, then the same richer synthetic scene from
psf_convolution.py (reused directly, not re-implemented), and show how D and
wavelength change the result -- and one case an Airy-only simulator could
never produce.

A GRID-SAMPLING TRAP, worth naming explicitly: the pupil sampling dx must be
held FIXED while D or wavelength is swept. Scaling dx with D (e.g. "always
put 64 pixels across the aperture") is exactly right for Phase II, where the
only goal is a well-resolved PSF SHAPE -- but it silently cancels D's effect
on the output pixel scale (d_theta = wavelength/(N dx) with dx = D/64 makes
D drop out of the first-null-in-pixels ratio algebraically). A fixed
detector doesn't re-focus itself when you swap lenses; dx must model that.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from aperture import make_grid, circular_aperture, annular_aperture
from fraunhofer import fraunhofer_psf

_MODULE8 = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _MODULE8 not in sys.path:
    sys.path.insert(0, _MODULE8)
from psf_convolution import make_ideal_image           # noqa: E402  (reused, not duplicated)

D0 = 50.0e-3
WAVELENGTH0 = 0.5e-6
N_PUPIL = 1024
DX = D0 / 64.0                # FIXED pupil sampling -- the same for every
                               # case below, regardless of that case's D


def make_psf_kernel(aperture_fn, D, wavelength, dx=DX, n_pupil=N_PUPIL,
                    **aperture_kwargs):
    """Pupil -> Phase II FFT PSF -> a re-normalised convolution kernel, sized
    to comfortably contain the first few Airy rings for THIS D and wavelength."""
    X, Y = make_grid(n_pupil, dx)
    P = aperture_fn(X, Y, D, **aperture_kwargs)
    PSF, d_theta = fraunhofer_psf(P, dx, wavelength)

    theta1 = 1.22 * wavelength / D
    first_null_px = theta1 / d_theta
    half = int(np.clip(np.ceil(4 * first_null_px), 20, n_pupil // 2 - 1))
    c = n_pupil // 2
    kernel = PSF[c - half:c + half + 1, c - half:c + half + 1].copy()
    kernel /= kernel.sum()
    return kernel, d_theta, first_null_px


def point_source_fwhm_px(img, center, window=90):
    cx, cy = center
    lo = max(0, cy - window)
    sub = img[lo:cy + window, max(0, cx - window):cx + window]
    half_max = sub.max() / 2.0
    n_above = np.sum(sub >= half_max)
    return 2.0 * np.sqrt(n_above / np.pi)


def main():
    ideal = make_ideal_image()

    cases = [
        ("2. small aperture", circular_aperture, D0 / 4.0, WAVELENGTH0, {}),
        ("3. large aperture", circular_aperture, D0 * 4.0, WAVELENGTH0, {}),
        ("4. short wavelength", circular_aperture, D0, WAVELENGTH0 / 2.0, {}),
        ("5. long wavelength", circular_aperture, D0, WAVELENGTH0 * 2.0, {}),
        ("6. annular (40% obstruction)", annular_aperture, D0, WAVELENGTH0,
         {"D_inner": 0.4 * D0}),
    ]

    print("=" * 88)
    print("OPTICS SIMULATOR -- PHASE III: IMAGE FORMATION  (I_optical = I_ideal * PSF)")
    print("=" * 88)
    print(f"baseline: D0 = {D0*1e3:.0f} mm, wavelength0 = {WAVELENGTH0*1e9:.0f} nm, "
          f"circular aperture")
    print(f"pupil sampling dx = {DX*1e3:.4f} mm/px HELD FIXED across every case below")
    print(f"{'case':>30} {'D (mm)':>8} {'wavelength (nm)':>16} "
          f"{'d_theta (arcsec/px)':>20} {'first null (px)':>16} "
          f"{'point FWHM (px)':>16}")
    print("-" * 108)

    formed = {"1. ideal (no blur)": ideal}
    fwhm0 = point_source_fwhm_px(ideal, (40, 40))
    print(f"{'1. ideal (no blur)':>30} {D0*1e3:>8.0f} {WAVELENGTH0*1e9:>16.0f} "
          f"{'--':>20} {'--':>16} {fwhm0:>16.2f}")

    for name, aperture_fn, D, wl, kwargs in cases:
        kernel, d_theta, first_null_px = make_psf_kernel(aperture_fn, D, wl, **kwargs)
        img = fftconvolve(ideal, kernel, mode="same")
        formed[name] = img
        fwhm = point_source_fwhm_px(img, (40, 40))
        arcsec = np.degrees(d_theta) * 3600
        print(f"{name:>30} {D*1e3:>8.1f} {wl*1e9:>16.0f} {arcsec:>20.4f} "
              f"{first_null_px:>16.2f} {fwhm:>16.2f}")

    print()
    print("d_theta stays the SAME for cases 2 and 3 (it depends only on the")
    print("FIXED pupil sampling and wavelength, never on D) -- so the first")
    print("null, measured in those same native pixels, scales as 1/D exactly")
    print("as 1.22 wavelength/D predicts, and the measured FWHM follows it:")
    print("aperture size genuinely changes the picture here.")
    print()
    print("Cases 4 and 5 look IDENTICAL in raw pixels (30 px FWHM either way)")
    print("even though wavelength doubled -- not a bug, a real limitation of")
    print("this file: the 'image' IS the PSF's own native FFT grid, and that")
    print("grid's own pixel scale d_theta = wavelength/(N dx) is ITSELF")
    print("proportional to wavelength, so wavelength's effect on angle and on")
    print("the ruler measuring it cancel exactly. The arcsec column still")
    print("shows the real 4x change -- it just never reaches the picture.")
    print("A real detector's pixel pitch does not stretch with the color of")
    print("light; giving the image a FIXED, wavelength-independent pixel grid")
    print("is precisely what Phase IV's sensor model does next.")
    print()
    print("Cases 2-5 reproduce psf_convolution.py's aperture result through")
    print("the general FFT propagator instead of the closed-form Airy formula.")
    print("Case 6 is new: an annular (obstructed) pupil has no clean")
    print("analytical PSF at all -- Phase II's FFT approach doesn't care, it")
    print("just transforms whatever P(x,y) it's given.")
    print("=" * 88)

    _plot(formed)


def _plot(formed):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.2))
    order = ["1. ideal (no blur)", "2. small aperture", "3. large aperture",
            "4. short wavelength", "5. long wavelength",
            "6. annular (40% obstruction)"]
    for ax, name in zip(axes.ravel(), order):
        ax.imshow(formed[name], cmap="inferno", origin="upper")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("I_optical = I_ideal * PSF, PSF from the FFT propagator "
                 "(Phase II) -- case 6 has no closed form", fontsize=12.5)
    fig.tight_layout()
    fig.savefig("image_formation.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
