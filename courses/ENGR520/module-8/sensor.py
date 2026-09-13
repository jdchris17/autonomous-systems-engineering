"""optics_sim/sensor.py  (Optics Simulator, Phase IV: Focal Length and Sensor Sampling)

Phase III's "image" was the PSF's own native FFT pixel grid, whose angular
scale d_theta = wavelength/(N dx) is itself proportional to wavelength --
which is exactly why swapping wavelength there left the raw-pixel picture
unchanged (the ruler stretched along with what it was measuring). A real
detector doesn't do that: its pixel pitch is fixed hardware, set once.

Add:
    f                      focal length (m)
    Ws, Hs, Nx, Ny         sensor size (m) and resolution (px)
    px = Ws/Nx, py = Hs/Ny  pixel pitch (m/px)

This is exactly module-7/camera/camera_model.py's `Camera` -- reused here
directly, not reimplemented, via a sys.path insert to that folder.

The physical size of a diffraction feature on the focal plane is

    size_physical = f * theta                (small-angle, radians -> metres)
    size_pixels   = f * theta / p             (metres -> pixels)

So a point on the sensor's OWN angular pixel grid has spacing

    d_theta_sensor = p / f                    (rad/px -- independent of D
                                                and wavelength: it's hardware)

Phase II's PSF (native spacing d_theta_pupil = wavelength/(N dx)) is
resampled onto this fixed sensor grid before convolving with the image --
now D, wavelength, f AND p all genuinely show up in the final picture.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve

from aperture import make_grid, circular_aperture
from fraunhofer import fraunhofer_psf

_MODULE7 = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "module-7"))
if _MODULE7 not in sys.path:
    sys.path.insert(0, _MODULE7)
from camera_model import Camera                          # noqa: E402  (Module 7)

_MODULE8 = os.path.dirname(__file__)
if _MODULE8 not in sys.path:
    sys.path.insert(0, _MODULE8)
from psf_convolution import make_ideal_image              # noqa: E402

D0 = 50.0e-3
WAVELENGTH0 = 0.5e-6
N_PUPIL = 1024
DX = D0 / 64.0


def pupil_psf(D, wavelength, dx=DX, n_pupil=N_PUPIL, aperture_fn=circular_aperture,
              **aperture_kwargs):
    X, Y = make_grid(n_pupil, dx)
    P = aperture_fn(X, Y, D, **aperture_kwargs)
    return fraunhofer_psf(P, dx, wavelength)          # PSF, d_theta_pupil


def resample_psf_to_sensor(PSF, d_theta_pupil, px_ang, py_ang, n_out):
    """Interpolate a PSF sampled at d_theta_pupil (rad/px, isotropic) onto a
    NEW n_out x n_out grid with the sensor's own (possibly anisotropic)
    angular pixel pitch px_ang, py_ang = p/f."""
    n_in = PSF.shape[0]
    coords_in = (np.arange(n_in) - n_in // 2) * d_theta_pupil
    interp = RegularGridInterpolator((coords_in, coords_in), PSF,
                                     bounds_error=False, fill_value=0.0)
    xs_out = (np.arange(n_out) - n_out // 2) * px_ang
    ys_out = (np.arange(n_out) - n_out // 2) * py_ang
    Yo, Xo = np.meshgrid(ys_out, xs_out, indexing="ij")
    pts = np.stack([Yo.ravel(), Xo.ravel()], axis=-1)
    out = interp(pts).reshape(n_out, n_out)
    total = out.sum()
    return out / total if total > 0 else out


def main():
    # same astro-camera pixel pitch (~2.4 um) as camera_model.py's own demo;
    # two focal lengths so the SAME optical blur lands well under a pixel for
    # one and comfortably resolved for the other
    cam_short_f = Camera(f=500.0, Ws=13.2, Hs=8.8, Nx=5496, Ny=3672, name="500mm scope")
    cam_long_f = Camera(f=4000.0, Ws=13.2, Hs=8.8, Nx=5496, Ny=3672, name="4000mm scope")

    print("=" * 88)
    print("OPTICS SIMULATOR -- PHASE IV: FOCAL LENGTH AND SENSOR SAMPLING")
    print("=" * 88)
    print(f"optics fixed: D0 = {D0*1e3:.0f} mm, wavelength0 = {WAVELENGTH0*1e9:.0f} nm")
    theta1 = 1.22 * WAVELENGTH0 / D0
    print(f"diffraction-limited first null theta1 = 1.22 wavelength/D = "
          f"{theta1:.4e} rad = {np.degrees(theta1)*3600:.4f} arcsec")
    print(f"(this is the SAME number for both cameras below -- it's a fact")
    print(f" about the optics, before any sensor is even chosen)")
    print()

    print(f"{'camera':>14} {'f (mm)':>8} {'px pitch (um)':>14} "
          f"{'d_theta_sensor':>18} {'first null (px)':>16} "
          f"{'first null (um)':>16}")
    print("-" * 92)
    for cam in (cam_short_f, cam_long_f):
        # Camera stores f, px, py all in mm -- their ratio is already the
        # dimensionless angle (radians) with no unit conversion needed
        d_theta_sensor = cam.px / cam.f
        null_px = theta1 / d_theta_sensor
        null_um = cam.f * theta1 * 1e3                # f(mm)*theta(rad) -> mm -> um
        print(f"{cam.name:>14} {cam.f:>8.0f} {cam.px*1e3:>14.3f} "
              f"{np.degrees(d_theta_sensor)*3600:>15.4f}\" "
              f"{null_px:>16.2f} {null_um:>16.2f}")

    print()
    print("Same optics -> same blur ANGLE theta1 on both rows above. The two")
    print("cameras share the same pixel pitch p, so going from 500mm to")
    print("4000mm (8x) scales the physical blur size f*theta1 by the same 8x")
    print("(6.1 um -> 48.8 um) -- everything in the image scales with focal")
    print("length, the blur included. What actually matters for whether you")
    print("SEE it is the blur size relative to a pixel: size_px = f theta1/p,")
    print("so the same 8x in f gives 8x more pixels across it (2.5 -> 20.3).")
    print("D and wavelength set theta1 (the optics' job); f turns that angle")
    print("into a physical size (the lens' job); p turns that size into a")
    print("pixel count (the sensor's job) -- three separable knobs in one model.")
    print("=" * 88)

    _plot(cam_short_f, cam_long_f, theta1)


def _plot(cam_short_f, cam_long_f, theta1):
    ideal = make_ideal_image()
    PSF_pupil, d_theta_pupil = pupil_psf(D0, WAVELENGTH0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax_row, cam in zip(axes, (cam_short_f, cam_long_f)):
        px_ang = cam.px / cam.f          # mm/mm -> radians, no conversion needed
        py_ang = cam.py / cam.f
        kernel = resample_psf_to_sensor(PSF_pupil, d_theta_pupil, px_ang, py_ang,
                                        n_out=121)
        img = fftconvolve(ideal, kernel, mode="same")

        ax_row[0].imshow(kernel, cmap="inferno", origin="upper")
        ax_row[0].set_title(f"{cam.name}: PSF resampled onto sensor pixels\n"
                            f"({theta1/px_ang:.1f} px to first null)", fontsize=9.5)
        ax_row[0].set_xticks([]); ax_row[0].set_yticks([])

        ax_row[1].imshow(img, cmap="inferno", origin="upper")
        ax_row[1].set_title(f"{cam.name}: I_ideal * (resampled PSF)", fontsize=9.5)
        ax_row[1].set_xticks([]); ax_row[1].set_yticks([])

    fig.suptitle("Phase IV: SAME optics (D, wavelength), two focal lengths -- "
                 "the sensor decides how much of the blur you actually see",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig("sensor.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
