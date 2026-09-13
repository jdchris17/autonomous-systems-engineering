"""physics_to_pixel/simulator.py  (Connect It to the Module 7 Camera Model)

The two major projects become one pipeline. Module 7's camera does

    s_hat_i  ->  (u_i, v_i)                     ideal geometric projection

Module 8 takes each (u_i, v_i) and produces a PSF there instead of a bare
point, so the combined simulator is

    3-D/celestial scene
          |
    camera attitude              module-7/camera/star_field.py, camera_orientation.py
          v
    ideal geometric projection   module-7/camera/projection.py  (project_points)
          v
    aperture diffraction/PSF     module-8/optics_sim/fraunhofer.py (airy_analytic)
          v
    sensor sampling              module-7/camera/camera_model.py (Camera: f, p)
          v
    synthetic image

Every stage below is IMPORTED, not reimplemented -- this file is the wiring,
not new physics. The one new piece of work is `add_star_stamp`: place a
properly-normalised, sub-pixel-positioned PSF footprint for EACH star into a
single shared image array (the many-stars generalisation of
module-8/star_centroid.py's single-star `sample_star`).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_MODULE7_CAMERA = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "module-7"))
_MODULE8_OPTICS = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "module-8"))
for _p in (_MODULE7_CAMERA, _MODULE8_OPTICS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from camera_model import Camera                              # noqa: E402  (Module 7)
from camera_orientation import (make_star_field, project,     # noqa: E402
                                attitude_matrix)               # (Module 7)
from fraunhofer import airy_analytic                          # noqa: E402  (Module 8)

D0 = 100.0e-3         # 100 mm aperture
WAVELENGTH0 = 550.0e-9


def demo_camera():
    # 4000 mm focal length, 10 um pixels -- an f/40 "scope" (matches the
    # numbers module-8/optics_sim/sensor.py already validated the sensor
    # model against)
    return Camera(f=4000.0, Ws=4.4, Hs=3.3, Nx=440, Ny=330,
                  name="4000mm scope, 10um px, D=100mm")


def mag_to_flux(mag, mag_zero=0.0):
    """Standard astronomical flux-magnitude relation, mag_zero -> flux 1."""
    return 10.0 ** (-0.4 * (mag - mag_zero))


def add_star_stamp(image, u0, v0, flux, D, wavelength, px_ang, py_ang,
                   half_window, oversample=8):
    """Add ONE star's properly-integrated, sub-pixel-positioned Airy PSF
    footprint into `image`, clipped to the array bounds. Same fine-grid,
    block-average sampling as star_centroid.py's `sample_star`, generalised
    to accumulate (not just render) into a shared frame."""
    Ny_img, Nx_img = image.shape
    x0 = int(round(u0)) - half_window
    y0 = int(round(v0)) - half_window
    n = 2 * half_window + 1

    xs_fine = x0 - 0.5 + (np.arange(n * oversample) + 0.5) / oversample
    ys_fine = y0 - 0.5 + (np.arange(n * oversample) + 0.5) / oversample
    X, Y = np.meshgrid(xs_fine, ys_fine)
    theta = np.hypot((X - u0) * px_ang, (Y - v0) * py_ang)
    fine = airy_analytic(theta, D, wavelength)
    stamp = fine.reshape(n, oversample, n, oversample).mean(axis=(1, 3))

    total = stamp.sum()
    if total <= 0:
        return
    stamp *= flux / total

    xlo, xhi = max(0, x0), min(Nx_img, x0 + n)
    ylo, yhi = max(0, y0), min(Ny_img, y0 + n)
    if xlo >= xhi or ylo >= yhi:
        return                                    # entirely off-frame
    image[ylo:yhi, xlo:xhi] += stamp[ylo - y0:yhi - y0, xlo - x0:xhi - x0]


def render_image(stars_world, mags, R_cam_world, cam, D, wavelength):
    """The full pipeline for one attitude: project, then paint a PSF at
    every surviving star. Returns the image plus the projection (for
    diagnostics)."""
    uv, valid, reason = project(stars_world, R_cam_world, cam)

    px_ang = cam.px / cam.f            # rad/px -- Module 7's Camera, Module 8's physics
    py_ang = cam.py / cam.f
    theta1 = 1.22 * wavelength / D
    half_window = int(np.clip(np.ceil(4 * theta1 / px_ang), 8, 80))

    image = np.zeros((cam.Ny, cam.Nx))
    for i in np.where(valid)[0]:
        add_star_stamp(image, uv[i, 0], uv[i, 1], mag_to_flux(mags[i]),
                       D, wavelength, px_ang, py_ang, half_window)
    return image, uv, valid


def main():
    cam = demo_camera()
    theta1 = 1.22 * WAVELENGTH0 / D0
    px_ang = cam.px / cam.f

    print("=" * 82)
    print("PHYSICS-TO-PIXEL SIMULATOR")
    print("=" * 82)
    print("3-D/celestial scene   -> module-7/camera/camera_orientation.make_star_field")
    print("camera attitude       -> module-7/camera/camera_orientation.attitude_matrix")
    print("ideal projection      -> module-7/camera/camera_orientation.project"
          "  (wraps projection.project_points)")
    print("aperture PSF          -> module-8/optics_sim/fraunhofer.airy_analytic"
          "  (validated vs FFT in Phase II)")
    print("sensor sampling       -> module-7/camera/camera_model.Camera (f, p)"
          "  +  physics_to_pixel.add_star_stamp")
    print()
    print(cam.summary())
    print()
    print(f"optics: D = {D0*1e3:.0f} mm, wavelength = {WAVELENGTH0*1e9:.0f} nm")
    print(f"theta1 = 1.22 wavelength/D = {np.degrees(theta1)*3600:.4f} arcsec "
          f"= {theta1/px_ang:.2f} sensor px")
    print()

    stars = make_star_field(cam, n=60, frac=0.85, depth=1e6, seed=5)
    rng = np.random.default_rng(5)
    mags = rng.uniform(0.0, 6.0, len(stars))
    print(f"{len(stars)} stars scattered across the field, magnitudes 0-6")
    print()

    R0 = np.eye(3)
    dyaw = 0.35 * cam.fov_x / 2.0
    dpitch = -0.25 * cam.fov_y / 2.0
    R1 = attitude_matrix(dyaw, dpitch, 0.0)

    img0, uv0, valid0 = render_image(stars, mags, R0, cam, D0, WAVELENGTH0)
    img1, uv1, valid1 = render_image(stars, mags, R1, cam, D0, WAVELENGTH0)
    img2, uv2, valid2 = render_image(stars, mags, R0, cam, D0 / 3.0, WAVELENGTH0)

    for name, img, uv, valid in [("baseline attitude", img0, uv0, valid0),
                                 ("slewed attitude", img1, uv1, valid1),
                                 ("baseline attitude, D/3 aperture", img2, uv2, valid2)]:
        flux_in = mag_to_flux(mags[valid]).sum()
        flux_out = img.sum()
        print(f"{name:>32}: {valid.sum():>3d} stars in frame, "
              f"flux in = {flux_in:.4f}, flux recovered in image = {flux_out:.4f} "
              f"({100*flux_out/flux_in:.2f}%)")

    print()
    print("Flux recovered is slightly under 100% only for stars near the edge")
    print("of the frame, where the PSF stamp is clipped by the sensor boundary")
    print("-- everywhere else, sum(stamp) = flux exactly, because every stamp")
    print("is explicitly renormalised to its star's flux before being added in.")
    print("=" * 82)

    _plot(img0, img1, img2)


def _plot(img0, img1, img2):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    vmax = max(img0.max(), img1.max(), img2.max())
    for ax, img, title in zip(
        axes, (img0, img1, img2),
        ("baseline attitude", "slewed attitude\n(same optics)",
         "baseline attitude, D/3 aperture\n(same pointing, more diffraction blur)")):
        ax.imshow(np.sqrt(np.clip(img, 0, None)), cmap="inferno", origin="upper",
                 vmax=np.sqrt(vmax))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("3-D scene -> attitude -> projection -> PSF -> sensor -> "
                 "synthetic image  (sqrt stretch)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig("simulator.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
