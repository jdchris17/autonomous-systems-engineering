"""module-9 -- synthetic_celestial_camera.py  (Section 38: Synthetic Celestial Camera)

Combines Modules 3, 7, 8 and this one. Ten steps, each traceable to a
specific already-built, already-validated piece:

  1. star direction                  frames.star_celestial          (C)
  2. -> observer frame                 frames.star_earth_fixed/local  (C->E->L)
  3. reject below horizon              altitude <= 0
  4. -> camera frame                    frames.R_BL / star_camera      (L->B)
  5. reject behind camera              module-7 project_points (Zc<=0)
  6. project into image                 module-7 project_points (pinhole)
  7. reject outside FOV/sensor          module-7 project_points (image_size)
  8. relative brightness from magnitude magnitude_flux.relative_flux
  9. generate PSF                       module-8 airy_value (Airy disk)
  10. sample onto sensor                module-8's fine-subgrid integration
                                        technique (star_centroid.py), + one
                                        Poisson draw per pixel ("photon
                                        realization")

DISTORTION: not modelled. Step 6 is the ideal pinhole from Module 7 --
a stated simplification, not a silent omission.

PHOTON BUDGET: a single order-of-magnitude zero point,
ZERO_MAG_PHOTON_RATE = 1e10 photons/s/m^2 for m=0 (roughly the textbook
~1000 photons/s/cm^2/Angstrom V-band value integrated over a ~1000
Angstrom bandpass), times a flat quantum efficiency. Not a photometric
calibration -- a believable order of magnitude, clearly labelled as such.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(__file__)
for rel in [("..", "module-7"), ("..", "module-8")]:
    p = os.path.normpath(os.path.join(_HERE, *rel))
    if p not in sys.path:
        sys.path.insert(0, p)

from camera_model import Camera                       # noqa: E402  (Module 7)
from projection import project_points                  # noqa: E402  (Module 7)
from star_centroid import airy_value                   # noqa: E402  (Module 8)

from frames import (star_celestial, star_earth_fixed, star_local,        # this module
                    altaz_from_local, R_BL)
from spherical_astronomy import julian_date
from magnitude_flux import relative_flux
from star_catalog import generate_catalog, MAG_MAX

ZERO_MAG_PHOTON_RATE = 1.0e10     # photons / s / m^2, m = 0  (order-of-magnitude)
QUANTUM_EFFICIENCY = 0.7

# a few real named stars (RA hours, Dec deg, apparent V magnitude), added to
# the synthetic field catalog so there is something recognisable in it
NAMED_STARS = [
    ("Vega", 18.615, 38.784, 0.03),
    ("Polaris", 2.530, 89.264, 1.98),
    ("Deneb", 20.690, 45.280, 1.25),
]


def expected_photons(mag, D_m, exposure_s, qe=QUANTUM_EFFICIENCY):
    """Expected photon count for a star of apparent magnitude `mag`."""
    area_m2 = np.pi * (D_m / 2.0) ** 2
    return ZERO_MAG_PHOTON_RATE * area_m2 * relative_flux(mag) * exposure_s * qe


def _airy_kernel_sum(half_window, D_m, wavelength_m, pixel_scale, oversample):
    """Total flux (raw airy_value units) captured within +/- half_window
    pixels, via the SAME fine-grid block-average used per star -- so the
    normalisation is resolved the same way the rendering is, not aliased."""
    n = 2 * half_window + 1
    coords = (np.arange(n * oversample) + 0.5) / oversample - half_window - 0.5
    X, Y = np.meshgrid(coords, coords)
    fine = airy_value(X, Y, D_m, wavelength_m, pixel_scale)
    return fine.reshape(n, oversample, n, oversample).mean(axis=(1, 3)).sum()


def render_image(cam, D_m, wavelength_m, u, v, n_photons, rng, oversample=8):
    """Sum every star's flux onto the sensor, then draw ONE Poisson
    realization of the whole image ("photon realization").

    u, v, n_photons: (N,) arrays -- already-projected pixel positions and
    each star's expected photon count.

    Two regimes, chosen by how many pixels fit across the diffraction
    limit (first_null_px), NOT hand-picked per config:

      * first_null_px >= 0.5 -- the PSF is resolved by the pixel grid.
        Render the real Airy kernel with Module 8's fine-grid, block-
        averaged integration (the SAME technique as star_centroid.py).

      * first_null_px < 0.5 -- deeply undersampled (Config B's regime).
        The Airy pattern oscillates many times within one pixel; trying to
        "resolve" that on a pixel grid this coarse is neither meaningful
        nor affordable, and it isn't what a real sensor does either: all
        the light from a source this much smaller than a pixel lands in
        that pixel. Point-source limit: split each star's flux by
        bilinear weight onto its 4 nearest pixels -- exact in this limit,
        and it sidesteps trying to numerically integrate a function
        oscillating faster than any affordable grid can sample.
    """
    pixel_scale = np.deg2rad(cam.angular_scale_x / 3600.0)   # rad/pixel
    first_null_px = 1.22 * wavelength_m / D_m / pixel_scale

    expected = np.zeros((cam.Ny, cam.Nx))
    n_rendered, flux_captured = 0, []

    if first_null_px < 0.5:
        half_window = 1
        for ui, vi, ni in zip(u, v, n_photons):
            x0, y0 = int(np.floor(ui)), int(np.floor(vi))
            fx, fy = ui - x0, vi - y0
            for dx, dy, w in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                              (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]:
                xx, yy = x0 + dx, y0 + dy
                if 0 <= xx < cam.Nx and 0 <= yy < cam.Ny:
                    expected[yy, xx] += ni * w
            n_rendered += 1
            flux_captured.append(1.0)          # exact, in this point-source limit
    else:
        half_window = int(np.clip(np.ceil(6 * first_null_px), 2, 40))
        hw_norm = max(half_window, 60)
        norm = _airy_kernel_sum(hw_norm, D_m, wavelength_m, pixel_scale, oversample)

        for ui, vi, ni in zip(u, v, n_photons):
            x0, x1 = int(np.round(ui)) - half_window, int(np.round(ui)) + half_window + 1
            y0, y1 = int(np.round(vi)) - half_window, int(np.round(vi)) + half_window + 1
            xs0, xs1 = max(x0, 0), min(x1, cam.Nx)
            ys0, ys1 = max(y0, 0), min(y1, cam.Ny)
            if xs0 >= xs1 or ys0 >= ys1:
                continue                                   # PSF entirely off-frame

            xs_fine = xs0 - 0.5 + (np.arange((xs1 - xs0) * oversample) + 0.5) / oversample
            ys_fine = ys0 - 0.5 + (np.arange((ys1 - ys0) * oversample) + 0.5) / oversample
            X, Y = np.meshgrid(xs_fine, ys_fine)
            fine = airy_value(X - ui, Y - vi, D_m, wavelength_m, pixel_scale)
            kernel = fine.reshape(ys1 - ys0, oversample, xs1 - xs0, oversample).mean(axis=(1, 3))

            captured = kernel.sum() / norm
            expected[ys0:ys1, xs0:xs1] += ni * kernel / norm
            n_rendered += 1
            flux_captured.append(captured)

    realized = rng.poisson(expected)
    return expected, realized, n_rendered, np.array(flux_captured), half_window


def run_config(name, cam, D_m, wavelength_m, exposure_s, catalog, jd, lat_deg, lon_deg,
               point_alt, point_az, rng, mag_render_limit=None):
    ra, dec, mag = catalog
    n_total = len(ra)

    # 1: celestial direction
    s_c = star_celestial(ra, dec)
    # 2: -> observer frame (C -> E -> L)
    s_e = star_earth_fixed(s_c, jd)
    s_l = star_local(s_e, lat_deg, lon_deg)
    alt, az = altaz_from_local(s_l)

    # 3: reject below horizon
    above = alt > 0.0
    n_above = int(above.sum())

    # 4: -> camera frame (L -> B)
    s_b = s_l[above] @ R_BL(point_alt, point_az).T

    # 5,6,7: behind camera / project / outside FOV+sensor  (Module 7)
    uv, valid, reason = project_points(s_b, np.eye(3), np.zeros(3), cam.K,
                                       image_size=(cam.Nx, cam.Ny))
    n_infront = int(np.sum(reason != "behind_camera"))
    n_infov = int(valid.sum())

    mag_v = mag[above][valid]
    u, v = uv[valid, 0], uv[valid, 1]

    # 8: relative brightness -> expected photon count
    n_photons = expected_photons(mag_v, D_m, exposure_s)
    if mag_render_limit is not None:
        keep = mag_v <= mag_render_limit
    else:
        keep = np.ones_like(mag_v, dtype=bool)

    # 9, 10: PSF + sensor sampling + photon realization
    expected_img, realized_img, n_rendered, captured, half_window = render_image(
        cam, D_m, wavelength_m, u[keep], v[keep], n_photons[keep], rng)

    print(f"--- {name} ---")
    print(f"  catalog                    : {n_total:>8,} stars")
    print(f"  3. above horizon           : {n_above:>8,}")
    print(f"  5. in front of camera      : {n_infront:>8,}")
    print(f"  7. inside FOV + sensor     : {n_infov:>8,}")
    print(f"  rendered (mag<={mag_render_limit if mag_render_limit is not None else 'inf'}) "
          f": {n_rendered:>8,}")
    if n_rendered:
        print(f"  brightest rendered star    : {mag_v[keep].min():.2f} mag, "
              f"{n_photons[keep].max():,.0f} expected photons")
        if half_window == 1 and np.allclose(captured, 1.0):
            print("  regime                     : point-source (first null < 0.5 px) "
                  "-- bilinear split onto 4 nearest pixels, exact in this limit")
        else:
            print(f"  window flux captured       : {captured.mean()*100:.2f}% "
                  f"(half-window={half_window} px; rest is outer Airy rings, "
                  f"truncated -- same effect star_centroid.py noted)")
    print(f"  peak expected counts/pixel  : {expected_img.max():,.1f}")
    print(f"  total photons realized      : {realized_img.sum():,.0f}")
    print()

    s_l_final = s_l[above][valid]         # true local-frame direction, RENDERED subset only
    return dict(alt=alt, az=az, above=above, valid=valid,
               u=u[keep], v=v[keep], mag=mag_v[keep], n_photons=n_photons[keep],
               s_l=s_l_final[keep],
               expected=expected_img, realized=realized_img, cam=cam,
               point_alt=point_alt, point_az=point_az)


def main():
    LAT_DEG, LON_DEG = 40.0, -74.0
    YEAR, MONTH, DAY, UT_HOUR = 2026, 9, 14, 22.0
    jd = julian_date(YEAR, MONTH, DAY, hour=UT_HOUR)

    print("=" * 90)
    print("SYNTHETIC CELESTIAL CAMERA -- Modules 3, 7, 8, 9 combined")
    print("=" * 90)
    print(f"observer: lat={LAT_DEG}, lon={LON_DEG}, {YEAR}-{MONTH:02d}-{DAY:02d} "
          f"{UT_HOUR:.1f}h UT")

    # point the camera at Vega's actual sky position right now (frames.py,
    # already validated) -- both configs below aim at the same real target
    vega_ra, vega_dec, vega_mag = 18.615 * 15.0, 38.784, 0.03
    s_c = star_celestial(vega_ra, vega_dec)
    s_e = star_earth_fixed(s_c, jd)
    s_l = star_local(s_e, LAT_DEG, LON_DEG)
    point_alt, point_az = altaz_from_local(s_l)
    print(f"pointing: Vega's current (alt, az) = ({point_alt:.3f}, {point_az:.3f}) deg\n")

    rng = np.random.default_rng(7)

    # build the field catalog once: real all-sky synthetic stars down to the
    # catalog's faint limit, plus a few named bright stars for recognisability
    n_total = 60_000
    ra, dec, mag = generate_catalog(n_total, plane_biased=True, rng=rng)
    for name, ra_h, dec_d, m in NAMED_STARS:
        ra = np.append(ra, ra_h * 15.0)
        dec = np.append(dec, dec_d)
        mag = np.append(mag, m)
    catalog = (ra, dec, mag)

    # ---- Config A: narrow "telescope" (well-sampled PSF), pointed at Vega -
    camA = Camera(f=1000.0, Ws=0.603, Hs=0.603, Nx=201, Ny=201, name="telescope")
    resA = run_config("Config A: 50mm telescope, narrow FOV", camA,
                      D_m=0.050, wavelength_m=550e-9, exposure_s=0.05,
                      catalog=catalog, jd=jd, lat_deg=LAT_DEG, lon_deg=LON_DEG,
                      point_alt=point_alt, point_az=point_az, rng=rng)

    # ---- Config B: wide-field camera lens (undersampled), same target -----
    camB = Camera(f=50.0, Ws=36.0, Hs=24.0, Nx=960, Ny=640, name="wide-field")
    resB = run_config("Config B: 50mm f/2 camera lens, wide FOV", camB,
                      D_m=0.025, wavelength_m=550e-9, exposure_s=2.0,
                      catalog=catalog, jd=jd, lat_deg=LAT_DEG, lon_deg=LON_DEG,
                      point_alt=point_alt, point_az=point_az, rng=rng,
                      mag_render_limit=9.0)

    for name, res in [("A", resA), ("B", resB)]:
        cam = res["cam"]
        vega_mask = (np.abs(res["mag"] - vega_mag) < 1e-9)
        if vega_mask.any():
            uv_v = (res["u"][vega_mask][0], res["v"][vega_mask][0])
            print(f"Config {name}: Vega lands at (u,v)=({uv_v[0]:.3f},{uv_v[1]:.3f}), "
                  f"principal point=({cam.cx:.3f},{cam.cy:.3f}) "
                  f"-- diff {np.hypot(uv_v[0]-cam.cx, uv_v[1]-cam.cy):.2e} px "
                  f"(pointed straight at it, so it should be dead centre)")
    print()
    print(f"Config A FOV = {np.degrees(camA.fov_x)*3600:.1f} arcsec "
          f"({np.degrees(camA.fov_x)*60:.2f} arcmin); "
          f"first null = {1.22*550e-9/0.050/np.deg2rad(camA.angular_scale_x/3600):.2f} px "
          f"-> sharp, well-sampled, essentially one star.")
    print(f"Config B FOV = {np.degrees(camB.fov_x):.1f} x {np.degrees(camB.fov_y):.1f} deg; "
          f"first null = {1.22*550e-9/0.025/np.deg2rad(camB.angular_scale_x/3600):.3f} px "
          f"-> far under 1 pixel, every star is a point -- a rich field, no")
    print("resolved PSF. Same code, same physics, two regimes -- exactly the")
    print("tension telescope_trade_study.py and star_density_vs_fov.py already")
    print("quantified: aperture and FOV trade against each other, they don't")
    print("both max out for free.")
    print("=" * 90)

    _plot(resA, resB)
    return resA, resB


def _plot(resA, resB):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    im0 = axes[0].imshow(resA["realized"], cmap="inferno", origin="upper")
    axes[0].set_title(f"Config A: telescope on Vega\n"
                      f"({resA['cam'].Nx}x{resA['cam'].Ny} px, "
                      f"FOV={np.degrees(resA['cam'].fov_x)*3600:.0f} arcsec) "
                      f"-- real rendered pixels")
    fig.colorbar(im0, ax=axes[0], shrink=0.7, label="photons/pixel")

    # Config B's point sources are 1 real pixel wide -- correct, but
    # invisible at this display size/DPI. Show star POSITIONS (scatter,
    # size/colour by magnitude) rather than the raw image so the rich field
    # is actually visible; the printed diagnostics above are the real numbers.
    camB = resB["cam"]
    ax = axes[1]
    ax.set_facecolor("black")
    order = np.argsort(-resB["mag"])           # draw brightest last (on top)
    m = resB["mag"][order]
    size = np.clip(60.0 * 10 ** (-0.2 * m), 0.6, 80.0)
    ax.scatter(resB["u"][order], resB["v"][order], s=size, c="white",
              edgecolors="none")
    ax.set_xlim(0, camB.Nx); ax.set_ylim(camB.Ny, 0)
    ax.set_aspect("equal")
    ax.set_title(f"Config B: wide-field lens on Vega's field\n"
                 f"({camB.Nx}x{camB.Ny} px, FOV={np.degrees(camB.fov_x):.0f}x"
                 f"{np.degrees(camB.fov_y):.0f} deg) -- star positions, "
                 f"marker size ~ brightness\n(real PSFs are sub-pixel here; "
                 f"see printed peak counts/pixel for the actual numbers)",
                 fontsize=9.5)

    fig.suptitle("Synthetic celestial camera: the whole chain, two instruments,\n"
                 "the same real sky", fontsize=12)
    fig.tight_layout()
    fig.savefig("synthetic_celestial_camera.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
