"""
Synthetic test data ONLY (never used during inference): Module 13 catalogue generator, pointing sampler
and the "celestial positions first -> camera -> render" image simulator (render13.render), with extra
knobs for the capstone stress cases. With default arguments the outputs are identical to Module 13.
"""

import numpy as np

from ._m11 import FWHM_TO_SIGMA, pixel_integrated_gaussian
from .catalog import StarCatalog
from .geometry import CameraModel, attitude_from_pointing, vec_to_radec

CENTER = (150.0, 20.0)
PATCH_RADIUS = 7.0


def make_catalog(seed=13, n=180, center=CENTER, radius=PATCH_RADIUS, mag_range=(3.0, 8.0), name=None):
    rng = np.random.default_rng(seed)
    cos_t = rng.uniform(np.cos(np.radians(radius)), 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    sin_t = np.sqrt(1 - cos_t**2)
    local = np.column_stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])
    R = attitude_from_pointing(center[0], center[1], 0.0).T
    vec = local @ R.T
    ra, dec = vec_to_radec(vec)
    k = 0.35 * np.log(10)
    u = rng.uniform(0, 1, n)
    m0, m1 = mag_range
    mag = np.log(np.exp(k * m0) + u * (np.exp(k * m1) - np.exp(k * m0))) / k
    order = np.argsort(mag)
    return StarCatalog(np.arange(1, n + 1), ra[order], dec[order], mag[order], vec[order], name or f"synthetic-seed{seed}")


def random_pointing(rng, max_offset=2.0, center=CENTER, roll=None):
    R0 = attitude_from_pointing(center[0], center[1], 0.0)
    rho, phi = np.radians(max_offset) * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    local = np.array([np.sin(rho) * np.cos(phi), np.sin(rho) * np.sin(phi), np.cos(rho)])
    ra, dec = vec_to_radec(R0.T @ local)
    roll = rng.uniform(0, 360) if roll is None else roll
    return attitude_from_pointing(float(ra), float(dec), roll), (float(ra), float(dec), roll)


def mag_to_flux(m, zp_mag=8.0, zp_flux=1000.0):
    return zp_flux * 10 ** (-0.4 * (np.asarray(m) - zp_mag))


def render(cat, R, seed, camera=None, fwhm=3.0, background=100.0, read_noise=5.0, n_field=40, field_mag=(8.0, 9.5),
           n_hot=(10, 20), n_streaks=(2, 6), zp_flux=1000.0, f=None):
    """Module 13 render13.render. Returns (image float32, truth{x, y, flux, cat_idx}); cat_idx = -1 for
    uncatalogued field stars. Positions are zero-based pixel centres."""
    camera = camera or CameraModel()
    f = camera.focal_length_px if f is None else f
    rng = np.random.default_rng(seed)
    size = camera.width
    u, v, front = camera.project(R, cat.vec, f)
    vis = np.nonzero(camera.in_frame(u, v, front, margin=-8))[0]
    xs, ys, fl, cid = list(u[vis]), list(v[vis]), list(mag_to_flux(cat.mag[vis], zp_flux=zp_flux)), list(vis)
    fm = rng.uniform(*field_mag, n_field)
    xs += list(rng.uniform(-0.5, size - 0.5, n_field))
    ys += list(rng.uniform(-0.5, size - 0.5, n_field))
    fl += list(mag_to_flux(fm, zp_flux=zp_flux))
    cid += [-1] * n_field
    xs, ys, fl, cid = map(np.array, (xs, ys, fl, cid))
    sigma = fwhm * FWHM_TO_SIGMA
    expected = np.full((size, size), background, float)
    half = int(np.ceil(6 * sigma))
    for xk, yk, fk in zip(xs, ys, fl):
        i0, j0 = int(round(xk)) - half, int(round(yk)) - half
        stamp = pixel_integrated_gaussian((2 * half + 1, 2 * half + 1), xk, yk, sigma, fk, i0, j0)
        ys_ = slice(max(j0, 0), min(j0 + 2 * half + 1, size))
        xs_ = slice(max(i0, 0), min(i0 + 2 * half + 1, size))
        if ys_.stop > ys_.start and xs_.stop > xs_.start:
            expected[ys_, xs_] += stamp[ys_.start - j0:ys_.stop - j0, xs_.start - i0:xs_.stop - i0]
    nh = int(rng.integers(n_hot[0], n_hot[1] + 1))
    hx, hy = rng.integers(0, size, nh), rng.integers(0, size, nh)
    expected[hy, hx] += rng.uniform(500, 5000, nh)
    YY, XX = np.mgrid[0:size, 0:size]
    for _ in range(int(rng.integers(n_streaks[0], n_streaks[1] + 1))):
        cx, cy = rng.uniform(20, size - 20, 2)
        L, th, F = rng.uniform(6, 20), rng.uniform(0, np.pi), rng.uniform(500, 5000)
        j0, j1, i0, i1 = int(cy - 15), int(cy + 16), int(cx - 15), int(cx + 16)
        dx, dy = XX[j0:j1, i0:i1] - cx, YY[j0:j1, i0:i1] - cy
        along = dx * np.cos(th) + dy * np.sin(th)
        across = -dx * np.sin(th) + dy * np.cos(th)
        prof = np.exp(-0.5 * (across / 0.6) ** 2) * (np.abs(along) <= L / 2)
        if prof.sum() > 0:
            expected[j0:j1, i0:i1] += F * prof / prof.sum()
    image = (rng.poisson(expected) + rng.normal(0, read_noise, (size, size))).astype(np.float32)
    inside = camera.in_frame(xs, ys, np.ones(len(xs), bool))
    truth = {"x": xs[inside], "y": ys[inside], "flux": fl[inside], "cat_idx": cid[inside],
             "hot_xy": np.column_stack([hx, hy])}
    return image, truth
