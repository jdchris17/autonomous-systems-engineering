"""
Experiment C simulation change: celestial positions FIRST -> camera projection -> pixel rendering.

Module 12's simulator drew stars at random pixel positions with no celestial identity. Here the star
list comes from the catalogue (RA/Dec unit vectors) projected through the true camera (R_CI_true, f, c),
then rendered with the same Module 11/12 machinery: pixel-integrated Gaussian PSF (FWHM 3 px), sky
background, Poisson + read noise, hot pixels and cosmic-ray streaks. Optionally uncatalogued faint
"field" stars (real sources that are NOT in the catalogue - the solver must treat them as extras).
Flux scale: flux = 1000 e- * 10^(-0.4 (m - 8)), so catalogue stars (m 3-8) have 1e3-1e5 e-.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-12"))          # data12 adds module-11 itself

from data12 import FWHM_TO_SIGMA, pixel_integrated_gaussian  # noqa: E402

import sky13 as s  # noqa: E402

ZP_MAG, ZP_FLUX = 8.0, 1000.0


def mag_to_flux(m):
    return ZP_FLUX * 10 ** (-0.4 * (np.asarray(m) - ZP_MAG))


def render(cat, R, seed, f=s.F_TRUE, fwhm=3.0, background=100.0, read_noise=5.0, n_field=40,
           field_mag=(8.0, 9.5), n_hot=(10, 20), n_streaks=(2, 6)):
    rng = np.random.default_rng(seed)
    size = s.SIZE
    u, v, front = s.project(R, cat["vec"], f)
    vis = np.nonzero(s.in_frame(u, v, front, margin=-8))[0]            # include stars just off the edge
    xs, ys, fl, cid = list(u[vis]), list(v[vis]), list(mag_to_flux(cat["mag"][vis])), list(vis)
    fm = rng.uniform(*field_mag, n_field)
    xs += list(rng.uniform(-0.5, size - 0.5, n_field))
    ys += list(rng.uniform(-0.5, size - 0.5, n_field))
    fl += list(mag_to_flux(fm))
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
    inside = s.in_frame(xs, ys, np.ones(len(xs), bool))
    truth = {"x": xs[inside], "y": ys[inside], "flux": fl[inside], "cat_idx": cid[inside]}
    return image, truth
