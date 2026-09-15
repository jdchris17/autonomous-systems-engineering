"""module-10 -- pipeline.py  (Stages I-VII of the final-project simulator)

Stage I  -- Generate the Celestial Scene
    (alpha, delta)_i  ->  C_shat_i        Module 9 frames.star_celestial
    C_shat -> E_shat -> L_shat            Module 9 frames.star_earth_fixed / star_local
    reject stars below the horizon

Stage II -- Observer Frame -> Camera Frame
    B_shat = R_BL  L_shat                 R_BL from state.attitude (Module 3 rotation math)
    reject stars behind the camera        s_B,z <= 0

Stage III -- Geometric Image Projection
    x_n = s_x / s_z,  y_n = s_y / s_z     normalized image coordinates
    [u, v, 1]^T ~ K [x_n, y_n, 1]^T        Module 7's intrinsic matrix
    (u, v) is the IDEAL geometric star position -- no diffraction, no
    detector. That separation is deliberate: later stages add distortion
    and a PSF on TOP of this, they never touch it.

Stage IV -- Optical Distortion
    r^2 = x_n^2 + y_n^2
    x_d = x_n (1 + k1 r^2 + k2 r^4),  y_d = y_n (1 + k1 r^2 + k2 r^4)
    (u, v)_distorted = K [x_d, y_d, 1]^T   state.distortion's k1, k2 (Section
    3 carried these; Stage III deliberately left them unused -- this is
    where they finally apply)

Stage V -- Diffraction / PSF
    theta_Airy = 1.22 lambda / D           Module 8's Airy formula
    star -> PSF(u, v), not star -> pixel point: each surviving star's
    DISTORTED (u, v) (Stage IV's answer -- the geometric ray's real landing
    spot, whatever produced it) sets the PSF's center; state.optics'
    (D_mm, wavelength_optical_nm) and state.intrinsics.fx/fy (pixel scale =
    1/fx, small-angle) set its shape. Module 8's star_centroid.py already
    solved "how do you sample an Airy disk onto a finite pixel grid without
    aliasing" (fine sub-pixel grid, block-averaged) -- reused directly, not
    re-derived.

Stage VI -- Brightness and Photon Distribution
    F_i proportional to 10^(-0.4 m_i)      Carroll & Ostlie magnitude (Module 9's
                                           magnitude_flux.relative_flux)
    N_i                                    expected detected photons for star i,
                                           a chosen reference (m=0) photon count
                                           scaled by F_i -- same zero-point idea
                                           module-9/synthetic_celestial_camera.py
                                           used, given directly here (see
                                           `expected_photons` below) rather than
                                           imported, since importing that file
                                           would re-import its own `frames.py`
                                           by the ambiguous bare name -- the
                                           exact collision _pathutil.py exists
                                           to avoid (see README).
    PSF_ij normalized so sum_j = 1         Stage V's Airy kernel, called again
                                           here with real photon-count weights
                                           instead of Stage V's relative-flux
                                           placeholders (stage5_psf's new
                                           `weights` argument)
    lambda_ij = N_i PSF_ij                 expected photon COUNT per pixel --
                                           Carroll & Ostlie + Module 8 PSF +
                                           Module 9 photon statistics, joined

Stage VII -- Sensor Measurement
    N_ij ~ Poisson(lambda_ij)              one photon-count realization per
                                           pixel (Module 9's "photon
                                           realization" technique)
    N_e = QE N_gamma                       optional one-line "more realistic"
                                           step the exercise offers; nothing
                                           past it -- no read noise, dark
                                           current, ADC modeling, nonlinearity,
                                           thermal modeling, or rolling
                                           shutter, exactly as instructed.

Every stage takes and returns a plain dict of aligned numpy arrays (one
row per surviving star) -- easy to inspect, easy to print, easy to feed
into the next stage or into a plot.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(__file__)
_MODULE9 = os.path.normpath(os.path.join(_HERE, "..", "module-9"))
if _MODULE9 not in sys.path:
    sys.path.insert(0, _MODULE9)
_MODULE8 = os.path.normpath(os.path.join(_HERE, "..", "module-8"))
if _MODULE8 not in sys.path:
    sys.path.insert(0, _MODULE8)

# module-3 also has its own, unrelated frames.py; load module-9's by
# explicit file path (see _pathutil.py) rather than a bare `from frames
# import ...`, so this never depends on sys.path insertion order.
from _pathutil import load_module
_frames9 = load_module("module9_frames", os.path.join(_MODULE9, "frames.py"))
star_celestial = _frames9.star_celestial
star_earth_fixed = _frames9.star_earth_fixed
star_local = _frames9.star_local
altaz_from_local = _frames9.altaz_from_local

from magnitude_flux import relative_flux              # noqa: E402  (Module 9)
from star_centroid import airy_value, first_null_px    # noqa: E402  (Module 8)


def stage1_celestial_scene(state):
    """(RA, Dec) for every catalog star -> C -> E -> L; reject below horizon."""
    ra, dec, mag, wl = state.scene.arrays()
    n_total = len(ra)

    s_c = star_celestial(ra, dec)
    s_e = star_earth_fixed(s_c, state.observer.jd)
    s_l = star_local(s_e, state.observer.lat_deg, state.observer.lon_deg)
    alt, az = altaz_from_local(s_l)

    above = alt > 0.0
    return dict(
        ra=ra[above], dec=dec[above], mag=mag[above], wavelength_nm=wl[above],
        s_l=s_l[above], alt=alt[above], az=az[above],
        n_total=n_total, n_above_horizon=int(above.sum()),
    )


def stage2_camera_frame(state, s1):
    """L -> B via R_BL (state.attitude); reject stars behind the camera."""
    R_BL = state.attitude.R_BL
    s_b = s1["s_l"] @ R_BL.T                     # R_BL @ v for every row v

    in_front = s_b[:, 2] > 0.0
    out = {k: (v[in_front] if isinstance(v, np.ndarray) and v.ndim >= 1
              and len(v) == len(in_front) else v)
          for k, v in s1.items()}
    out["s_b"] = s_b[in_front]
    out["n_in_front"] = int(in_front.sum())
    return out


def stage3_projection(state, s2):
    """Normalized coordinates -> K -> ideal (u, v). No FOV/sensor cull yet
    (that belongs to whichever stage explicitly asks for it) and no
    distortion (state.distortion is carried but deliberately unused here)."""
    K = state.intrinsics.K
    s_b = s2["s_b"]
    x_n = s_b[:, 0] / s_b[:, 2]
    y_n = s_b[:, 1] / s_b[:, 2]

    uv1 = np.stack([x_n, y_n, np.ones_like(x_n)], axis=-1) @ K.T
    out = dict(s2)
    out["x_n"], out["y_n"] = x_n, y_n
    out["u"], out["v"] = uv1[:, 0], uv1[:, 1]
    return out


def run_stages_1_to_3(state):
    """Convenience: run Stage I, II, III in sequence, return each result."""
    s1 = stage1_celestial_scene(state)
    s2 = stage2_camera_frame(state, s1)
    s3 = stage3_projection(state, s2)
    return s1, s2, s3


def stage4_distortion(state, s3):
    """Radial distortion, applied in normalized coordinates (before K, same
    as Stage III used to project): r^2 = x_n^2+y_n^2, then scale (x_n,y_n)
    by (1 + k1 r^2 + k2 r^4). k1=k2=0 (state.distortion's default) makes
    this an identity -- (u,v)_distorted == (u,v)_ideal exactly, which is
    the cheapest correctness check this stage has."""
    k1, k2 = state.distortion.k1, state.distortion.k2
    x_n, y_n = s3["x_n"], s3["y_n"]
    r2 = x_n ** 2 + y_n ** 2
    factor = 1.0 + k1 * r2 + k2 * r2 ** 2
    x_d, y_d = x_n * factor, y_n * factor

    K = state.intrinsics.K
    uvd1 = np.stack([x_d, y_d, np.ones_like(x_d)], axis=-1) @ K.T
    out = dict(s3)
    out["x_d"], out["y_d"] = x_d, y_d
    out["u_d"], out["v_d"] = uvd1[:, 0], uvd1[:, 1]
    out["du"] = out["u_d"] - s3["u"]
    out["dv"] = out["v_d"] - s3["v"]
    out["distortion_px"] = np.hypot(out["du"], out["dv"])
    return out


def _airy_kernel_norm(half_window, D_m, wavelength_m, pixel_scale, oversample):
    """Total flux (raw airy_value units) within +/- half_window pixels, via
    the SAME fine-grid block-average used to render each star -- so the
    normalization is resolved the same way the rendering is, not aliased
    (same technique module-9's synthetic_celestial_camera.py validated)."""
    n = 2 * half_window + 1
    coords = (np.arange(n * oversample) + 0.5) / oversample - half_window - 0.5
    X, Y = np.meshgrid(coords, coords)
    fine = airy_value(X, Y, D_m, wavelength_m, pixel_scale)
    return fine.reshape(n, oversample, n, oversample).mean(axis=(1, 3)).sum()


def stage5_psf(state, s4, oversample=8, weights=None):
    """star -> PSF(u, v): sum every surviving star's Airy kernel, centered
    on its Stage IV (distorted) position and weighted by `weights`, onto
    the sensor grid. `weights` defaults to relative brightness alone
    (Stage V's original, uncalibrated placeholder -- "how bright is this
    star relative to the others", no photon budget yet); Stage VI passes
    real expected photon counts instead, through this SAME kernel-building
    code, to get lambda_ij = N_i * PSF_ij rather than duplicating the PSF
    machinery a second time. No detector/noise model yet either way
    (Sensor's qe/exposure_s stay unused here -- same "define it, don't use
    it early" pattern Stage III used for state.distortion; Stage VI/VII
    are where they finally apply).

    Regime picked by first_null_px vs one pixel, not hand-picked per
    config (module-8/module-9's already-validated split):
      * first_null_px <  0.5 -- point-source limit: bilinear-split each
        star's weight onto its 4 nearest pixels (exact in this limit).
      * first_null_px >= 0.5 -- render the real Airy kernel via Module 8's
        fine-grid, block-averaged integration.
    """
    fx, fy = state.intrinsics.fx, state.intrinsics.fy
    pixel_scale = 2.0 / (fx + fy)          # rad/px, small-angle (fx ~= fy here)
    D_m = state.optics.D_mm / 1000.0
    wavelength_m = state.optics.wavelength_optical_nm * 1e-9
    fn_px = first_null_px(D_m, wavelength_m, pixel_scale)

    Nx, Ny = state.intrinsics.Nx, state.intrinsics.Ny
    u, v = s4["u_d"], s4["v_d"]
    weight = relative_flux(s4["mag"]) if weights is None else np.asarray(weights, dtype=float)
    expected = np.zeros((Ny, Nx))
    n_rendered, flux_captured = 0, []

    if fn_px < 0.5:
        half_window = 1
        in_range = (u > -1) & (u < Nx + 1) & (v > -1) & (v < Ny + 1)
        for ui, vi, wi in zip(u[in_range], v[in_range], weight[in_range]):
            x0, y0 = int(np.floor(ui)), int(np.floor(vi))
            fxp, fyp = ui - x0, vi - y0
            for dx, dy, w in [(0, 0, (1 - fxp) * (1 - fyp)), (1, 0, fxp * (1 - fyp)),
                              (0, 1, (1 - fxp) * fyp), (1, 1, fxp * fyp)]:
                xx, yy = x0 + dx, y0 + dy
                if 0 <= xx < Nx and 0 <= yy < Ny:
                    expected[yy, xx] += wi * w
            n_rendered += 1
            flux_captured.append(1.0)
        regime = ("point-source (first_null < 0.5 px) -- bilinear split onto "
                  "4 nearest pixels, exact in this limit")
    else:
        half_window = int(np.clip(np.ceil(6 * fn_px), 2, 40))
        in_range = (u > -half_window) & (u < Nx + half_window) & \
                  (v > -half_window) & (v < Ny + half_window)
        norm = _airy_kernel_norm(max(half_window, 60), D_m, wavelength_m,
                                 pixel_scale, oversample)
        for ui, vi, wi in zip(u[in_range], v[in_range], weight[in_range]):
            x0, x1 = int(np.round(ui)) - half_window, int(np.round(ui)) + half_window + 1
            y0, y1 = int(np.round(vi)) - half_window, int(np.round(vi)) + half_window + 1
            xs0, xs1 = max(x0, 0), min(x1, Nx)
            ys0, ys1 = max(y0, 0), min(y1, Ny)
            if xs0 >= xs1 or ys0 >= ys1:
                continue
            xs_fine = xs0 - 0.5 + (np.arange((xs1 - xs0) * oversample) + 0.5) / oversample
            ys_fine = ys0 - 0.5 + (np.arange((ys1 - ys0) * oversample) + 0.5) / oversample
            X, Y = np.meshgrid(xs_fine, ys_fine)
            fine = airy_value(X - ui, Y - vi, D_m, wavelength_m, pixel_scale)
            kernel = fine.reshape(ys1 - ys0, oversample, xs1 - xs0, oversample).mean(axis=(1, 3))
            expected[ys0:ys1, xs0:xs1] += wi * kernel / norm
            n_rendered += 1
            flux_captured.append(kernel.sum() / norm)
        regime = (f"resolved Airy PSF (first_null = {fn_px:.2f} px) -- "
                  f"Module 8 fine-grid, block-averaged kernel")

    out = dict(s4)
    out["expected_image"] = expected
    out["n_rendered"] = n_rendered
    out["flux_captured"] = np.array(flux_captured)
    out["half_window"] = half_window
    out["first_null_px"] = fn_px
    out["pixel_scale_rad"] = pixel_scale
    out["regime"] = regime
    return out


def run_stages_1_to_5(state, oversample=8):
    """Convenience: run Stage I through Stage V in sequence."""
    s1 = stage1_celestial_scene(state)
    s2 = stage2_camera_frame(state, s1)
    s3 = stage3_projection(state, s2)
    s4 = stage4_distortion(state, s3)
    s5 = stage5_psf(state, s4, oversample=oversample)
    return s1, s2, s3, s4, s5


# m=0 photon rate and the quantum efficiency default this reference count
# assumes -- the SAME order-of-magnitude, admittedly-uncalibrated zero
# point module-9/synthetic_celestial_camera.py used (roughly the textbook
# ~1000 photons/s/cm^2/Angstrom V-band value, integrated over a ~1000
# Angstrom bandpass). Given directly here rather than imported: importing
# that file would execute its own `from frames import ...`, reintroducing
# the exact module-3/module-9 frames.py name collision _pathutil.py exists
# to avoid (see README) -- not worth it for two constants and four lines
# of arithmetic.
ZERO_MAG_PHOTON_RATE = 1.0e10      # photons / s / m^2, m = 0 (order-of-magnitude)


def expected_photons(mag, D_m, exposure_s):
    """Reference (m=0) photon count, scaled by Carroll & Ostlie relative
    flux -- "choose a reference star photon count and scale the others
    appropriately." QE is deliberately NOT applied here (Stage VII's job,
    N_e = QE N_gamma -- these ARE incident photons, not yet detected ones)."""
    area_m2 = np.pi * (D_m / 2.0) ** 2
    return ZERO_MAG_PHOTON_RATE * area_m2 * relative_flux(mag) * exposure_s


def stage6_photon_distribution(state, s4, oversample=8):
    """Carroll & Ostlie magnitude -> N_i (expected_photons above) fed
    through Stage V's own PSF-kernel machinery (stage5_psf's `weights`
    argument), instead of Stage V's relative-brightness placeholder --
    joining magnitude, Module 8's PSF, and photon statistics into one
    model, exactly as the exercise asks, by composition rather than a
    second implementation of the kernel code."""
    D_m = state.optics.D_mm / 1000.0
    n_photons = expected_photons(s4["mag"], D_m, state.sensor.exposure_s)
    out = stage5_psf(state, s4, oversample=oversample, weights=n_photons)
    out["n_photons"] = n_photons          # N_i, one expected photon count per star
    out["lambda_image"] = out["expected_image"]   # lambda_ij = N_i PSF_ij, in real photon units
    return out


def stage7_sensor_measurement(state, s6, rng):
    """N_ij ~ Poisson(lambda_ij) -- one photon-count realization per pixel,
    sampled directly onto the Nx x Ny grid lambda_ij already lives on
    (Module 9's "photon realization" technique). Optionally scales by QE
    for a "slightly more realistic" detected-electron count -- the one
    extra line the exercise offers, and no further: no read noise, dark
    current, ADC modeling, sensor nonlinearity, thermal modeling, or
    rolling-shutter behavior (explicitly out of scope here)."""
    n_photons_image = rng.poisson(s6["lambda_image"])
    n_electrons_image = state.sensor.qe * n_photons_image
    out = dict(s6)
    out["n_photons_image"] = n_photons_image
    out["n_electrons_image"] = n_electrons_image
    return out


def run_stages_1_to_7(state, rng, oversample=8):
    """Convenience: run Stage I through Stage VII in sequence."""
    s1 = stage1_celestial_scene(state)
    s2 = stage2_camera_frame(state, s1)
    s3 = stage3_projection(state, s2)
    s4 = stage4_distortion(state, s3)
    s5 = stage5_psf(state, s4, oversample=oversample)
    s6 = stage6_photon_distribution(state, s4, oversample=oversample)
    s7 = stage7_sensor_measurement(state, s6, rng)
    return s1, s2, s3, s4, s5, s6, s7
