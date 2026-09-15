"""module-9 -- star_catalog.py

A synthetic all-sky star catalog, used by star_density_vs_fov.py.

Two ingredients, deliberately kept independent (a stated simplification, not
a claim of a fully self-consistent 3-D galaxy model):

  MAGNITUDES -- drawn so cumulative whole-sky counts follow the classical
  "Euclidean" slope for a uniform density of stars in space:

      N(<=m) = C * 10 ^ (0.6 m)

  (flux ~ 1/d^2, m ~ -2.5 log10(flux)  ->  d ~ 10^(0.2 m)  ->  volume, and
  hence count at uniform space density, ~ d^3 ~ 10^(0.6 m)). Calibrated so
  N(<=6) ~= 5000 -- roughly the classical "naked-eye whole-sky" star count.

  POSITIONS -- uniform in azimuth about the real North Galactic Pole
  (RA=192.859, Dec=27.128, J2000 -- the one fact about the galactic
  coordinate system this file needs), but with the density enhanced near
  the galactic PLANE (b=0), the same way the real Milky Way's stars are
  concentrated in a disk rather than spread isotropically:

      p(b) ~ cos(b) * (1 + A exp(-|b| / b0))

  the cos(b) is the sphere's own solid-angle weighting (needed so A=0
  reproduces a plain uniform-on-sphere catalog); A, b0 set how much brighter
  the disk is and how wide it is. No galactic LONGITUDE is used anywhere --
  this model is symmetric about the pole axis by construction, which is all
  the density-vs-sky-direction question in star_density_vs_fov.py needs.
"""

import numpy as np

# North Galactic Pole, J2000 (the one real-astronomy constant this file uses)
NGP_RA_DEG = 192.85948
NGP_DEC_DEG = 27.12825

# magnitude model
MAG_SLOPE = 0.6          # log10 N(<=m) per magnitude (Euclidean/uniform-density slope)
MAG_MIN = -1.5
MAG_MAX = 9.0
N_AT_M6 = 5000.0         # calibration: ~5000 stars visible to m<=6 over the whole sky

# galactic-plane density enhancement
PLANE_ENHANCEMENT = 8.0      # relative density AT the plane, above the isotropic floor
PLANE_HALFWIDTH_DEG = 15.0   # angular scale of the enhancement


def radec_to_xyz(ra_deg, dec_deg):
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)], axis=-1)


def xyz_to_radec(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    dec = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    return ra, dec


def galactic_latitude(ra_deg, dec_deg):
    """b: angle above/below the galactic plane. Same 'angle from a pole'
    formula as spherical_astronomy's altitude -- NGP plays the role of the
    zenith, RA - RA_NGP plays the role of hour angle."""
    dec, dec0 = np.deg2rad(dec_deg), np.deg2rad(NGP_DEC_DEG)
    dra = np.deg2rad(ra_deg - NGP_RA_DEG)
    sinb = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(dra)
    return np.degrees(np.arcsin(np.clip(sinb, -1.0, 1.0)))


def _pole_frame():
    """Orthonormal (e1, e2, pole) basis, equatorial Cartesian, pole = NGP."""
    pole = radec_to_xyz(NGP_RA_DEG, NGP_DEC_DEG)
    ref = np.array([0.0, 0.0, 1.0])           # safe: NGP is 63 deg from this
    e1 = np.cross(ref, pole)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(pole, e1)
    return e1, e2, pole


def sample_magnitudes(n, rng):
    """Inverse-CDF sample of m in [MAG_MIN, MAG_MAX] with N(<=m) ~ 10^(0.6 m)."""
    lo, hi = 10 ** (MAG_SLOPE * MAG_MIN), 10 ** (MAG_SLOPE * MAG_MAX)
    u = rng.uniform(lo, hi, size=n)
    return np.log10(u) / MAG_SLOPE


def sample_positions_uniform(n, rng):
    """Plain uniform-on-the-sphere positions (the 'for now' baseline catalog)."""
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    ra = rng.uniform(0.0, 360.0, size=n)
    return ra, dec


def sample_positions_plane_biased(n, rng, A=PLANE_ENHANCEMENT, b0=PLANE_HALFWIDTH_DEG):
    """Positions with p(b) ~ cos(b) (1 + A exp(-|b|/b0)), uniform in azimuth."""
    p_max = 1.0 + A                                   # cos(0)*(1+A), the peak
    b_accepted = np.empty(0)
    while b_accepted.size < n:
        batch = max(4096, 2 * (n - b_accepted.size))
        cand = rng.uniform(-90.0, 90.0, size=batch)
        p = np.cos(np.deg2rad(cand)) * (1.0 + A * np.exp(-np.abs(cand) / b0))
        keep = rng.uniform(0.0, p_max, size=batch) <= p
        b_accepted = np.concatenate([b_accepted, cand[keep]])
    b = np.deg2rad(b_accepted[:n])
    phi = rng.uniform(0.0, 2 * np.pi, size=n)

    e1, e2, pole = _pole_frame()
    theta = np.pi / 2 - b                              # colatitude from the pole
    xyz = (np.sin(theta)[:, None] * np.cos(phi)[:, None] * e1
          + np.sin(theta)[:, None] * np.sin(phi)[:, None] * e2
          + np.cos(theta)[:, None] * pole)
    ra, dec = xyz_to_radec(xyz)
    return ra, dec


def generate_catalog(n_total, plane_biased, rng):
    """Returns (ra_deg, dec_deg, mag) arrays, length n_total."""
    mag = sample_magnitudes(n_total, rng)
    if plane_biased:
        ra, dec = sample_positions_plane_biased(n_total, rng)
    else:
        ra, dec = sample_positions_uniform(n_total, rng)
    return ra, dec, mag


def whole_sky_count(m_lim):
    """N(<=m_lim) predicted by the calibrated magnitude model (for printing)."""
    C = N_AT_M6 / 10 ** (MAG_SLOPE * 6.0)
    return C * 10 ** (MAG_SLOPE * m_lim)
