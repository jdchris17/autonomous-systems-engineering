"""
Module 13 core: synthetic local star catalogue, pinhole camera, Wahba / weighted-SVD attitude solution.

Conventions
  inertial frame I : unit vector s = (cos d cos a, cos d sin a, sin d)  for RA a, Dec d
  camera frame C   : +z = boresight, +x = increasing column u, +y = increasing row v
  R_CI             : rotates inertial vectors into the camera frame,  c = R_CI s
  pixels           : ZERO-BASED pixel-centre convention (pixel i spans [i - 0.5, i + 0.5]); the image
                     centre of a 512 x 512 sensor is (255.5, 255.5). This is the Module 11/12 convention.
  projection       : u = cx + f c_x / c_z,   v = cy + f c_y / c_z      (pinhole = gnomonic / FITS TAN)
Camera: 512 x 512, 30 arcsec / pixel at the centre -> f = 206264.8 / 30 = 6875.5 px, field 4.27 deg
square (half-diagonal 3.02 deg).
"""

import numpy as np

ARCSEC = np.pi / 180 / 3600
SIZE = 512
PIXSCALE = 30.0                                  # arcsec / px
F_TRUE = 1 / (PIXSCALE * ARCSEC)                 # focal length in pixels
CX = CY = (SIZE - 1) / 2
HALF_DIAG = np.degrees(np.arctan(np.hypot(CX + 0.5, CY + 0.5) / F_TRUE))
CENTER = (150.0, 20.0)                           # catalogue patch centre (RA, Dec) deg
PATCH_RADIUS = 7.0                               # deg


def radec_to_vec(ra, dec):
    a, d = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], -1)


def vec_to_radec(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360, np.degrees(np.arcsin(v[..., 2]))


def make_catalog(seed=13, n=180, center=CENTER, radius=PATCH_RADIUS, mag_range=(3.0, 8.0)):
    """n stars uniform on the spherical cap; magnitudes with counts rising ~ 10^(0.35 m)."""
    rng = np.random.default_rng(seed)
    cos_t = rng.uniform(np.cos(np.radians(radius)), 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    sin_t = np.sqrt(1 - cos_t**2)
    local = np.column_stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])   # around +z
    R = attitude(center[0], center[1], 0.0).T                                     # camera->inertial
    vec = local @ R.T
    ra, dec = vec_to_radec(vec)
    k = 0.35 * np.log(10)
    u = rng.uniform(0, 1, n)
    m0, m1 = mag_range
    mag = np.log(np.exp(k * m0) + u * (np.exp(k * m1) - np.exp(k * m0))) / k
    order = np.argsort(mag)
    return {"id": np.arange(1, n + 1), "ra": ra[order], "dec": dec[order], "vec": vec[order], "mag": mag[order]}


def attitude(ra, dec, roll_deg):
    """R_CI for a boresight at (ra, dec) with the camera +x axis rotated by `roll` from local east."""
    z = radec_to_vec(ra, dec)
    east = np.array([-np.sin(np.radians(ra)), np.cos(np.radians(ra)), 0.0])
    north = np.cross(z, east)
    r = np.radians(roll_deg)
    x = np.cos(r) * east + np.sin(r) * north
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def project(R, s, f=F_TRUE, cx=CX, cy=CY):
    c = s @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cx + f * c[:, 0] / c[:, 2]
        v = cy + f * c[:, 1] / c[:, 2]
    return u, v, c[:, 2] > 0


def in_frame(u, v, front, margin=0.0):
    return front & (u > -0.5 + margin) & (u < SIZE - 0.5 - margin) & (v > -0.5 + margin) & (v < SIZE - 0.5 - margin)


def pixel_to_vec(u, v, f=F_TRUE, cx=CX, cy=CY):
    b = np.column_stack([(np.asarray(u) - cx), (np.asarray(v) - cy), np.full(np.shape(u), f)])
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def wahba_svd(b, r, w=None):
    """Weighted SVD solution of Wahba's problem: R minimising sum w_i |b_i - R r_i|^2 with det R = +1.
    b: camera-frame unit vectors, r: inertial unit vectors (N x 3). Returns R_CI."""
    w = np.ones(len(b)) if w is None else np.asarray(w, float)
    B = (b * w[:, None]).T @ r
    U, _, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def rotation_angle(R_est, R_true):
    Re = R_est @ R_true.T
    return float(np.arccos(np.clip((np.trace(Re) - 1) / 2, -1.0, 1.0)))


def error_components(R_est, R_true):
    """Small-angle error vector in the camera frame -> (cross-boresight, roll) in radians."""
    Re = R_est @ R_true.T
    phi = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    return float(np.hypot(phi[0], phi[1])), float(abs(phi[2]))


def angular_sigma(u, v, sigma_px, f=F_TRUE, cx=CX, cy=CY):
    """1-sigma angular uncertainty (rad) of a direction measured with an isotropic pixel error sigma_px.
    For a pinhole camera d(angle)/d(pixel) is cos^2(alpha)/f radially and cos(alpha)/f tangentially,
    alpha = off-axis angle; use the geometric mean cos^1.5(alpha)/f (within 0.4 % of 1/f in this field)."""
    cosa = f / np.sqrt((np.asarray(u) - cx) ** 2 + (np.asarray(v) - cy) ** 2 + f**2)
    return np.asarray(sigma_px) * cosa**1.5 / f


def random_pointing(rng, max_offset=2.0, center=CENTER, roll=None):
    """True boresight within max_offset deg of the patch centre; roll uniform unless given."""
    R0 = attitude(center[0], center[1], 0.0)
    rho, phi = np.radians(max_offset) * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    local = np.array([np.sin(rho) * np.cos(phi), np.sin(rho) * np.sin(phi), np.cos(rho)])
    ra, dec = vec_to_radec(R0.T @ local)
    roll = rng.uniform(0, 360) if roll is None else roll
    return attitude(float(ra), float(dec), roll), (float(ra), float(dec), roll)


def observe(cat, R, rng, sigma_px=0.1, f=F_TRUE, margin=0.0):
    """Visible catalogue stars projected with the TRUE camera, plus Gaussian centroid noise.
    sigma_px may be a scalar or per-star array (aligned with the catalogue)."""
    u, v, front = project(R, cat["vec"], f)
    vis = np.nonzero(in_frame(u, v, front, margin))[0]
    sig = np.broadcast_to(np.asarray(sigma_px, float), (len(cat["vec"]),))[vis] if np.ndim(sigma_px) else np.full(len(vis), float(sigma_px))
    uo = u[vis] + rng.normal(0, 1, len(vis)) * sig
    vo = v[vis] + rng.normal(0, 1, len(vis)) * sig
    return {"u": uo, "v": vo, "cat_idx": vis, "sigma_px": sig, "u_true": u[vis], "v_true": v[vis]}
