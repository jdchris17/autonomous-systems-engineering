"""
Camera model, coordinate frames and attitude mathematics (from Module 13 sky13.py, parameterised by a
CameraModel instead of module constants; expressions kept identical so results are bit-for-bit equal).

FRAMES AND UNITS
  inertial (I)  ICRS-like celestial frame; unit vector s = (cos d cos a, cos d sin a, sin d), RA a / Dec d
  camera (C)    +z boresight, +x along increasing column u, +y along increasing row v
  attitude      R_CI (3 x 3, dimensionless): c = R_CI s
  pixels        ZERO-BASED pixel-centre convention: pixel (row j, column i) spans [i-0.5, i+0.5] x
                [j-0.5, j+0.5]; the centre of a 512 x 512 sensor is (255.5, 255.5). FITS / Astrometry.net
                use 1-based coordinates: add exactly 1.0 when exporting (validated in Module 13).
  projection    pinhole = gnomonic (FITS TAN):  u = cx + f c_x / c_z,  v = cy + f c_y / c_z  (f in px)
  angles        radians internally; degrees for RA/Dec/roll in results; arcsec for uncertainties
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ARCSEC = np.pi / 180 / 3600


@dataclass(frozen=True)
class CameraModel:
    width: int = 512
    height: int = 512
    focal_length_px: float = 1 / (30.0 * ARCSEC)      # 30 arcsec / px at the centre
    cx: float = 255.5                                   # principal point, zero-based pixel coordinates
    cy: float = 255.5
    psf_fwhm_px: float = 3.0
    focal_length_tolerance: float = 0.03                # permitted fractional focal-length error
    name: str = "module13-synthetic"

    @classmethod
    def from_json(cls, path_or_dict):
        d = path_or_dict if isinstance(path_or_dict, dict) else json.loads(Path(path_or_dict).read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)

    @property
    def pixel_scale_arcsec(self):
        return 1 / (self.focal_length_px * ARCSEC)

    def project(self, R, s, f=None):
        f = self.focal_length_px if f is None else f
        c = s @ R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + f * c[:, 0] / c[:, 2]
            v = self.cy + f * c[:, 1] / c[:, 2]
        return u, v, c[:, 2] > 0

    def in_frame(self, u, v, front, margin=0.0):
        return (front & (u > -0.5 + margin) & (u < self.width - 0.5 - margin)
                & (v > -0.5 + margin) & (v < self.height - 0.5 - margin))

    def pixel_to_vec(self, u, v, f=None):
        f = self.focal_length_px if f is None else f
        b = np.column_stack([(np.asarray(u) - self.cx), (np.asarray(v) - self.cy), np.full(np.shape(u), f)])
        return b / np.linalg.norm(b, axis=1, keepdims=True)

    def angular_sigma(self, u, v, sigma_px, f=None):
        """1-sigma angular uncertainty (rad) for isotropic pixel error sigma_px (geometric mean of the
        radial cos^2 / f and tangential cos / f pinhole derivatives)."""
        f = self.focal_length_px if f is None else f
        cosa = f / np.sqrt((np.asarray(u) - self.cx) ** 2 + (np.asarray(v) - self.cy) ** 2 + f**2)
        return np.asarray(sigma_px) * cosa**1.5 / f

    def half_diagonal_rad(self, f=None):
        f = self.focal_length_px if f is None else f
        return float(np.arctan(np.hypot(self.cx + 0.5, self.cy + 0.5) / f))


def radec_to_vec(ra, dec):
    a, d = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], -1)


def vec_to_radec(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360, np.degrees(np.arcsin(v[..., 2]))


def east_north(ra, dec):
    z = radec_to_vec(ra, dec)
    east = np.array([-np.sin(np.radians(ra)), np.cos(np.radians(ra)), 0.0])
    return east, np.cross(z, east)


def attitude_from_pointing(ra, dec, roll_deg):
    """R_CI for boresight (ra, dec) with camera +x rotated by `roll` from local east towards north."""
    z = radec_to_vec(ra, dec)
    east = np.array([-np.sin(np.radians(ra)), np.cos(np.radians(ra)), 0.0])
    north = np.cross(z, east)
    r = np.radians(roll_deg)
    x = np.cos(r) * east + np.sin(r) * north
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def pointing_from_attitude(R):
    """(boresight RA deg, Dec deg, roll deg) - inverse of attitude_from_pointing."""
    ra, dec = vec_to_radec(R[2])
    east, north = east_north(float(ra), float(dec))
    roll = np.degrees(np.arctan2(R[0] @ north, R[0] @ east)) % 360
    return float(ra), float(dec), float(roll)


def quaternion_from_rotation(R):
    """Unit quaternion (w, x, y, z), scalar first, w >= 0, representing R_CI."""
    m = R
    t = np.trace(m)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        q = [0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S]
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def wahba_svd(b, r, w=None):
    """Weighted SVD solution of Wahba's problem: R minimising sum w_i |b_i - R r_i|^2, det R = +1."""
    w = np.ones(len(b)) if w is None else np.asarray(w, float)
    B = (b * w[:, None]).T @ r
    U, _, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def rotation_angle(R_est, R_true):
    Re = R_est @ R_true.T
    return float(np.arccos(np.clip((np.trace(Re) - 1) / 2, -1.0, 1.0)))


def error_vector(R_est, R_true):
    """Small-angle error vector (rad, camera frame): R_est = exp([phi]x) R_true."""
    Re = R_est @ R_true.T
    return 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])


def tan_wcs(R, camera, f=None):
    """FITS TAN WCS equivalent to the pinhole camera with attitude R (exact, no distortion).
    Standard coordinates (xi east, eta north) = M^T (du, dv) / f with M rows = camera x, y axes in the
    local (east, north) basis. CRPIX is 1-based (FITS); astropy's pixel API with origin 0 then uses our
    zero-based convention directly."""
    from astropy.wcs import WCS
    f = camera.focal_length_px if f is None else f
    ra, dec, _ = pointing_from_attitude(R)
    east, north = east_north(ra, dec)
    M = np.array([[R[0] @ east, R[0] @ north], [R[1] @ east, R[1] @ north]])
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [camera.cx + 1.0, camera.cy + 1.0]
    w.wcs.cd = np.degrees(M.T / f)
    w.wcs.radesys = "ICRS"
    w.pixel_shape = (camera.width, camera.height)
    return w
