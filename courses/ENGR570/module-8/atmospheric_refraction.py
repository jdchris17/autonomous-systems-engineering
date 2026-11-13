"""
atmospheric_refraction.py - Space Systems Engineering, Module 8

Atmospheric refraction of star directions, in the observer's local frame.

Refraction is a property of the ENVIRONMENT along the line of sight. It
acts in the local east/north/up frame: it raises every star toward the
zenith and leaves azimuth unchanged. It is applied here, to directions in
that frame, and not as an adjustment to attitude or position.

Model
-----
ERFA's refraction constants A and B (erfa.refco, the same model used by
erfa.atco13 and Astropy's AltAz frame):

    z_vacuum - z_observed = A tan(z_observed) + B tan^3(z_observed)

A depends on pressure, temperature, humidity and wavelength; B is a small
negative correction for the curvature of the atmosphere. ERFA states the
model is accurate to better than about 0.1 arcsec for zenith distances up
to about 70 degrees and degrades toward the horizon, reaching arcseconds by
z = 85 degrees. Zenith distances beyond MAX_ZENITH_DEG are refused rather
than extrapolated.

simple_tan_refraction() is the one-term k tan(z) textbook form. It is
included only for comparison; it is not a precision model at low altitude.
"""

from dataclasses import dataclass

import erfa
import numpy as np

ARCSEC_PER_RAD = np.degrees(1.0) * 3600.0
MAX_ZENITH_DEG = 87.0


@dataclass(frozen=True)
class Atmosphere:
    """Observing conditions at the instrument. Defaults are illustrative."""
    pressure_hpa: float = 1013.0
    temperature_c: float = 15.0
    relative_humidity: float = 0.5        # 0 to 1
    wavelength_um: float = 0.55           # 550 nm, visible


VACUUM = Atmosphere(pressure_hpa=0.0)
STANDARD_ATMOSPHERE = Atmosphere()


def refraction_coefficients(atmosphere):
    """ERFA refraction constants (A, B) in radians."""
    if atmosphere is None or atmosphere.pressure_hpa <= 0.0:
        return 0.0, 0.0
    a, b = erfa.refco(atmosphere.pressure_hpa, atmosphere.temperature_c,
                      atmosphere.relative_humidity, atmosphere.wavelength_um)
    return float(a), float(b)


def _check_zenith(zenith_rad):
    z = np.asarray(zenith_rad, dtype=float)
    if np.any(z > np.radians(MAX_ZENITH_DEG)):
        raise ValueError(
            f"zenith distance above {MAX_ZENITH_DEG} deg: the refraction "
            "model is not valid this close to the horizon")
    return z


def refraction_from_observed(zenith_observed_rad, atmosphere):
    """z_vacuum - z_observed [rad] for an OBSERVED zenith distance."""
    z = _check_zenith(zenith_observed_rad)
    a, b = refraction_coefficients(atmosphere)
    tan_z = np.tan(z)
    return a * tan_z + b * tan_z**3


def observed_zenith_distance(zenith_vacuum_rad, atmosphere):
    """
    Observed zenith distance for a vacuum (geometric) one.

    Solves z_obs + A tan z_obs + B tan^3 z_obs = z_vac by Newton iteration;
    it converges to machine precision in a few steps over the valid range.
    """
    z_vac = _check_zenith(zenith_vacuum_rad)
    a, b = refraction_coefficients(atmosphere)
    if a == 0.0 and b == 0.0:
        return z_vac.copy() if z_vac.ndim else float(z_vac)
    z = z_vac.copy()
    for _ in range(8):
        tan_z = np.tan(z)
        residual = z + a * tan_z + b * tan_z**3 - z_vac
        slope = 1.0 + (a + 3.0 * b * tan_z**2) / np.cos(z) ** 2
        z = z - residual / slope
    return z if z.ndim else float(z)


def refraction_arcsec(geometric_altitude_deg, atmosphere=STANDARD_ATMOSPHERE):
    """
    Refractive displacement [arcsec] of a star at a given GEOMETRIC
    (vacuum) altitude: how far above its true place it appears.
    """
    z_vac = np.radians(90.0 - np.asarray(geometric_altitude_deg, dtype=float))
    return (z_vac - observed_zenith_distance(z_vac, atmosphere)) * ARCSEC_PER_RAD


def simple_tan_refraction_arcsec(geometric_altitude_deg,
                                 atmosphere=STANDARD_ATMOSPHERE):
    """The one-term approximation A tan(z), with z the geometric zenith
    distance. Comparison only."""
    a, _ = refraction_coefficients(atmosphere)
    z = np.radians(90.0 - np.asarray(geometric_altitude_deg, dtype=float))
    return a * np.tan(z) * ARCSEC_PER_RAD


def saemundsson_refraction_arcsec(geometric_altitude_deg,
                                  atmosphere=STANDARD_ATMOSPHERE):
    """
    Saemundsson's empirical formula for refraction from true altitude,
    scaled for pressure and temperature. An independent cross-check that
    remains sensible down to the horizon (to roughly 0.1 arcminute).
    """
    h = np.asarray(geometric_altitude_deg, dtype=float)
    arcmin = 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11)))
    scale = (atmosphere.pressure_hpa / 1010.0) * (283.0 / (273.0 + atmosphere.temperature_c))
    return arcmin * 60.0 * scale


# ---------------------------------------------------- directions in the ENU frame
def _split(vectors_enu):
    v = np.atleast_2d(np.asarray(vectors_enu, dtype=float))
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    horizontal = np.hypot(v[:, 0], v[:, 1])
    zenith = np.arctan2(horizontal, v[:, 2])
    return v, horizontal, zenith


def _with_zenith(v, horizontal, zenith):
    """Same azimuth, new zenith distance."""
    safe = np.where(horizontal > 0.0, horizontal, 1.0)
    scale = np.where(horizontal > 0.0, np.sin(zenith) / safe, 0.0)
    return np.column_stack((v[:, 0] * scale, v[:, 1] * scale, np.cos(zenith)))


def refract_enu(vectors_enu, atmosphere):
    """
    Apply refraction: vacuum (topocentric apparent) directions in local
    east/north/up become observed directions. Shape (N, 3) or (3,).
    """
    single = np.ndim(vectors_enu) == 1
    v, horizontal, z_vac = _split(vectors_enu)
    out = _with_zenith(v, horizontal, observed_zenith_distance(z_vac, atmosphere))
    return out[0] if single else out


def unrefract_enu(vectors_enu, atmosphere):
    """Remove refraction: observed directions become vacuum directions."""
    single = np.ndim(vectors_enu) == 1
    v, horizontal, z_obs = _split(vectors_enu)
    out = _with_zenith(v, horizontal,
                       z_obs + refraction_from_observed(z_obs, atmosphere))
    return out[0] if single else out
