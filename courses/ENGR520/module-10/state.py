"""module-10 -- state.py  (Section 3: Define the State of the Simulator)

The final-project simulator's configuration, as a handful of small, named
dataclasses -- so "what does the simulator need to know" is answered by
reading this file, not by hunting for module-level constants scattered
across the Module 9 capstone script.

    CelestialScene   : list of Star(ra_deg, dec_deg, mag, wavelength_nm)
    Observer         : lat_deg, lon_deg, jd
    CameraAttitude   : q OR R (either is accepted; .R_BL is always available)
    CameraIntrinsics : fx, fy, cx, cy, Nx, Ny  (-> .K)
    PhysicalOptics   : f_mm, D_mm, wavelength_optical_nm
    Distortion       : k1, k2               (carried, not yet applied)
    Sensor           : px_um, py_um, qe, exposure_s
    SimulatorState   : the above seven, together

Everything geometric reuses Module 9's frames.py and Module 3's
quaternions.py directly (imported below, not re-implemented).
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(__file__)
_MODULE3 = os.path.normpath(os.path.join(_HERE, "..", "module-3"))
if _MODULE3 not in sys.path:
    sys.path.insert(0, _MODULE3)

from quaternions import (quaternion_normalize, quaternion_to_rotation_matrix,   # noqa: E402
                         rotation_matrix_to_quaternion)             # (Module 3)

# module-3 and module-9 each have their own, unrelated frames.py; load
# module-9's by explicit file path (see _pathutil.py) so the ambiguous
# bare name "frames" never has to be resolved off sys.path.
from _pathutil import load_module                                   # noqa: E402
_MODULE9 = os.path.normpath(os.path.join(_HERE, "..", "module-9"))
if _MODULE9 not in sys.path:
    sys.path.insert(0, _MODULE9)
_frames9 = load_module("module9_frames", os.path.join(_MODULE9, "frames.py"))
_frames_R_BL = _frames9.R_BL


# ---------------------------------------------------------------------------
# Celestial scene
# ---------------------------------------------------------------------------
@dataclass
class Star:
    ra_deg: float
    dec_deg: float
    mag: float
    wavelength_nm: float = 550.0     # optional; default = a generic visual band
    name: str = ""


@dataclass
class CelestialScene:
    stars: list                      # list[Star]

    def arrays(self):
        """(ra_deg, dec_deg, mag, wavelength_nm) as aligned numpy arrays."""
        ra = np.array([s.ra_deg for s in self.stars])
        dec = np.array([s.dec_deg for s in self.stars])
        mag = np.array([s.mag for s in self.stars])
        wl = np.array([s.wavelength_nm for s in self.stars])
        return ra, dec, mag, wl


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------
@dataclass
class Observer:
    lat_deg: float
    lon_deg: float
    jd: float                        # Julian Date; how you got it is not this state's concern


# ---------------------------------------------------------------------------
# Camera attitude -- represented as EITHER q or R, .R_BL always available
# ---------------------------------------------------------------------------
@dataclass
class CameraAttitude:
    """Camera attitude, in the SAME sense Module 3 always used the word:
    camera(body) -> local(world), i.e. R_LB. Give exactly one of `q`
    (scalar-first Hamilton quaternion) or `R_LB` (3x3 matrix); the other
    representation is derived on demand, not stored redundantly.

    Stage II's projection needs the OPPOSITE direction (local -> camera),
    exposed as `.R_BL` = R_LB^T -- getting this transpose backward is the
    single most common attitude bug (Module 7 said so first; still true).
    """
    q: np.ndarray = None
    R_LB_matrix: np.ndarray = None

    def __post_init__(self):
        have_q, have_R = self.q is not None, self.R_LB_matrix is not None
        if have_q == have_R:
            raise ValueError("CameraAttitude needs exactly one of q or R_LB_matrix")
        if have_q:
            self.q = quaternion_normalize(np.asarray(self.q, dtype=float))
        else:
            self.R_LB_matrix = np.asarray(self.R_LB_matrix, dtype=float)

    @property
    def R_LB(self):
        """camera -> local (this is what "attitude" means)."""
        if self.R_LB_matrix is not None:
            return self.R_LB_matrix
        return quaternion_to_rotation_matrix(self.q)

    @property
    def R_BL(self):
        """local -> camera; what the Stage II projection formula needs."""
        return self.R_LB.T

    @property
    def quaternion(self):
        if self.q is not None:
            return self.q
        return rotation_matrix_to_quaternion(self.R_LB_matrix)

    @classmethod
    def from_pointing(cls, alt_deg, az_deg, roll_deg=0.0):
        """Build from an alt/az/roll pointing, via Module 9's frames.R_BL
        (local -> camera directly); stored here as its transpose, R_LB."""
        R_BL = _frames_R_BL(alt_deg, az_deg, roll_deg)
        return cls(R_LB_matrix=R_BL.T)


# ---------------------------------------------------------------------------
# Camera intrinsics / physical optics / distortion / sensor
# ---------------------------------------------------------------------------
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    Nx: int
    Ny: int

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @classmethod
    def from_camera_model(cls, cam):
        """Build from a Module 7 `camera_model.Camera` instance directly."""
        return cls(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy, Nx=cam.Nx, Ny=cam.Ny)


@dataclass
class PhysicalOptics:
    f_mm: float
    D_mm: float
    wavelength_optical_nm: float = 550.0


@dataclass
class Distortion:
    """Radial distortion, k1/k2: x_d = x(1+k1 r^2+k2 r^4), same for y (r^2 =
    x^2+y^2 in normalized image coordinates). Carried since Section 3, but
    deliberately unused through Stage III -- that stage's job is the IDEAL
    geometric position (see pipeline.py). Stage IV (pipeline.stage4_distortion)
    is where these finally apply; the default k1=k2=0 makes that stage an
    identity map, which is itself a cheap correctness check."""
    k1: float = 0.0
    k2: float = 0.0


@dataclass
class Sensor:
    px_um: float
    py_um: float
    qe: float = 0.7
    exposure_s: float = 1.0


# ---------------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------------
@dataclass
class SimulatorState:
    scene: CelestialScene
    observer: Observer
    attitude: CameraAttitude
    intrinsics: CameraIntrinsics
    optics: PhysicalOptics
    distortion: Distortion
    sensor: Sensor
