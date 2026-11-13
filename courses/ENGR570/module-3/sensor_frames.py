"""
sensor_frames.py - Space Systems Engineering, Module 3, Experiment B

Named-frame coordinate transformations for a star-camera / IMU instrument,
plus the pinhole camera model that turns camera-frame directions into pixels.

Frames
------
C  camera           +z out along the boresight, +x toward increasing pixel
                    column (image right), +y toward increasing pixel row
                    (image down). Right-handed.
B  instrument body  mechanical reference of the instrument.
I  IMU              the inertial measurement unit's sensing axes.
G  celestial        GCRS axes (aligned with ICRS): +x to RA 0, +z north.
E  Earth-fixed      rotating Earth frame of Module 2.

DCM convention
--------------
A Transform named C_AB converts COORDINATES from frame B into frame A:

    v_A = C_AB v_B          to_frame = A, from_frame = B

Its columns are the B axes written in A coordinates. Adjacent letters must
match when composing, and the outer letters name the result:

    C_CG = C_CB C_BG        (G -> B, then B -> C)

The inverse is the transpose: C_BA = C_AB^T.

This is a passive (coordinate) transformation of a fixed physical vector. It
is related to rotation_engine's active rotations by: if R actively rotates
frame A's axes onto frame B's axes (both written in A), then C_AB = R.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_COURSE / "module-1"))
sys.path.insert(0, str(_COURSE / "module-2"))
from celestial_geometry import radec_to_unit  # noqa: E402
from reference_frames import celestial_to_earth_fixed  # noqa: E402

from rotation_engine import (  # noqa: E402
    check_rotation_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

FRAMES = {
    "C": "camera",
    "B": "instrument body",
    "I": "IMU",
    "G": "celestial (GCRS axes)",
    "E": "Earth-fixed",
}


@dataclass(frozen=True, eq=False)
class Transform:
    """
    Direction-cosine matrix with named frames:  v_to = matrix @ v_from.

    The matrix is checked for orthogonality and det = +1 on construction and
    stored read-only.
    """
    to_frame: str
    from_frame: str
    matrix: np.ndarray

    def __post_init__(self):
        for frame in (self.to_frame, self.from_frame):
            if frame not in FRAMES:
                raise ValueError(
                    f"unknown frame {frame!r}; expected one of {sorted(FRAMES)}")
        matrix = np.array(check_rotation_matrix(self.matrix), dtype=float)
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @property
    def name(self):
        return f"C_{self.to_frame}{self.from_frame}"

    def __repr__(self):
        return (f"Transform {self.name}: {FRAMES[self.from_frame]} -> "
                f"{FRAMES[self.to_frame]}")

    def apply(self, vectors):
        """Convert one 3-vector, or an (..., 3) array, from from_frame
        coordinates to to_frame coordinates."""
        v = np.asarray(vectors, dtype=float)
        if v.shape[-1] != 3:
            raise ValueError(f"expected (..., 3) vectors, got shape {v.shape}")
        return v @ self.matrix.T

    def inverse(self):
        """The reverse transform, C_BA = C_AB^T (transpose, never inv())."""
        return Transform(self.from_frame, self.to_frame, self.matrix.T)

    def __matmul__(self, other):
        """C_AB @ C_BD -> C_AD. The inner frames must be the same frame."""
        if not isinstance(other, Transform):
            return NotImplemented
        if self.from_frame != other.to_frame:
            raise ValueError(
                f"cannot compose {self.name} @ {other.name}: the left "
                f"transform expects {self.from_frame}-frame input but the "
                f"right one outputs {other.to_frame}-frame coordinates")
        return Transform(self.to_frame, other.from_frame,
                         self.matrix @ other.matrix)

    @classmethod
    def identity(cls, to_frame, from_frame):
        """Two frames with coincident axes."""
        return cls(to_frame, from_frame, np.eye(3))


class FrameSet:
    """
    A small store of known DCMs that composes them on request.

    Add each physically known transform once (either direction); the
    inverse is implied. get("C", "G") then chains whatever stored
    transforms connect G to C. Adding a transform between two frames that
    are already connected is refused, so the set can never hold two
    inconsistent routes between the same pair of frames.
    """

    def __init__(self):
        self._edges = {}            # (to_frame, from_frame) -> Transform

    def add(self, transform, replace=False):
        key = (transform.to_frame, transform.from_frame)
        if transform.to_frame == transform.from_frame:
            raise ValueError("a transform must connect two different frames")
        direct = key in self._edges
        if direct and not replace:
            raise ValueError(f"{transform.name} is already defined; pass "
                             "replace=True to overwrite it")
        if not direct and self._path(*key) is not None:
            raise ValueError(
                f"{transform.to_frame} and {transform.from_frame} are already "
                f"connected through other frames; adding {transform.name} "
                "could make the set inconsistent")
        self._edges[key] = transform
        self._edges[key[::-1]] = transform.inverse()

    def _path(self, to_frame, from_frame):
        """Breadth-first search; returns the list of frames visited."""
        paths = [[from_frame]]
        seen = {from_frame}
        while paths:
            path = paths.pop(0)
            if path[-1] == to_frame:
                return path
            for (dst, src) in self._edges:
                if src == path[-1] and dst not in seen:
                    seen.add(dst)
                    paths.append(path + [dst])
        return None

    def get(self, to_frame, from_frame):
        """Return the Transform C_{to}{from}, composing stored DCMs."""
        for frame in (to_frame, from_frame):
            if frame not in FRAMES:
                raise ValueError(f"unknown frame {frame!r}")
        if to_frame == from_frame:
            return Transform(to_frame, from_frame, np.eye(3))
        path = self._path(to_frame, from_frame)
        if path is None:
            raise KeyError(f"no known transform chain from {from_frame} "
                           f"to {to_frame}")
        result = self._edges[(path[1], path[0])]
        for src, dst in zip(path[1:], path[2:]):
            result = self._edges[(dst, src)] @ result
        return result

    def transform(self, vectors, to_frame, from_frame):
        """Convert vectors from from_frame coordinates to to_frame."""
        return self.get(to_frame, from_frame).apply(vectors)


# ------------------------------------------------------------ frame builders
def earth_from_celestial(earth_angle_rad):
    """
    C_EG for a given Earth rotation angle, built from Module 2's
    celestial_to_earth_fixed() so there is only one astronomical rotation
    model in the course code. It inherits that function's idealization
    (rotation about z only; no precession, nutation or polar motion).
    """
    # Transforming the three basis vectors gives the matrix's columns as rows.
    matrix = celestial_to_earth_fixed(np.eye(3), earth_angle_rad).T
    return Transform("E", "G", matrix)


def attitude_to_transform(q_camera_attitude):
    """
    C_GC from a camera attitude quaternion.

    The attitude quaternion q is the active rotation that carries the
    celestial axes onto the camera axes, so the columns of R(q) are the
    camera axes in celestial coordinates and R(q) = C_GC. Celestial-to-
    camera is the inverse: attitude_to_transform(q).inverse().
    """
    return Transform("G", "C", quaternion_to_matrix(q_camera_attitude))


def camera_attitude_from_boresight(ra_deg, dec_deg, roll_deg=0.0):
    """
    Attitude quaternion for a camera pointed at (RA, Dec).

    With roll = 0 the image has celestial north up: +z = boresight,
    +y (image down) = south, +x (image right) = west. East is to the LEFT
    because the camera looks outward at the inside of the celestial sphere.
    Positive roll rotates the camera right-handed about its boresight.
    """
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    boresight = radec_to_unit(ra_deg, dec_deg)
    north = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra),
                      np.cos(dec)])
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])

    camera_axes_in_celestial = np.column_stack((-east, -north, boresight))
    roll = np.radians(roll_deg)
    c, s = np.cos(roll), np.sin(roll)
    roll_about_boresight = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    # Right-multiplication: the roll is about the camera's own z axis.
    return matrix_to_quaternion(camera_axes_in_celestial @ roll_about_boresight)


# -------------------------------------------------------------- camera model
@dataclass(frozen=True)
class PinholeCamera:
    """
    Ideal pinhole camera (no distortion, square pixels).

        u = cx + fx * X / Z          column, increasing to the right
        v = cy + fy * Y / Z          row, increasing downward

    for a camera-frame direction (X, Y, Z) with Z > 0. Pixel coordinates
    are continuous; the sensor covers 0 <= u < width, 0 <= v < height.
    """
    width: int = 1920
    height: int = 1080
    horizontal_fov_deg: float = 20.0
    cx: float = 960.0
    cy: float = 540.0

    @property
    def fx(self):
        """Focal length in pixels: W / (2 tan(HFOV / 2))."""
        return self.width / (2.0 * np.tan(np.radians(self.horizontal_fov_deg) / 2.0))

    @property
    def fy(self):
        return self.fx

    @property
    def vertical_fov_deg(self):
        return float(np.degrees(2.0 * np.arctan(self.height / (2.0 * self.fy))))

    def project(self, camera_vectors):
        """
        Project (N, 3) camera-frame directions to pixels.

        Returns (pixels, visible): pixels is (N, 2) [u, v], NaN for
        directions at or behind the camera plane (Z <= 0); visible is True
        for stars that are in front of the camera AND land on the sensor.
        """
        v = np.atleast_2d(np.asarray(camera_vectors, dtype=float))
        if v.shape[-1] != 3:
            raise ValueError(f"expected (N, 3) vectors, got shape {v.shape}")
        in_front = v[:, 2] > 0.0

        pixels = np.full((len(v), 2), np.nan)
        z = v[in_front, 2]
        pixels[in_front, 0] = self.cx + self.fx * v[in_front, 0] / z
        pixels[in_front, 1] = self.cy + self.fy * v[in_front, 1] / z

        on_sensor = np.zeros(len(v), dtype=bool)
        on_sensor[in_front] = ((pixels[in_front, 0] >= 0.0)
                               & (pixels[in_front, 0] < self.width)
                               & (pixels[in_front, 1] >= 0.0)
                               & (pixels[in_front, 1] < self.height))
        return pixels, on_sensor

    def unproject(self, pixels):
        """Unit camera-frame direction for each [u, v] pixel coordinate."""
        p = np.atleast_2d(np.asarray(pixels, dtype=float))
        rays = np.column_stack(((p[:, 0] - self.cx) / self.fx,
                                (p[:, 1] - self.cy) / self.fy,
                                np.ones(len(p))))
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)
