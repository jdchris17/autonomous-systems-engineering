"""
Controlled positive and negative scenarios (Experiment B and tests/test_negative.py).

Each builder returns a Scenario: an image (or a raw source list), the catalogue the solver uses, an
optional camera override, whether a verified solution is CORRECT, and the failure codes that would be an
acceptable explanation if it is not.
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stellar_vision import CameraModel, StarCatalog  # noqa: E402
from stellar_vision.geometry import attitude_from_pointing, radec_to_vec, vec_to_radec  # noqa: E402
from stellar_vision.result import FailureCode as F  # noqa: E402
from stellar_vision.simulation import CENTER, make_catalog, random_pointing, render  # noqa: E402

CAT = StarCatalog.load(ROOT / "stellar_vision" / "data" / "catalog_m13.npz")
CAM = CameraModel.from_json(ROOT / "stellar_vision" / "data" / "camera_m13.json")
EMPTY = StarCatalog.from_arrays(np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 3)), "empty")


@dataclass
class Scenario:
    name: str
    purpose: str
    image: np.ndarray | None
    sources: dict | None
    catalog: StarCatalog
    camera: CameraModel
    R_true: np.ndarray | None
    expect_solution: bool
    acceptable_codes: tuple = ()


def _visible(cat, R):
    u, v, front = CAM.project(R, cat.vec)
    return np.nonzero(CAM.in_frame(u, v, front))[0]


def nominal(seed=200):
    rng = np.random.default_rng(seed)
    R, _ = random_pointing(rng)
    img, _ = render(CAT, R, seed)
    return Scenario("nominal field", "positive control", img, None, CAT, CAM, R, True)


def calibration_within_tolerance(seed=201, err=0.02):
    rng = np.random.default_rng(seed)
    R, _ = random_pointing(rng)
    img, _ = render(CAT, R, seed)
    cam = replace(CAM, focal_length_px=CAM.focal_length_px * (1 + err))
    return Scenario(f"focal length {err:+.0%} (inside 3% tolerance)", "positive control: approximate intrinsics",
                    img, None, CAT, cam, R, True)


def blank(seed=500):
    img, _ = render(EMPTY, attitude_from_pointing(*CENTER, 0.0), seed, n_field=0, n_hot=(0, 0), n_streaks=(0, 0))
    return Scenario("blank image", "background + noise only", img, None, CAT, CAM, None, False,
                    (F.NO_SOURCES, F.INSUFFICIENT_SOURCES, F.NO_HYPOTHESIS))


def artifacts_only(seed=501):
    img, _ = render(EMPTY, attitude_from_pointing(*CENTER, 0.0), seed, n_field=0, n_hot=(20, 30), n_streaks=(5, 8))
    return Scenario("near-blank: hot pixels + streaks", "no stars, only detector artifacts", img, None, CAT, CAM, None, False,
                    (F.NO_SOURCES, F.INSUFFICIENT_SOURCES, F.NO_HYPOTHESIS, F.INSUFFICIENT_MATCHES))


def few_stars(seed=502, n=4):
    rng = np.random.default_rng(seed)
    R, _ = random_pointing(rng)
    vis = _visible(CAT, R)
    keep = np.zeros(len(CAT), bool)
    keep[vis[np.argsort(CAT.mag[vis])[:n]]] = True
    img, _ = render(CAT.subset(keep), R, seed, n_field=0)
    return Scenario(f"insufficient stars ({n} in field)", "too few stars for an independently verified solution",
                    img, None, CAT, CAM, R, False, (F.INSUFFICIENT_SOURCES, F.NO_HYPOTHESIS, F.INSUFFICIENT_MATCHES))


def random_detections(seed=503, n=30):
    rng = np.random.default_rng(seed)
    src = {"x": rng.uniform(-0.5, 511.5, n), "y": rng.uniform(-0.5, 511.5, n), "flux": np.sort(rng.uniform(1e3, 1e5, n))[::-1],
           "sigma_px": np.full(n, 0.07)}
    return Scenario("random false detections", "no coherent celestial geometry", None, src, CAT, CAM, None, False,
                    (F.NO_HYPOTHESIS, F.INSUFFICIENT_MATCHES, F.LOW_CATALOG_FRACTION))


def wrong_catalog(seed=504):
    rng = np.random.default_rng(seed)
    R, _ = random_pointing(rng)
    img, _ = render(make_catalog(seed=999), R, seed)
    return Scenario("wrong catalogue", "real-looking field that is not in our catalogue", img, None, CAT, CAM, None, False,
                    (F.NO_HYPOTHESIS, F.INSUFFICIENT_MATCHES, F.LOW_CATALOG_FRACTION))


def wrong_region(seed=505, offset=5.5):
    rng = np.random.default_rng(seed)
    R0 = attitude_from_pointing(CENTER[0], CENTER[1], 0.0)
    phi = rng.uniform(0, 2 * np.pi)
    local = np.array([np.sin(np.radians(offset)) * np.cos(phi), np.sin(np.radians(offset)) * np.sin(phi), np.cos(np.radians(offset))])
    ra, dec = vec_to_radec(R0.T @ local)
    R = attitude_from_pointing(float(ra), float(dec), rng.uniform(0, 360))
    img, _ = render(CAT, R, seed)
    return Scenario("wrong sky region", f"boresight {offset} deg from the search centre (search radius 2.5 deg)", img, None,
                    CAT, CAM, R, False, (F.NO_HYPOTHESIS, F.INSUFFICIENT_MATCHES, F.LOW_CATALOG_FRACTION))


def calibration_outside_tolerance(seed=506, err=0.06):
    rng = np.random.default_rng(seed)
    R, _ = random_pointing(rng)
    img, _ = render(CAT, R, seed)
    cam = replace(CAM, focal_length_px=CAM.focal_length_px * (1 + err))
    return Scenario(f"focal length {err:+.0%} (outside 3% tolerance)", "camera calibration outside the permitted range", img,
                    None, CAT, cam, R, False, (F.CAMERA_MODEL_INVALID, F.NO_HYPOTHESIS, F.INSUFFICIENT_MATCHES,
                                               F.LOW_CATALOG_FRACTION, F.RESIDUALS_INCONSISTENT))


def ambiguous_twin(seed=507, angle=120.0):
    """Catalogue = the field's own stars + an exact copy rotated 120 deg about the search-centre axis
    (a duplicated pattern inside the search region): two equally good, distinct solutions."""
    rng = np.random.default_rng(seed)
    while True:
        R, pose = random_pointing(rng)
        if np.degrees(np.arccos(R[2] @ radec_to_vec(*CENTER))) > 1.5:
            break
    vis = _visible(CAT, R)
    c = radec_to_vec(*CENTER)
    a = np.radians(angle)
    K = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
    Q = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    vec = np.vstack([CAT.vec[vis], CAT.vec[vis] @ Q.T])
    ra, dec = vec_to_radec(vec)
    twin = StarCatalog(np.concatenate([CAT.id[vis], 10000 + CAT.id[vis]]), ra, dec, np.concatenate([CAT.mag[vis]] * 2), vec,
                       "twin-pattern")
    img, _ = render(CAT.subset(np.isin(np.arange(len(CAT)), vis)), R, seed)
    return Scenario("competing ambiguous hypotheses", "catalogue contains the field's pattern twice", img, None, twin, CAM, R,
                    False, (F.AMBIGUOUS,))


def weak_geometry(seed=508, n=9, radius_deg=0.42, mags=(8.3, 8.8)):
    """A tight cluster of faint stars and nothing else: identification works, but roll is barely observable."""
    rng = np.random.default_rng(seed)
    R, _ = random_pointing(rng)
    rho = np.radians(radius_deg) * np.sqrt(rng.uniform(0, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    local = np.column_stack([np.sin(rho) * np.cos(phi), np.sin(rho) * np.sin(phi), np.cos(rho)])
    vec = local @ R                                        # camera-frame cluster around the boresight -> inertial
    ra, dec = vec_to_radec(vec)
    cl = StarCatalog(np.arange(1, n + 1), ra, dec, rng.uniform(*mags, n), vec, "tight-cluster")
    img, _ = render(cl, R, seed, n_field=0)
    return Scenario("weak star geometry", f"{n} faint stars within {radius_deg} deg: poorly constrained roll", img, None, cl,
                    CAM, R, False, (F.WEAK_GEOMETRY,))


ALL_NEGATIVE = (blank, artifacts_only, few_stars, random_detections, wrong_catalog, wrong_region,
                calibration_outside_tolerance, ambiguous_twin, weak_geometry)
ALL_POSITIVE = (nominal, calibration_within_tolerance)
