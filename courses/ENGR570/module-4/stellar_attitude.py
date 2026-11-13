"""
stellar_attitude.py - Space Systems Engineering, Module 4

Camera attitude from matched star directions.

Conventions
-----------
* reference_vectors: shape (N, 3). Each ROW is a catalog star direction in
  CELESTIAL coordinates (GCRS-oriented axes), frame G.
* measured_vectors: shape (N, 3). Each ROW is the same star's observed
  direction in CAMERA coordinates, frame C (+z boresight, +x image right,
  +y image down, as in Module 3).
* Row i of both arrays is the same star. Rows are normalized internally.
* The attitude is the DCM C_CG of Module 3:  b_i = C_CG r_i.
* Attitude error is a rotation vector phi in CAMERA axes, defined by
      C_estimated = R(phi) C_true
  so phi_x, phi_y are pointing errors (tilts of the boresight) and phi_z is
  roll about the optical axis.
* Angles are radians throughout this file.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module-3"))
from rotation_engine import check_rotation_matrix, matrix_to_quaternion  # noqa: E402

# Wahba solution is declared non-unique when (s2 + d*s3) / s1 falls below
# this. For two stars the ratio is about (separation / 2)^2, so this rejects
# pairs closer than roughly 0.4 arcsec.
GEOMETRY_TOLERANCE = 1e-12

# Two directions are "parallel" for TRIAD when |sin(separation)| is below this.
PARALLEL_TOLERANCE = 1e-9


class InsufficientGeometryError(ValueError):
    """The star directions do not determine a unique three-axis attitude."""


# ------------------------------------------------------------------ validation
def _unit_rows(vectors, name):
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} contains NaN or infinite values")
    norms = np.linalg.norm(v, axis=1)
    if np.any(norms == 0.0):
        raise ValueError(f"{name} contains a zero-length vector "
                         f"(row {int(np.argmin(norms))})")
    return v / norms[:, None]


def _matched_pair(reference_vectors, measured_vectors):
    r = _unit_rows(reference_vectors, "reference_vectors")
    b = _unit_rows(measured_vectors, "measured_vectors")
    if len(r) != len(b):
        raise ValueError(f"reference_vectors has {len(r)} rows but "
                         f"measured_vectors has {len(b)}; they must match")
    return r, b


def _positive_per_star(values, n, name):
    a = np.asarray(values, dtype=float)
    if a.ndim == 0:
        a = np.full(n, float(a))
    if a.shape != (n,):
        raise ValueError(f"{name} must be a scalar or have shape ({n},), "
                         f"got {a.shape}")
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return a


# ------------------------------------------------------------------- estimators
def _triad_core(r1, r2, b1, b2):
    """TRIAD without input checks. Returns (C_CG, sin_ref, sin_meas)."""
    rn, bn = np.cross(r1, r2), np.cross(b1, b2)
    sin_r, sin_b = np.linalg.norm(rn), np.linalg.norm(bn)
    if sin_r < PARALLEL_TOLERANCE or sin_b < PARALLEL_TOLERANCE:
        return None, sin_r, sin_b
    t2, s2 = rn / sin_r, bn / sin_b
    ref_triad = np.column_stack((r1, t2, np.cross(r1, t2)))
    cam_triad = np.column_stack((b1, s2, np.cross(b1, s2)))
    return cam_triad @ ref_triad.T, sin_r, sin_b


def triad(reference_vectors, measured_vectors):
    """
    Estimate C_CG from two matched star directions.

    Reference vectors: GCRS-oriented unit vectors.
    Measured vectors: camera-frame unit vectors.

    Both arrays have shape (2, 3), one star per row. An orthonormal triad is
    built from each pair (first star, normal to the pair, and their cross
    product) and C_CG maps the celestial triad onto the camera triad.

    TRIAD is not symmetric: the first star's direction is reproduced
    exactly, and the second only fixes the rotation about the first. Put
    the more trusted star first.
    """
    r, b = _matched_pair(reference_vectors, measured_vectors)
    if len(r) != 2:
        raise ValueError(f"TRIAD needs exactly 2 stars, got {len(r)}")
    C, sin_r, sin_b = _triad_core(r[0], r[1], b[0], b[1])
    if C is None:
        raise InsufficientGeometryError(
            "TRIAD needs two non-parallel directions "
            f"(|sin separation|: reference {sin_r:.1e}, measured {sin_b:.1e})")
    return C


def wahba_svd(reference_vectors, measured_vectors, weights=None):
    """
    Find the weighted least-squares camera attitude C_CG.

    Return a proper 3x3 rotation matrix.

    Minimizes Wahba's loss  L(C) = 1/2 sum_i w_i |b_i - C r_i|^2  over
    proper rotations, with reference_vectors (N, 3) celestial-frame rows
    r_i, measured_vectors (N, 3) camera-frame rows b_i, and positive
    weights w_i (default all 1; the statistically optimal choice is
    1 / sigma_i^2).

    Method: B = sum_i w_i b_i r_i^T = U S V^T, then
        C = U diag(1, 1, det(U) det(V)) V^T.
    The last factor forces det C = +1; without it noisy data can return a
    reflection.

    Uniqueness: the attitude is unique when s2 + d*s3 > 0, where
    s1 >= s2 >= s3 are the singular values and d = det(U) det(V). Two
    non-parallel stars already satisfy this even though B then has rank 2,
    so B is NOT required to have full rank. Fewer than two independent
    directions raises InsufficientGeometryError.
    """
    r, b = _matched_pair(reference_vectors, measured_vectors)
    if len(r) < 2:
        raise InsufficientGeometryError(
            "at least two star directions are needed for a three-axis attitude")
    w = (np.ones(len(r)) if weights is None
         else _positive_per_star(weights, len(r), "weights"))

    profile = (w[:, None] * b).T @ r                       # sum w b r^T
    U, s, Vt = np.linalg.svd(profile)
    d = float(np.sign(np.linalg.det(U) * np.linalg.det(Vt)))
    if s[0] == 0.0 or (s[1] + d * s[2]) <= GEOMETRY_TOLERANCE * s[0]:
        raise InsufficientGeometryError(
            "star directions are (nearly) parallel: rotation about their "
            "common direction is undetermined")
    return U @ np.diag([1.0, 1.0, d]) @ Vt


# -------------------------------------------------------------- error measures
def rotation_matrix_to_vector(R):
    """Rotation vector (axis * angle, radians) of a proper rotation matrix."""
    q = matrix_to_quaternion(R)                 # scalar-first, w >= 0
    vector_norm = np.linalg.norm(q[1:])
    if vector_norm == 0.0:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vector_norm, q[0])
    return q[1:] * (angle / vector_norm)


def attitude_error(C_estimated, C_true):
    """
    Return the rotation-vector and total-angle error
    between estimated and true attitudes.

    Returns (phi, angle): phi is the 3-vector [rad] in camera axes with
    C_estimated = R(phi) C_true, and angle = |phi| >= 0. phi[0], phi[1] are
    pointing errors about camera x and y; phi[2] is roll about the boresight.
    """
    C_estimated = check_rotation_matrix(C_estimated)
    C_true = check_rotation_matrix(C_true)
    phi = rotation_matrix_to_vector(C_estimated @ C_true.T)
    return phi, float(np.linalg.norm(phi))


def attitude_information(measured_vectors, sigma_rad):
    """
    Construct the approximate 3x3 attitude information
    matrix from camera-frame star directions.

        J = sum_i (1 / sigma_i^2) (I - b_i b_i^T)

    measured_vectors: (N, 3) camera-frame rows b_i. sigma_rad: per-axis
    angular measurement noise of each star direction [rad], a scalar or
    (N,). J is in camera axes and its inverse is the covariance of the
    attitude-error rotation vector phi (first order, small errors,
    isotropic noise about each line of sight, weights 1/sigma^2).

    Each star contributes information about rotations perpendicular to its
    own direction and none about rotation around it. Stars near the
    boresight therefore say almost nothing about roll.
    """
    b = _unit_rows(measured_vectors, "measured_vectors")
    sigma = _positive_per_star(sigma_rad, len(b), "sigma_rad")
    inverse_variance = 1.0 / sigma**2
    return (np.sum(inverse_variance) * np.eye(3)
            - (inverse_variance[:, None] * b).T @ b)


def attitude_covariance(information):
    """Inverse of the information matrix [rad^2], refusing singular input."""
    information = np.asarray(information, dtype=float)
    if (not np.all(np.isfinite(information))
            or np.linalg.cond(information) > 1.0 / np.finfo(float).eps):
        raise InsufficientGeometryError(
            "information matrix is singular: some rotation is unobservable")
    return np.linalg.inv(information)


def angular_residuals(C_CG, reference_vectors, measured_vectors):
    """
    Angle [rad] between each measured direction and its prediction C_CG r_i.

    This is r_i = acos(b_i^T C r_i), computed as atan2(|b x Cr|, b . Cr),
    which is the same angle but keeps full precision for the arcsecond-level
    residuals that acos cannot resolve.
    """
    r, b = _matched_pair(reference_vectors, measured_vectors)
    predicted = r @ np.asarray(C_CG, dtype=float).T
    cross = np.linalg.norm(np.cross(b, predicted), axis=1)
    dot = np.einsum("ij,ij->i", b, predicted)
    return np.arctan2(cross, dot)


# ----------------------------------------------------------- robust estimation
@dataclass
class RobustResult:
    """Outcome of robust_wahba() or ransac_attitude()."""
    C: np.ndarray                 # C_CG from the accepted stars
    inliers: np.ndarray           # boolean (N,): stars used in the final fit
    residuals: np.ndarray         # (N,) angular residuals of ALL stars [rad]
    thresholds: np.ndarray        # (N,) residual limit applied per star [rad]
    converged: bool               # False = the method itself reports failure
    status: str
    iterations: int
    rejected: list = field(default_factory=list)   # indices, in rejection order


def residual_threshold_factor(false_alarm_probability):
    """
    Multiple of sigma above which a residual is called an outlier.

    A correct match has independent Gaussian errors of sigma on two axes, so
    its angular residual is Rayleigh distributed: P(r > k sigma) =
    exp(-k^2 / 2). Solving for the chosen false-alarm probability gives k.
    """
    if not 0.0 < false_alarm_probability < 1.0:
        raise ValueError("false_alarm_probability must be between 0 and 1")
    return float(np.sqrt(-2.0 * np.log(false_alarm_probability)))


def robust_wahba(reference_vectors, measured_vectors, sigma_rad,
                 false_alarm_probability=1e-3, min_stars=3):
    """
    Wahba/SVD attitude with iterative outlier rejection.

    Loop:
      1. fit C_CG to the currently accepted stars (weights 1/sigma^2);
      2. compute every accepted star's angular residual;
      3. compare each with its own limit k*sigma_i, where k comes from the
         Rayleigh noise model and the false-alarm probability;
      4. if the worst offender exceeds its limit, drop that ONE star and
         refit. One at a time, because a single bad match drags the
         solution and inflates the residuals of good stars too;
      5. stop when all accepted residuals are within limits.

    It reports failure (converged=False) if it would have to go below
    min_stars or the remaining stars no longer determine an attitude. It
    can also converge to a WRONG attitude without noticing when the bad
    matches are numerous enough to dominate the first fit; converged=True
    means "self-consistent", not "correct".
    """
    r, b = _matched_pair(reference_vectors, measured_vectors)
    n = len(r)
    sigma = _positive_per_star(sigma_rad, n, "sigma_rad")
    if n < max(min_stars, 2):
        raise InsufficientGeometryError(
            f"need at least {max(min_stars, 2)} stars, got {n}")
    thresholds = residual_threshold_factor(false_alarm_probability) * sigma
    weights = 1.0 / sigma**2

    active = np.ones(n, dtype=bool)
    rejected = []
    C = None
    iterations = 0
    while True:
        iterations += 1
        try:
            C_new = wahba_svd(r[active], b[active], weights[active])
        except InsufficientGeometryError:
            converged, status = False, "remaining stars are degenerate"
            break
        C = C_new
        residuals = angular_residuals(C, r, b)
        ratio = np.where(active, residuals / thresholds, -np.inf)
        worst = int(np.argmax(ratio))
        if ratio[worst] <= 1.0:
            converged, status = True, "all accepted residuals within limits"
            break
        if active.sum() - 1 < max(min_stars, 2):
            converged = False
            status = (f"residuals still exceed limits with only "
                      f"{int(active.sum())} stars left")
            break
        active[worst] = False
        rejected.append(worst)

    if C is None:                                  # first fit already failed
        raise InsufficientGeometryError(status)
    return RobustResult(C=C, inliers=active,
                        residuals=angular_residuals(C, r, b),
                        thresholds=thresholds, converged=converged,
                        status=status, iterations=iterations,
                        rejected=rejected)


def ransac_attitude(reference_vectors, measured_vectors, sigma_rad, rng=None,
                    max_hypotheses=500, false_alarm_probability=1e-3,
                    hypothesis_slack=3.0, min_inliers=3):
    """
    RANSAC attitude estimate using two matched stars as the minimal sample.

    For each sampled pair of stars:
      * skip it unless the angle between the two stars agrees in the catalog
        and in the image (a wrong match almost never preserves it) and the
        pair is not nearly parallel;
      * TRIAD gives an attitude hypothesis;
      * count the stars whose residual is within hypothesis_slack * k *
        sigma_i (slack because a two-star hypothesis is itself noisy).
    The largest consensus set wins. It is then refined: SVD fit on the
    set, re-select inliers with the strict limit k*sigma_i, repeat until
    the set stops changing.

    If the number of distinct pairs is at most max_hypotheses, every pair is
    tried and the result is deterministic; otherwise pairs are sampled with
    `rng`. converged=False when fewer than min_inliers stars agree.
    """
    r, b = _matched_pair(reference_vectors, measured_vectors)
    n = len(r)
    sigma = _positive_per_star(sigma_rad, n, "sigma_rad")
    if n < 2:
        raise InsufficientGeometryError("need at least 2 stars")
    k = residual_threshold_factor(false_alarm_probability)
    thresholds = k * sigma
    loose = hypothesis_slack * thresholds
    weights = 1.0 / sigma**2

    first, second = np.triu_indices(n, k=1)
    if len(first) > max_hypotheses:
        rng = np.random.default_rng() if rng is None else rng
        chosen = rng.choice(len(first), size=max_hypotheses, replace=False)
        first, second = first[chosen], second[chosen]

    # Pre-filter: inter-star angles must match between catalog and image.
    ref_angle = np.arccos(np.clip(np.einsum("ij,ij->i", r[first], r[second]), -1, 1))
    cam_angle = np.arccos(np.clip(np.einsum("ij,ij->i", b[first], b[second]), -1, 1))
    pair_sigma = np.hypot(sigma[first], sigma[second])
    plausible = ((np.abs(ref_angle - cam_angle) <= hypothesis_slack * k * pair_sigma)
                 & (cam_angle > 20.0 * pair_sigma))

    best_count, best_cost, best_mask = 0, np.inf, None
    tested = 0
    for i, j in zip(first[plausible], second[plausible]):
        C, _, _ = _triad_core(r[i], r[j], b[i], b[j])
        if C is None:
            continue
        tested += 1
        predicted = r @ C.T
        residuals = np.arctan2(np.linalg.norm(np.cross(b, predicted), axis=1),
                               np.einsum("ij,ij->i", b, predicted))
        mask = residuals <= loose
        count = int(mask.sum())
        cost = float(np.sum((residuals[mask] / sigma[mask]) ** 2))
        if count > best_count or (count == best_count and cost < best_cost):
            best_count, best_cost, best_mask = count, cost, mask

    if best_mask is None or best_count < max(min_inliers, 2):
        return RobustResult(C=np.eye(3), inliers=np.zeros(n, dtype=bool),
                            residuals=np.full(n, np.nan), thresholds=thresholds,
                            converged=False, iterations=tested,
                            status="no consistent set of stars found",
                            rejected=list(range(n)))

    # Refinement with the strict, noise-model threshold.
    mask = best_mask
    C = None
    status = "consensus set refined"
    converged = True
    for _ in range(10):
        try:
            C = wahba_svd(r[mask], b[mask], weights[mask])
        except InsufficientGeometryError:
            converged, status = False, "consensus set is degenerate"
            break
        new_mask = angular_residuals(C, r, b) <= thresholds
        if new_mask.sum() < max(min_inliers, 2):
            converged, status = False, "too few stars pass the strict limit"
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    if C is None:
        C = np.eye(3)
        residuals = np.full(n, np.nan)
    else:
        residuals = angular_residuals(C, r, b)
    return RobustResult(C=C, inliers=mask, residuals=residuals,
                        thresholds=thresholds, converged=converged,
                        status=status, iterations=tested,
                        rejected=[int(i) for i in np.flatnonzero(~mask)])
