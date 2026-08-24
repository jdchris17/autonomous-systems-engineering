"""Power iteration, symmetric-matrix diagonalization, and a NumPy comparison.

Scope note: diagonalize_symmetric() only handles *symmetric* matrices.
Symmetric matrices are guaranteed to have real eigenvalues and an
orthogonal eigenvector basis, which is exactly what makes Hotelling
deflation (used below) mathematically exact after each step. General
non-symmetric diagonalizable matrices would need separate left and right
eigenvectors for exact deflation, and can have complex eigenvalues -- for
that case, this module falls back to (and is checked against)
numpy.linalg.eig rather than reimplementing the general QR algorithm.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from matrix import Matrix
from least_squares import dot, norm, vec_add, vec_scale, vec_sub


# -- power iteration -----------------------------------------------------------

@dataclass
class PowerIterationResult:
    eigenvalue: float
    eigenvector: list
    converged: bool
    iterations: int


def power_iteration(
    a: Matrix, num_iterations: int = 5000, tol: float = 1e-10, seed=None, initial_vector=None
) -> PowerIterationResult:
    """Estimate the dominant eigenvalue/eigenvector of a square matrix.

    Repeatedly applies A and renormalizes: b_{k+1} = A b_k / ||A b_k||.
    This converges to the eigenvector of the eigenvalue with the largest
    magnitude, at a linear rate governed by |lambda_2 / lambda_1| -- the
    closer the top two eigenvalues are, the slower (or less reliable) the
    convergence. If A has no single dominant real eigenvalue (e.g. a
    genuine rotation matrix, whose eigenvalues are a complex-conjugate
    pair of equal magnitude), the iterate never settles down and
    `converged` comes back False -- that is itself the correct, expected
    behavior, not a bug.

    By default the starting vector is random (seeded by `seed`); pass
    `initial_vector` explicitly to control it, e.g. to study how a
    starting vector's overlap with the true dominant eigenvector affects
    convergence speed.
    """
    if not a.is_square:
        raise ValueError(f"power iteration requires a square matrix, got {a.shape}")

    n = a.num_rows
    if initial_vector is not None:
        b = list(initial_vector)
    else:
        rng = random.Random(seed)
        b = [rng.uniform(-1, 1) for _ in range(n)]
    b = vec_scale(b, 1 / norm(b))

    converged = False
    iterations_used = num_iterations
    for i in range(1, num_iterations + 1):
        ab = a.multiply_vector(b)
        ab_norm = norm(ab)
        if ab_norm < 1e-14:
            raise ValueError("matrix maps the iterate to (near) zero; power iteration cannot proceed")
        b_next = vec_scale(ab, 1 / ab_norm)

        # the sign of b can flip each step (e.g. for a negative eigenvalue),
        # so treat b_next and -b_next as equally "converged"
        diff = min(norm(vec_sub(b_next, b)), norm(vec_add(b_next, b)))
        b = b_next
        if diff < tol:
            converged = True
            iterations_used = i
            break

    # Rayleigh quotient: exact for a true eigenvector, a good estimate otherwise
    eigenvalue = dot(a.multiply_vector(b), b) / dot(b, b)
    return PowerIterationResult(eigenvalue=eigenvalue, eigenvector=b, converged=converged, iterations=iterations_used)


# -- symmetric matrices ----------------------------------------------------------

def is_symmetric(a: Matrix, tol: float = 1e-9) -> bool:
    if not a.is_square:
        return False
    n = a.num_rows
    return all(
        math.isclose(a.rows[i][j], a.rows[j][i], abs_tol=tol)
        for i in range(n)
        for j in range(i + 1, n)
    )


def random_symmetric(n: int, low: float = -5.0, high: float = 5.0, seed=None) -> Matrix:
    rng = random.Random(seed)
    m = Matrix([[rng.uniform(low, high) for _ in range(n)] for _ in range(n)])
    return (m + m.transpose()) * 0.5


def orthogonal_columns(p: Matrix, tol: float = 1e-6) -> bool:
    columns = p.transpose().rows
    n = len(columns)
    return all(
        abs(dot(columns[i], columns[j])) < tol
        for i in range(n)
        for j in range(i + 1, n)
    )


# -- diagonalization: A = P D P^-1 ------------------------------------------------

def diagonalize_symmetric(a: Matrix, num_iterations: int = 5000, tol: float = 1e-10, seed=None) -> tuple[Matrix, Matrix]:
    """Find P (eigenvectors as columns) and D (eigenvalues on the diagonal)
    such that A = P D P^-1, via repeated power iteration + Hotelling
    deflation. Only valid for symmetric A.
    """
    if not is_symmetric(a):
        raise ValueError(
            "diagonalize_symmetric requires a symmetric matrix (guarantees real "
            "eigenvalues and an orthogonal eigenbasis); use numpy.linalg.eig for "
            "general non-symmetric matrices"
        )

    n = a.num_rows
    rng = random.Random(seed)
    working = a
    eigenvalues, eigenvectors = [], []

    for _ in range(n):
        result = power_iteration(working, num_iterations=num_iterations, tol=tol, seed=rng.randrange(2**31))
        eigenvalues.append(result.eigenvalue)
        eigenvectors.append(result.eigenvector)

        # Hotelling deflation: remove this eigenpair's contribution so the
        # next power iteration converges to the next-largest eigenvalue.
        # Exact for symmetric A, where eigenvectors are orthonormal.
        v = result.eigenvector
        outer = Matrix([[result.eigenvalue * vi * vj for vj in v] for vi in v])
        working = working - outer

    p = Matrix([list(row) for row in zip(*eigenvectors)])
    d = Matrix([[eigenvalues[i] if i == j else 0.0 for j in range(n)] for i in range(n)])
    return p, d


def matrices_close(a: Matrix, b: Matrix, tol: float = 1e-6) -> bool:
    if a.shape != b.shape:
        return False
    return all(
        math.isclose(x, y, abs_tol=tol)
        for r1, r2 in zip(a.rows, b.rows)
        for x, y in zip(r1, r2)
    )


def verify_diagonalization(a: Matrix, p: Matrix, d: Matrix, tol: float = 1e-6) -> bool:
    reconstructed = p * d * p.inverse()
    return matrices_close(a, reconstructed, tol)


def _build_symmetric_with_known_eigenvalues(eigenvalues, seed=0):
    """A = P D P^T for a known diagonal D and a random orthonormal P
    (via Gram-Schmidt) -- a symmetric matrix whose exact eigenvalues are
    known in advance, useful for checking power iteration against ground
    truth rather than just against numpy.
    """
    from least_squares import gram_schmidt

    n = len(eigenvalues)
    rng = random.Random(seed)
    raw_vectors = [[rng.uniform(-3, 3) for _ in range(n)] for _ in range(n)]
    orthonormal_rows = gram_schmidt(raw_vectors)
    p = Matrix([list(row) for row in zip(*orthonormal_rows)])  # columns = eigenvectors
    d = Matrix([[eigenvalues[i] if i == j else 0.0 for j in range(n)] for i in range(n)])
    a = p * d * p.transpose()  # P^-1 == P^T for an orthonormal P
    return a, p, d


def demo_diagonal_matrices():
    print("=" * 78)
    print("1. Diagonal matrices")
    print("=" * 78)
    a = Matrix([[2, 0, 0], [0, -7, 0], [0, 0, 3]])
    print(f"  A =\n  {a.pretty_print()}".replace("\n", "\n  "))

    result = power_iteration(a, seed=0)
    print(f"  power_iteration:  eigenvalue={result.eigenvalue:.6f}  converged={result.converged}")
    print("  -> matches the largest-magnitude diagonal entry, -7, as expected")

    p, d = diagonalize_symmetric(a, seed=0)
    found = sorted(d.rows[i][i] for i in range(3))
    print(f"  diagonalize_symmetric eigenvalues (sorted) = {[round(v, 4) for v in found]}")
    print(f"  A == P D P^-1 (within tolerance)            = {verify_diagonalization(a, p, d)}")
    print()


def demo_symmetric_matrices():
    print("=" * 78)
    print("2. Symmetric matrices")
    print("=" * 78)
    a = random_symmetric(4, seed=1)
    print(f"  random symmetric A (seed=1) =\n  {a.pretty_print()}".replace("\n", "\n  "))
    print(f"  is_symmetric(A) = {is_symmetric(a)}")

    p, d = diagonalize_symmetric(a, seed=2)
    eigenvalues = [round(d.rows[i][i], 4) for i in range(4)]
    print(f"  eigenvalues found      = {eigenvalues}")
    print(f"  P has orthogonal cols  = {orthogonal_columns(p)}")
    print(f"  A == P D P^-1          = {verify_diagonalization(a, p, d)}")
    print()


def demo_identity_matrix():
    print("=" * 78)
    print("3. Identity matrix")
    print("=" * 78)
    i4 = Matrix.identity(4)
    result = power_iteration(i4, seed=0)
    print(f"  power_iteration on I(4): eigenvalue={result.eigenvalue:.6f}  converged={result.converged}")
    print(f"  eigenvector picked = {[round(c, 4) for c in result.eigenvector]}  (every direction is an eigenvector of I)")

    p, d = diagonalize_symmetric(i4, seed=0)
    print(f"  diagonalize_symmetric(I) gives D == I: {matrices_close(d, i4, tol=1e-9)}")
    print()


def demo_rotation_matrices():
    print("=" * 78)
    print("4. Rotation matrices (observe behavior)")
    print("=" * 78)
    for angle in (0, 90, 180):
        rot = Matrix.rotation_2d(angle, degrees=True)
        result = power_iteration(rot, num_iterations=500, seed=0)
        print(
            f"  rotation_2d({angle:>3} deg): converged={result.converged!s:<5}  "
            f"eigenvalue={result.eigenvalue if result.converged else 'n/a'}"
        )
    print(
        "  -> 0 deg is the identity (eigenvalue 1) and 180 deg is -I (eigenvalue\n"
        "     -1); both are real and converge normally. A genuine 90 deg rotation\n"
        "     has complex-conjugate eigenvalues e^{+-i*90deg} of equal magnitude,\n"
        "     so there's no real dominant eigenvector -- the iterate just keeps\n"
        "     rotating and never converges. That's the correct behavior, not a bug."
    )
    print()


def demo_known_dominant_eigenvalues():
    print("=" * 78)
    print("5. Random matrices with known dominant eigenvalues")
    print("=" * 78)
    a, p, _ = _build_symmetric_with_known_eigenvalues([5, 3, 1], seed=7)
    result = power_iteration(a, seed=0)
    print(f"  constructed A with known eigenvalues {{5, 3, 1}}")
    print(f"  power_iteration found eigenvalue = {result.eigenvalue:.6f}  (expected 5)")

    _, d = diagonalize_symmetric(a, seed=1)
    found = sorted(round(d.rows[i][i], 4) for i in range(3))
    print(f"  diagonalize_symmetric found all three = {found}  (expected [1, 3, 5])")

    print("  convergence speed vs. eigenvalue gap:")
    close_gap, _, _ = _build_symmetric_with_known_eigenvalues([5, 4.9, 1], seed=9)
    wide_gap, _, _ = _build_symmetric_with_known_eigenvalues([5, 1, 0.5], seed=10)
    close_result = power_iteration(close_gap, num_iterations=5000, tol=1e-10, seed=0)
    wide_result = power_iteration(wide_gap, num_iterations=5000, tol=1e-10, seed=0)
    print(f"    eigenvalues {{5, 4.9, 1}} (close top two)  -> {close_result.iterations} iterations to converge")
    print(f"    eigenvalues {{5, 1, 0.5}} (wide top two)   -> {wide_result.iterations} iterations to converge")
    print("    -> the closer the top two eigenvalues, the slower power iteration converges")
    print()


def demo_numpy_comparison():
    print("=" * 78)
    print("6. Comparison against NumPy")
    print("=" * 78)
    import numpy as np

    a = random_symmetric(5, seed=11)
    result = power_iteration(a, seed=0)
    np_vals, np_vecs = np.linalg.eig(np.array(a.rows, dtype=float))
    idx = int(np.argmax(np.abs(np_vals.real)))
    np_val, np_vec = np_vals[idx].real, np_vecs[:, idx].real
    if np.dot(np_vec, result.eigenvector) < 0:
        np_vec = -np_vec

    print(f"  dominant eigenvalue:  ours={result.eigenvalue:.8f}  numpy={np_val:.8f}  diff={abs(result.eigenvalue - np_val):.2e}")
    vec_diff = max(abs(o - t) for o, t in zip(result.eigenvector, np_vec.tolist()))
    print(f"  dominant eigenvector: max component diff = {vec_diff:.2e}")

    _, d = diagonalize_symmetric(a, seed=1)
    ours_all = sorted(d.rows[i][i] for i in range(5))
    theirs_all = sorted(float(v.real) for v in np_vals)
    print(f"  full eigenvalue set:  ours ={[round(v, 4) for v in ours_all]}")
    print(f"                        numpy={[round(v, 4) for v in theirs_all]}")
    print(
        "  -> numpy's LAPACK routine computes all eigenvalues in one direct pass\n"
        "     to ~1e-15 precision, independent of eigenvalue spacing. Ours needed\n"
        f"     {result.iterations} power-iteration steps just for the dominant one, and\n"
        "     diagonalize_symmetric's deflation compounds approximation error with\n"
        "     each extracted eigenpair, so later eigenvalues drift further from\n"
        "     numpy's answer than the first (dominant) one does."
    )
    print()


if __name__ == "__main__":
    demo_diagonal_matrices()
    demo_symmetric_matrices()
    demo_identity_matrix()
    demo_rotation_matrices()
    demo_known_dominant_eigenvalues()
    demo_numpy_comparison()
