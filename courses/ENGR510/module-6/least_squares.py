"""Orthogonal projection, Gram-Schmidt, and least squares regression --
built entirely on Matrix / list vector operations, no numpy involved.

Vectors here are plain lists (or tuples) of numbers rather than the 3D
Vector class, since projection subspaces and regression design matrices
routinely have more than 3 dimensions (one per data point).
"""

from __future__ import annotations

import math

from matrix import Matrix
from vector import Vector


# -- generic n-dimensional vector helpers ------------------------------------

def _components(v):
    """Allow the 3D Vector class to be passed in anywhere a list is expected."""
    return [v.x, v.y, v.z] if isinstance(v, Vector) else list(v)


def dot(u, v) -> float:
    u, v = _components(u), _components(v)
    return sum(a * b for a, b in zip(u, v))


def norm(v) -> float:
    return math.sqrt(dot(v, v))


def vec_add(u, v):
    u, v = _components(u), _components(v)
    return [a + b for a, b in zip(u, v)]


def vec_sub(u, v):
    u, v = _components(u), _components(v)
    return [a - b for a, b in zip(u, v)]


def vec_scale(v, scalar: float):
    return [scalar * a for a in _components(v)]


# -- orthogonal projection ---------------------------------------------------

def project_onto_vector(v, onto):
    """Project vector v onto vector `onto`: ((v . onto) / (onto . onto)) * onto."""
    denom = dot(onto, onto)
    if denom == 0:
        raise ValueError("cannot project onto the zero vector")
    scalar = dot(v, onto) / denom
    return vec_scale(onto, scalar)


def project_onto_subspace(v, basis):
    """Project v onto the subspace spanned by `basis`.

    `basis` must be an orthogonal set of vectors (e.g. the output of
    gram_schmidt) -- for an orthogonal basis, the projection onto the
    subspace is just the sum of the projections onto each basis vector.
    """
    v = _components(v)
    projection = [0.0] * len(v)
    for u in basis:
        projection = vec_add(projection, project_onto_vector(v, u))
    return projection


# -- Gram-Schmidt --------------------------------------------------------------

def gram_schmidt(vectors):
    """Convert a list of linearly independent vectors into an orthonormal basis.

    Modified Gram-Schmidt: at each step, subtract off the component along
    every orthonormal vector found so far, then normalize what's left.
    """
    basis = []
    for v in vectors:
        w = _components(v)
        for u in basis:
            w = vec_sub(w, project_onto_vector(w, u))
        length = norm(w)
        if length < 1e-10:
            raise ValueError("vectors are linearly dependent")
        basis.append(vec_scale(w, 1 / length))
    return basis


# -- QR decomposition (stretch goal) ------------------------------------------

def qr_decompose(a: Matrix) -> tuple[Matrix, Matrix]:
    """Thin QR decomposition A = QR via Gram-Schmidt on A's columns.

    Q has orthonormal columns (same shape as A), R is square upper
    triangular. Requires A's columns to be linearly independent
    (num_rows >= num_cols).
    """
    columns = a.transpose().rows
    orthonormal_columns = gram_schmidt(columns)
    q = Matrix(orthonormal_columns).transpose()
    r = q.transpose() * a
    return q, r


def back_substitution(r: Matrix, y):
    """Solve the upper-triangular system R x = y."""
    n = r.num_rows
    x = [0.0] * n
    for i in reversed(range(n)):
        residual = y[i] - sum(r.rows[i][j] * x[j] for j in range(i + 1, n))
        x[i] = residual / r.rows[i][i]
    return x


# -- least squares --------------------------------------------------------------

def least_squares(a: Matrix, b):
    """Solve the least squares problem min ||Ax - b||^2 via the normal
    equations: x_hat = (A^T A)^-1 A^T b.
    """
    b = _components(b)
    at = a.transpose()
    ata = at * a
    atb = at.multiply_vector(b)
    return ata.inverse().multiply_vector(atb)


def least_squares_qr(a: Matrix, b):
    """Solve the same least squares problem via QR instead of an explicit
    matrix inverse: A = QR, so R x = Q^T b, solved by back substitution.
    """
    b = _components(b)
    q, r = qr_decompose(a)
    qtb = q.transpose().multiply_vector(b)
    return back_substitution(r, qtb)


if __name__ == "__main__":
    print("Projection onto a vector")
    v, onto = [3, 4], [1, 0]
    print(f"  project {v} onto {onto} = {project_onto_vector(v, onto)}")

    print("\nProjection onto a subspace (the xy-plane, spanned by e1, e2)")
    e1, e2 = [1, 0, 0], [0, 1, 0]
    point = [2, 5, 9]
    print(f"  project {point} onto span(e1, e2) = {project_onto_subspace(point, [e1, e2])}")

    print("\nGram-Schmidt: turning 3 independent vectors into an orthonormal basis")
    raw = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    basis = gram_schmidt(raw)
    for u in basis:
        print(f"  {[round(c, 4) for c in u]}  (|u| = {norm(u):.4f})")
    print(f"  u0 . u1 = {dot(basis[0], basis[1]):.2e}  (should be ~0)")

    print("\nLeast squares regression line fit: y = m*x + c")
    xs = [0, 1, 2, 3, 4]
    ys = [1.1, 2.9, 4.9, 7.2, 8.8]  # ~ y = 2x + 1, with noise
    a = Matrix([[x, 1] for x in xs])  # columns: [x, 1] -> solves for [m, c]
    m_normal, c_normal = least_squares(a, ys)
    m_qr, c_qr = least_squares_qr(a, ys)
    print(f"  normal equations: m={m_normal:.4f}, c={c_normal:.4f}")
    print(f"  QR decomposition: m={m_qr:.4f}, c={c_qr:.4f}")
