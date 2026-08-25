"""Numerical experiments: floating-point surprises, condition numbers,
solver stability (inverse vs LU vs QR vs SVD), and error propagation.
"""

from __future__ import annotations

import math
import pathlib
import random
import sys
import time
from decimal import Decimal

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matrix import Matrix
from least_squares import back_substitution, norm, qr_decompose, vec_add, vec_sub
from eigen import diagonalize_symmetric, power_iteration
from svd import svd, singular_values_of

OUTPUT_DIR = pathlib.Path(__file__).parent


def hilbert_matrix(n: int) -> Matrix:
    """The classic textbook example of a poorly-conditioned matrix."""
    return Matrix([[1 / (i + j + 1) for j in range(n)] for i in range(n)])


def symmetric_with_known_eigenvalues(eigenvalues, seed=0):
    from least_squares import gram_schmidt

    n = len(eigenvalues)
    rng = random.Random(seed)
    raw_vectors = [[rng.uniform(-3, 3) for _ in range(n)] for _ in range(n)]
    orthonormal_rows = gram_schmidt(raw_vectors)
    p = Matrix([list(row) for row in zip(*orthonormal_rows)])
    d = Matrix([[eigenvalues[i] if i == j else 0.0 for j in range(n)] for i in range(n)])
    return p * d * p.transpose(), p, d


# ============================================================
# 1. Floating-point experiments
# ============================================================

def demo_floating_point():
    print("=" * 78)
    print("1. Floating-point experiments")
    print("=" * 78)

    total = 0.1 + 0.2
    print(f"  0.1 + 0.2            = {total!r}")
    print(f"  0.1 + 0.2 == 0.3     = {total == 0.3}")
    print(f"  (0.1+0.2) - 0.3      = {total - 0.3:.3e}")
    print(f"  math.isclose(...,0.3)= {math.isclose(total, 0.3)}")

    print("\n  the EXACT decimal value each double actually stores:")
    for value in (0.1, 0.2, 0.3, total):
        print(f"    {value!r:<6} is really {Decimal(value)}")

    print("\n  raw hex (bit-exact) representation:")
    for value in (0.1, 0.2, 0.3, total):
        print(f"    {value!r:<6} -> {value.hex()}")

    print(
        "\n  why this is surprising: 0.1 and 0.2 look like simple, exact numbers on\n"
        "  paper, so '+' feels like it should just work. But doubles store numbers\n"
        "  in BASE 2, and 1/10 has no finite binary expansion -- exactly like 1/3\n"
        "  has no finite DECIMAL expansion. So '0.1' is already silently rounded to\n"
        "  the nearest representable double before any arithmetic happens, and so\n"
        "  is '0.2'. Adding those two already-rounded values gives a sum whose\n"
        "  nearest double lands just above 0.3's own rounded value. Nothing is\n"
        "  'broken' -- '0.1', '0.2', and '0.3' were each never exactly those\n"
        "  numbers to begin with; only their DISPLAYED decimal text made them look\n"
        "  exact."
    )

    eps = sys.float_info.epsilon
    print(f"\n  machine epsilon = {eps:.3e}  (smallest e with 1.0 + e != 1.0)")
    print(f"    1.0 + eps/2 == 1.0 ? {1.0 + eps / 2 == 1.0}  (rounds away entirely)")
    print(f"    1.0 + eps   == 1.0 ? {1.0 + eps == 1.0}")

    x = (0.1 + 0.2) + 0.3
    y = 0.1 + (0.2 + 0.3)
    print(f"\n  addition isn't even associative in floating point:")
    print(f"    (0.1+0.2)+0.3 = {x!r}")
    print(f"    0.1+(0.2+0.3) = {y!r}")
    print(f"    equal? {x == y}")

    print("\n  summation order matters when magnitudes differ a lot:")
    values = [1e16] + [1.0] * 1_000_000 + [-1e16]
    forward = 0.0
    for v in values:
        forward += v
    backward = 0.0
    for v in reversed(values):
        backward += v
    print(f"    true mathematical sum = 1,000,000")
    print(f"    naive forward sum  (1e16 first) = {forward}")
    print(f"    naive backward sum (1e16 last)  = {backward}")
    print(
        "    -> adding 1.0 to 1e16 has literally NO EFFECT (the gap between\n"
        "       representable doubles near 1e16 is already bigger than 1), so\n"
        "       summing the million 1.0's LAST -- once they're a plain 1,000,000 --\n"
        "       preserves them; summing them FIRST into a running total near 1e16\n"
        "       silently discards every one."
    )
    print()


# ============================================================
# 2. Condition number
# ============================================================

def condition_number(a: Matrix) -> float:
    """kappa(A) = sigma_max / sigma_min, via our own svd() from svd.py."""
    _, sigma, _ = svd(a)
    values = singular_values_of(sigma)
    sigma_max, sigma_min = values[0], values[-1]
    if sigma_min < 1e-14:
        return math.inf
    return sigma_max / sigma_min


def demo_condition_numbers():
    print("=" * 78)
    print("2. Condition number kappa(A)")
    print("=" * 78)
    cases = {
        "well-conditioned":   Matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]]),
        "poorly-conditioned (5x5 Hilbert)": hilbert_matrix(5),
        "nearly singular":    Matrix([[1, 1], [1, 1 + 1e-10]]),
    }
    print(f"  {'case':<34}{'ours':>14}{'numpy':>14}")
    for label, a in cases.items():
        ours = condition_number(a)
        theirs = np.linalg.cond(np.array(a.rows, dtype=float))
        print(f"  {label:<34}{ours:>14.4e}{theirs:>14.4e}")
    print(
        "\n  -> a well-conditioned matrix's kappa is close to 1: every direction is\n"
        "     stretched by roughly the same amount, so inverting it doesn't amplify\n"
        "     error much. The Hilbert matrix's rows/columns are nearly linearly\n"
        "     dependent by construction, so kappa is enormous (~1e5-1e6 for just a\n"
        "     5x5!). The near-singular pair's two rows are almost identical, so its\n"
        "     kappa is enormous too -- both signal that ANY small change to the\n"
        "     matrix or the right-hand side can produce a wildly different solution.\n"
        "\n"
        "  worth noting honestly: OURS reports inf for both ill-conditioned cases,\n"
        "  while numpy reports a huge-but-finite number. Checking our own svd()'s\n"
        "  raw singular values against numpy's shows why -- for the Hilbert matrix,\n"
        "  ours: [1.567, 0.2085, 0.01141, 0.0003059, 0.0], numpy's smallest is\n"
        "  ~3.3e-6, not 0. By the time Hotelling deflation reaches the 5th (tiny)\n"
        "  eigenvalue, the accumulated rounding error from removing the previous 4\n"
        "  is already LARGER than the true remaining eigenvalue, so our method\n"
        "  can't resolve it and reports exactly 0 instead. That's not a bug to\n"
        "  paper over -- it's the same ill-conditioning story playing out one\n"
        "  level down, inside our own algorithm's ability to measure it at all."
    )
    print()


# ============================================================
# 3. Stable vs unstable computation: inverse, LU, QR, SVD
# ============================================================

def lu_decompose(a: Matrix):
    """Partial-pivoting LU: P A = L U. Returns (P, L, U)."""
    if not a.is_square:
        raise ValueError(f"LU decomposition requires a square matrix, got {a.shape}")
    n = a.num_rows
    u_rows = [row[:] for row in a.rows]
    l_rows = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    perm = list(range(n))

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(u_rows[r][col]))
        if abs(u_rows[pivot_row][col]) < 1e-14:
            raise ValueError("matrix is numerically singular; LU decomposition failed")
        if pivot_row != col:
            u_rows[col], u_rows[pivot_row] = u_rows[pivot_row], u_rows[col]
            perm[col], perm[pivot_row] = perm[pivot_row], perm[col]
            for k in range(col):
                l_rows[col][k], l_rows[pivot_row][k] = l_rows[pivot_row][k], l_rows[col][k]
        for row in range(col + 1, n):
            factor = u_rows[row][col] / u_rows[col][col]
            l_rows[row][col] = factor
            for k in range(col, n):
                u_rows[row][k] -= factor * u_rows[col][k]

    p = Matrix([[1.0 if perm[i] == j else 0.0 for j in range(n)] for i in range(n)])
    return p, Matrix(l_rows), Matrix(u_rows)


def forward_substitution(l: Matrix, y):
    n = l.num_rows
    x = [0.0] * n
    for i in range(n):
        residual = y[i] - sum(l.rows[i][j] * x[j] for j in range(i))
        x[i] = residual / l.rows[i][i]
    return x


def solve_via_inverse(a: Matrix, b):
    return a.inverse().multiply_vector(b)


def solve_via_lu(a: Matrix, b):
    p, l, u = lu_decompose(a)
    return back_substitution(u, forward_substitution(l, p.multiply_vector(b)))


def solve_via_qr(a: Matrix, b):
    q, r = qr_decompose(a)
    return back_substitution(r, q.transpose().multiply_vector(b))


def solve_via_svd(a: Matrix, b, rcond: float = 1e-12):
    """x = V Sigma^-1 U^T b. Singular values below rcond * sigma_max are
    treated as zero (the standard pseudo-inverse convention, matching
    numpy.linalg.pinv/lstsq) instead of dividing by them -- an SVD-based
    solver's actual defense against a near-singular A, at the cost of
    silently dropping whatever the matrix can't resolve in that direction.
    """
    u, sigma, vt = svd(a)
    values = singular_values_of(sigma)
    threshold = rcond * values[0] if values else 0.0
    utb = u.transpose().multiply_vector(b)
    y = [utb[i] / values[i] if values[i] > threshold else 0.0 for i in range(len(values))]
    return vt.transpose().multiply_vector(y)


SOLVERS = {
    "explicit inverse": solve_via_inverse,
    "LU decomposition": solve_via_lu,
    "QR decomposition": solve_via_qr,
    "SVD":              solve_via_svd,
}


def demo_stable_vs_unstable():
    print("=" * 78)
    print("3. Solving Ax=b: explicit inverse vs LU vs QR vs SVD")
    print("=" * 78)
    cases = {
        "well-conditioned (3x3)":  Matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]]),
        "poorly-conditioned (5x5 Hilbert)": hilbert_matrix(5),
        "nearly singular (2x2)":   Matrix([[1, 1], [1, 1 + 1e-10]]),
    }
    repeats = 50

    for label, a in cases.items():
        n = a.num_rows
        rng = random.Random(0)
        x_true = [rng.uniform(1, 5) for _ in range(n)]
        b = a.multiply_vector(x_true)
        kappa = condition_number(a)

        print(f"\n  {label}  (kappa = {kappa:.3e})")
        print(f"    {'method':<20}{'time (us)':>12}{'||x-x_true||':>16}{'sensitivity':>16}")
        for name, solve in SOLVERS.items():
            try:
                start = time.perf_counter()
                for _ in range(repeats):
                    x = solve(a, b)
                elapsed_us = (time.perf_counter() - start) / repeats * 1e6

                accuracy = norm(vec_sub(x, x_true))

                rel_noise = 1e-8
                delta_b = [rel_noise * norm(b) * rng.uniform(-1, 1) for _ in range(n)]
                x_perturbed = solve(a, vec_add(b, delta_b))
                rel_db = norm(delta_b) / norm(b)
                rel_dx = norm(vec_sub(x_perturbed, x)) / norm(x) if norm(x) > 0 else float("nan")
                sensitivity = rel_dx / rel_db if rel_db > 0 else float("nan")

                print(f"    {name:<20}{elapsed_us:>12.2f}{accuracy:>16.3e}{sensitivity:>16.3e}")
            except ValueError as e:
                print(f"    {name:<20} FAILED: {e}")
    print(
        "\n  -> 'sensitivity' is the empirical amplification factor: (relative\n"
        "     change in x) / (relative change in b), for a tiny nudge to b. Theory\n"
        "     says this should be bounded by roughly kappa(A). For the\n"
        "     well-conditioned case it stays near 1 (output noise ~= input noise);\n"
        "     for the nearly-singular case it explodes toward kappa's huge value --\n"
        "     the same solve, the same tiny nudge, but a vastly more violent\n"
        "     response, regardless of which of the four methods is used. The\n"
        "     METHOD affects accuracy/runtime; the CONDITION NUMBER controls how\n"
        "     much any method can possibly help.\n"
        "\n"
        "     also notice QR FAILED outright on the nearly-singular case ('vectors\n"
        "     are linearly dependent'): our qr_decompose() is Gram-Schmidt-based,\n"
        "     and when two columns are nearly parallel, the leftover residual after\n"
        "     subtracting off the first column's projection is tiny enough to trip\n"
        "     our dependency check. That's a real, known weakness of\n"
        "     (even modified) Gram-Schmidt on ill-conditioned input -- production\n"
        "     QR solvers use Householder reflections instead specifically because\n"
        "     they don't have this failure mode, at the cost of more complex code\n"
        "     than we've built here."
    )
    print()


# ============================================================
# 4. Error propagation
# ============================================================

def demo_error_propagation():
    print("=" * 78)
    print("4. Error propagation: noisy measurements vs. condition number")
    print("=" * 78)
    cases = {
        "well-conditioned": Matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]]),
        "poorly-conditioned (5x5 Hilbert)": hilbert_matrix(5),
    }
    noise_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    rng = random.Random(0)
    results = {}

    for label, a in cases.items():
        n = a.num_rows
        x_true = [1.0] * n
        b_clean = a.multiply_vector(x_true)
        kappa = condition_number(a)
        mean_errors = []
        for noise_std in noise_levels:
            trial_errors = []
            for _ in range(50):
                noisy_b = [bi + rng.gauss(0, noise_std * max(abs(bi), 1e-6)) for bi in b_clean]
                x_est = solve_via_qr(a, noisy_b)
                trial_errors.append(norm(vec_sub(x_est, x_true)) / norm(x_true))
            mean_errors.append(sum(trial_errors) / len(trial_errors))
        results[label] = mean_errors
        print(f"\n  {label} (kappa={kappa:.2e}):")
        for noise, err in zip(noise_levels, mean_errors):
            print(f"    input noise std={noise:<8.0e} -> mean relative solution error = {err:.3e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, errors in results.items():
        ax.plot(noise_levels, errors, marker="o", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relative noise on b")
    ax.set_ylabel("mean relative error in x")
    ax.set_title("Error propagation vs. measurement noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = OUTPUT_DIR / "numerical_4_error_propagation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved plot -> {out_path}")
    print(
        "  -> both curves rise with noise, but the poorly-conditioned system's\n"
        "     curve sits far higher at every noise level -- the same measurement\n"
        "     noise produces a solution error roughly kappa(A) times larger. This\n"
        "     is the same 'sensitivity' number from section 3, just observed\n"
        "     directly through repeated noisy trials instead of a single nudge."
    )
    print()


# ============================================================
# 5. Suggested testing: large matrices, precision limits,
#    repeated multiplication, long iterative computations
# ============================================================

def demo_large_matrices():
    print("=" * 78)
    print("5a. Large matrices (numpy backend -- our own SVD doesn't scale)")
    print("=" * 78)
    # our own svd()/diagonalize_symmetric is a pure-Python power-iteration +
    # deflation method: benchmarks show it takes ~7s already at n=40, so a
    # "large" matrix here uses numpy directly, same call as demo_condition_numbers
    # makes for verification -- exactly how a real engineer would reach for
    # LAPACK at this scale rather than re-deriving it.
    for n in (100, 400):
        rng = np.random.default_rng(0)
        a = rng.uniform(-1, 1, size=(n, n))
        start = time.perf_counter()
        kappa = np.linalg.cond(a)
        cond_time = time.perf_counter() - start

        b = rng.uniform(-1, 1, size=n)
        start = time.perf_counter()
        x = np.linalg.solve(a, b)
        solve_time = time.perf_counter() - start
        residual = np.linalg.norm(a @ x - b)
        print(
            f"  n={n:<4} kappa={kappa:.2e}  cond() time={cond_time*1000:.1f}ms  "
            f"solve() time={solve_time*1000:.1f}ms  ||Ax-b||={residual:.2e}"
        )
    print(
        "  -> a random matrix stays well-conditioned regardless of size, so\n"
        "     residuals stay tiny even at n=400; direct LAPACK solves scale\n"
        "     roughly with n^3 but stay fast because they're implemented in\n"
        "     optimized, vectorized native code rather than nested Python loops."
    )
    print()


def demo_floating_point_precision_limits():
    print("=" * 78)
    print("5b. Floating-point precision limits: catastrophic cancellation")
    print("=" * 78)
    # classic quadratic-formula instability: when b is huge relative to
    # sqrt(disc), one root's naive formula subtracts two nearly-equal huge
    # numbers, destroying precision in the result.
    a_coef, b_coef, c_coef = 1.0, -1e8, 1.0

    disc = b_coef**2 - 4 * a_coef * c_coef
    sqrt_disc = math.sqrt(disc)
    naive_small_root = (-b_coef - sqrt_disc) / (2 * a_coef)

    # stable: compute the well-conditioned root directly, get the other via
    # Vieta's formula (product of roots = c/a), which never subtracts two
    # nearly-equal numbers
    stable_large_root = (-b_coef + sqrt_disc) / (2 * a_coef)
    stable_small_root = c_coef / (a_coef * stable_large_root)

    # ground truth via arbitrary-precision decimal arithmetic
    from decimal import getcontext
    getcontext().prec = 50
    da, db, dc = Decimal(a_coef), Decimal(b_coef), Decimal(c_coef)
    d_disc = (db * db - 4 * da * dc).sqrt()
    true_small_root = float((-db - d_disc) / (2 * da))

    print(f"  solving x^2 - 1e8 x + 1 = 0  (roots are ~1e8 and ~1e-8)")
    print(f"  true small root (50-digit decimal) = {true_small_root:.15e}")
    print(f"  naive formula's small root         = {naive_small_root:.15e}")
    print(f"  stable (Vieta's formula) small root = {stable_small_root:.15e}")
    print(f"  naive relative error  = {abs(naive_small_root - true_small_root) / abs(true_small_root):.3e}")
    print(f"  stable relative error = {abs(stable_small_root - true_small_root) / abs(true_small_root):.3e}")
    print(
        "  -> both formulas are algebraically correct, but the naive one\n"
        "     subtracts two nearly-equal ~1e8 values to get a ~1e-8 result,\n"
        "     wiping out most of its significant digits (catastrophic\n"
        "     cancellation). The stable version never subtracts nearly-equal\n"
        "     numbers, so it keeps full precision."
    )
    print()


def demo_repeated_multiplication():
    print("=" * 78)
    print("5c. Repeated matrix multiplication: naive vs. closed-form power")
    print("=" * 78)
    a = Matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    p, d = diagonalize_symmetric(a, seed=0)
    dominant = d.rows[0][0]
    a_norm = a * (1 / dominant)  # rescale so the dominant eigenvalue is ~1, avoiding overflow
    p_norm, d_norm = diagonalize_symmetric(a_norm, seed=0)

    print(f"  {'k':>7}  {'naive vs numpy':>16}  {'closed-form vs numpy':>22}")
    for k in (10, 100, 1000, 10000):
        naive = Matrix.identity(3)
        for _ in range(k):
            naive = naive * a_norm

        d_k = Matrix([[d_norm.rows[i][i] ** k if i == j else 0.0 for j in range(3)] for i in range(3)])
        closed_form = p_norm * d_k * p_norm.inverse()

        ground_truth = np.linalg.matrix_power(np.array(a_norm.rows), k)
        naive_error = np.linalg.norm(np.array(naive.rows) - ground_truth)
        closed_form_error = np.linalg.norm(np.array(closed_form.rows) - ground_truth)
        print(f"  {k:>7}  {naive_error:>16.3e}  {closed_form_error:>22.3e}")
    print(
        "  -> naive repeated multiplication does k sequential floating-point\n"
        "     matmuls, so rounding error accumulates roughly with k. The\n"
        "     closed-form P D^k P^-1 does a FIXED number of operations no matter\n"
        "     how large k is (only D^k, a per-entry power, grows with k), so its\n"
        "     error stays flat instead of climbing."
    )
    print()


def demo_long_iterative_computation():
    print("=" * 78)
    print("5d. Long iterative computations: power iteration's noise floor")
    print("=" * 78)
    # a small eigenvalue gap so convergence is naturally slow, making the
    # floating-point noise floor visible at achievable iteration counts
    a, _, _ = symmetric_with_known_eigenvalues([10.0, 9.9, 1.0], seed=0)
    true_eigenvalue = 10.0

    print(f"  {'max iterations':>15}  {'|estimate - true|':>20}")
    for max_iter in (10, 100, 1000, 10000, 100000):
        # seed=1 here, deliberately NOT the matrix-construction seed (0):
        # a freshly-seeded Random() drawing from a range symmetric about
        # zero always lands on the same *direction* for matching seeds
        # regardless of the range's width, so reusing seed=0 would silently
        # start already at the eigenvector and converge trivially
        result = power_iteration(a, num_iterations=max_iter, tol=0.0, seed=1)  # tol=0 -> always uses the full budget
        error = abs(result.eigenvalue - true_eigenvalue)
        print(f"  {max_iter:>15}  {error:>20.3e}")
    print(
        "  -> error shrinks steadily as iterations increase, then hits double\n"
        "     precision's noise floor (typically ~1e-14 to 1e-16 relative, and\n"
        "     here even rounding down to exactly 0.0) rather than shrinking\n"
        "     forever -- running the loop even longer can't out-run the rounding\n"
        "     error already present in every floating-point matrix-vector multiply."
    )
    print()


if __name__ == "__main__":
    demo_floating_point()
    demo_condition_numbers()
    demo_stable_vs_unstable()
    demo_error_propagation()
    demo_large_matrices()
    demo_floating_point_precision_limits()
    demo_repeated_multiplication()
    demo_long_iterative_computation()
