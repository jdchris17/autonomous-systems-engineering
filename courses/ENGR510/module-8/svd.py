"""Singular Value Decomposition: our own implementation (built on
diagonalize_symmetric from eigen.py), low-rank compression, and an
image-compression demo.

Performance note: our own svd() computes eigenvectors of A^T A via
repeated power iteration + deflation (eigen.diagonalize_symmetric), which
is roughly cubic-and-up in pure Python and becomes impractical much
beyond a few dozen rows/columns. That's fine for the small "conceptual"
matrices below, where the point is to see every piece of the algorithm
we already built. For the image compression section, the pixel matrix is
far too large for that approach to finish in reasonable time, so that
section uses numpy.linalg.svd as the computational backend instead --
exactly how a real engineer would reach for a battle-tested LAPACK
routine at scale rather than re-deriving it. compress_matrix() itself,
the actual "keep the top k singular values" logic, is ours either way.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from matrix import Matrix
from eigen import diagonalize_symmetric

OUTPUT_DIR = pathlib.Path(__file__).parent


# -- our own SVD, built on diagonalize_symmetric -----------------------------

def svd(a: Matrix, num_iterations: int = 5000, tol: float = 1e-10, seed: int = 0):
    """Thin SVD: A = U Sigma V^T, via eigendecomposition of A^T A.

    V's columns are eigenvectors of A^T A; the singular values are the
    square roots of A^T A's (non-negative) eigenvalues; U's columns are
    recovered as u_i = A v_i / sigma_i. Returns (U, Sigma, Vt) as Matrix
    objects, with r = min(A.num_rows, A.num_cols) columns/rows kept.
    """
    m, n = a.shape
    ata = a.transpose() * a  # n x n, symmetric -> diagonalize_symmetric applies
    p, d = diagonalize_symmetric(ata, num_iterations=num_iterations, tol=tol, seed=seed)

    eigenpairs = [
        (max(d.rows[i][i], 0.0), [p.rows[row][i] for row in range(n)]) for i in range(n)
    ]
    eigenpairs.sort(key=lambda pair: pair[0], reverse=True)

    r = min(m, n)
    singular_values, u_columns, v_columns = [], [], []
    for eigenvalue, v in eigenpairs[:r]:
        sigma = math.sqrt(eigenvalue)
        singular_values.append(sigma)
        v_columns.append(v)
        if sigma > 1e-10:
            av = a.multiply_vector(v)
            u_columns.append([c / sigma for c in av])
        else:
            u_columns.append([0.0] * m)  # degenerate direction for a rank-deficient A

    u = Matrix([list(row) for row in zip(*u_columns)])
    v = Matrix([list(row) for row in zip(*v_columns)])
    sigma_matrix = Matrix([[singular_values[i] if i == j else 0.0 for j in range(r)] for i in range(r)])
    return u, sigma_matrix, v.transpose()


def reconstruct(u: Matrix, sigma: Matrix, vt: Matrix) -> Matrix:
    return u * sigma * vt


def singular_values_of(sigma: Matrix) -> list:
    return [sigma.rows[i][i] for i in range(sigma.num_rows)]


def effective_rank(sigma: Matrix, tol: float = 1e-8) -> int:
    values = singular_values_of(sigma)
    return sum(1 for v in values if v > tol * values[0])


def compress_matrix(a: Matrix, k: int, **svd_kwargs) -> Matrix:
    """Rank-k approximation of A: keep only the k largest singular values
    (and their matching U/V columns), then reconstruct."""
    r = min(a.shape)
    if not (1 <= k <= r):
        raise ValueError(f"k must be between 1 and {r} for a {a.shape} matrix, got {k}")

    u, sigma, vt = svd(a, **svd_kwargs)
    u_k = Matrix([row[:k] for row in u.rows])
    sigma_k = Matrix([row[:k] for row in sigma.rows[:k]])
    vt_k = Matrix(vt.rows[:k])
    return u_k * sigma_k * vt_k


def frobenius_norm(a: Matrix) -> float:
    return math.sqrt(sum(v * v for row in a.rows for v in row))


def matrices_diff(a: Matrix, b: Matrix) -> Matrix:
    return a - b


# -- visualization -------------------------------------------------------------

def plot_singular_values(series: dict, title: str, out_path: pathlib.Path, logy: bool = True):
    """series: {label: [singular values, descending]}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, values in series.items():
        ax.plot(range(1, len(values) + 1), values, marker="o", markersize=4, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("magnitude" + (" (log scale)" if logy else ""))
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -- sample grayscale image ---------------------------------------------------

def generate_sample_image(path: pathlib.Path, size=(160, 160)):
    """Synthesize and save a grayscale PNG (a smooth gradient, a few flat
    shapes, and some sharp text) since no real photo was supplied. Point
    load_grayscale_image() at any real image file instead to use one.
    """
    w, h = size
    img = Image.new("L", size)
    pixels = img.load()
    for x in range(w):
        for y in range(h):
            # smooth radial gradient background
            dx, dy = (x - w / 2) / w, (y - h / 2) / h
            pixels[x, y] = int(60 + 120 * (1 - min(1.0, math.hypot(dx, dy) * 1.6)))

    draw = ImageDraw.Draw(img)
    draw.ellipse([w * 0.15, h * 0.15, w * 0.55, h * 0.55], fill=210)
    draw.rectangle([w * 0.5, h * 0.55, w * 0.9, h * 0.9], fill=40)
    draw.ellipse([w * 0.55, h * 0.1, w * 0.85, h * 0.4], outline=250, width=4)
    draw.text((w * 0.08, h * 0.85), "SVD DEMO", fill=255)

    img.save(path)
    return path


def load_grayscale_image(path: pathlib.Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=float)


def compress_image_numpy(pixels: np.ndarray, k: int):
    """Same rank-k idea as compress_matrix(), but backed by numpy.linalg.svd
    since a real image is far too large for our pure-Python power
    iteration to factor in reasonable time."""
    u, s, vt = np.linalg.svd(pixels, full_matrices=False)
    reconstructed = (u[:, :k] * s[:k]) @ vt[:k, :]
    return np.clip(reconstructed, 0, 255), s


def compression_ratio(m: int, n: int, k: int) -> float:
    original = m * n
    compressed = k * (m + n + 1)  # U_k (m x k) + singular values (k) + V_k^T (k x n)
    return original / compressed


# -- demos ----------------------------------------------------------------------

def demo_conceptual_svd():
    print("=" * 78)
    print("1. Conceptual SVD on a small matrix")
    print("=" * 78)
    a = Matrix([[3, 1, 1], [1, 3, 1], [1, 1, 3], [2, 0, 1]])  # 4x3, not symmetric, not square
    print(f"  A ({a.shape[0]}x{a.shape[1]}) =\n  {a.pretty_print()}".replace("\n", "\n  "))

    u, sigma, vt = svd(a)
    reconstructed = reconstruct(u, sigma, vt)
    error = frobenius_norm(matrices_diff(a, reconstructed))
    print(f"\n  our singular values = {[round(v, 6) for v in singular_values_of(sigma)]}")
    print(f"  ||A - U Sigma V^T||_F = {error:.2e}  (should be ~0)")

    np_u, np_s, np_vt = np.linalg.svd(np.array(a.rows, dtype=float), full_matrices=False)
    print(f"  numpy singular values = {[round(v, 6) for v in np_s.tolist()]}")
    print(
        f"  max diff vs numpy      = {max(abs(o - t) for o, t in zip(singular_values_of(sigma), np_s.tolist())):.2e}"
    )

    print(
        "\n  interpreting the pieces:\n"
        "    V's columns (rows of V^T) are orthonormal directions in A's INPUT\n"
        "    space -- the directions a vector can point in that get mapped\n"
        "    cleanly, without rotating into some other direction.\n"
        "    U's columns are orthonormal directions in A's OUTPUT space -- where\n"
        "    each of those input directions actually lands.\n"
        "    Sigma's diagonal entries are the stretch factors: v_i maps to\n"
        f"    sigma_i * u_i. In short, A = U Sigma V^T says every linear map is\n"
        "    'rotate, then stretch along orthogonal axes, then rotate again.'"
    )

    plot_singular_values(
        {"A's singular values": singular_values_of(sigma)},
        "Conceptual SVD: singular value magnitudes",
        OUTPUT_DIR / "svd_1_conceptual_singular_values.png",
        logy=False,
    )
    print(f"\n  full-rank compress_matrix(A, {min(a.shape)}) reconstructs A exactly:")
    full_rank = compress_matrix(a, min(a.shape))
    print(f"    ||A - compress_matrix(A, full rank)||_F = {frobenius_norm(matrices_diff(a, full_rank)):.2e}")
    print()


def demo_low_rank_approximation():
    print("=" * 78)
    print("2. Low-rank approximation with compress_matrix(A, k)")
    print("=" * 78)
    # deliberately low-rank structure (2 outer products) plus a little noise
    import random

    rng = random.Random(0)
    u1 = [1, 2, 1, 0, -1, 1]
    u2 = [0, 1, -1, 2, 1, 0]
    v1 = [2, 1, 0, -1, 1]
    v2 = [1, -1, 2, 0, 1]
    rows = []
    for i in range(6):
        row = [u1[i] * v1[j] + 0.5 * u2[i] * v2[j] + rng.gauss(0, 0.05) for j in range(5)]
        rows.append(row)
    a = Matrix(rows)
    print(f"  A (6x5, built from 2 outer products + small noise)")

    r = min(a.shape)
    print(f"  {'k':>3}  {'||A - A_k||_F':>15}  {'relative error':>15}")
    full_norm = frobenius_norm(a)
    for k in range(1, r + 1):
        a_k = compress_matrix(a, k)
        error = frobenius_norm(matrices_diff(a, a_k))
        print(f"  {k:>3}  {error:>15.6f}  {error / full_norm:>14.4%}")
    print(
        "  -> error drops sharply after k=2 and is ~0 by full rank: this matrix's\n"
        "     real information content is essentially rank 2, and compress_matrix\n"
        "     recovers that almost exactly once k reaches it. The remaining\n"
        "     singular values are basically capturing only the injected noise."
    )
    print()


def demo_singular_value_decay():
    print("=" * 78)
    print("3. Singular value decay: structured vs. random data")
    print("=" * 78)
    import random

    rng = random.Random(1)
    n = 8

    # structured: built from just 2 outer products (same idea as challenge 2)
    u1 = [rng.uniform(-1, 1) for _ in range(n)]
    u2 = [rng.uniform(-1, 1) for _ in range(n)]
    v1 = [rng.uniform(-1, 1) for _ in range(n)]
    v2 = [rng.uniform(-1, 1) for _ in range(n)]
    structured = Matrix(
        [[3 * u1[i] * v1[j] + u2[i] * v2[j] for j in range(n)] for i in range(n)]
    )

    # unstructured: every entry independently random, no low-rank structure to exploit
    random_matrix = Matrix([[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)])

    _, sigma_structured, _ = svd(structured)
    _, sigma_random, _ = svd(random_matrix)

    print(f"  structured (rank-2-ish) singular values = {[round(v, 3) for v in singular_values_of(sigma_structured)]}")
    print(f"  random (full-rank) singular values       = {[round(v, 3) for v in singular_values_of(sigma_random)]}")

    plot_singular_values(
        {
            "structured (built from 2 outer products)": singular_values_of(sigma_structured),
            "unstructured random matrix": singular_values_of(sigma_random),
        },
        "Singular value decay: structured vs. random data",
        OUTPUT_DIR / "svd_3_decay_structured_vs_random.png",
    )
    print(
        "  -> the structured matrix's singular values fall off a cliff after the\n"
        "     first 2 (its true information content), while the random matrix's\n"
        "     values decay much more gently, since there's no low-dimensional\n"
        "     structure to concentrate the energy into a handful of directions.\n"
        "     Real-world data (images, sensor readings) usually looks much more\n"
        "     like the structured case -- which is exactly why low-rank\n"
        "     compression works so well on it in practice."
    )
    print()


def demo_image_compression():
    print("=" * 78)
    print("4. Image compression via SVD")
    print("=" * 78)
    image_path = OUTPUT_DIR / "svd_sample_image.png"
    generate_sample_image(image_path)
    pixels = load_grayscale_image(image_path)
    m, n = pixels.shape
    print(f"  loaded grayscale image: {image_path.name}  ({m}x{n} pixels)")

    ks = [1, 5, 10, 20, 50]
    full_norm = np.linalg.norm(pixels)

    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(3 * (len(ks) + 1), 3.4))
    axes[0].imshow(pixels, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("original")
    axes[0].axis("off")

    print(f"  {'k':>3}  {'rel. error':>11}  {'compression ratio':>18}")
    singular_values = np.linalg.svd(pixels, compute_uv=False)
    for i, k in enumerate(ks):
        reconstructed, _ = compress_image_numpy(pixels, k)
        rel_error = np.linalg.norm(pixels - reconstructed) / full_norm
        ratio = compression_ratio(m, n, k)
        print(f"  {k:>3}  {rel_error:>10.2%}  {ratio:>17.2f}x")
        axes[i + 1].imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
        axes[i + 1].set_title(f"k={k}")
        axes[i + 1].axis("off")

    fig.suptitle("Image reconstruction at increasing rank k")
    out_path = OUTPUT_DIR / "svd_4_image_compression.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved comparison grid -> {out_path}")

    plot_singular_values(
        {"image singular values": singular_values.tolist()},
        f"Singular value decay for {image_path.name}",
        OUTPUT_DIR / "svd_4_image_singular_values.png",
    )
    print(
        "  -> quality improves fast at first (k=1 to k=10 captures the coarse\n"
        "     gradient and shapes) then levels off -- the sharp text/edges are\n"
        "     what need the most additional singular values to render cleanly,\n"
        "     since sharp edges spread their energy across many more directions\n"
        "     than smooth regions do."
    )
    print()


if __name__ == "__main__":
    demo_conceptual_svd()
    demo_low_rank_approximation()
    demo_singular_value_decay()
    demo_image_compression()
