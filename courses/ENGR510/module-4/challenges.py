"""Least-squares challenge problems: line fitting, GPS trilateration,
Gram-Schmidt verification, and a design note on camera calibration.
"""

import math
import pathlib
import random

import matplotlib

matplotlib.use("Agg")  # headless: save PNGs instead of opening a window
import matplotlib.pyplot as plt
import numpy as np

from matrix import Matrix
from least_squares import dot, gram_schmidt, least_squares, norm

OUTPUT_DIR = pathlib.Path(__file__).parent


# -- Challenge 1: line fitting -----------------------------------------------
# Math: same normal-equations least squares built in least_squares.py.
# 100 noisy points is a heavily overdetermined 2-unknown (m, c) system --
# exactly the regime least squares is for.

def challenge_1():
    print("Challenge 1: fitting a line to 100 noisy points")
    random.seed(0)
    true_m, true_c = 2.5, -1.0
    xs = [random.uniform(0, 10) for _ in range(100)]
    ys = [true_m * x + true_c + random.gauss(0, 1.5) for x in xs]

    a_rows = [[x, 1] for x in xs]
    m_hat, c_hat = least_squares(Matrix(a_rows), ys)

    m_np, c_np = np.linalg.lstsq(np.array(a_rows), np.array(ys), rcond=None)[0]

    print(f"  true line:      y = {true_m}x + {true_c}")
    print(f"  our solver:     y = {m_hat:.4f}x + {c_hat:.4f}")
    print(f"  numpy lstsq:    y = {m_np:.4f}x + {c_np:.4f}")
    print(f"  agreement:      max diff = {max(abs(m_hat - m_np), abs(c_hat - c_np)):.2e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=15, alpha=0.6, label="noisy samples")
    line_x = [min(xs), max(xs)]
    ax.plot(
        line_x, [true_m * x + true_c for x in line_x],
        color="green", linestyle="--", label="true line"
    )
    ax.plot(
        line_x, [m_hat * x + c_hat for x in line_x],
        color="red", label=f"our fit: y={m_hat:.2f}x+{c_hat:.2f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Challenge 1: least-squares line fit vs. ground truth")
    ax.legend()
    out_path = OUTPUT_DIR / "challenge_1_line_fit.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {out_path}")
    print()


# -- Challenge 2: 2D GPS trilateration ---------------------------------------
# Math: each range measurement gives a circle equation
# (x-xi)^2 + (y-yi)^2 = ri^2, which is quadratic in the unknowns (x, y).
# Subtracting satellite 1's equation from each of the others cancels the
# x^2+y^2 term and leaves a LINEAR equation in (x, y) -- the standard
# trilateration linearization trick. Four satellites give 3 linear
# equations for 2 unknowns: an overdetermined system, solved by the same
# least_squares() routine used for line fitting.

def trilaterate(satellites, ranges):
    (x1, y1), r1 = satellites[0], ranges[0]
    rows, b = [], []
    for (xi, yi), ri in zip(satellites[1:], ranges[1:]):
        rows.append([2 * (x1 - xi), 2 * (y1 - yi)])
        b.append(ri**2 - r1**2 - xi**2 + x1**2 - yi**2 + y1**2)
    return least_squares(Matrix(rows), b)


def challenge_2():
    print("Challenge 2: 2D GPS trilateration from 4 noisy satellites")
    random.seed(1)
    satellites = [(-8, 6), (10, 9), (7, -8), (-9, -6)]
    true_position = (2.0, 3.0)
    noise_std = 0.3

    true_ranges = [math.hypot(true_position[0] - sx, true_position[1] - sy) for sx, sy in satellites]
    measured_ranges = [r + random.gauss(0, noise_std) for r in true_ranges]

    est_x, est_y = trilaterate(satellites, measured_ranges)
    error = math.hypot(est_x - true_position[0], est_y - true_position[1])
    print(f"  true position      = {true_position}")
    print(f"  estimated position = ({est_x:.4f}, {est_y:.4f})")
    print(f"  position error     = {error:.4f}")

    print("  effect of measurement noise (averaged over 200 trials each):")
    for std in (0.01, 0.1, 0.3, 1.0, 2.0):
        errors = []
        for _ in range(200):
            ranges = [
                math.hypot(true_position[0] - sx, true_position[1] - sy) + random.gauss(0, std)
                for sx, sy in satellites
            ]
            ex, ey = trilaterate(satellites, ranges)
            errors.append(math.hypot(ex - true_position[0], ey - true_position[1]))
        mean_error = sum(errors) / len(errors)
        print(f"    range noise std={std:>4}  ->  mean position error = {mean_error:.4f}")
    print(
        "  -> position error grows roughly in step with range noise: least\n"
        "     squares blends the (over-determined) noisy circles into one\n"
        "     estimate rather than trusting any single circle exactly, but\n"
        "     it can't remove noise, only average it out across measurements.\n"
        "     Geometry matters too -- satellites clustered close together or\n"
        "     nearly collinear make the linear system ill-conditioned, which\n"
        "     amplifies the same amount of range noise into a larger position\n"
        "     error (poor \"geometric dilution of precision\")."
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    for (sx, sy), ri in zip(satellites, measured_ranges):
        ax.add_patch(plt.Circle((sx, sy), ri, fill=False, linestyle="--", alpha=0.5, color="steelblue"))
        ax.plot(sx, sy, "^", color="black", markersize=8)
    ax.plot(*true_position, "go", markersize=10, label="true position")
    ax.plot(est_x, est_y, "rx", markersize=12, mew=3, label="estimated position")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect("equal")
    ax.set_title("Challenge 2: GPS trilateration geometry")
    ax.legend(loc="upper left")
    out_path = OUTPUT_DIR / "challenge_2_trilateration.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {out_path}")
    print()


# -- Challenge 3: Gram-Schmidt on random vectors -----------------------------

def challenge_3():
    print("Challenge 3: Gram-Schmidt on randomly generated vectors")
    random.seed(2)
    dimension, num_vectors = 5, 4
    random_vectors = [[random.uniform(-5, 5) for _ in range(dimension)] for _ in range(num_vectors)]
    basis = gram_schmidt(random_vectors)

    print("  unit length check:")
    all_unit = True
    for i, u in enumerate(basis):
        length = norm(u)
        ok = math.isclose(length, 1.0, abs_tol=1e-9)
        all_unit &= ok
        print(f"    |u{i}| = {length:.10f}  {'OK' if ok else 'FAIL'}")

    print("  pairwise orthogonality check:")
    all_orthogonal = True
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            d = dot(basis[i], basis[j])
            ok = math.isclose(d, 0.0, abs_tol=1e-9)
            all_orthogonal &= ok
            print(f"    u{i} . u{j} = {d:.2e}  {'OK' if ok else 'FAIL'}")

    print(f"  result: {'orthonormal basis confirmed' if all_unit and all_orthogonal else 'CHECK FAILED'}")
    print()


# -- Challenge 4: design note -------------------------------------------------

DESIGN_NOTE = """
Design Note: Why Camera Orientation Estimation Is a Least-Squares Problem
============================================================================

Setup: a camera observes 50 known stars. For an assumed camera orientation,
each star's true direction predicts where its image should land on the
sensor. The camera's orientation has only 3 degrees of freedom (e.g. a
rotation), but there are 50 stars x 2 pixel coordinates = 100 observed
values. Each observation is corrupted by a small, independent error:
centroiding error (the star's light blob isn't a perfect point, so its
measured center is off by a pixel or so) and lens distortion (a
systematic geometric warp that isn't perfectly modeled).

Why not just solve it exactly?

With 3 unknowns, only 2 stars (4 equations) are needed to pin down an
exact solution -- pick any 2 observations, solve for the orientation that
satisfies them exactly, and ignore the other 48. This is a real option,
and it's a bad one:

  1. It's arbitrary. Which 2 stars you pick determines the answer, and
     different pairs give different (and equally "exact") answers, since
     each pair carries its own noise.
  2. It throws away information. 48 independent noisy measurements of the
     same 3 unknowns contain far more information about the true
     orientation than any single pair does -- discarding them discards
     that information for no benefit.
  3. It's fragile. If the 2 chosen stars happen to have unusually large
     centroiding error, the "exact" solution is exactly wrong, with no
     way to detect or correct for it from those 2 points alone.

More fundamentally, an exact solution usually doesn't exist at all. With
100 noisy equations and only 3 unknowns, the system is massively
overdetermined -- there is generically no single orientation that
satisfies all 50 star observations simultaneously, because the noise on
each star is independent. Asking to "solve" the system exactly is asking
for a solution to a system that, in general, has none.

Why least squares is the right framing.

Least squares reframes the problem correctly: instead of solving
"predicted = observed" for every star, minimize the total squared
disagreement, sum_i ||predicted_i - observed_i||^2, over all 50 stars at
once. Every observation contributes to the estimate, weighted by how much
it disagrees with the current orientation guess, and no single star can
unilaterally force an exact (but noisy) fit.

This has a statistical justification, not just a practical one: if the
centroiding/distortion errors are independent and roughly Gaussian
(a reasonable assumption for small optical/measurement noise), the
least-squares solution is the maximum-likelihood estimate of the true
orientation, and by the Gauss-Markov theorem it is the best linear
unbiased estimator available. In other words, minimizing squared
residuals isn't a convenient approximation -- for this noise model, it is
provably the best use of all 50 noisy measurements.

Mechanically, this is the same machinery as the line-fit in Challenge 1:
build a residual for every observation, stack them into an overdetermined
system, and solve x_hat = (A^T A)^-1 A^T b (or the QR-based solver) for
the orientation parameters that make the residuals as small as possible
in aggregate. The only difference is that a full 3D camera-orientation
model is nonlinear (rotations don't add linearly), so in practice the
system is linearized around a current estimate and solved iteratively
(Gauss-Newton) -- but the underlying reason to use least squares at each
step is exactly the argument above: many noisy, mutually inconsistent
observations, no unknowns-count justification for trusting any small
subset exactly, and a noise model under which "minimize the sum of
squared errors" is the statistically correct thing to do.
"""


def challenge_4():
    print("Challenge 4: design note")
    print(DESIGN_NOTE)


if __name__ == "__main__":
    challenge_1()
    challenge_2()
    challenge_3()
    challenge_4()
