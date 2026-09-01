"""module-3 -- check_rotations.py  (Coding Exercise: Rotation Matrices)

Numerically verify, for R_x(phi), R_y(theta), R_z(psi):

    R^T R  ~ I          (orthogonal  -> preserves lengths and angles)
    det(R) ~ +1         (proper      -> no reflection, right-handedness kept)

then rotate known vectors and check the geometry by hand.

"Why is a rotation matrix orthogonal?" -- because its columns are the images
of the orthonormal basis vectors, and a rigid rotation keeps them orthonormal.
Here that abstract fact becomes physical: it is exactly what lets the same
physical vector carry different coordinates in two frames without changing
length.
"""

import numpy as np

from rotations import rot_x, rot_y, rot_z, orthonormality_error, determinant_error


def near(a, b, tol=1e-12):
    return np.allclose(a, b, atol=tol)


def main():
    print("=" * 70)
    print("1. ORTHOGONALITY AND DETERMINANT OVER A RANGE OF ANGLES")
    print("=" * 70)
    angles = np.deg2rad([-179, -90, -33, 0, 17, 45, 90, 150, 179])
    print(f"{'axis':>5} {'angle':>10} {'max|R^T R - I|':>16} {'|det R - 1|':>14}")
    worst = 0.0
    for name, R_fn in (("x", rot_x), ("y", rot_y), ("z", rot_z)):
        for a in angles:
            R = R_fn(a)
            oe, de = orthonormality_error(R), determinant_error(R)
            worst = max(worst, oe, de)
            print(f"{name:>5} {np.rad2deg(a):>8.1f}deg {oe:>16.2e} {de:>14.2e}")
    print(f"\nworst error across all {3*len(angles)} matrices: {worst:.2e}  "
          f"(= floating-point noise)")

    print()
    print("=" * 70)
    print("2. ROTATE KNOWN VECTORS -- check by hand")
    print("=" * 70)
    checks = [
        ("R_z(90) . xhat  ->  yhat", rot_z(np.pi / 2) @ [1, 0, 0], [0, 1, 0]),
        ("R_z(90) . yhat  -> -xhat", rot_z(np.pi / 2) @ [0, 1, 0], [-1, 0, 0]),
        ("R_x(90) . yhat  ->  zhat", rot_x(np.pi / 2) @ [0, 1, 0], [0, 0, 1]),
        ("R_y(90) . zhat  ->  xhat", rot_y(np.pi / 2) @ [0, 0, 1], [1, 0, 0]),
        ("R_z(180) . (1,1,0) -> (-1,-1,0)",
         rot_z(np.pi) @ [1, 1, 0], [-1, -1, 0]),
    ]
    for label, got, want in checks:
        ok = "OK  " if near(got, want) else "FAIL"
        print(f"  [{ok}] {label:<33}  got ({got[0]:+.2f}, {got[1]:+.2f}, {got[2]:+.2f})")

    print()
    print("=" * 70)
    print("3. PROPERTIES THAT MAKE THEM PHYSICAL")
    print("=" * 70)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(3)
    R = rot_z(0.7) @ rot_y(-0.4) @ rot_x(1.1)

    print(f"  length preserved : |v| = {np.linalg.norm(v):.6f}   "
          f"|R v| = {np.linalg.norm(R @ v):.6f}")

    a, b = rng.standard_normal(3), rng.standard_normal(3)
    ang_before = np.degrees(np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    ra, rb = R @ a, R @ b
    ang_after = np.degrees(np.arccos(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb))))
    print(f"  angle preserved  : {ang_before:.4f} deg  ->  {ang_after:.4f} deg")

    inv_err = np.abs(R.T @ R - np.eye(3)).max()
    print(f"  inverse is transpose : |R^T R - I| = {inv_err:.2e}  "
          f"(so  v_B = R_IB^T v_I  costs nothing to compute)")

    lhs = rot_x(0.3) @ rot_y(0.5)
    rhs = rot_y(0.5) @ rot_x(0.3)
    print(f"  rotations do NOT commute : |R_x R_y - R_y R_x| = "
          f"{np.abs(lhs - rhs).max():.3f}  (order matters -- hence the need to "
          f"fix a convention)")
    print("=" * 70)


if __name__ == "__main__":
    main()
