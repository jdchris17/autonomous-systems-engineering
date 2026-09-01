"""module-3 -- check_quaternions.py  (Quaternion Coding Exercises, verification)

For quaternions built from axis-angle, verify:

    R(q)^T R(q) = I           (orthogonal)
    det R(q)    = +1          (proper rotation)

then the main point of the exercise:

    axis-angle  ->  q  ->  R(q)     must equal
    axis-angle  ->  R   directly (Rodrigues, rotations.axis_angle_to_R)

plus the algebra homomorphism  R(q1 (x) q2) = R(q1) R(q2),  the double cover
q ~ -q,  q* as the inverse rotation,  and rotate_vector == R @ v.
"""

import numpy as np

from rotations import (rot_z, axis_angle_to_R,
                       orthonormality_error, determinant_error)
from quaternions import (quaternion_multiply, quaternion_conjugate,
                         axis_angle_to_quaternion, quaternion_to_rotation_matrix,
                         rotate_vector)


def main():
    rng = np.random.default_rng(2)
    print("=" * 74)
    print("1. R(q) IS A VALID ROTATION, AND MATCHES THE DIRECT CONSTRUCTION")
    print("=" * 74)
    print(f"{'#':>3} {'angle(deg)':>11} {'|R^T R - I|':>14} {'|det-1|':>12} "
          f"{'|R(q) - Rodrigues|':>20}")
    worst_valid = worst_match = 0.0
    for i in range(8):
        axis = rng.standard_normal(3)
        angle = rng.uniform(-np.pi, np.pi)
        q = axis_angle_to_quaternion(axis, angle)
        Rq = quaternion_to_rotation_matrix(q)
        Rd = axis_angle_to_R(axis, angle)
        oe, de = orthonormality_error(Rq), determinant_error(Rq)
        md = np.abs(Rq - Rd).max()
        worst_valid = max(worst_valid, oe, de)
        worst_match = max(worst_match, md)
        print(f"{i:>3} {np.rad2deg(angle):>11.2f} {oe:>14.2e} {de:>12.2e} {md:>20.2e}")
    print(f"\nworst validity error : {worst_valid:.2e}")
    print(f"worst q-vs-direct gap : {worst_match:.2e}   -- the two routes agree")

    print()
    print("=" * 74)
    print("2. ELEMENTARY CHECK:  axis = z  ->  R(q) = rot_z(theta)")
    print("=" * 74)
    for deg in (30, 90, 175):
        th = np.deg2rad(deg)
        Rq = quaternion_to_rotation_matrix(axis_angle_to_quaternion([0, 0, 1], th))
        print(f"  theta = {deg:>4} deg : |R(q) - rot_z| = {np.abs(Rq - rot_z(th)).max():.2e}")

    print()
    print("=" * 74)
    print("3. ALGEBRA MATCHES GEOMETRY")
    print("=" * 74)
    qa = axis_angle_to_quaternion(rng.standard_normal(3), 0.9)
    qb = axis_angle_to_quaternion(rng.standard_normal(3), -1.7)

    lhs = quaternion_to_rotation_matrix(quaternion_multiply(qa, qb))
    rhs = quaternion_to_rotation_matrix(qa) @ quaternion_to_rotation_matrix(qb)
    print(f"  R(qa (x) qb) vs R(qa) R(qb)   : {np.abs(lhs - rhs).max():.2e}  "
          f"(quaternion product = rotation composition)")

    Rq = quaternion_to_rotation_matrix(qa)
    Rinv = quaternion_to_rotation_matrix(quaternion_conjugate(qa))
    print(f"  R(q*) vs R(q)^T               : {np.abs(Rinv - Rq.T).max():.2e}  "
          f"(conjugate = inverse rotation)")

    q_neg = -qa
    print(f"  R(q) vs R(-q)                 : "
          f"{np.abs(Rq - quaternion_to_rotation_matrix(q_neg)).max():.2e}  "
          f"(double cover: q and -q are the same rotation)")

    v = rng.standard_normal(3)
    print(f"  rotate_vector(q, v) vs R(q) v : "
          f"{np.abs(rotate_vector(qa, v) - Rq @ v).max():.2e}")
    print("=" * 74)


if __name__ == "__main__":
    main()
