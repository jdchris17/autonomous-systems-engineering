"""module-3 -- quaternion_kinematics.py  (Quaternion Attitude Kinematics)

Angular velocity enters. For our convention the attitude quaternion evolves as

    q_dot = 0.5 * q (x) [0, omega_B]

CONVENTION (must be stated -- this is where attitude bugs live):

  * q encodes R_IB : body -> inertial  (same as everywhere in this module)
  * omega_B is the body's angular velocity expressed in the BODY frame
  * Hamilton product, scalar-first quaternion

  If instead omega were given in the INERTIAL frame, the rule would be
      q_dot = 0.5 * [0, omega_I] (x) q
  and if q encoded I -> B the factor would flip sign / swap order. We use the
  first form and nothing else.

This file only defines q_dot and checks it three ways; the actual integration
is in integrate_omega.py.
"""

import numpy as np

from rotations import rot_x
from quaternions import (quaternion_multiply, quaternion_conjugate,
                         quaternion_normalize, axis_angle_to_quaternion,
                         quaternion_to_rotation_matrix)


def quaternion_derivative(q, omega_body):
    """q_dot = 0.5 * q (x) [0, omega_body]  (body-frame rate, q: body->inertial)."""
    omega_quat = np.concatenate([[0.0], np.asarray(omega_body, dtype=float)])
    return 0.5 * quaternion_multiply(q, omega_quat)


def skew(w):
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def main():
    rng = np.random.default_rng(3)
    q = quaternion_normalize(axis_angle_to_quaternion([1.0, 1.0, 0.0], 0.7))
    omega_B = np.array([0.10, -0.25, 0.40])          # rad/s, body frame
    qd = quaternion_derivative(q, omega_B)

    print("=" * 74)
    print("QUATERNION KINEMATICS   q_dot = 0.5 q (x) [0, omega_B]")
    print("=" * 74)
    print(f"q       = {np.array2string(q, precision=5)}")
    print(f"omega_B = {omega_B}  rad/s   (body frame)")
    print(f"q_dot   = {np.array2string(qd, precision=6)}")
    print()

    # 1. q_dot keeps ||q|| constant  ->  q_dot must be orthogonal to q
    print(f"1. d/dt ||q||^2 = 2 q.q_dot = {2 * q @ qd:.2e}   "
          f"-> norm is preserved, unit quaternion stays unit")

    # 2. recover omega_B back out:  omega = 2 * (q* (x) q_dot)_vec
    omega_back = 2.0 * quaternion_multiply(quaternion_conjugate(q), qd)[1:]
    print(f"2. recovered omega_B = {np.array2string(omega_back, precision=6)}   "
          f"(max err {np.abs(omega_back - omega_B).max():.2e})")

    # 3. consistency with rotation-matrix kinematics:  R_dot = R skew(omega_B)
    h = 1e-7
    R = quaternion_to_rotation_matrix(q)
    R_plus = quaternion_to_rotation_matrix(quaternion_normalize(q + h * qd))
    R_minus = quaternion_to_rotation_matrix(quaternion_normalize(q - h * qd))
    Rdot_num = (R_plus - R_minus) / (2 * h)
    Rdot_kin = R @ skew(omega_B)
    print(f"3. R_dot from q  vs  R skew(omega_B) : "
          f"{np.abs(Rdot_num - Rdot_kin).max():.2e}   "
          f"-> quaternion and matrix kinematics agree")

    print()
    print("If you use omega in the WRONG frame the error is silent and large:")
    omega_I = R @ omega_B                              # same rate, inertial coords
    qd_wrong = quaternion_derivative(q, omega_I)       # feeding inertial omega
    omega_chk = 2.0 * quaternion_multiply(quaternion_conjugate(q), qd_wrong)[1:]
    print(f"   feeding omega_I into the body-frame formula implies body rate "
          f"{np.array2string(omega_chk, precision=4)}")
    print(f"   which differs from the true {omega_B} by "
          f"{np.abs(omega_chk - omega_B).max():.3f} rad/s. Hence: document the "
          f"convention.")
    print("=" * 74)


if __name__ == "__main__":
    main()
