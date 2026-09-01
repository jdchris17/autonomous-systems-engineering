"""module-3 -- euler_angles.py  (Euler Angles)

"Euler angles" means nothing until you pin down four choices. This file makes
all four explicit and sticks to them everywhere:

  axis sequence            : Z-Y-X   (aerospace 3-2-1: yaw, then pitch, then roll)
  intrinsic vs extrinsic   : intrinsic (each rotation about the new body axis)
  active vs passive         : active elementary matrices (rotations.py)
  frame convention          : R_IB maps BODY coords -> INERTIAL coords,
                              v_I = R_IB @ v_B

With active elementary matrices the intrinsic 3-2-1 body->inertial DCM is

    R_IB(phi, theta, psi) = R_z(psi) @ R_y(theta) @ R_x(phi)

    phi   = roll  about body x
    theta = pitch about body y
    psi   = yaw   about body z

The columns of R_IB are the body axes x_B, y_B, z_B written in inertial
coordinates.
"""

import numpy as np

from rotations import rot_x, rot_y, rot_z, assert_rotation


# ---------------------------------------------------------------------------
# angles  <->  matrix
# ---------------------------------------------------------------------------
def euler_zyx_to_R(phi, theta, psi):
    """Body->inertial rotation for the 3-2-1 (yaw-pitch-roll) sequence."""
    return rot_z(psi) @ rot_y(theta) @ rot_x(phi)


def R_to_euler_zyx(R):
    """Recover (phi, theta, psi) from a body->inertial rotation matrix.

    Returns radians. At theta = +/-90 deg the split between phi and psi is
    arbitrary (gimbal lock); we then set phi = 0 and put everything in psi,
    and also return a `singular` flag.
    """
    cos_theta = np.hypot(R[0, 0], R[1, 0])          # = |cos(theta)|
    singular = cos_theta < 1e-9

    theta = np.arctan2(-R[2, 0], cos_theta)
    if not singular:
        phi = np.arctan2(R[2, 1], R[2, 2])
        psi = np.arctan2(R[1, 0], R[0, 0])
    else:
        phi = 0.0
        psi = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([phi, theta, psi]), singular


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def main():
    d = np.rad2deg
    r = np.deg2rad

    print("=" * 72)
    print("EULER ANGLES -- Z-Y-X (3-2-1), intrinsic, active, R_IB: body -> inertial")
    print("=" * 72)

    phi, theta, psi = r(25), r(-10), r(40)     # roll, pitch, yaw
    R = euler_zyx_to_R(phi, theta, psi)
    oe, de = assert_rotation(R, "R_IB")
    print(f"angles in : roll {d(phi):.1f} deg, pitch {d(theta):.1f} deg, "
          f"yaw {d(psi):.1f} deg")
    print(f"R_IB =\n{np.array2string(R, precision=4, suppress_small=True)}")
    print(f"valid rotation: |R^T R - I| = {oe:.1e}, |det - 1| = {de:.1e}")
    print()
    print("columns of R_IB = body axes in inertial coordinates:")
    for axis, col in zip("xyz", R.T):
        print(f"   {axis}_B -> ({col[0]:+.4f}, {col[1]:+.4f}, {col[2]:+.4f})_I")

    print()
    print("round-trip  angles -> R -> angles  over 20000 random attitudes")
    print("(|pitch| < 89.5 deg so we stay clear of gimbal lock):")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20000):
        a = np.array([rng.uniform(-np.pi, np.pi),
                      rng.uniform(r(-89.5), r(89.5)),
                      rng.uniform(-np.pi, np.pi)])
        back, sing = R_to_euler_zyx(euler_zyx_to_R(*a))
        # compare as rotations, not raw angles (avoids +/-pi wrap artefacts)
        err = np.abs(euler_zyx_to_R(*a) - euler_zyx_to_R(*back)).max()
        worst = max(worst, err)
    print(f"   worst matrix reconstruction error: {worst:.2e}")

    print()
    print("frame check: a vector fixed in the body, expressed both ways")
    v_B = np.array([1.0, 0.0, 0.0])            # points along body x (nose)
    v_I = R @ v_B
    print(f"   v_B = {v_B}   ->   v_I = R_IB v_B = "
          f"({v_I[0]:+.4f}, {v_I[1]:+.4f}, {v_I[2]:+.4f})")
    print(f"   back: R_IB^T v_I = {np.round(R.T @ v_I, 12)}   (recovers v_B)")
    print(f"   |v_B| = |v_I| = {np.linalg.norm(v_I):.6f}  -- same physical vector")
    print("=" * 72)


if __name__ == "__main__":
    main()
