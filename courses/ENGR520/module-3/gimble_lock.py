"""module-3 -- gimbal_lock.py  (Gimbal Lock)

Don't just read about it. In the 3-2-1 (roll phi, pitch theta, yaw psi)
convention the representation goes singular at theta = +/- 90 deg.

Two independent demonstrations, both as functions of theta approaching 90 deg:

  (A) Kinematic singularity.
      Body angular velocity and Euler-angle rates are linked by
          omega_B = A(phi, theta) @ [phi_dot, theta_dot, psi_dot]
      with det A(phi, theta) = cos(theta). Inverting to get the Euler rates
      needs 1/cos(theta), so cond(A) -> infinity as theta -> 90 deg: some
      body rates cannot be represented as finite Euler-angle rates.

  (B) Loss of an independent direction.
      Near theta = 90 deg, changing phi and changing psi produce almost the
      SAME change in R_IB: the roll axis and the yaw axis have become nearly
      collinear. d R/d phi and d R/d psi turn parallel.

Key point: the rigid body still has three rotational degrees of freedom. It
is the (phi, theta, psi) chart that has become singular -- a property of the
coordinates, not the physics.
"""

import numpy as np
import matplotlib.pyplot as plt

from rotations import rot_x, rot_y, rot_z
from euler_angles import euler_zyx_to_R, R_to_euler_zyx


# ---------------------------------------------------------------------------
# (A) Euler-rate kinematics
# ---------------------------------------------------------------------------
def euler_rate_to_omega_matrix(phi, theta):
    """A(phi, theta):  omega_B = A @ [phi_dot, theta_dot, psi_dot].

    Built from the 3-2-1 sequence: phi_dot about body x, theta_dot about the
    once-rotated y, psi_dot about the twice-rotated z, all expressed in body
    axes. Always finite; det(A) = cos(theta).
    """
    e_phi = np.array([1.0, 0.0, 0.0])
    e_theta = rot_x(phi).T @ np.array([0.0, 1.0, 0.0])
    e_psi = (rot_y(theta) @ rot_x(phi)).T @ np.array([0.0, 0.0, 1.0])
    return np.column_stack([e_phi, e_theta, e_psi])


# ---------------------------------------------------------------------------
# (B) sensitivity of R to phi and psi
# ---------------------------------------------------------------------------
def dR_dangle(which, phi, theta, psi, h=1e-6):
    """Central-difference derivative of R_IB w.r.t. one Euler angle."""
    a = {"phi": phi, "theta": theta, "psi": psi}
    ap, am = dict(a), dict(a)
    ap[which] += h
    am[which] -= h
    return (euler_zyx_to_R(ap["phi"], ap["theta"], ap["psi"])
            - euler_zyx_to_R(am["phi"], am["theta"], am["psi"])) / (2 * h)


def parallelism(theta, phi=np.deg2rad(20.0), psi=np.deg2rad(-35.0)):
    """|cos angle| between dR/dphi and dR/dpsi as flat 9-vectors. -> 1 at lock."""
    u = dR_dangle("phi", phi, theta, psi).ravel()
    w = dR_dangle("psi", phi, theta, psi).ravel()
    return abs(u @ w) / (np.linalg.norm(u) * np.linalg.norm(w))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
TEST_THETAS_DEG = [80.0, 85.0, 89.0, 89.9, 90.0]


def main():
    print("=" * 74)
    print("GIMBAL LOCK  --  3-2-1 roll/pitch/yaw, singular at pitch = 90 deg")
    print("=" * 74)

    phi0, psi0 = np.deg2rad(20.0), np.deg2rad(-35.0)
    print(f"holding roll = {np.rad2deg(phi0):.0f} deg, yaw = {np.rad2deg(psi0):.0f} deg; "
          f"sweeping pitch toward 90 deg\n")
    print(f"{'pitch':>9} {'det A = cos(th)':>16} {'cond(A)':>12} "
          f"{'|cos angle(dR/dphi, dR/dpsi)|':>30}")
    print("-" * 74)
    for tdeg in TEST_THETAS_DEG:
        th = np.deg2rad(tdeg)
        A = euler_rate_to_omega_matrix(phi0, th)
        cond = np.linalg.cond(A)
        par = parallelism(th, phi0, psi0)
        print(f"{tdeg:>7.1f}deg {np.linalg.det(A):>16.3e} {cond:>12.3e} {par:>30.6f}")

    print()
    print("Reading it: at pitch = 90 deg, A is singular (det = 0), so the Euler")
    print("rates needed to follow some body angular velocities are infinite, and")
    print("dR/dphi and dR/dpsi point along the SAME line -- roll and yaw now do")
    print("the same thing. Two of the three knobs have collapsed onto one.")

    # ---- recover angles right at the singularity --------------------------
    print()
    print("Feeding attitudes built at pitch = 90 deg back through R_to_euler_zyx:")
    th = np.deg2rad(90.0)
    for phi_deg, psi_deg in [(10, 0), (0, -10), (30, 20), (5, 5)]:
        R = euler_zyx_to_R(np.deg2rad(phi_deg), th, np.deg2rad(psi_deg))
        (rphi, rth, rpsi), sing = R_to_euler_zyx(R)
        print(f"   built (roll {phi_deg:>3}deg, yaw {psi_deg:>4}deg)  ->  recovered "
              f"(roll {np.rad2deg(rphi):>6.1f}deg, yaw {np.rad2deg(rpsi):>6.1f}deg)   "
              f"[yaw-roll] in = {psi_deg-phi_deg:>4}deg, out = "
              f"{np.rad2deg(rpsi-rphi):>6.1f}deg   [singular={sing}]")
    print("   At pitch = +90 deg only the combination (yaw - roll) is determined;")
    print("   the individual split between roll and yaw is gone.")

    # ---- the body itself is fine ----------------------------------------
    print()
    R_lock = euler_zyx_to_R(np.deg2rad(20), th, np.deg2rad(-35))
    print(f"But R_IB at the lock is still a perfectly good rotation: "
          f"|R^T R - I| = {np.abs(R_lock.T @ R_lock - np.eye(3)).max():.1e}, "
          f"det = {np.linalg.det(R_lock):.6f}.")
    print("The body has all three DOF. The chart lost one. That is the whole point.")
    print("=" * 74)

    _plot(phi0, psi0)


def _plot(phi0, psi0):
    thetas = np.deg2rad(np.linspace(0, 89.99, 400))
    cond = [np.linalg.cond(euler_rate_to_omega_matrix(phi0, t)) for t in thetas]
    par = [parallelism(t, phi0, psi0) for t in thetas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.semilogy(np.rad2deg(thetas), cond)
    for tdeg in TEST_THETAS_DEG[:-1]:
        ax1.axvline(tdeg, color="grey", ls=":", lw=0.8)
    ax1.set_xlabel("pitch theta (deg)")
    ax1.set_ylabel("cond(A)  (log scale)")
    ax1.set_title("Euler-rate kinematics blow up as theta -> 90 deg")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(np.rad2deg(thetas), par)
    ax2.set_xlabel("pitch theta (deg)")
    ax2.set_ylabel(r"$|\cos\angle(\partial R/\partial\phi,\ \partial R/\partial\psi)|$")
    ax2.set_title("Roll and yaw axes become collinear")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("gimbal_lock.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
