"""module-3 -- sim_torque_axis.py  (Simulation Experiment 1: Torque About a Principal Axis)

Rectangular body, I1 != I2 != I3. Start at rest. Apply a constant torque about
the body z (a principal axis):

    tau = [0, 0, tau_z]   (body frame)

PREDICTION (before running):
Starting from rest with the torque along a principal axis, omega never leaves
that axis: omega x (I omega) stays zero, so the rotational dynamics decouple
and

    omega_z(t) = omega_z(0) + (tau_z / I_z) * t          (a straight line)
    omega_x(t) = omega_y(t) = 0

This is the basic validation case: linear spin-up, no coupling.
"""

import numpy as np
import matplotlib.pyplot as plt

from inertia_tensor import box_principal_moments
from rigid_body import RigidBody, simulate, initial_state, W

M, A, B, C = 6.0, 0.40, 0.30, 0.20
TAU_Z = 0.5
DT, T_END = 0.002, 10.0


def main():
    Ip = box_principal_moments(M, A, B, C)          # [I1, I2, I3], all distinct
    body = RigidBody(M, Ip)
    print("=" * 72)
    print("EXPERIMENT 1 -- constant torque about principal axis z")
    print("=" * 72)
    print(f"box m={M}, dims=({A},{B},{C})  ->  principal moments "
          f"I = {np.array2string(Ip, precision=5)}")
    print(f"tau = [0, 0, {TAU_Z}] N*m (body),  start at rest")
    print(f"predicted:  omega_z(t) = 0 + tau_z/I_z * t = {TAU_Z/Ip[2]:.6f} * t")
    print()

    torque = lambda t, x: np.array([0.0, 0.0, TAU_Z])
    t, X = simulate(body, initial_state(omega=(0, 0, 0)), T_END, DT, torque=torque)

    w = X[:, W]
    w_analytic = TAU_Z / Ip[2] * t

    err_z = np.abs(w[:, 2] - w_analytic).max()
    leak = np.abs(w[:, :2]).max()
    print(f"max |omega_z_num - omega_z_analytic| : {err_z:.2e}")
    print(f"max |omega_x|, |omega_y| (should be 0): {leak:.2e}")
    print(f"omega_z at t={T_END:g}s : num {w[-1,2]:.5f} rad/s,  "
          f"analytic {w_analytic[-1]:.5f} rad/s")
    print()
    print("Straight-line spin-up, no coupling into x or y. Integrator validated")
    print("against the one case with a closed-form answer.")
    print("=" * 72)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(t, w[:, 2], label="omega_z  (numerical)")
    ax1.plot(t, w_analytic, "k--", lw=1, label="omega_z(0) + tau_z/I_z * t")
    ax1.set_xlabel("t (s)"); ax1.set_ylabel("omega_z (rad/s)")
    ax1.set_title("Linear spin-up about a principal axis")
    ax1.legend(); ax1.grid(True)

    ax2.plot(t, w[:, 0], label="omega_x")
    ax2.plot(t, w[:, 1], label="omega_y")
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("rad/s")
    ax2.set_title(f"Off-axis rates stay at zero (max {leak:.1e})")
    ax2.legend(); ax2.grid(True)

    fig.tight_layout()
    fig.savefig("sim_torque_axis.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
