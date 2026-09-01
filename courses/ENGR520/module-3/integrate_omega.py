"""module-3 -- integrate_omega.py  (Coding Exercise: Integrate Constant Angular Velocity)

Start simple:

    omega   = [0, 0, 1] rad/s   (constant, body frame)
    q(0)    = [1, 0, 0, 0]      (identity attitude)

Integrate q(t) with RK4 on  q_dot = 0.5 q (x) [0, omega]  and renormalize
q <- q / ||q|| every step to control drift. Then convert q(t) -> R(t), rotate
the body x-axis into the inertial frame, and plot its components.

Analytical prediction: omega is a pure z-spin at 1 rad/s, so
    R(t) = rot_z(t),   R(t) @ [1,0,0] = [cos t, sin t, 0].
The plotted components should land exactly on those curves.
"""

import numpy as np
import matplotlib.pyplot as plt

from quaternions import (quaternion_normalize, axis_angle_to_quaternion,
                         quaternion_to_rotation_matrix, quaternion_angle)
from quaternion_kinematics import quaternion_derivative

OMEGA_B = np.array([0.0, 0.0, 1.0])      # rad/s, body frame
Q0 = np.array([1.0, 0.0, 0.0, 0.0])
DT = 0.01
T_END = 8.0                              # a bit more than one 2*pi revolution


def euler_step(q, dt, omega):
    return q + dt * quaternion_derivative(q, omega)


def rk4_step(q, dt, omega):
    k1 = quaternion_derivative(q, omega)
    k2 = quaternion_derivative(q + 0.5 * dt * k1, omega)
    k3 = quaternion_derivative(q + 0.5 * dt * k2, omega)
    k4 = quaternion_derivative(q + dt * k3, omega)
    return q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(step, renormalize):
    n = int(round(T_END / DT))
    times = np.linspace(0.0, n * DT, n + 1)
    qs = np.empty((n + 1, 4))
    qs[0] = Q0
    for k in range(n):
        q = step(qs[k], DT, OMEGA_B)
        qs[k + 1] = quaternion_normalize(q) if renormalize else q
    return times, qs


def main():
    t, q = integrate(rk4_step, renormalize=True)              # the recommended one
    _, q_eul_raw = integrate(euler_step, renormalize=False)   # naive
    _, q_eul_fix = integrate(euler_step, renormalize=True)    # naive + the fix

    # analytic: constant z-spin -> q_true(t) = axis_angle(z, t)
    q_true = np.array([axis_angle_to_quaternion([0, 0, 1], ti) for ti in t])
    # resolve the +/-q sign ambiguity before comparing
    flip = np.sign(np.sum(q * q_true, axis=1))[:, None]
    q_err = np.abs(q - flip * q_true).max(axis=1)

    body_x_I = np.array([quaternion_to_rotation_matrix(qi) @ [1.0, 0.0, 0.0]
                         for qi in q])
    analytic_x = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])

    print("=" * 74)
    print("INTEGRATE CONSTANT ANGULAR VELOCITY  omega = [0,0,1] rad/s")
    print("=" * 74)
    print(f"dt = {DT}, t_end = {T_END} s ({int(T_END/DT)} RK4 steps), "
          f"renormalized each step")
    print()
    print(f"rotation angle at t_end : {np.rad2deg(quaternion_angle(q[-1])):.4f} deg")
    print(f"analytic  ({np.rad2deg(T_END % (2*np.pi)):.4f} deg, i.e. "
          f"{T_END:.2f} rad wrapped)")
    print(f"max |q - q_true| over the run          : {q_err.max():.2e}")
    print(f"max |bodyX_in_I - [cos t, sin t, 0]|   : "
          f"{np.abs(body_x_I - analytic_x).max():.2e}")
    print()
    print("why renormalize -- forward Euler on the same ODE:")
    norm_raw = np.linalg.norm(q_eul_raw, axis=1)
    norm_fix = np.linalg.norm(q_eul_fix, axis=1)
    print(f"   Euler, no renorm  : ||q|| at t_end = {norm_raw[-1]:.6f}  "
          f"(drift {norm_raw[-1]-1:+.2e}) -- each tangent step lands outside "
          f"the unit sphere")
    print(f"   Euler, renorm     : ||q|| at t_end = {norm_fix[-1]:.6f}  "
          f"(drift {norm_fix[-1]-1:+.2e}) -- the  q <- q/||q||  step pins it back")
    print()
    print("The rotated body x-axis traces (cos t, sin t, 0): a unit vector")
    print("going in a circle in the inertial xy-plane, exactly the constant")
    print("z-spin predicted analytically.")
    print("=" * 74)

    _plot(t, body_x_I, analytic_x, q, q_err, norm_raw, norm_fix)


def _plot(t, body_x_I, analytic_x, q, q_err, norm_raw, norm_fix):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    for k, lab in enumerate(["x", "y", "z"]):
        ax.plot(t, body_x_I[:, k], label=f"num {lab}")
        ax.plot(t, analytic_x[:, k], "k--", lw=0.8)
    ax.set_title("body x-axis expressed in the inertial frame\n"
                 "(dashed = cos t, sin t, 0)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("component")
    ax.legend(ncol=3, fontsize=8); ax.grid(True)

    ax = axes[0, 1]
    for k, lab in enumerate(["w", "x", "y", "z"]):
        ax.plot(t, q[:, k], label=lab)
    ax.set_title("attitude quaternion q(t)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("component")
    ax.legend(ncol=4, fontsize=8); ax.grid(True)

    ax = axes[1, 0]
    ax.semilogy(t, np.maximum(q_err, 1e-18))
    ax.set_title("|q(t) - q_analytic(t)|  (RK4 + renormalize)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("max abs error")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, norm_raw - 1.0, label="Euler, no renorm")
    ax.plot(t, norm_fix - 1.0, label="Euler, renorm")
    ax.set_title("||q|| - 1 : the renormalization step earns its keep")
    ax.set_xlabel("t (s)"); ax.set_ylabel("norm drift")
    ax.legend(fontsize=8); ax.grid(True)

    fig.tight_layout()
    fig.savefig("integrate_omega.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
