"""
Module 16, Part 1: noise-free reference trajectory.

Module 15's initial state [0, 0, 2 m/s, 170 deg], acceleration schedule (+0.2 / -0.2 / 0 m/s^2),
turn schedule (+0.2 / 0 / -0.2 rad/s) and discrete motion model (no drag, no noise):
    p_{k+1} = p_k + h v_k [cos psi_k, sin psi_k],  v_{k+1} = v_k + h a_k,  psi_{k+1} = wrap(psi_k + h omega_k)
Saved at every timestamp k = 0..N: p_ref, v_ref, psi_ref, and the interval inputs a_ref, omega_ref
(for k = N, where there is no following interval, the last inputs are repeated).
"""

import numpy as np

from nav_model import N_STEPS, X0_TRUE, accel_cmd, f_truth, yaw_cmd


def build_reference():
    x = np.zeros((N_STEPS + 1, 4))
    x[0] = X0_TRUE
    a = np.array([accel_cmd(k) for k in range(N_STEPS)] + [accel_cmd(N_STEPS - 1)])
    w = np.array([yaw_cmd(k) for k in range(N_STEPS)] + [yaw_cmd(N_STEPS - 1)])
    for k in range(N_STEPS):
        x[k + 1] = f_truth(x[k], a[k], w[k])
    return {"p": x[:, :2], "v": x[:, 2], "psi": x[:, 3], "a": a, "w": w}


if __name__ == "__main__":
    ref = build_reference()
    for t in (0, 5, 10, 15, 25, 40):
        k = int(round(t / 0.05))
        print(f"t = {t:2d} s: p = ({ref['p'][k, 0]:7.2f}, {ref['p'][k, 1]:7.2f}), v = {ref['v'][k]:.2f}, "
              f"psi = {np.degrees(ref['psi'][k]):7.1f} deg, a = {ref['a'][k]:+.1f}, w = {ref['w'][k]:+.1f}")
