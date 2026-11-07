"""
Module 15, Parts 2-3: inertial-propagation EKF with position and wheel-speed aiding.

At each tick k:
    1. if k > 0: propagate from k-1 with that interval's inertial reading (Jacobian at x^+_{k-1})
    2. aiding readings stamped at t_k, fixed order: wheel speed, then position (H recomputed
       each time; linear here, but kept general), each with its own R (position R follows the fix's
       SD: 0.5 m before 30 s, 2 m after)
    3. log the estimate and covariance
Configurations choose which aiding streams are used and whether biases are states:
    estimate_biases=True  -> 6-state model [p_x, p_y, v, psi, b_a, b_g]
    estimate_biases=False -> Module 14 four-state model [p_x, p_y, v, psi] fed the same biased
                             readings, with the same inertial-noise Q (nothing models the bias).
"""

import numpy as np

from nav_model import F6, H_POS, H_WHEEL, N_STEPS, Q6, f6, wrap

X0_OFFSET = np.array([1.0, -1.0, 0.3, np.radians(10.0)])        # Module 14 case A
P0_SD = np.array([2.0, 2.0, 0.5, np.radians(20.0), 0.1, 0.02])


def correct(x, P, nu, H, R):
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T
    x = x + K @ nu
    I_KH = np.eye(len(x)) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T
    return x, P, S


def run(x_true0, imu, aiding, use_position=True, use_wheel=True, estimate_biases=True):
    n = 6 if estimate_biases else 4
    x = np.r_[x_true0 + X0_OFFSET, 0.0, 0.0][:n]
    x[3] = wrap(x[3])
    P = np.diag(P0_SD[:n] ** 2)
    xs = np.zeros((N_STEPS + 1, n))
    Ps = np.zeros((N_STEPS + 1, n, n))
    log = []
    for k in range(N_STEPS + 1):
        if k > 0:
            xp = np.r_[x, np.zeros(6 - n)]               # 4-state: biases fixed at zero
            F = F6(xp)[:n, :n]
            x = f6(xp, imu[k - 1])[:n]
            P = F @ P @ F.T + Q6[:n, :n]
        for sensor, z, sd in sorted(aiding.get(k, []), key=lambda r: r[0] != "wheel"):
            if (sensor == "position" and not use_position) or (sensor == "wheel" and not use_wheel):
                continue
            H = (H_POS if sensor == "position" else H_WHEEL)[:, :n]
            R = sd**2 * np.eye(len(z))
            nu = z - H @ x
            prior = x.copy()
            x, P, S = correct(x, P, nu, H, R)
            log.append({"k": k, "t": round(k * 0.05, 4), "sensor": sensor, "nu": nu, "S": S,
                        "nis": float(nu @ np.linalg.solve(S, nu)), "prior": prior})
        x[3] = float(wrap(x[3]))
        xs[k], Ps[k] = x, P
    return xs, Ps, log
