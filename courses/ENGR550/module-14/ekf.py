"""
Module 14, Part 3: extended Kalman filter for the planar vehicle.

Predict (from the corrected estimate):
    F_k = F_jac(x^+_k)          (Jacobian evaluated BEFORE moving the mean)
    x^-_{k+1} = f(x^+_k, u_k),  P^-_{k+1} = F_k P^+_k F_k^T + Q,  heading wrapped
Correct, for each reading at tick k, in a fixed order (position, then bearing), each one using the
CURRENT prior (the output of the previous correction at the same tick):
    nu = z - g(x^-)              (bearing: nu = wrap(z - g(x^-)))
    H  = Jacobian of g at x^-    (recomputed before every correction)
    S = H P^- H^T + R,  K = P^- H^T S^{-1} (linear solve),  x^+ = x^- + K nu,  heading wrapped
    P^+ = (I - K H) P^- (I - K H)^T + K R K^T   (Joseph)
Module 5's correct() forms nu = z - H x, which is only right for a linear sensor; here the
innovation is computed from the nonlinear g and passed in explicitly (correct_innovation).
"""

import numpy as np

from planar_model import (H_POS, N_STEPS, Q, R_BRG, R_POS, F_jac, H_bearing, f, g_bearing,
                          g_position, wrap)


def correct_innovation(x_pred, P_pred, nu, H, R):
    """Measurement update given an explicit innovation (nonlinear prediction already applied)."""
    H = np.atleast_2d(H)
    S = H @ P_pred @ H.T + R
    K = np.linalg.solve(S, H @ P_pred).T            # P H^T S^-1 without forming S^-1
    x = x_pred + K @ np.atleast_1d(nu)
    I_KH = np.eye(len(x)) - K @ H
    P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
    return x, P, S, K


def run_ekf(x0, P0, u, meas, wrap_bearing_innovation=True):
    """Returns per-tick corrected estimates/covariances and a log of every bearing update."""
    xs = np.zeros((N_STEPS + 1, 4))
    Ps = np.zeros((N_STEPS + 1, 4, 4))
    blog = []
    x, P = x0.copy(), P0.copy()
    x[3] = wrap(x[3])
    for k in range(N_STEPS + 1):
        if k > 0:
            Fk = F_jac(x)
            x = f(x, u[k - 1])
            P = Fk @ P @ Fk.T + Q
        for sensor, z in meas.get(k, []):
            if sensor == "position":
                nu = z - g_position(x)
                x, P, _, _ = correct_innovation(x, P, nu, H_POS, R_POS)
            else:
                pred = g_bearing(x)
                raw = z - pred
                nu = float(wrap(raw)) if wrap_bearing_innovation else raw
                x_before = x.copy()
                x, P, S, K = correct_innovation(x, P, nu, H_bearing(x_before), R_BRG)
                blog.append({"k": k, "z": z, "pred": pred, "raw": raw, "nu": nu, "S": S[0, 0],
                             "nis": nu**2 / S[0, 0], "dx": x - x_before})
            x[3] = float(wrap(x[3]))
        xs[k], Ps[k] = x, P
    return xs, Ps, blog
