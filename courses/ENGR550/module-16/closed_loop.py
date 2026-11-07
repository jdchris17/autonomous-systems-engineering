"""
Module 16, Part 2: the closed planar tracking loop (navigation EKF + guidance + LQI speed control).

Physical vehicle (per interval k, inputs held):
    a_actual = u_a - c_true v + d_a + w_a,         w_a ~ N(0, 0.2^2)
    w_actual = yaw_gain u_w + d_w + w_w,           w_w ~ N(0, 0.02^2)
    x_{k+1}  = Module 15 discrete motion model with (a_actual, w_actual)
    IMU readings for the interval: a_m = a_actual + b_a + n_a, w_m = w_actual + b_g + n_g.
Aiding readings at t_k from this scenario's own truth: position 1 Hz (0.5 m, or 2 m late in the
sensor-failure case; none during its outage), wheel speed 10 Hz (0.1 m/s).
EKF: Module 15 six-state filter, prediction equations unchanged (it knows nothing about drag).

Guidance and control (from the corrected estimate, or the prediction when nothing arrived):
    dp = p_ref - p_hat;  e_par = [cos psi_ref, sin psi_ref] dp;  e_perp = [-sin psi_ref, cos psi_ref] dp
    v_target = clip(v_ref + k_par e_par, 0.5, 4)
    u_a,raw  = a_ref + c_nom v_target - k_e (v_hat - v_target) - k_xi xi,  u_a = clip(+-2)
    xi_{k+1} = xi_k + h (v_target - v_hat), frozen when u_a,raw is past a limit and the integral
               contribution -k_xi dxi would push further past it (conditional integration)
    psi_target = wrap(psi_ref + atan2(k_perp e_perp, v_ref))
    u_w = clip(w_ref + k_psi wrap(psi_target - psi_hat), +-u_w_lim)

Timing at each tick k: 1 correct with readings at t_k -> 2 commands -> 3 propagate truth ->
4 that interval's inertial readings -> 5 EKF prediction to t_{k+1}.
"""

import numpy as np

import config as C
from nav_ekf import P0_SD, X0_OFFSET, correct
from nav_model import (B_A_TRUE, B_G_TRUE, F6, H_POS, H_WHEEL, N_STEPS, POS_EVERY, Q6, SIGMA_POS,
                       SIGMA_WHEEL, WHEEL_EVERY, X0_TRUE, draws, f6, f_truth, wrap)
from reference import build_reference

REF = build_reference()


def in_window(t, win):
    return win is not None and win[0] - 1e-9 <= t < win[1] - 1e-9


def simulate(name, seed):
    cfg = C.scenario(name)
    nz = draws(seed)
    h = C.H_STEP
    n_t = N_STEPS + 1

    L = {key: np.zeros(n_t) for key in (
        "u_a_raw", "u_a", "u_w_raw", "u_w", "xi", "integ", "v_target", "psi_target",
        "e_par_hat", "e_perp_hat", "pos_sd", "min_eig", "asym")}
    L["truth"] = np.zeros((n_t, 4))
    L["est"] = np.zeros((n_t, 6))
    L["P"] = np.zeros((n_t, 6, 6))
    events = []

    x = X0_TRUE.copy()
    xh = np.r_[X0_TRUE + X0_OFFSET, 0.0, 0.0]
    xh[3] = wrap(xh[3])
    P = np.diag(P0_SD**2)
    xi = 0.0

    for k in range(n_t):
        t = k * h
        # 1. corrections with readings stamped at t_k (wheel, then position)
        if k % WHEEL_EVERY == 0:
            z = np.array([x[2] + nz["wheel"][k]])
            nu = z - H_WHEEL @ xh
            xh, P, S = correct(xh, P, nu, H_WHEEL, np.array([[SIGMA_WHEEL**2]]))
            events.append((t, "wheel", nu[0], np.nan, float(nu @ np.linalg.solve(S, nu))))
        if k % POS_EVERY == 0 and not in_window(t, cfg["outage"]):
            sd = cfg["pos_sd_late"] if cfg["late_from"] is not None and t >= cfg["late_from"] - 1e-9 else SIGMA_POS
            z = x[:2] + sd * nz["pos"][k]
            nu = z - H_POS @ xh
            xh, P, S = correct(xh, P, nu, H_POS, sd**2 * np.eye(2))
            events.append((t, "position", nu[0], nu[1], float(nu @ np.linalg.solve(S, nu))))
        xh[3] = float(wrap(xh[3]))

        # 2. commands
        if cfg["controller_input"] == "truth":
            p_c, v_c, psi_c = x[:2], x[2], x[3]
        else:
            p_c, v_c, psi_c = xh[:2], xh[2], xh[3]
        pr, vr, sr = REF["p"][k], REF["v"][k], REF["psi"][k]
        dp = pr - p_c
        e_par = np.cos(sr) * dp[0] + np.sin(sr) * dp[1]
        e_perp = -np.sin(sr) * dp[0] + np.cos(sr) * dp[1]
        v_tgt = float(np.clip(vr + C.K_PAR * e_par, *C.V_TARGET_LIM))
        u_a_raw = REF["a"][k] + C.C_NOM * v_tgt - C.K_E * (v_c - v_tgt) - C.K_XI * xi
        u_a = float(np.clip(u_a_raw, -C.U_A_LIM, C.U_A_LIM))
        psi_tgt = float(wrap(sr + np.arctan2(C.K_PERP * e_perp, vr)))
        u_w_raw = REF["w"][k] + C.K_PSI * float(wrap(psi_tgt - psi_c))
        u_w = float(np.clip(u_w_raw, -cfg["u_w_lim"], cfg["u_w_lim"]))

        # log the state at t_k (after corrections) and this tick's commands
        L["truth"][k], L["est"][k], L["P"][k] = x, xh, P
        for key, val in (("u_a_raw", u_a_raw), ("u_a", u_a), ("u_w_raw", u_w_raw), ("u_w", u_w), ("xi", xi),
                         ("integ", -C.K_XI * xi), ("v_target", v_tgt), ("psi_target", psi_tgt),
                         ("e_par_hat", e_par), ("e_perp_hat", e_perp),
                         ("pos_sd", np.sqrt(P[0, 0] + P[1, 1])),
                         ("min_eig", np.linalg.eigvalsh(0.5 * (P + P.T)).min()),
                         ("asym", np.max(np.abs(P - P.T)))):
            L[key][k] = val
        if k == N_STEPS:
            break

        # conditional integration, then update the speed integral
        dxi = h * (v_tgt - v_c)
        push = -C.K_XI * dxi
        if not ((u_a_raw > C.U_A_LIM and push > 0) or (u_a_raw < -C.U_A_LIM and push < 0)):
            xi += dxi

        # 3. propagate truth over [t_k, t_k+1)
        dist = in_window(t, cfg["dist_window"])
        a_act = u_a - cfg["c_true"] * x[2] + (cfg["d_a"] if dist else 0.0) + nz["w"][k, 0]
        w_act = cfg["yaw_gain"] * u_w + (cfg["d_w"] if dist else 0.0) + nz["w"][k, 1]
        x = f_truth(x, a_act, w_act)

        # 4. inertial readings for that interval
        um = np.array([a_act + B_A_TRUE + nz["imu"][k, 0], w_act + B_G_TRUE + nz["imu"][k, 1]])

        # 5. EKF prediction to t_{k+1}
        F = F6(xh)
        xh = f6(xh, um)
        P = F @ P @ F.T + Q6

    L["t"] = np.arange(n_t) * h
    L["events"] = events
    L["scenario"], L["seed"], L["cfg"] = name, seed, cfg
    return L


def score(L):
    """Navigation and tracking metrics. Cross-track is scored with TRUE position in the reference frame."""
    t, x, xh, P = L["t"], L["truth"], L["est"], L["P"]
    h = C.H_STEP
    nav_e = x[:, :2] - xh[:, :2]
    head_e = wrap(x[:, 3] - xh[:, 3])
    dp = REF["p"] - x[:, :2]
    along = np.cos(REF["psi"]) * dp[:, 0] + np.sin(REF["psi"]) * dp[:, 1]
    cross = -np.sin(REF["psi"]) * dp[:, 0] + np.cos(REF["psi"]) * dp[:, 1]
    track = np.hypot(dp[:, 0], dp[:, 1])
    out = (t >= 18 - 1e-9) & (t < 28 - 1e-9)
    late = t >= C.OUTAGE_SCORE_FROM - 1e-9
    cfg = L["cfg"]
    return {
        # navigation
        "nav_pos_rmse": np.sqrt(np.mean(np.sum(nav_e**2, axis=1))),
        "nav_head_rmse_deg": np.degrees(np.sqrt(np.mean(head_e**2))),
        "nav_speed_rmse": np.sqrt(np.mean((x[:, 2] - xh[:, 2])**2)),
        "pos_sd_max_18_28": np.max(L["pos_sd"][out]),
        "b_a_err": B_A_TRUE - xh[-1, 4], "b_a_sd": np.sqrt(P[-1, 4, 4]),
        "b_g_err": B_G_TRUE - xh[-1, 5], "b_g_sd": np.sqrt(P[-1, 5, 5]),
        # tracking
        "track_rmse": np.sqrt(np.mean(track**2)),
        "track_peak_from_18": np.max(track[late]),
        "track_peak": np.max(track),
        "along_rmse": np.sqrt(np.mean(along**2)),
        "cross_rmse": np.sqrt(np.mean(cross**2)),
        "cross_max": np.max(np.abs(cross)),
        "speed_err_rmse": np.sqrt(np.mean((x[:, 2] - REF["v"])**2)),
        "effort_a": h * np.sum(L["u_a"][:-1]**2),
        "effort_w": h * np.sum(L["u_w"][:-1]**2),
        "t_at_a_lim": h * np.sum(np.abs(L["u_a_raw"][:-1]) > C.U_A_LIM),
        "t_at_w_lim": h * np.sum(np.abs(L["u_w_raw"][:-1]) > cfg["u_w_lim"]),
        # numerical health
        "finite": bool(np.all(np.isfinite(xh)) and np.all(np.isfinite(P)) and np.all(np.isfinite(x))),
        "min_eig_P": np.min(L["min_eig"]),
        "max_asym_P": np.max(L["asym"]),
        "_along": along, "_cross": cross, "_track": track,
    }


if __name__ == "__main__":
    for name in C.SCENARIOS:
        s = score(simulate(name, C.DEMO_SEED))
        print(f"{name:<22} track RMSE {s['track_rmse']:.3f} m, peak from 18 s {s['track_peak_from_18']:.3f} m, "
              f"cross RMSE {s['cross_rmse']:.3f}, nav pos RMSE {s['nav_pos_rmse']:.3f}, "
              f"yaw-limit time {s['t_at_w_lim']:.2f} s, accel-limit time {s['t_at_a_lim']:.2f} s")
