"""
Module 16: frozen configuration. Everything a run depends on lives here; run_all.py snapshots it
to outputs/config_used.json alongside the results.

PASS TARGETS — recorded before the final suite was evaluated (adopting the proposed values):
    nominal position-tracking RMSE (true position vs timed reference, whole run)  < 1.0 m
    outage peak position-tracking error (max |p_ref - p_true| from 18 s to 40 s)   < 3.0 m
    The outage window is scored to the END of the run, not just 18-28 s, so the recovery after
    fixes resume and the 2 m fixes after 30 s are included. Both targets must hold on every one
    of the 10 seeds, not just on average.

Design history (see REPORT.md): starting gains k_par = 0.3, k_perp = 0.3, k_psi = 1. The
nominal-tracking check (seed 2) gave position-tracking RMSE 0.39 m with the true-state benchmark
at 0.03 m, so the remaining error is navigation error, which no guidance gain removes. No gain
adjustment was justified; the starting gains were frozen unchanged. Module 12 LQI gains (R_cost = 1)
were designed at h = 0.02 s and are reused at h = 0.05 s as the problem set directs.
"""

import numpy as np

from lqi_design import lqi_gain

# ---- timing and plant ----
H_STEP = 0.05
T_END = 40.0
C_NOM = 0.2                        # drag the controller assumes (feedforward)
SIGMA_WA, SIGMA_WW = 0.2, 0.02     # truth motion disturbances (Module 14/15)

# ---- controller (frozen) ----
K_E, K_XI = lqi_gain(1.0)[0]       # Module 12, R_cost = 1: [1.2208, -0.4939]
K_PAR = 0.3                        # 1/s, along-track progress gain
K_PERP = 0.3                       # 1/s, cross-track gain
K_PSI = 1.0                        # 1/s, heading gain
V_TARGET_LIM = (0.5, 4.0)          # m/s
U_A_LIM = 2.0                      # m/s^2
U_W_LIM = 0.5                      # rad/s

# ---- estimator: Module 15 EKF unchanged (Q from IMU noise, case-A initial offsets) ----

# ---- scenarios ----
SCENARIOS = {
    "true-state benchmark":   dict(controller_input="truth"),
    "nominal":                dict(),
    "disturbance":            dict(d_a=-0.3, d_w=0.03, dist_window=(15.0, 22.0)),
    "sensor failure":         dict(outage=(18.0, 28.0), pos_sd_late=2.0, late_from=30.0),
    "model mismatch":         dict(c_true=0.35, yaw_gain=0.8),
    "authority limit":        dict(u_w_lim=0.05),
}
DEFAULTS = dict(controller_input="estimate", c_true=0.2, yaw_gain=1.0, d_a=0.0, d_w=0.0,
                dist_window=None, outage=None, pos_sd_late=None, late_from=None, u_w_lim=U_W_LIM)

# ---- seeds ----
DEMO_SEED = 2                      # demonstration plots and the full scenario table
MULTI_SEEDS = list(range(10))      # nominal and sensor-failure runs over 10 seeds (0-9)

# ---- targets (frozen before evaluation) ----
TARGET_NOMINAL_RMSE = 1.0          # m
TARGET_OUTAGE_PEAK = 3.0           # m
OUTAGE_SCORE_FROM = 18.0           # s, to end of run


def scenario(name):
    cfg = dict(DEFAULTS)
    cfg.update(SCENARIOS[name])
    return cfg


def snapshot():
    """JSON-serializable copy of the frozen configuration."""
    return {
        "h": H_STEP, "T_end": T_END, "c_nom": C_NOM, "sigma_w": [SIGMA_WA, SIGMA_WW],
        "controller": {"k_e": float(K_E), "k_xi": float(K_XI), "k_par": K_PAR, "k_perp": K_PERP,
                       "k_psi": K_PSI, "v_target_lim": V_TARGET_LIM, "u_a_lim": U_A_LIM, "u_w_lim": U_W_LIM},
        "scenarios": {n: scenario(n) for n in SCENARIOS},
        "seeds": {"demo": DEMO_SEED, "multi": MULTI_SEEDS},
        "targets": {"nominal_position_rmse_m": TARGET_NOMINAL_RMSE,
                    "outage_peak_position_error_m": TARGET_OUTAGE_PEAK,
                    "outage_peak_scored_from_s": OUTAGE_SCORE_FROM},
    }
