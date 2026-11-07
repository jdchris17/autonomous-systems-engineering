"""
Module 10, Part 1: the plant and the digital PID controller (shared by every experiment).

Plant (speed with linear drag):  v' = -c v + u + d,   c = 0.2 1/s,   v0 = 0.
Controller updates every h = 0.02 s and holds its output (zero-order hold); d is also held over
each interval, so the exact sampled plant is
    v_{k+1} = a v_k + (1 - a)/c (u_k + d_k),   a = e^{-c h}.

Controller, with y_k = v_k + n_k and e_k = r_k - y_k:
    u_raw,k = Kp e_k + I_k - Kd D_k
    D_raw,k = (y_k - y_{k-1}) / h                    derivative of the MEASUREMENT, not the error,
    D_k     = alpha D_{k-1} + (1 - alpha) D_raw,k    so a reference step gives no derivative kick;
    alpha   = tau_d / (tau_d + h), tau_d = 0.1 s     alpha = 0 means unfiltered. D_0 = 0.
    u_k     = clip(u_raw,k, u_min, u_max)            (no clip when limits are None)
    I_{k+1} = I_k + Ki h e_k                         after computing u_k
Conditional integration (anti-windup): skip the increment when
    (u_raw,k > u_max and e_k > 0)  or  (u_raw,k < u_min and e_k < 0),
i.e. whenever integrating would push the command further into the limit it is already beyond.

Everything is indexed by integer tick k (t_k = k h), so reference and disturbance switches
happen at exact samples.
"""

from collections import namedtuple

import numpy as np

C_DRAG = 0.2      # 1/s
H_STEP = 0.02     # s
TAU_D = 0.1       # s

Gains = namedtuple("Gains", "Kp Ki Kd")
GAINS = {"P": Gains(1.0, 0.0, 0.0), "PI": Gains(1.0, 0.5, 0.0), "PID": Gains(1.0, 0.5, 0.1)}


def simulate(gains, r, d, u_lim=None, anti_windup=False, noise=None, filtered=True):
    """r, d: arrays over ticks k = 0..N-1. noise: array (same length) or None.

    Returns a dict of logged arrays: t, v (true speed at t_k), y, r, e (= r - y), u_raw, u, I, D,
    plus v_end (speed at t_N).
    """
    n = len(r)
    a = np.exp(-C_DRAG * H_STEP)
    b = (1.0 - a) / C_DRAG
    alpha = TAU_D / (TAU_D + H_STEP) if filtered else 0.0
    noise = np.zeros(n) if noise is None else noise

    log = {key: np.zeros(n) for key in ("v", "y", "e", "u_raw", "u", "I", "D")}
    v, I, D, y_prev = 0.0, 0.0, 0.0, None
    for k in range(n):
        y = v + noise[k]
        e = r[k] - y
        if y_prev is not None:                      # D_0 = 0: no previous sample at k = 0
            D = alpha * D + (1.0 - alpha) * (y - y_prev) / H_STEP
        u_raw = gains.Kp * e + I - gains.Kd * D
        u = u_raw if u_lim is None else float(np.clip(u_raw, -u_lim, u_lim))
        for key, val in zip(log, (v, y, e, u_raw, u, I, D)):
            log[key][k] = val
        saturating_further = u_lim is not None and (
            (u_raw > u_lim and e > 0) or (u_raw < -u_lim and e < 0))
        if not (anti_windup and saturating_further):
            I += gains.Ki * H_STEP * e
        v = a * v + b * (u + d[k])
        y_prev = y
    log["t"] = np.arange(n) * H_STEP
    log["r"] = np.asarray(r, dtype=float)
    log["v_end"] = v
    return log


def ticks(T):
    return int(round(T / H_STEP))


def window(t, t0, t1):
    return (t >= t0 - 1e-9) & (t < t1 - 1e-9)


def p_steady_error(r, d, Kp, c=C_DRAG):
    """P-only equilibrium: 0 = -c v + Kp (r - v) + d  ->  e = r - v = (c r - d) / (c + Kp)."""
    return (c * r - d) / (c + Kp)
