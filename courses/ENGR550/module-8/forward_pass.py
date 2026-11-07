"""
Module 8, Part 1: forward Kalman pass that saves the history the smoother needs.

Setup (from Module 7): two-state model x = [p, v], unbiased accelerometer as the known input,
position outage 8 <= t < 12 s, no outliers, gating off. With gating off and a fixed schedule
of readings, the filter is purely linear, so the RTS smoother applies exactly.

Saved at every base tick k = 0..N (t_k = k h), including ticks with no readings:
    x_pred[k], P_pred[k]   prediction before processing readings at t_k
                           (at k = 0 this is the prior x0, P0)
    x_corr[k], P_corr[k]   estimate after processing all readings at t_k
                           (equal to the prediction when there are none)
    F_steps[k]             transition from t_k to t_{k+1}  (k = 0..N-1)

Prediction:  x_pred[k+1] = F_k x_corr[k] + G_k a_m,k,   P_pred[k+1] = F_k P_corr[k] F_k^T + Q_k
The known acceleration enters x_pred, so the smoother never needs it separately.
Corrections use the Joseph form (kalman.correct), applied one reading at a time.
"""

from collections import namedtuple

import numpy as np

from kalman import correct
from sensor_timeline import (F, G, H_STEP, N_INTERVALS, SIGMA_ETA, SIGMA_P, SIGMA_V,
                             build_events, draw_noise, simulate_truth)

H = {"velocity": np.array([[0.0, 1.0]]), "position": np.array([[1.0, 0.0]])}
R = {"velocity": np.array([[SIGMA_V**2]]), "position": np.array([[SIGMA_P**2]])}
X0 = np.zeros(2)
P0 = np.diag([1.0, 1.0])

History = namedtuple("History", "t x_pred P_pred x_corr P_corr F_steps")


def make_events(outage=True, last_tick=N_INTERVALS, seed=None):
    """Module 8 data: bias 0, outage on, outliers off. last_tick truncates to a prefix recording."""
    x_true, u = simulate_truth()
    noise = draw_noise() if seed is None else draw_noise(seed)
    events = build_events(x_true, u, noise, bias=0.0, outage=outage, outliers=False)
    events = [e for e in events
              if e.k <= last_tick and not (e.sensor == "accel" and e.k >= last_tick)]
    return x_true, events


def forward_pass(events):
    """Run the filter over events and return the full history (ticks 0..last tick in events)."""
    last = max(e.k for e in events)
    accel = {e.k: e for e in events if e.sensor == "accel"}
    readings = {}
    for e in events:
        if e.sensor != "accel":
            readings.setdefault(e.k, []).append(e)

    n_ticks = last + 1
    x_pred = np.zeros((n_ticks, 2)); P_pred = np.zeros((n_ticks, 2, 2))
    x_corr = np.zeros((n_ticks, 2)); P_corr = np.zeros((n_ticks, 2, 2))
    F_steps = np.zeros((n_ticks - 1, 2, 2))

    x, P = X0.copy(), P0.copy()
    for k in range(n_ticks):
        if k > 0:
            a = accel[k - 1]
            dt = k * H_STEP - a.t                   # elapsed time from timestamps
            Fk, Gk = F(dt), G(dt)
            F_steps[k - 1] = Fk
            x = Fk @ x + Gk * a.value
            P = Fk @ P @ Fk.T + SIGMA_ETA**2 * np.outer(Gk, Gk)
        x_pred[k], P_pred[k] = x, P
        for e in readings.get(k, []):               # velocity then position (event order)
            x, P, _, _, _ = correct(x, P, e.value, H[e.sensor], R[e.sensor])
        x_corr[k], P_corr[k] = x, P
    t = np.arange(n_ticks) * H_STEP
    return History(t, x_pred, P_pred, x_corr, P_corr, F_steps)


def main():
    x_true, events = make_events()
    hist = forward_pass(events)
    n = len(hist.t)
    has_reading = {e.k for e in events if e.sensor != "accel"}
    no_reading = [k for k in range(n) if k not in has_reading]
    same = all(np.array_equal(hist.x_pred[k], hist.x_corr[k]) and
               np.array_equal(hist.P_pred[k], hist.P_corr[k]) for k in no_reading)
    print(f"Forward history: {n} timestamps (0 to {hist.t[-1]:.0f} s), {len(hist.F_steps)} transitions")
    print(f"  ticks with readings: {len(has_reading)},  without: {len(no_reading)}")
    print(f"  corrected == predicted at every tick without readings: {same}")
    for k in (0, 1, 400, 500, 600):
        print(f"  t = {hist.t[k]:5.2f} s  x- = [{hist.x_pred[k, 0]:7.3f}, {hist.x_pred[k, 1]:+.3f}]"
              f"  x+ = [{hist.x_corr[k, 0]:7.3f}, {hist.x_corr[k, 1]:+.3f}]"
              f"  SD_p- = {np.sqrt(hist.P_pred[k, 0, 0]):.3f}  SD_p+ = {np.sqrt(hist.P_corr[k, 0, 0]):.3f}")


if __name__ == "__main__":
    main()
