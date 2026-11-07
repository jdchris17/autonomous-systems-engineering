"""
Module 7, Part 1: truth and the multi-rate sensor timeline.

Truth: 20 s at base interval h = 0.02 s, exact propagation, true acceleration held over each
interval:  u = +0.5 (0-4 s), 0 (4-12 s), -0.5 (12-16 s), 0 (16-20 s) m/s^2.

Streams (all on the base grid, generated from integer tick indices k, t = k h):
    accel     50 Hz  a_m,k = u_k + b_a + eta_k   SD 0.1 m/s^2   one per interval [t_k, t_k+1)
    velocity   5 Hz  z_v = v + n_v               SD 0.2 m/s     every 10 ticks
    position   1 Hz  z_p = p + n_p               SD 0.5 m       every 50 ticks

Part 3 edits (applied to the position stream only):
    outage:   no position readings for 8 <= t < 12 s   (ticks 400 <= k < 600)
    outliers: +8 m added at t = 4 s and t = 14 s        (ticks 200 and 700)

The noise draws are made once from the seed and reused, so every run sees the same truth and
the same noise; only the bias and the edits change.
"""

from collections import namedtuple

import numpy as np

H_STEP = 0.02      # s
N_INTERVALS = 1000  # 20 s
VEL_EVERY = 10     # ticks -> 5 Hz
POS_EVERY = 50     # ticks -> 1 Hz
SIGMA_ETA = 0.1    # m/s^2
SIGMA_V = 0.2      # m/s
SIGMA_P = 0.5      # m
OUTAGE_TICKS = (400, 600)      # [start, end) -> 8 <= t < 12 s
OUTLIER_TICKS = (200, 700)     # t = 4 s and 14 s
OUTLIER_SIZE = 8.0             # m
SEED = 2

Event = namedtuple("Event", "k t sensor value")


def F(h):
    return np.array([[1.0, h],
                     [0.0, 1.0]])


def G(h):
    return np.array([h**2 / 2, h])


def true_accel(k):
    """Acceleration held over interval k, chosen by integer tick (no float time comparisons)."""
    if k < 200:        # 0 <= t < 4
        return 0.5
    if k < 600:        # 4 <= t < 12
        return 0.0
    if k < 800:        # 12 <= t < 16
        return -0.5
    return 0.0         # 16 <= t <= 20


def simulate_truth():
    u = np.array([true_accel(k) for k in range(N_INTERVALS)])
    x = np.zeros((N_INTERVALS + 1, 2))
    for k in range(N_INTERVALS):
        x[k + 1] = F(H_STEP) @ x[k] + G(H_STEP) * u[k]
    return x, u


def draw_noise(seed=SEED):
    """One set of noise draws, reused by every run."""
    rng = np.random.default_rng(seed)
    return {
        "accel": rng.normal(0.0, SIGMA_ETA, N_INTERVALS),
        "velocity": rng.normal(0.0, SIGMA_V, N_INTERVALS // VEL_EVERY + 1),
        "position": rng.normal(0.0, SIGMA_P, N_INTERVALS // POS_EVERY + 1),
    }


def build_events(x, u, noise, bias=0.0, outage=True, outliers=True):
    """All sensor events, sorted by tick. Each event has a tick, timestamp, sensor and value."""
    events = []
    for k in range(N_INTERVALS):
        events.append(Event(k, k * H_STEP, "accel", u[k] + bias + noise["accel"][k]))
    for i, k in enumerate(range(0, N_INTERVALS + 1, VEL_EVERY)):
        events.append(Event(k, k * H_STEP, "velocity", x[k, 1] + noise["velocity"][i]))
    for i, k in enumerate(range(0, N_INTERVALS + 1, POS_EVERY)):
        if outage and OUTAGE_TICKS[0] <= k < OUTAGE_TICKS[1]:
            continue
        value = x[k, 0] + noise["position"][i]
        if outliers and k in OUTLIER_TICKS:
            value += OUTLIER_SIZE
        events.append(Event(k, k * H_STEP, "position", value))
    order = {"accel": 0, "velocity": 1, "position": 2}
    return sorted(events, key=lambda e: (e.k, order[e.sensor]))


def main():
    x, u = simulate_truth()
    noise = draw_noise()
    events = build_events(x, u, noise)

    print(f"Truth at t = 4, 12, 16, 20 s:")
    for k in (200, 600, 800, 1000):
        print(f"  t = {k * H_STEP:4.1f} s   p = {x[k, 0]:8.3f} m   v = {x[k, 1]:+.3f} m/s")

    counts = {s: sum(e.sensor == s for e in events) for s in ("accel", "velocity", "position")}
    print(f"\nEvent counts: {counts}  (position: 21 scheduled - 4 in outage = 17)")
    pos_times = [e.t for e in events if e.sensor == "position"]
    print(f"Position timestamps: {[round(t, 2) for t in pos_times]}")
    print("Outlier readings:")
    for e in events:
        if e.sensor == "position" and e.k in OUTLIER_TICKS:
            print(f"  t = {e.t:.1f} s  z_p = {e.value:.3f} m  truth = {x[e.k, 0]:.3f} m")
    print("\nFirst events:")
    for e in events[:4]:
        print(f"  k = {e.k:4d}  t = {e.t:5.2f} s  {e.sensor:<8} {e.value:+.4f}")


if __name__ == "__main__":
    main()
