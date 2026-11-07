"""
Module 13, Parts 1-2: the complete loop. LQI controller driven by a Kalman speed estimate.

Truth (true drag c_true may differ from the nominal c the controller/estimator assume):
    v_{k+1} = a_t v_k + b_t (u_k + d_k + eta_k),  eta_k ~ N(0, 0.3^2),  v_0 = 0
Sensor:  y_k = v_k + n_k at measurement ticks only.
Estimator (scalar Kalman filter on the NOMINAL model, a = e^{-ch}, b = (1-a)/c, c = 0.2):
    prior v^-_0 = 0.5 m/s, P^-_0 = 1 (m/s)^2
    correct (only if a reading at tick k):  K = P^- / (P^- + R_f),  v^+ = v^- + K (y - v^-),
                                            P^+ = (1-K)^2 P^- + K^2 R_f   (Joseph form)
    predict with the APPLIED (clipped) command:  v^-_{k+1} = a v^+ + b u_k,  P^-_{k+1} = a^2 P^+ + Q_f
    Q_f = b^2 0.3^2 (the random acceleration enters through b),  R_f = sigma_n^2 of that sensor.
Controller (Module 12 LQI, K = [k_e, k_xi]; the R_cost = 1 design is used):
    u_raw = c r - k_e (v_hat - r) - k_xi xi,   u = clip(u_raw, -2, 2)
    xi_{k+1} = xi_k + h (r - v_hat), frozen when u_raw is beyond a limit and -k_xi dxi would push
    further beyond it (conditional integration).
Order at each tick: 1 correct (if reading) -> 2 command from v^+ (or v^- if no reading, which is
the same number since v^+ = v^- then) -> 3 integral update -> 4 propagate truth, predict estimator.
The random disturbance eta and the true speed are never available to the estimated-state controller.

Part 2, fixed-gain observer (noise-free, a reading every tick):
    v^+_{k+1} = v^-_{k+1} + l (y_{k+1} - v^-_{k+1}), v^-_{k+1} = a v^+_k + b u_k, and
    v_{k+1} = a v_k + b u_k, so e = v - v^+ obeys e_{k+1} = (1 - l)(v_{k+1} - v^-_{k+1}) = (1 - l) a e_k.
    The known command cancels exactly. With l = 0.2: |(1 - l) a| = 0.8 x 0.99601 = 0.7968 < 1, so the
    error shrinks by ~20% per tick (time constant ~0.09 s) regardless of what the controller does.
"""

from collections import namedtuple

import numpy as np

from lqi_design import A_PLANT, B_PLANT, C_DRAG, H_STEP, lqi_gain

T_END = 30.0
N = int(round(T_END / H_STEP))
SIGMA_ETA = 0.3
U_LIM = 2.0
R_DESIGN = 1.0
K_LQI = lqi_gain(R_DESIGN)[0]           # [k_e, k_xi]
Q_F = B_PLANT**2 * SIGMA_ETA**2
V0_PRIOR, P0_PRIOR = 0.5, 1.0
SEED = 2

Sensor = namedtuple("Sensor", "every sigma outage")     # outage: (t0, t1) or None
SENSORS = {
    "true-state benchmark": None,                          # controller uses the true speed
    "estimated, 10 Hz": Sensor(5, 0.1, None),
    "degraded, 2 Hz + outage": Sensor(25, 0.2, (12.0, 16.0)),
}


def plant(c):
    a = np.exp(-c * H_STEP)
    return a, (1 - a) / c


def draws(seed=SEED):
    """One process-noise sequence and one standard-normal measurement sequence on the base grid."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, SIGMA_ETA, N), rng.normal(0.0, 1.0, N)


def has_reading(sensor, k):
    if sensor is None:
        return True
    if k % sensor.every:
        return False
    if sensor.outage and sensor.outage[0] - 1e-9 <= k * H_STEP < sensor.outage[1] - 1e-9:
        return False
    return True


def simulate(sensor, eta, z_std, c_true=C_DRAG, load=0.0, t_load=12.0, K=K_LQI):
    """sensor None -> true-state feedback. Returns per-tick logs."""
    a_t, b_t = plant(c_true)
    k_e, k_xi = K
    k = np.arange(N)
    r = np.where(k >= int(round(1.0 / H_STEP)), 3.0, 0.0)
    d = np.where(k >= int(round(t_load / H_STEP)), load, 0.0)
    R_f = sensor.sigma**2 if sensor else 0.0

    keys = ("v", "v_hat", "P", "u_raw", "u", "integ", "reading", "gain")
    log = {key: np.zeros(N) for key in keys}
    v, xi = 0.0, 0.0
    v_hat, P = V0_PRIOR, P0_PRIOR
    for i in range(N):
        gain = 0.0
        if sensor is None:
            v_hat, P = v, 0.0
        elif has_reading(sensor, i):
            y = v + sensor.sigma * z_std[i]
            gain = P / (P + R_f)
            v_hat = v_hat + gain * (y - v_hat)
            P = (1 - gain)**2 * P + gain**2 * R_f
        u_raw = C_DRAG * r[i] - k_e * (v_hat - r[i]) - k_xi * xi
        u = float(np.clip(u_raw, -U_LIM, U_LIM))
        for key, val in zip(keys, (v, v_hat, P, u_raw, u, -k_xi * xi,
                                   float(sensor is None or has_reading(sensor, i)), gain)):
            log[key][i] = val
        dxi = H_STEP * (r[i] - v_hat)
        push = -k_xi * dxi
        if not ((u_raw > U_LIM and push > 0) or (u_raw < -U_LIM and push < 0)):
            xi += dxi
        v = a_t * v + b_t * (u + d[i] + eta[i])
        v_hat = A_PLANT * v_hat + B_PLANT * u
        P = A_PLANT**2 * P + Q_F
    log["t"] = k * H_STEP
    log["r"] = r
    return log


def fixed_gain_check(ell=0.2, e0=0.5, steps=25):
    factor = (1 - ell) * A_PLANT
    print("Part 2: fixed-gain observer, l = 0.2")
    print(f"  |(1 - l) a| = {abs(factor):.5f} < 1: stable; per-tick error factor {factor:.4f}, "
          f"time constant {-H_STEP / np.log(factor):.3f} s")
    # noise-free check with an arbitrary command sequence: the command cancels from the error
    rng = np.random.default_rng(0)
    v, vh = 0.0, e0 * -1          # true 0, estimate -0.5 -> e = +0.5
    errs = []
    for _ in range(steps):
        u = rng.normal(0, 1)
        v = A_PLANT * v + B_PLANT * u
        vh_pred = A_PLANT * vh + B_PLANT * u
        vh = vh_pred + ell * (v - vh_pred)
        errs.append(v - vh)
    pred = e0 * factor ** np.arange(1, steps + 1)
    print(f"  simulated error after 1, 5, 25 ticks: {errs[0]:.4f}, {errs[4]:.4f}, {errs[-1]:.2e}"
          f"   (predicted {pred[0]:.4f}, {pred[4]:.4f}, {pred[-1]:.2e}; max diff {np.max(np.abs(errs - pred)):.1e})")


def steady_kalman_gain(sensor):
    """Periodic steady state: run the covariance recursion alone, report the gain at each reading."""
    P, gains = P0_PRIOR, []
    for i in range(20 * N):
        if i % sensor.every == 0:
            g = P / (P + sensor.sigma**2)
            P = (1 - g)**2 * P + g**2 * sensor.sigma**2
            gains.append((g, np.sqrt(P), None))
        P_pred = A_PLANT**2 * P + Q_F
        if (i + 1) % sensor.every == 0:
            gains[-1] = (gains[-1][0], gains[-1][1], np.sqrt(P_pred))
        P = P_pred
    return gains[-1]


def main():
    fixed_gain_check()
    print(f"\nKalman settings: Q_f = b^2 (0.3)^2 = {Q_F:.3e} (m/s)^2 per tick (SD {np.sqrt(Q_F):.4f} m/s), "
          f"LQI K = [{K_LQI[0]:.4f}, {K_LQI[1]:.4f}] (R_cost = {R_DESIGN:g})")
    for name, s in SENSORS.items():
        if s is None:
            continue
        g, sd_post, sd_pre = steady_kalman_gain(s)
        print(f"  {name:<26} steady-state gain at each reading {g:.3f};  SD just before a reading "
              f"{sd_pre:.4f} m/s, just after {sd_post:.4f} m/s (sensor SD {s.sigma})")
    print("  Compare l = 0.2 every tick: the Kalman gain is set by the ratio of model to sensor "
          "uncertainty instead of chosen by hand.")


if __name__ == "__main__":
    main()
