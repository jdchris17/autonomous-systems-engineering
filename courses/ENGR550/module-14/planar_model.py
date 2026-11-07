"""
Module 14, Parts 1-2: planar vehicle model, sensors, Jacobians and finite-difference checks.

State x = [p_x, p_y, v, psi], psi in rad, counterclockwise from +x. Input u = [a, omega].
    f(x, u) = [p_x + h v cos psi,  p_y + h v sin psi,  v + h a,  wrap(psi + h omega)]
    wrap(theta) = ((theta + pi) mod 2 pi) - pi
Disturbances w = [w_a, w_omega] ~ N(0, diag(0.2^2, 0.02^2)) per interval enter speed and heading:
    x_{k+1} = f(x_k, u_k + w_k)  ->  L = [[0,0],[0,0],[h,0],[0,h]],  Q = L diag(0.04, 0.0004) L^T.

Process Jacobian (d f / d x, evaluated at the corrected estimate):
    d p_x+ / d v = h cos psi,  d p_x+ / d psi = -h v sin psi
    d p_y+ / d v = h sin psi,  d p_y+ / d psi =  h v cos psi
    F = [[1, 0, h cos psi, -h v sin psi], [0, 1, h sin psi, h v cos psi], [0, 0, 1, 0], [0, 0, 0, 1]]
    (wrap has slope 1 everywhere except at the cut, which a derivative never sees.)

Sensors:
    position  1 Hz  z = [p_x, p_y] + n, SD 0.5 m per axis   H_p = [[1,0,0,0],[0,1,0,0]]
    bearing   5 Hz  z = wrap(atan2(l_y - p_y, l_x - p_x) - psi + n), SD 3 deg, landmark (30, 20)
    With d_x = l_x - p_x, d_y = l_y - p_y, rho^2 = d_x^2 + d_y^2:
        d atan2(d_y, d_x) / d p_x = d_y / rho^2,   d / d p_y = -d_x / rho^2,   d / d psi = -1
        H_b = [d_y / rho^2, -d_x / rho^2, 0, -1]
    (bearing is undefined at the landmark itself, rho = 0.)

Truth: 40 s at h = 0.05 s from [0, 0, 2 m/s, 170 deg]; a = 0; omega = +0.2 (0-15 s), 0 (15-25 s),
-0.2 rad/s (25-40 s), chosen by integer tick. Truth and estimator use the same transition f.
"""

import numpy as np

H_STEP = 0.05
N_STEPS = 800                      # 40 s
X0_TRUE = np.array([0.0, 0.0, 2.0, np.radians(170.0)])
SIGMA_A, SIGMA_W = 0.2, 0.02
LANDMARK = np.array([30.0, 20.0])
POS_EVERY, BRG_EVERY = 20, 4        # 1 Hz, 5 Hz
SIGMA_POS, SIGMA_BRG = 0.5, np.radians(3.0)
SEED = 2

L_MAT = np.array([[0.0, 0.0], [0.0, 0.0], [H_STEP, 0.0], [0.0, H_STEP]])
Q = L_MAT @ np.diag([SIGMA_A**2, SIGMA_W**2]) @ L_MAT.T
R_POS = SIGMA_POS**2 * np.eye(2)
R_BRG = np.array([[SIGMA_BRG**2]])
H_POS = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


def wrap(theta):
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def yaw_rate(k):
    if k < 300:          # 0 <= t < 15
        return 0.2
    if k < 500:          # 15 <= t < 25
        return 0.0
    return -0.2          # 25 <= t < 40


def commands():
    return np.array([[0.0, yaw_rate(k)] for k in range(N_STEPS)])


def f(x, u, h=H_STEP):
    px, py, v, psi = x
    return np.array([px + h * v * np.cos(psi), py + h * v * np.sin(psi), v + h * u[0],
                     float(wrap(psi + h * u[1]))])


def F_jac(x, h=H_STEP):
    _, _, v, psi = x
    return np.array([[1, 0, h * np.cos(psi), -h * v * np.sin(psi)],
                     [0, 1, h * np.sin(psi), h * v * np.cos(psi)],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1.0]])


def g_bearing(x):
    dx, dy = LANDMARK[0] - x[0], LANDMARK[1] - x[1]
    return float(wrap(np.arctan2(dy, dx) - x[3]))


def H_bearing(x):
    dx, dy = LANDMARK[0] - x[0], LANDMARK[1] - x[1]
    rho2 = dx**2 + dy**2
    return np.array([[dy / rho2, -dx / rho2, 0.0, -1.0]])


def g_position(x):
    return x[:2].copy()


def draws(seed=SEED):
    """Standard-normal draws shared by every run: process noise, position noise, bearing noise."""
    rng = np.random.default_rng(seed)
    return {"w": rng.normal(size=(N_STEPS, 2)) * [SIGMA_A, SIGMA_W],
            "pos": rng.normal(size=(N_STEPS + 1, 2)) * SIGMA_POS,
            "brg": rng.normal(size=N_STEPS + 1) * SIGMA_BRG}


def simulate_truth(noise):
    u = commands()
    x = np.zeros((N_STEPS + 1, 4))
    x[0] = X0_TRUE
    for k in range(N_STEPS):
        x[k + 1] = f(x[k], u[k] + noise["w"][k])
    return x, u


def measurements(x, noise):
    """Dict tick -> list of (sensor, z), position first then bearing (fixed order at shared ticks)."""
    meas = {}
    for k in range(N_STEPS + 1):
        if k % POS_EVERY == 0:
            meas.setdefault(k, []).append(("position", g_position(x[k]) + noise["pos"][k]))
        if k % BRG_EVERY == 0:
            meas.setdefault(k, []).append(("bearing", float(wrap(g_bearing(x[k]) + noise["brg"][k]))))
    return meas


# ---------- Part 2: finite-difference checks ----------

def fd_jacobian(fun, x, eps=1e-6, angular_out=()):
    y0 = np.atleast_1d(fun(x))
    J = np.zeros((len(y0), len(x)))
    for j in range(len(x)):
        dx = np.zeros(len(x))
        dx[j] = eps
        diff = np.atleast_1d(fun(x + dx)) - np.atleast_1d(fun(x - dx))
        for i in angular_out:
            diff[i] = wrap(diff[i])            # wrap angular differences before dividing
        J[:, j] = diff / (2 * eps)
    return J


def jacobian_checks(n=500, seed=0):
    rng = np.random.default_rng(seed)
    worst_F = worst_H = 0.0
    tested = 0
    for _ in range(n):
        x = np.array([rng.uniform(-20, 50), rng.uniform(-20, 40), rng.uniform(0, 4), rng.uniform(-np.pi, np.pi)])
        if np.linalg.norm(LANDMARK - x[:2]) < 1.0:          # bearing undefined at the landmark
            continue
        u = np.array([0.0, rng.uniform(-0.3, 0.3)])
        worst_F = max(worst_F, np.max(np.abs(fd_jacobian(lambda s: f(s, u), x, angular_out=(3,)) - F_jac(x))))
        worst_H = max(worst_H, np.max(np.abs(fd_jacobian(g_bearing, x, angular_out=(0,)) - H_bearing(x))))
        tested += 1
    # states sitting on the +-pi cut: wrapping the finite difference is what makes these pass
    edge = np.array([5.0, 3.0, 2.0, np.pi - 0.2 * H_STEP - 1e-7])   # psi + h omega sits on the cut
    u = np.array([0.0, 0.2])
    raw = (f(edge + [0, 0, 0, 1e-6], u)[3] - f(edge - [0, 0, 0, 1e-6], u)[3]) / 2e-6
    wrapped = fd_jacobian(lambda s: f(s, u), edge, angular_out=(3,))[3, 3]
    return tested, worst_F, worst_H, raw, wrapped


def main():
    tested, wF, wH, raw, wrapped = jacobian_checks()
    print("Part 2: Jacobians vs central finite differences (eps = 1e-6), "
          f"{tested} random states (landmark neighbourhood excluded)")
    print(f"  max |F - F_fd| = {wF:.1e}    max |H_b - H_b,fd| = {wH:.1e}")
    print(f"  at psi + h omega = pi - 1e-7 (heading cut): d psi+/d psi unwrapped difference = {raw:.3e}, "
          f"wrapped = {wrapped:.6f} (true value 1)")
    noise = draws()
    x, _ = simulate_truth(noise)
    for t in (0, 15, 25, 40):
        k = int(round(t / H_STEP))
        print(f"  truth t = {t:2d} s: p = ({x[k, 0]:6.2f}, {x[k, 1]:6.2f}) m, v = {x[k, 2]:.2f} m/s, "
              f"psi = {np.degrees(x[k, 3]):7.1f} deg")
    meas = measurements(x, noise)
    print(f"  readings: {sum(s == 'position' for v in meas.values() for s, _ in v)} position, "
          f"{sum(s == 'bearing' for v in meas.values() for s, _ in v)} bearing")


if __name__ == "__main__":
    main()
