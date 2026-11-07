"""
Module 15, Parts 1-2: frames, sensor contracts, the bias-augmented model and its Jacobian.

Frame: local level world frame, position (p_x, p_y) in m, heading psi counterclockwise from +x,
forward speed v along the body axis. Level travel, forward-axis acceleration, no lateral slip.

Truth (Module 14 trajectory, h = 0.05 s, 40 s, start [0, 0, 2 m/s, 170 deg]):
    commanded a(t) = +0.2 (0-5 s), -0.2 (5-10 s), 0 after;  omega(t) = +0.2 (0-15 s), 0 (15-25 s),
    -0.2 rad/s (25-40 s); Module 14 disturbances w_a ~ N(0, 0.2^2), w_omega ~ N(0, 0.02^2) per interval.
    Actual interval inputs: a_k = a(t_k) + w_a,k,  omega_k = omega(t_k) + w_omega,k.

Sensor streams (all generated from the truth, indexed by integer tick k, t_k = k h):
    accelerometer  20 Hz  a_m,k = a_k + b_a + n_a,  SD 0.1 m/s^2   describes interval [t_k, t_k+1)
    gyro           20 Hz  w_m,k = omega_k + b_g + n_g, SD 0.01 rad/s describes interval [t_k, t_k+1)
    position fix    1 Hz  [p_x, p_y](t_k) + n_p,  SD 0.5 m/axis (2 m/axis from t = 30 s)
                          none for 18 <= t < 28 s (outage)                     describes t_k
    wheel speed    10 Hz  v(t_k) + n_v,  SD 0.1 m/s                           describes t_k
    true biases b_a = 0.05 m/s^2, b_g = 0.005 rad/s (constant).

Filter model, x = [p_x, p_y, v, psi, b_a, b_g]:
    f(x, u_m) = [p_x + h v cos psi, p_y + h v sin psi, v + h (a_m - b_a), wrap(psi + h (w_m - b_g)), b_a, b_g]
    F = [[1, 0, h cos psi, -h v sin psi,  0,  0],
         [0, 1, h sin psi,  h v cos psi,  0,  0],
         [0, 0, 1,          0,           -h,  0],
         [0, 0, 0,          1,            0, -h],
         [0, 0, 0,          0,            1,  0],
         [0, 0, 0,          0,            0,  1]]
    Q = L diag(0.1^2, 0.01^2) L^T, L (6x2) with L[v,0] = h, L[psi,1] = h: only the inertial
    SENSOR noise. The truth disturbances are already inside a_m and w_m (they are what the IMU
    measures), so adding their variance again would double count.
"""

import numpy as np

H_STEP = 0.05
N_STEPS = 800
X0_TRUE = np.array([0.0, 0.0, 2.0, np.radians(170.0)])
SIGMA_WA, SIGMA_WW = 0.2, 0.02            # truth motion disturbances
SIGMA_ACC, SIGMA_GYRO = 0.1, 0.01         # inertial sensor noise
B_A_TRUE, B_G_TRUE = 0.05, 0.005
SIGMA_POS, SIGMA_POS_LATE, T_POS_LATE = 0.5, 2.0, 30.0
SIGMA_WHEEL = 0.1
POS_EVERY, WHEEL_EVERY = 20, 2            # 1 Hz, 10 Hz
OUTAGE = (18.0, 28.0)
SEED = 2

H_POS = np.array([[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]])
H_WHEEL = np.array([[0, 0, 1.0, 0, 0, 0]])


def wrap(theta):
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def accel_cmd(k):
    return 0.2 if k < 100 else (-0.2 if k < 200 else 0.0)


def yaw_cmd(k):
    return 0.2 if k < 300 else (0.0 if k < 500 else -0.2)


def f_truth(x, a, w, h=H_STEP):
    px, py, v, psi = x
    return np.array([px + h * v * np.cos(psi), py + h * v * np.sin(psi), v + h * a, float(wrap(psi + h * w))])


def f6(x, um, h=H_STEP):
    px, py, v, psi, ba, bg = x
    return np.array([px + h * v * np.cos(psi), py + h * v * np.sin(psi), v + h * (um[0] - ba),
                     float(wrap(psi + h * (um[1] - bg))), ba, bg])


def F6(x, h=H_STEP):
    _, _, v, psi, _, _ = x
    return np.array([[1, 0, h * np.cos(psi), -h * v * np.sin(psi), 0, 0],
                     [0, 1, h * np.sin(psi), h * v * np.cos(psi), 0, 0],
                     [0, 0, 1, 0, -h, 0],
                     [0, 0, 0, 1, 0, -h],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1.0]])


L6 = np.zeros((6, 2))
L6[2, 0] = H_STEP
L6[3, 1] = H_STEP
Q6 = L6 @ np.diag([SIGMA_ACC**2, SIGMA_GYRO**2]) @ L6.T


def draws(seed=SEED):
    rng = np.random.default_rng(seed)
    return {"w": rng.normal(size=(N_STEPS, 2)) * [SIGMA_WA, SIGMA_WW],
            "imu": rng.normal(size=(N_STEPS, 2)) * [SIGMA_ACC, SIGMA_GYRO],
            "pos": rng.normal(size=(N_STEPS + 1, 2)),          # standard normal, scaled per epoch
            "wheel": rng.normal(size=N_STEPS + 1) * SIGMA_WHEEL}


def simulate(noise):
    """Truth states (N+1 x 4), inertial readings (N x 2), and timestamped aiding readings."""
    x = np.zeros((N_STEPS + 1, 4))
    x[0] = X0_TRUE
    imu = np.zeros((N_STEPS, 2))
    for k in range(N_STEPS):
        a = accel_cmd(k) + noise["w"][k, 0]
        w = yaw_cmd(k) + noise["w"][k, 1]
        imu[k] = [a + B_A_TRUE + noise["imu"][k, 0], w + B_G_TRUE + noise["imu"][k, 1]]
        x[k + 1] = f_truth(x[k], a, w)
    aiding = {}
    for k in range(N_STEPS + 1):
        t = k * H_STEP
        if k % WHEEL_EVERY == 0:
            aiding.setdefault(k, []).append(("wheel", np.array([x[k, 2] + noise["wheel"][k]]), SIGMA_WHEEL))
        if k % POS_EVERY == 0 and not (OUTAGE[0] - 1e-9 <= t < OUTAGE[1] - 1e-9):
            sd = SIGMA_POS_LATE if t >= T_POS_LATE - 1e-9 else SIGMA_POS
            aiding.setdefault(k, []).append(("position", x[k, :2] + sd * noise["pos"][k], sd))
    return x, imu, aiding


def fd_check(n=300, eps=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        x = np.array([rng.uniform(-30, 30), rng.uniform(-50, 10), rng.uniform(0, 4),
                      rng.uniform(-np.pi, np.pi), rng.normal(0, 0.1), rng.normal(0, 0.02)])
        um = np.array([rng.normal(0, 0.3), rng.normal(0, 0.2)])
        J = np.zeros((6, 6))
        for j in range(6):
            d = np.zeros(6)
            d[j] = eps
            diff = f6(x + d, um) - f6(x - d, um)
            diff[3] = wrap(diff[3])
            J[:, j] = diff / (2 * eps)
        worst = max(worst, np.max(np.abs(J - F6(x))))
    return worst


def main():
    print(f"Part 2: max |F6 - finite difference| over 300 random states = {fd_check():.1e}")
    noise = draws()
    x, imu, aiding = simulate(noise)
    for t in (0, 5, 10, 18, 28, 40):
        k = int(round(t / H_STEP))
        print(f"  truth t = {t:2d} s: p = ({x[k, 0]:7.2f}, {x[k, 1]:7.2f}) m, v = {x[k, 2]:.2f} m/s, "
              f"psi = {np.degrees(x[k, 3]):7.1f} deg")
    n_pos = sum(s == "position" for v in aiding.values() for s, _, _ in v)
    n_wh = sum(s == "wheel" for v in aiding.values() for s, _, _ in v)
    print(f"  streams: {len(imu)} inertial intervals, {n_pos} position fixes (41 - 10 in outage), {n_wh} wheel readings")
    print(f"  mean accel reading - actual = {np.mean(imu[:, 0] - [accel_cmd(k) + noise['w'][k, 0] for k in range(N_STEPS)]):.4f}"
          f" (bias {B_A_TRUE}),  mean gyro - actual = {np.mean(imu[:, 1] - [yaw_cmd(k) + noise['w'][k, 1] for k in range(N_STEPS)]):.5f} (bias {B_G_TRUE})")


if __name__ == "__main__":
    main()
