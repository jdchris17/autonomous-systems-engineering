"""
Module 3, Parts 1-2: the Module 1 model in state-space form, plus model checks.

Two-state model (acceleration is a known command):
    x = [p, v]^T,   x_dot = A x + B u,   z = H x + n
    A = [[0, 1], [0, 0]],   B = [0, 1]^T,   H = [1, 0]
    A x + B u = [v, u]^T  ->  p_dot = v, v_dot = u;   H x = p  ->  z = p + n

Discretization with u held constant over [t_k, t_k + h]:
    v(t_k + s) = v_k + u s                                    (integrate v_dot = u)
    p(t_k + s) = p_k + v_k s + u s^2 / 2                      (integrate p_dot = v)
    at s = h:  x_{k+1} = F(h) x_k + G(h) u_k
    F(h) = [[1, h], [0, 1]],   G(h) = [h^2/2, h]^T

Units check (p in m, v in m/s, u in m/s^2, h in s):
    position row:  p_k [m] + h*v_k [s * m/s = m] + (h^2/2)*u_k [s^2 * m/s^2 = m]
    velocity row:  v_k [m/s] + h*u_k [s * m/s^2 = m/s]

General formulas: F(h) = e^{Ah}, G(h) = int_0^h e^{A tau} B dtau.
A^2 = 0, so e^{Ah} = I + Ah exactly and e^{A tau} B = [tau, 1]^T, whose integral is [h^2/2, h]^T.

Three-state model (acceleration is an unknown, persistent state):
    x = [p, v, a]^T,   A_CA = [[0,1,0],[0,0,1],[0,0,0]],   H_CA = [1, 0, 0]
    A_CA^3 = 0, so F_CA(h) = I + A_CA h + A_CA^2 h^2 / 2 = [[1, h, h^2/2], [0, 1, h], [0, 0, 1]]

Running this file prints numerical checks of all of the above, and the two
Part 3 checks: (1) matrix propagation reproduces the Module 1 trajectory, and
(2) the three-state model started at [0, 0, 1] matches the two-state model with u = 1.
"""

import numpy as np

# ---------------------------------------------------------------- settings (Module 1 trajectory)
DURATION = 10.0    # s
DT = 0.1           # s
P0 = 0.0           # m
V0 = 0.0           # m/s

# ---------------------------------------------------------------- continuous-time models
A = np.array([[0.0, 1.0],
              [0.0, 0.0]])
B = np.array([0.0, 1.0])
H = np.array([1.0, 0.0])

A_CA = np.array([[0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0]])
H_CA = np.array([1.0, 0.0, 0.0])


# ---------------------------------------------------------------- discrete-time models
def F(h):
    """Exact two-state transition matrix, e^{Ah} = I + Ah."""
    return np.array([[1.0, h],
                     [0.0, 1.0]])


def G(h):
    """Exact input matrix for acceleration held constant over the step."""
    return np.array([h**2 / 2, h])


def G_euler(h):
    """Forward-Euler input matrix: misses the position change from u within the step."""
    return np.array([0.0, h])


def F_CA(h):
    """Exact three-state (constant-acceleration) transition matrix."""
    return np.array([[1.0, h, h**2 / 2],
                     [0.0, 1.0, h],
                     [0.0, 0.0, 1.0]])


def expm_series(M, terms=20):
    """Matrix exponential by Taylor series (exact for nilpotent M once terms > its order)."""
    result = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, terms):
        term = term @ M / k
        result = result + term
    return result


def propagate(F_mat, G_vec, x0, u):
    """x_{k+1} = F x_k + G u_k for every u_k in u. Returns states at all len(u) + 1 samples."""
    x = np.zeros((len(u) + 1, len(x0)))
    x[0] = x0
    for k, uk in enumerate(u):
        x[k + 1] = F_mat @ x[k] + G_vec * uk
    return x


def propagate_autonomous(F_mat, x0, n_steps):
    """x_{k+1} = F x_k (no input)."""
    x = np.zeros((n_steps + 1, len(x0)))
    x[0] = x0
    for k in range(n_steps):
        x[k + 1] = F_mat @ x[k]
    return x


# ---------------------------------------------------------------- Module 1 reference (copied)
def accel_command(t):
    """Piecewise-constant acceleration schedule u(t) in m/s^2."""
    # Small tolerance so floating-point times such as 2.9999999 land in the right interval.
    t = t + 1e-9
    if t < 3.0:
        return 1.0
    if t < 6.0:
        return 0.0
    return -1.0


def simulate_module1(duration=DURATION, dt=DT, p0=P0, v0=V0):
    """Module 1 scalar updates, unchanged."""
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    p = np.zeros(n)
    v = np.zeros(n)
    u = np.zeros(n)

    p[0], v[0] = p0, v0
    for k in range(n):
        u[k] = accel_command(t[k])
        if k + 1 < n:
            p[k + 1] = p[k] + v[k] * dt + 0.5 * u[k] * dt**2
            v[k + 1] = v[k] + u[k] * dt
    return t, p, v, u


# ---------------------------------------------------------------- checks
def report(label, ok, detail=""):
    print(f"  [{'pass' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    return ok


def check_continuous_model(rng):
    print("Part 1a: multiplying out the continuous model")
    p, v, u = rng.normal(size=3)
    x = np.array([p, v])
    ok = report("A x + B u = [v, u]", np.allclose(A @ x + B * u, [v, u]))
    ok &= report("H x = p", np.isclose(H @ x, p))
    return ok


def check_discretization(hs=(0.1, 0.5, 1.0)):
    print("Part 1b: discretization")
    ok = report("A @ A = 0 (so e^{Ah} stops at I + Ah)", np.allclose(A @ A, 0))
    for h in hs:
        ok &= report(f"h = {h}: e^(Ah) by series = F(h) = I + Ah",
                     np.allclose(expm_series(A * h), F(h)) and np.allclose(F(h), np.eye(2) + A * h))

        # G(h) = int_0^h e^{A tau} B dtau by the trapezoid rule (the integrand is linear in tau,
        # so the trapezoid rule is exact up to roundoff).
        tau = np.linspace(0, h, 1001)
        integrand = np.array([expm_series(A * s) @ B for s in tau])
        G_num = np.sum((integrand[1:] + integrand[:-1]) / 2 * np.diff(tau)[:, None], axis=0)
        ok &= report(f"h = {h}: integral of e^(A tau) B = G(h) = [h^2/2, h]",
                     np.allclose(G_num, G(h)), f"numerical {np.round(G_num, 6)}")
    return ok


def check_constant_acceleration_model(hs=(0.1, 0.5, 1.0)):
    print("Part 2: three-state constant-acceleration model")
    A2 = A_CA @ A_CA
    ok = report("A_CA^2 != 0 but A_CA^3 = 0 (series stops at the h^2/2 term)",
                not np.allclose(A2, 0) and np.allclose(A2 @ A_CA, 0))
    for h in hs:
        ok &= report(f"h = {h}: e^(A_CA h) by series = F_CA(h)", np.allclose(expm_series(A_CA * h), F_CA(h)))
    return ok


def check_module1_trajectory():
    print("Check 1: F x + G u reproduces the Module 1 trajectory")
    t, p, v, u = simulate_module1()
    x = propagate(F(DT), G(DT), np.array([P0, V0]), u[:-1])
    err_p = np.max(np.abs(x[:, 0] - p))
    err_v = np.max(np.abs(x[:, 1] - v))
    ok = report("positions and velocities match at all 101 samples",
                err_p < 1e-12 and err_v < 1e-12, f"max |dp| = {err_p:.1e} m, max |dv| = {err_v:.1e} m/s")
    k = int(round(10.0 / DT))
    print(f"         final state from matrices: p = {x[k, 0]:.4f} m, v = {x[k, 1]:.4f} m/s (expected 17.5, -1)")
    return ok


def check_three_state_vs_two_state(T=10.0, hs=(0.1, 0.5, 1.0)):
    print("Check 2: three-state model from [0, 0, 1] vs two-state model with u = 1")
    ok = True
    for h in hs:
        n = int(round(T / h))
        x3 = propagate_autonomous(F_CA(h), np.array([0.0, 0.0, 1.0]), n)
        x2 = propagate(F(h), G(h), np.array([0.0, 0.0]), np.ones(n))
        diff = np.max(np.abs(x3[:, :2] - x2))
        ok &= report(f"h = {h}: p and v agree at every step, a stays 1",
                     diff < 1e-10 and np.allclose(x3[:, 2], 1.0),
                     f"max diff = {diff:.1e}, final p = {x3[-1, 0]:.4f} m, v = {x3[-1, 1]:.4f} m/s")
    return ok


def main():
    rng = np.random.default_rng(0)
    results = [
        check_continuous_model(rng),
        check_discretization(),
        check_constant_acceleration_model(),
        check_module1_trajectory(),
        check_three_state_vs_two_state(),
    ]
    print("\nAll model checks pass." if all(results) else "\nSOME CHECKS FAILED.")


if __name__ == "__main__":
    main()
