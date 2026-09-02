"""module-4 -- state_space.py

The state-space form of a linear mechanical system, plus the same RK4
model/integrator/driver split used in the earlier modules.

A scalar second-order system

    m x'' + c x' + k x = F(t)

becomes, with state  x = [position, velocity],

    x' = A x + B u,      u = F(t)

    A = [[    0   ,     1    ],        B = [[  0  ],
         [ -k/m   ,  -c/m    ]]             [ 1/m ]]

The eigenvalues of A are the roots of  lambda^2 + (c/m) lambda + k/m = 0;
their real parts are the decay rates and their imaginary parts the damped
frequencies, so the physical damping behaviour is read straight off A.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def oscillator_AB(m, k, c):
    """State/input matrices for  m x'' + c x' + k x = F."""
    A = np.array([[0.0, 1.0],
                  [-k / m, -c / m]])
    B = np.array([[0.0],
                  [1.0 / m]])
    return A, B


def linear_dynamics(A, B, u=None):
    """Return f(t, x) = A x + B u(t)  for the driver below.

    u may be None (autonomous), a constant, or a callable u(t).
    """
    if u is None:
        return lambda t, x: A @ x
    if callable(u):
        return lambda t, x: A @ x + (B @ np.atleast_1d(u(t)))
    u_vec = B @ np.atleast_1d(u)
    return lambda t, x: A @ x + u_vec


def eigenvalues(A):
    return np.linalg.eigvals(A)


# ---------------------------------------------------------------------------
# Integrator + driver  (same shape as modules 2-3)
# ---------------------------------------------------------------------------
def rk4_step(f, t, x, dt):
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(f, x0, t_end, dt):
    """Fixed-step RK4. Returns times (N+1,) and states (N+1, n)."""
    n = int(round(t_end / dt))
    times = np.linspace(0.0, n * dt, n + 1)
    xs = np.empty((n + 1, len(x0)))
    xs[0] = np.asarray(x0, dtype=float)
    for i in range(n):
        xs[i + 1] = rk4_step(f, times[i], xs[i], dt)
    return times, xs


# ---------------------------------------------------------------------------
# Helpers for analysis
# ---------------------------------------------------------------------------
def zero_crossing_period(t, x):
    """Estimate the oscillation period from sign changes of x(t).

    Consecutive zero crossings are half a period apart.
    """
    s = np.sign(x)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx) < 2:
        return np.nan
    tc = t[idx] - x[idx] * (t[idx + 1] - t[idx]) / (x[idx + 1] - x[idx])
    return 2.0 * np.mean(np.diff(tc))
