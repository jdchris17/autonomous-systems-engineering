"""module-3 -- rigid_body.py  (Major Project 1: Rigid-Body Simulator)

Full 6-DOF rigid body. 13-element state:

    x = [ r (3) | v (3) | q (4) | omega (3) ]
        r      : position of the centre of mass, INERTIAL frame
        v      : velocity of the centre of mass, INERTIAL frame
        q      : attitude quaternion, body -> inertial (scalar-first, Hamilton)
        omega  : angular velocity, BODY frame

Equations of motion:

    translational kinematics   r_dot     = v
    translational dynamics     v_dot     = F / m                 (F: inertial)
    rotational kinematics      q_dot     = 0.5 * q (x) [0, omega]
    rotational dynamics        omega_dot = I^-1 ( tau - omega x (I omega) )
                                                                 (tau: body)

The inertia tensor I is expressed in the body frame and is constant there.
Integrator: RK4, with the attitude quaternion renormalized after every step
(Module 3, Section 15).
"""

import numpy as np

from quaternions import quaternion_multiply, quaternion_normalize

# state slices
R, V, Q, W = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13)


def pack(r, v, q, w):
    return np.concatenate([r, v, q, w]).astype(float)


def unpack(x):
    return x[R], x[V], x[Q], x[W]


class RigidBody:
    def __init__(self, mass, inertia):
        self.mass = float(mass)
        self.inertia = np.asarray(inertia, dtype=float)
        if self.inertia.shape == (3,):
            self.inertia = np.diag(self.inertia)
        self.inertia_inv = np.linalg.inv(self.inertia)

    # -- forcing: override or pass callables to simulate() -----------------
    def force(self, t, x):
        return np.zeros(3)          # inertial frame

    def torque(self, t, x):
        return np.zeros(3)          # body frame

    # -- dynamics --------------------------------------------------------
    def derivative(self, t, x, force=None, torque=None):
        r, v, q, w = unpack(x)
        q = quaternion_normalize(q)

        F = (force or self.force)(t, x)
        tau = (torque or self.torque)(t, x)

        dr = v
        dv = F / self.mass
        dq = 0.5 * quaternion_multiply(q, np.concatenate([[0.0], w]))
        dw = self.inertia_inv @ (tau - np.cross(w, self.inertia @ w))
        return pack(dr, dv, dq, dw)

    # -- diagnostics ---------------------------------------------------
    def rotational_KE(self, x):
        w = x[W]
        return 0.5 * w @ (self.inertia @ w)

    def angular_momentum_body(self, x):
        return self.inertia @ x[W]

    def angular_momentum_inertial(self, x):
        from quaternions import quaternion_to_rotation_matrix
        return quaternion_to_rotation_matrix(x[Q]) @ (self.inertia @ x[W])


# ---------------------------------------------------------------------------
# integrator
# ---------------------------------------------------------------------------
def rk4_step(f, t, x, dt):
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next[Q] = quaternion_normalize(x_next[Q])
    return x_next


def simulate(body, x0, t_end, dt, force=None, torque=None):
    """Returns times (N+1,) and states (N+1, 13)."""
    f = lambda t, x: body.derivative(t, x, force=force, torque=torque)
    n = int(round(t_end / dt))
    times = np.linspace(0.0, n * dt, n + 1)
    states = np.empty((n + 1, 13))
    states[0] = pack(*unpack(np.asarray(x0, dtype=float)))
    states[0, Q] = quaternion_normalize(states[0, Q])
    for k in range(n):
        states[k + 1] = rk4_step(f, times[k], states[k], dt)
    return times, states


def initial_state(r=(0, 0, 0), v=(0, 0, 0), q=(1, 0, 0, 0), omega=(0, 0, 0)):
    return pack(np.array(r, float), np.array(v, float),
                np.array(q, float), np.array(omega, float))
