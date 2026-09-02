"""module-5 -- fields.py

Shared helpers for the electrostatics exercises: a 2-D grid, the potential of a
set of point charges, and two independent ways to get the electric field
(numerical gradient of V, and Coulomb's law directly).

Units are SI. The plane is the z = 0 slice; charges live in that plane.

Singularity handling: 1/r blows up at a charge. We (a) place charges at
non-grid locations so no sample sits exactly on one, and (b) clip the distance
at `r_min` (a small fraction of the cell size). That keeps V and E finite and
smooth for plotting without distorting anything more than a few cells away.
"""

import numpy as np

EPS0 = 8.8541878128e-12          # F/m
K_E = 1.0 / (4.0 * np.pi * EPS0)  # ~ 8.9875e9  N m^2 / C^2


def make_grid(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), n=241):
    """Return x, y (1-D) and X, Y (2-D meshgrid, shape (n, n))."""
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)          # X[i,j]=x[j], Y[i,j]=y[i]
    return x, y, X, Y


def _r_min(x, y, frac=0.5):
    """Clip radius: a fraction of the grid spacing."""
    return frac * min(x[1] - x[0], y[1] - y[0])


def potential(X, Y, charges, r_min=None, x=None, y=None):
    """Electrostatic potential V(x, y) from `charges` = [(q, xq, yq), ...]."""
    if r_min is None:
        r_min = _r_min(x if x is not None else X[0], y if y is not None else Y[:, 0])
    V = np.zeros_like(X, dtype=float)
    for q, xq, yq in charges:
        r = np.hypot(X - xq, Y - yq)
        V += K_E * q / np.maximum(r, r_min)
    return V


def field_coulomb(X, Y, charges, r_min=None, x=None, y=None):
    """Electric field (Ex, Ey) directly from Coulomb's law:  E = k q r_hat / r^2."""
    if r_min is None:
        r_min = _r_min(x if x is not None else X[0], y if y is not None else Y[:, 0])
    Ex = np.zeros_like(X, dtype=float)
    Ey = np.zeros_like(X, dtype=float)
    for q, xq, yq in charges:
        dx, dy = X - xq, Y - yq
        r = np.maximum(np.hypot(dx, dy), r_min)
        Ex += K_E * q * dx / r**3
        Ey += K_E * q * dy / r**3
    return Ex, Ey


def field_from_gradient(V, x, y):
    """E = -grad V, using centred finite differences (np.gradient)."""
    dVdy, dVdx = np.gradient(V, y, x)     # axis 0 is y, axis 1 is x
    return -dVdx, -dVdy


def magnitude(Ex, Ey):
    return np.hypot(Ex, Ey)
