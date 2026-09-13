"""module-7 -- abcd_optics.py  (Hecht 6.2: Analytical Ray Tracing)

Paraxial ray optics as linear algebra. A ray is

    r = [y, theta]      y = height above the axis, theta = slope dy/dz

Each component is a 2x2 matrix acting on r, and a whole optical system is
just the product of those matrices, applied right to left in the order light
travels through them:

    M = M_n ... M_2 M_1          r_out = M r_in

Units: y and every distance (d, f) in the same length unit (mm here);
theta is a dimensionless slope, so d*theta and y/f both come out in that
same length unit -- nothing to convert.
"""

import numpy as np


def propagation_matrix(d):
    """Free-space propagation over distance d."""
    return np.array([[1.0, d],
                     [0.0, 1.0]])


def thin_lens_matrix(f):
    """Ideal thin lens of focal length f (positive = converging)."""
    return np.array([[1.0, 0.0],
                     [-1.0 / f, 1.0]])


def element_matrix(element):
    kind, value = element
    if kind == "space":
        return propagation_matrix(value)
    if kind == "lens":
        return thin_lens_matrix(value)
    raise ValueError(f"unknown element kind: {kind!r}")


def system_matrix(elements):
    """M = M_n ... M_2 M_1 for `elements` = [(kind, value), ...] in travel order."""
    M = np.eye(2)
    for el in elements:
        M = element_matrix(el) @ M
    return M


def propagate(r0, elements):
    """Apply the whole system to one ray; returns the output [y, theta]."""
    return system_matrix(elements) @ np.asarray(r0, dtype=float)


def trace_ray(y0, theta0, elements, samples_per_space=60):
    """Ray height through the system, sampled finely for plotting.

    Returns (z, y) arrays. Inside a 'space' the ray is a straight line
    (sampled at `samples_per_space` points); at a 'lens' the height is
    unchanged but the angle kinks instantaneously, so z does not advance.
    """
    z, r = 0.0, np.array([y0, theta0], dtype=float)
    zs, ys = [z], [r[0]]
    for kind, value in elements:
        if kind == "space":
            d = value
            step = np.linspace(0.0, d, samples_per_space)[1:]
            ys.extend(r[0] + step * r[1])
            zs.extend(z + step)
            z += d
            r = np.array([r[0] + d * r[1], r[1]])
        elif kind == "lens":
            r = thin_lens_matrix(value) @ r
            zs.append(z)
            ys.append(r[0])
        else:
            raise ValueError(f"unknown element kind: {kind!r}")
    return np.array(zs), np.array(ys)


def lens_positions(elements):
    """z (from the start) of every 'lens' element, for drawing the diagram."""
    z, zs = 0.0, []
    for kind, value in elements:
        if kind == "space":
            z += value
        elif kind == "lens":
            zs.append(z)
    return zs
