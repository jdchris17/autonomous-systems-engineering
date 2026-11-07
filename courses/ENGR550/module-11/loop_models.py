"""
Module 11, Part 1: transfer functions of the speed loop, continuous and discrete.

Plant with uncertain drag and actuator effectiveness:  v' = -c v + beta u + d,  nominal c = 0.2, beta = 1.
    Continuous:  P(s) = beta / (s + c)      pole at -c, no finite zeros
                 C(s) = Kp + Ki / s = (Kp s + Ki) / s    pole at 0, zero at -Ki/Kp
    Discrete (h = 0.02 s, zero-order hold, as actually implemented in Module 10):
        v_{k+1} = a v_k + b u_k,  a = e^{-c h},  b = beta (1 - a) / c
            z-transform: (z - a) V = b U   ->   P(z) = b / (z - a)
        u_k = Kp e_k + I_k,  I_{k+1} = I_k + Ki h e_k
            (z - 1) I = Ki h E   ->   U = (Kp + Ki h / (z - 1)) E   ->   C(z) = Kp + Ki h / (z - 1)
        Command delay of m samples: L(z) = C(z) P(z) z^{-m}.
    Mapping: the plant pole -c maps to z = a = e^{-c h} (exact); the integrator pole 0 maps to z = 1.

Closed-loop characteristic polynomial (negative feedback, 1 + L = 0, multiplied through):
    z^m (z - a)(z - 1) + b [Kp (z - 1) + Ki h] = 0.

Controllers:
    baseline: Kp = 1, Ki = 0.5 (Module 10 starting PI)
    faster:   Kp = 3, Ki = 1.5 (baseline x 3: same zero at -0.5, ~3x the crossover frequency)
              The problem set names a "faster controller" without values in the part provided;
              this choice is an assumption, set in CONTROLLERS below.
"""

from collections import namedtuple

import numpy as np

H_STEP = 0.02
C_NOM, BETA_NOM = 0.2, 1.0
NYQUIST = np.pi / H_STEP          # 157.08 rad/s

PI = namedtuple("PI", "Kp Ki")
CONTROLLERS = {"baseline": PI(1.0, 0.5), "faster": PI(3.0, 1.5)}


def plant_coeffs(c=C_NOM, beta=BETA_NOM, h=H_STEP):
    a = np.exp(-c * h)
    return a, beta * (1.0 - a) / c


def delay_samples(delay_s, h=H_STEP):
    m = int(round(delay_s / h))
    assert abs(m * h - delay_s) < 1e-12, "delay must be an integer number of samples"
    return m


def C_z(z, ctrl, h=H_STEP):
    return ctrl.Kp + ctrl.Ki * h / (z - 1.0)


def P_z(z, c=C_NOM, beta=BETA_NOM):
    a, b = plant_coeffs(c, beta)
    return b / (z - a)


def L_z(z, ctrl, m=0, c=C_NOM, beta=BETA_NOM):
    return C_z(z, ctrl) * P_z(z, c, beta) * z ** (-m)


def char_poly(ctrl, m=0, c=C_NOM, beta=BETA_NOM, h=H_STEP):
    """Coefficients (highest power first) of z^m (z-a)(z-1) + b [Kp (z-1) + Ki h]."""
    a, b = plant_coeffs(c, beta)
    left = np.polymul(np.r_[1.0, np.zeros(m)], np.polymul([1.0, -a], [1.0, -1.0]))
    right = b * np.array([ctrl.Kp, ctrl.Ki * h - ctrl.Kp])
    poly = left.copy()
    poly[-2:] += right
    return poly


def max_pole(ctrl, m=0, c=C_NOM, beta=BETA_NOM):
    return np.max(np.abs(np.roots(char_poly(ctrl, m, c, beta))))
