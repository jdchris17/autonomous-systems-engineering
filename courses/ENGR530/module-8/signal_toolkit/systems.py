"""Discrete-time systems for the ENGR 530 toolkit."""

import numpy as np


def moving_average(x, length=5):
    """y[n] = (1/length) * sum_{k=0}^{length-1} x[n - k], with x[m] = 0 for m < 0.

    Causal: y[n] uses only the current and previous samples. The first
    length-1 outputs are a start-up transient (the window is not yet full).
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for k in range(length):
        y[k:] += x[: len(x) - k]
    return y / length
