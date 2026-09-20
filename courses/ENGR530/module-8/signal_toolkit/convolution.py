"""Discrete convolution written directly from the convolution sum."""

import numpy as np


def convolve_discrete(x, h):
    """y[n] = sum_k x[k] * h[n - k], for finite arrays x and h.

    Returns len(x) + len(h) - 1 samples. Terms where n - k falls outside
    0..len(h)-1 are zero (h is zero outside its support), so they are skipped.
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    n_out = len(x) + len(h) - 1
    y = np.zeros(n_out)
    for n in range(n_out):
        for k in range(len(x)):
            m = n - k
            if 0 <= m < len(h):
                y[n] += x[k] * h[m]
    return y
