"""
Module 3 shared pieces: data generators, feature construction, estimators, and metrics.

Reuses Module 2's gradient_descent (and through it Module 1's helpers) via sys.path.

Conventions
-----------
* Polynomial features without the constant: P(x) = [x, x^2, ..., x^d].
* Standardization uses TRAINING statistics only: z_j = (x^j - mu_j) / s_j.
* Every polynomial fit is reported in RAW coefficients c = [c0, c1, ..., cd] with
  y = c0 + c1 x + ... + cd x^d, so different methods can be compared directly.
* Ridge objective (implemented here):
      J(b, beta) = (1/n) ||y - b - Z beta||^2 + alpha ||beta||^2      (intercept b NOT penalized)
* Lasso objective (scikit-learn):
      J(b, beta) = (1/(2n)) ||y - b - Z beta||^2 + alpha ||beta||_1   (intercept not penalized)
  The two alphas therefore live on different scales; each is tuned on its own grid.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-2"))
sys.path.insert(0, str(HERE.parent / "module-1"))

from gd_core import gradient_descent, mse_hessian  # noqa: E402,F401

OUT = HERE / "outputs"

# --------------------------------------------------------------------------- Experiment A truth
QUAD_TRUE = np.array([1.5, -0.8, 0.35])        # y = 1.5 - 0.8 x + 0.35 x^2
QUAD_NOISE = 1.0


def generate_quadratic(n, rng):
    x = rng.uniform(-5.0, 5.0, n)
    y = QUAD_TRUE[0] + QUAD_TRUE[1] * x + QUAD_TRUE[2] * x**2 + rng.normal(0.0, QUAD_NOISE, n)
    return x, y


# --------------------------------------------------------------------------- Experiment B truth
CAL_TRUE = np.array([0.15, 0.4])               # delta_u = 0.15 + 0.4 r^2  (pixels)


def calibration_truth(r):
    return CAL_TRUE[0] + CAL_TRUE[1] * r**2


def generate_calibration_data(n, rng):
    """Course starter generator (unchanged draw order)."""
    r = rng.uniform(0.0, 1.0, n)
    intensity = rng.uniform(1.0, 100.0, n)
    true_bias = 0.15 + 0.4 * r**2
    sigma = 0.05 + 0.8 / np.sqrt(intensity)
    noise = rng.normal(0.0, sigma)
    observed_bias = true_bias + noise
    return {"r": r, "intensity": intensity, "sigma": sigma, "true_bias": true_bias,
            "observed_bias": observed_bias}


CORRUPT_MEAN = 1.0    # px
CORRUPT_STD = 2.0     # px  (vs measurement sigma of 0.13 - 0.85 px)


def corrupt(data, fraction, rng):
    """Add gross errors to a random `fraction` of observations.

    Offsets ~ N(1.0, 2.0^2) px: much wider than the measurement noise, and skewed to one side
    (a misassociated neighbour or a blended centroid tends to pull the same way). The reported
    sigma is NOT changed - the corruption is unrecognized. Returns a copy with an 'outlier' mask.
    """
    d = {k: v.copy() for k, v in data.items()}
    n = d["r"].size
    k = int(round(fraction * n))
    bad = np.zeros(n, dtype=bool)
    bad[rng.choice(n, size=k, replace=False)] = True
    d["observed_bias"][bad] += rng.normal(CORRUPT_MEAN, CORRUPT_STD, k)
    d["outlier"] = bad
    return d


# --------------------------------------------------------------------------- features
def poly(x, degree):
    """[x, x^2, ..., x^degree] (no constant column)."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x**p for p in range(1, degree + 1)])


def with_intercept(F):
    return np.column_stack([np.ones(F.shape[0]), F])


class Standardizer:
    """z = (F - mu) / s with mu, s from the training features only."""

    def __init__(self, F_train):
        self.mu = F_train.mean(axis=0)
        self.s = F_train.std(axis=0)

    def transform(self, F):
        return (F - self.mu) / self.s

    def to_raw(self, b, beta):
        """Map y = b + sum beta_j z_j back to y = c0 + sum c_j F_j."""
        c = beta / self.s
        return np.concatenate([[b - c @ self.mu], c])


# --------------------------------------------------------------------------- estimators
def ols(X, y):
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta


def wls(X, y, w):
    """argmin sum_i w_i (y_i - X_i theta)^2, solved as OLS on sqrt(w)-scaled rows."""
    sw = np.sqrt(w)
    theta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return theta


def ridge(Z, y, alpha):
    """Ridge on standardized (training-centred) features, unpenalized intercept.

    Minimises (1/n)||y - b - Z beta||^2 + alpha ||beta||^2.  Because the columns of Z have zero
    training mean, b = mean(y) and beta solves (Z^T Z / n + alpha I) beta = Z^T (y - ybar) / n.
    Solved as the equivalent augmented least-squares problem
        [ Z            ] beta ~= [ y - ybar ]
        [ sqrt(n a) I  ]         [ 0        ]
    which avoids forming Z^T Z (whose condition number is the square of Z's).
    """
    n, p = Z.shape
    yc = y - y.mean()
    if alpha == 0:
        beta, *_ = np.linalg.lstsq(Z, yc, rcond=None)
    else:
        A = np.vstack([Z, np.sqrt(n * alpha) * np.eye(p)])
        beta, *_ = np.linalg.lstsq(A, np.concatenate([yc, np.zeros(p)]), rcond=None)
    return y.mean(), beta


# --------------------------------------------------------------------------- metrics
def mse(y, yhat):
    r = np.asarray(y) - np.asarray(yhat)
    return float(np.mean(r**2))


def rmse(y, yhat):
    return float(np.sqrt(mse(y, yhat)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def medae(y, yhat):
    return float(np.median(np.abs(np.asarray(y) - np.asarray(yhat))))
