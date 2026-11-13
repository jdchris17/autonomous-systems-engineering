"""
Module 1 shared pieces: the synthetic sensor, the learning algorithm, and the error metric.

Data-generating process (Experiment A):
    y = 3 + 2 x + eps,   x ~ U(-5, 5),   eps ~ N(0, sigma^2),   sigma = 1.5 by default

Learning algorithm (Experiment B): ordinary least squares on the design matrix
    X = [1  x_i],   theta_hat = argmin ||X theta - y||^2   (solved with numpy.linalg.lstsq)

The fitting routine only ever receives (x, y) pairs. The TRUE_* constants are used to
generate data and to score the result afterward, never inside fit_linear().
"""

import numpy as np

SEED = 560

TRUE_INTERCEPT = 3.0
TRUE_SLOPE = 2.0
NOISE_STD = 1.5

X_LOW, X_HIGH = -5.0, 5.0
# Moments of x ~ U(-5, 5), used for exact (noise-free) risk calculations.
X_MEAN = 0.0
X_VAR = (X_HIGH - X_LOW) ** 2 / 12.0       # 25/3


def true_function(x):
    """The physical relationship, without measurement noise."""
    return TRUE_INTERCEPT + TRUE_SLOPE * x


def generate_data(n, rng, noise_std=NOISE_STD, quad_coef=0.0):
    """Draw n noisy sensor observations.

    quad_coef = 0 gives the course's starter generator exactly (same draw order:
    x first, then noise). A non-zero quad_coef adds quad_coef * x^2 to the truth,
    used only by the model-mismatch experiment.
    """
    x = rng.uniform(X_LOW, X_HIGH, size=n)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n)
    y = TRUE_INTERCEPT + TRUE_SLOPE * x + quad_coef * x**2 + noise
    return x, y


def starter_datasets():
    """The course's fixed experiment: rng(560), 30 training then 500 test observations."""
    rng = np.random.default_rng(SEED)
    x_train, y_train = generate_data(30, rng)
    x_test, y_test = generate_data(500, rng)
    return x_train, y_train, x_test, y_test


def design_matrix(x, degree=1):
    """Columns [1, x, x^2, ..., x^degree]. degree=1 is the straight-line model."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x**p for p in range(degree + 1)])


def fit_linear(x, y, degree=1):
    """Least-squares estimate of theta from observations only."""
    X = design_matrix(x, degree)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta


def predict(theta, x):
    return design_matrix(x, degree=len(theta) - 1) @ theta


def mse(y, y_hat):
    """Mean squared error, implemented directly from its definition."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    residual = y - y_hat
    return float(np.sum(residual**2) / residual.size)


def exact_excess_risk(theta):
    """E_x[(f_hat(x) - f_true(x))^2] for x ~ U(-5, 5), straight-line models only.

    With d0 = theta0 - 3 and d1 = theta1 - 2:
        E[(d0 + d1 x)^2] = d0^2 + 2 d0 d1 E[x] + d1^2 E[x^2] = d0^2 + d1^2 * 25/3
    Expected test MSE = sigma^2 + this term, so it isolates the part of the
    error that comes from imperfect learning rather than from sensor noise.
    """
    d0 = theta[0] - TRUE_INTERCEPT
    d1 = theta[1] - TRUE_SLOPE
    ex2 = X_VAR + X_MEAN**2
    return float(d0**2 + 2 * d0 * d1 * X_MEAN + d1**2 * ex2)
