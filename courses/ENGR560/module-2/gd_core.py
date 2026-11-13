"""
Module 2 core: the gradient-descent engine (Experiment A) and its stochastic extension (Experiment E).

Objective (mean squared error, any number of parameters p):
    J(theta) = (1/N) ||X theta - y||^2
    grad J   = (2/N) X^T (X theta - y)
    Hessian  = (2/N) X^T X               (constant: J is an exact quadratic)

Update:  theta_{k+1} = theta_k - eta * grad J(theta_k)

Stopping criterion (checked before each update):
    converged       ||grad J(theta_k)||_2 <= tolerance. The gradient is exactly zero at the
                    least-squares minimiser, so a small gradient norm means a stationary point.
                    For a quadratic, ||theta - theta*|| <= ||grad|| / lambda_min(H).
    max_iterations  the update budget ran out first.
    nonfinite       the loss, gradient or parameters became inf/NaN (divergence/overflow).

Module 1's data generator and fit_linear() are reused, not copied: module-1 is put on sys.path.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from learning_core import (NOISE_STD, SEED, TRUE_INTERCEPT, TRUE_SLOPE,  # noqa: E402,F401
                           design_matrix, fit_linear, generate_data, mse, predict, starter_datasets)

CONVERGED = "converged"
MAX_ITER = "max_iterations"
NONFINITE = "nonfinite"


# --------------------------------------------------------------------------- loss and gradient
def mse_loss(X, y, theta):
    """J(theta) = mean((X theta - y)^2)."""
    r = X @ theta - y
    return float(r @ r / y.size)


def mse_gradient(X, y, theta):
    """Analytical gradient: (2/N) X^T (X theta - y)."""
    return (2.0 / y.size) * (X.T @ (X @ theta - y))


def mse_hessian(X):
    return (2.0 / X.shape[0]) * (X.T @ X)


# --------------------------------------------------------------------------- Experiment A engine
def gradient_descent(X, y, theta_initial, learning_rate, max_iterations=1000, tolerance=1e-8,
                     loss_fn=mse_loss, grad_fn=mse_gradient):
    """Full-batch gradient descent.

    Returns (theta, loss_history, theta_history, info).
      theta_history[k] is the k-th iterate (theta_history[0] is a copy of the start) and
      loss_history[k] = J(theta_history[k]); both have the same length.
      info = {"status", "iterations" (updates performed), "grad_norm" (at the final theta)}.
    theta_initial is never modified.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = np.array(theta_initial, dtype=float).copy()

    loss_history = []
    theta_history = [theta.copy()]
    status = MAX_ITER
    grad_norm = np.nan
    iterations = 0

    with np.errstate(over="ignore", invalid="ignore"):
        for iterations in range(max_iterations + 1):
            # 1. loss and 2. gradient at the current iterate
            loss = loss_fn(X, y, theta)
            grad = grad_fn(X, y, theta)
            loss_history.append(loss)
            grad_norm = float(np.linalg.norm(grad))

            # 5. checks (numerical health first, then convergence, then budget)
            if not (np.isfinite(loss) and np.all(np.isfinite(grad))):
                status = NONFINITE
                break
            if grad_norm <= tolerance:
                status = CONVERGED
                break
            if iterations == max_iterations:
                status = MAX_ITER
                break

            # 3. update and 4. record the new iterate
            theta = theta - learning_rate * grad
            theta_history.append(theta.copy())
            if not np.all(np.isfinite(theta)):
                loss_history.append(np.inf)
                iterations += 1
                status = NONFINITE
                break

    info = {"status": status, "iterations": iterations, "grad_norm": grad_norm}
    return theta, np.array(loss_history), np.array(theta_history), info


# --------------------------------------------------------------------------- Experiment C helpers
def numerical_gradient(loss_fn, theta, h):
    """Central differences: dJ/dtheta_j ~ (J(theta + h e_j) - J(theta - h e_j)) / (2h)."""
    theta = np.asarray(theta, dtype=float)
    g = np.empty_like(theta)
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = h
        g[j] = (loss_fn(theta + e) - loss_fn(theta - e)) / (2.0 * h)
    return g


def relative_discrepancy(g_a, g_n):
    """E = ||g_a - g_n|| / max(1, ||g_a||, ||g_n||)."""
    return float(np.linalg.norm(g_a - g_n) / max(1.0, np.linalg.norm(g_a), np.linalg.norm(g_n)))


# --------------------------------------------------------------------------- Experiment E engine
def minibatch_gradient_descent(X, y, theta_initial, learning_rate, batch_size, max_updates, rng,
                               eval_every=1):
    """Batch (batch_size = N), stochastic (1) or mini-batch (b) gradient descent.

    Each epoch visits the training set in a fresh random order (sampling without replacement);
    a final short batch is used if N is not a multiple of batch_size.

    Records, every `eval_every` updates (and at the start and end):
      updates, examples (data-example gradient evaluations so far), full-dataset loss J(theta).
    Also records the loss of every mini-batch actually used (the noisy signal the optimizer sees).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    theta = np.array(theta_initial, dtype=float).copy()

    log = {"updates": [0], "examples": [0], "full_loss": [mse_loss(X, y, theta)],
           "theta": [theta.copy()], "batch_loss": []}
    updates = examples = 0
    status = MAX_ITER
    order = rng.permutation(n)
    pos = 0
    with np.errstate(over="ignore", invalid="ignore"):
        while updates < max_updates:
            if pos >= n:
                order = rng.permutation(n)
                pos = 0
            idx = order[pos:pos + batch_size]
            pos += batch_size
            Xb, yb = X[idx], y[idx]
            log["batch_loss"].append(mse_loss(Xb, yb, theta))
            theta = theta - learning_rate * mse_gradient(Xb, yb, theta)
            updates += 1
            examples += idx.size
            if not np.all(np.isfinite(theta)):
                status = NONFINITE
            if updates % eval_every == 0 or updates == max_updates or status == NONFINITE:
                log["updates"].append(updates)
                log["examples"].append(examples)
                log["full_loss"].append(mse_loss(X, y, theta))
                log["theta"].append(theta.copy())
            if status == NONFINITE:
                break

    out = {k: np.array(v) for k, v in log.items()}
    out["status"] = status
    return theta, out
