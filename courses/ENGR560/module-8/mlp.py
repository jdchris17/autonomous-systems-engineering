"""
A one-hidden-layer neural network in NumPy (Module 8).

    z1 = X W1 + b1            (N x H)      W1: D x H,  b1: H
    a1 = tanh(z1)
    z2 = a1 W2 + b2           (N,)         W2: H,      b2: scalar     (output logit)
    p  = sigmoid(z2)
    J  = mean( log(1 + e^z2) - y z2 )      binary cross-entropy from logits (np.logaddexp: stable)

Backpropagation (chain rule, one sample per row, averaged over N):
    dJ/dz2 = (p - y) / N
    dJ/dW2 = a1^T dz2          dJ/db2 = sum(dz2)
    dJ/da1 = dz2 W2^T          dJ/dz1 = dJ/da1 * (1 - a1^2)          (tanh' = 1 - tanh^2)
    dJ/dW1 = X^T dz1           dJ/db1 = sum over rows of dz1

Initialisation: W1 ~ N(0, 1/D) and W2 ~ N(0, 1/H) (Xavier/LeCun scale for tanh: keeps z1 and z2
of order one so tanh starts in its sensitive range), biases zero.
"""

import numpy as np

PARAMS = ("W1", "b1", "W2", "b2")


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))       # stable for any z


class MLP:
    def __init__(self, input_dim, hidden_dim, seed=0, init_scale=1.0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, init_scale / np.sqrt(input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0.0, init_scale / np.sqrt(hidden_dim), hidden_dim)
        self.b2 = 0.0

    # ---------------------------------------------------------------- core
    def forward(self, X):
        """Returns the output logits; keeps the intermediate quantities needed by backward()."""
        X = np.asarray(X, float)
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        self.cache = {"X": X, "a1": a1, "z2": z2}
        return z2

    def loss(self, X, y):
        logits = self.forward(X)
        return float(np.mean(np.logaddexp(0.0, logits) - y * logits))

    def backward(self, X, y):
        """Gradients of the mean binary cross-entropy w.r.t. W1, b1, W2, b2."""
        z2 = self.forward(X)
        Xc, a1 = self.cache["X"], self.cache["a1"]
        n = len(y)
        dz2 = (sigmoid(z2) - y) / n
        dW2 = a1.T @ dz2
        db2 = float(dz2.sum())
        da1 = np.outer(dz2, self.W2)
        dz1 = da1 * (1.0 - a1**2)
        dW1 = Xc.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, gradients, learning_rate):
        for k in PARAMS:
            setattr(self, k, getattr(self, k) - learning_rate * gradients[k])

    def predict_proba(self, X):
        return sigmoid(self.forward(X))

    # ---------------------------------------------------------------- helpers
    def get_flat(self):
        return np.concatenate([np.ravel(getattr(self, k)) for k in PARAMS])

    def set_flat(self, theta):
        i = 0
        for k in PARAMS:
            shape = np.shape(getattr(self, k))
            size = int(np.prod(shape)) if shape else 1
            v = theta[i:i + size]
            setattr(self, k, v.reshape(shape) if shape else float(v[0]))
            i += size

    def n_params(self):
        return self.get_flat().size


def train(model, X, y, learning_rate, iterations, X_val=None, y_val=None, patience=None, eval_every=1):
    """Full-batch gradient descent. With validation data and `patience`, stops when the validation
    loss has not improved for `patience` iterations and restores the best parameters."""
    hist = {"iter": [], "train": [], "val": []}
    best, best_theta, best_it, since = np.inf, model.get_flat(), 0, 0
    for it in range(iterations + 1):
        if it % eval_every == 0:
            hist["iter"].append(it)
            hist["train"].append(model.loss(X, y))
            if X_val is not None:
                v = model.loss(X_val, y_val)
                hist["val"].append(v)
                if v < best - 1e-9:
                    best, best_theta, best_it, since = v, model.get_flat(), it, 0
                else:
                    since += eval_every
                if patience is not None and since >= patience:
                    break
        if it < iterations:
            model.step(model.backward(X, y), learning_rate)
    if X_val is not None:
        model.set_flat(best_theta)
    return {k: np.array(v) for k, v in hist.items()}, best_it
