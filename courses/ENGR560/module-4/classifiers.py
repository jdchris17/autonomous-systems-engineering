"""
From-scratch classifiers and shared utilities for Module 4.

  Standardizer           training-only mean/std scaling
  Perceptron             mistake-driven updates, bias, mistakes per epoch, final weights
  LogisticRegressionGD   sigmoid + binary cross-entropy (softplus form), optional L2 on the
                         weights (never the bias), trained with Module 2's gradient_descent
  KNN                    brute-force Euclidean k-nearest neighbours (NumPy)
  PlattCalibrator        1-D logistic regression on a classifier score, fitted on validation data
  metrics                confusion matrix, precision/recall/FPR/F1, ROC, AUC, log loss, Brier, ECE
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-2"))
sys.path.insert(0, str(HERE.parent / "module-1"))

from gd_core import gradient_descent, numerical_gradient, relative_discrepancy  # noqa: E402,F401

OUT = HERE / "outputs"


# --------------------------------------------------------------------------- preprocessing
class Standardizer:
    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


def add_bias(X):
    return np.column_stack([np.ones(X.shape[0]), X])


# --------------------------------------------------------------------------- perceptron
class Perceptron:
    """Rosenblatt perceptron, labels in {0, 1} externally, {-1, +1} internally.

    Update on a mistake (y_i (w . x_i + b) <= 0):  w <- w + lr y_i x_i,  b <- b + lr y_i
    """

    def __init__(self, epochs=50, lr=1.0, seed=0):
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        t = np.where(y == 1, 1.0, -1.0)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.mistakes = []
        self.weight_history = [np.concatenate([[self.b], self.w])]
        for _ in range(self.epochs):
            m = 0
            for i in rng.permutation(X.shape[0]):
                if t[i] * (X[i] @ self.w + self.b) <= 0:
                    self.w += self.lr * t[i] * X[i]
                    self.b += self.lr * t[i]
                    m += 1
            self.mistakes.append(m)
            self.weight_history.append(np.concatenate([[self.b], self.w]))
            if m == 0:          # separable data: a full clean pass means we are done
                break
        self.final_weights = np.concatenate([[self.b], self.w])
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


# --------------------------------------------------------------------------- logistic regression
def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))          # stable for any z


def logistic_loss(Xb, y, theta, lam=0.0):
    """mean[ log(1 + e^z) - y z ] + (lam/2) ||w||^2,  z = Xb theta, bias theta[0] unpenalized."""
    z = Xb @ theta
    return float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * theta[1:] @ theta[1:])


def logistic_gradient(Xb, y, theta, lam=0.0):
    g = Xb.T @ (sigmoid(Xb @ theta) - y) / y.size
    g[1:] += lam * theta[1:]
    return g


class LogisticRegressionGD:
    """Binary logistic regression trained with Module 2's gradient_descent.

    Step size 1/L with L = lambda_max(X^T X)/(4n) + lam, the Lipschitz constant of the gradient
    (the logistic Hessian is X^T diag(p(1-p)) X / n <= X^T X / (4n)).
    """

    def __init__(self, lam=0.0, tol=1e-7, max_iter=50_000):
        self.lam, self.tol, self.max_iter = lam, tol, max_iter

    def fit(self, X, y):
        Xb = add_bias(X)
        L = np.linalg.eigvalsh(Xb.T @ Xb / Xb.shape[0]).max() / 4.0 + self.lam
        lam = self.lam
        theta, loss_h, theta_h, info = gradient_descent(
            Xb, y.astype(float), np.zeros(Xb.shape[1]), 1.0 / L, self.max_iter, self.tol,
            loss_fn=lambda A, b, t: logistic_loss(A, b, t, lam),
            grad_fn=lambda A, b, t: logistic_gradient(A, b, t, lam))
        self.theta, self.loss_history, self.info = theta, loss_h, info
        self.step = 1.0 / L
        return self

    def decision_function(self, X):
        return add_bias(X) @ self.theta

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# --------------------------------------------------------------------------- KNN
class KNN:
    """Brute-force k-nearest neighbours with Euclidean distance and majority vote."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y)
        self.sq = np.sum(self.X**2, axis=1)
        return self

    def neighbours(self, Q, batch=2000):
        out = []
        for s in range(0, Q.shape[0], batch):
            q = Q[s:s + batch]
            d2 = np.sum(q**2, axis=1)[:, None] + self.sq[None, :] - 2.0 * q @ self.X.T
            idx = np.argpartition(d2, self.k - 1, axis=1)[:, :self.k]
            # order the k neighbours by distance (ties broken by training index, like sklearn)
            dk = np.take_along_axis(d2, idx, axis=1)
            order = np.lexsort((idx, dk), axis=1)
            out.append(np.take_along_axis(idx, order, axis=1))
        return np.vstack(out)

    def predict_proba(self, Q):
        return self.y[self.neighbours(Q)].mean(axis=1)

    def predict(self, Q):
        return (self.predict_proba(Q) > 0.5).astype(int)


# --------------------------------------------------------------------------- calibration
def clip_logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """p = sigmoid(a s + b) fitted on a held-out set's scores s (our own logistic regression)."""

    def fit(self, s, y):
        self.m = LogisticRegressionGD(lam=0.0, tol=1e-8).fit(np.asarray(s).reshape(-1, 1), y)
        return self

    def transform(self, s):
        return self.m.predict_proba(np.asarray(s).reshape(-1, 1))


# --------------------------------------------------------------------------- metrics
def confusion(y, yhat):
    tp = int(np.sum((yhat == 1) & (y == 1)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    fn = int(np.sum((yhat == 0) & (y == 1)))
    tn = int(np.sum((yhat == 0) & (y == 0)))
    return np.array([[tn, fp], [fn, tp]])


def rates(y, yhat):
    (tn, fp), (fn, tp) = confusion(y, yhat)
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    acc = (tp + tn) / y.size
    bal = 0.5 * (rec + (1 - fpr))
    return {"accuracy": acc, "balanced_acc": bal, "precision": prec, "recall": rec, "fpr": fpr,
            "f1": f1, "accepted": tp + fp}


def roc_curve(y, s):
    """FPR, TPR for every distinct threshold (descending)."""
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.concatenate([[0], np.cumsum(ys)])
    fp = np.concatenate([[0], np.cumsum(1 - ys)])
    return fp / max(fp[-1], 1), tp / max(tp[-1], 1)


def auc(y, s):
    """Probability a random star outranks a random artifact (Mann-Whitney), ties count 1/2."""
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def reliability(y, p, bins=10):
    """Per-bin mean predicted probability, observed star fraction, count; and the ECE."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    conf, freq, cnt = [], [], []
    for b in range(bins):
        m = idx == b
        cnt.append(int(m.sum()))
        conf.append(p[m].mean() if m.any() else np.nan)
        freq.append(y[m].mean() if m.any() else np.nan)
    conf, freq, cnt = map(np.array, (conf, freq, cnt))
    ok = cnt > 0
    ece = float(np.sum(cnt[ok] * np.abs(conf[ok] - freq[ok])) / cnt.sum())
    return conf, freq, cnt, ece
