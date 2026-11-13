"""
Candidate classifiers with validation-selected hyperparameters (used by Experiments D, E, F).

Every candidate works on the same inputs: engineered features standardized with training stats.
Hyperparameters are chosen on the VALIDATION split only:
  perceptron   none (50 epochs, final weights)
  logistic     L2 strength lam in {1e-4, 1e-3, 1e-2, 1e-1}, chosen by validation log loss
  linear SVM   C in {0.01, 0.1, 1}, chosen by validation accuracy
  RBF SVM      C in {1, 10, 100} x gamma in {0.1, 0.3, 1, 3}, validation accuracy
  KNN          k in {1, 3, 5, 11, 21, 51} (own implementation), validation accuracy
  tree         max_depth in {2, 4, 6, 8, 16, None}, validation accuracy

Probabilities: every model also gets a Platt calibrator (sigmoid of a*score + b) fitted on the
validation split. For models with native probabilities (logistic, KNN, tree) the score is the
logit of the native probability. `proba` returns the calibrated probability; `native_proba`
returns the model's own probability where it has one.
"""

import time

import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classifiers import (KNN, LogisticRegressionGD, Perceptron, PlattCalibrator, Standardizer,
                         clip_logit, log_loss)
from star_data import engineered


class Candidate:
    def __init__(self, name, model, score_fn, native_fn=None, params="", fit_time=0.0):
        self.name, self.model, self.score_fn, self.native_fn = name, model, score_fn, native_fn
        self.params, self.fit_time = params, fit_time

    def score(self, Z):
        return self.score_fn(self.model, Z)

    def native_proba(self, Z):
        return None if self.native_fn is None else self.native_fn(self.model, Z)

    def calibrate(self, Zv, yv):
        self.cal = PlattCalibrator().fit(self.score(Zv), yv)
        return self

    def proba(self, Z):
        return self.cal.transform(self.score(Z))

    def predict(self, Z):
        """Model's own decision rule (score threshold 0 / native probability 0.5)."""
        p = self.native_proba(Z)
        return (p >= 0.5).astype(int) if p is not None else (self.score(Z) > 0).astype(int)


def _acc(pred, y):
    return float(np.mean(pred == y))


def build_candidates(Ztr, ytr, Zva, yva, verbose=True):
    out = {}

    t = time.perf_counter()
    p = Perceptron(epochs=50, seed=1).fit(Ztr, ytr)
    out["perceptron"] = Candidate("perceptron", p, lambda m, Z: m.decision_function(Z),
                                  params="50 epochs", fit_time=time.perf_counter() - t)

    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1):
        t = time.perf_counter()
        m = LogisticRegressionGD(lam=lam).fit(Ztr, ytr)
        dt = time.perf_counter() - t
        ll = log_loss(yva, m.predict_proba(Zva))
        if best is None or ll < best[0]:
            best = (ll, lam, m, dt)
    out["logistic"] = Candidate("logistic", best[2], lambda m, Z: m.decision_function(Z),
                                lambda m, Z: m.predict_proba(Z), f"lam={best[1]:g}", best[3])

    best = None
    for C in (0.01, 0.1, 1):
        t = time.perf_counter()
        m = SVC(kernel="linear", C=C).fit(Ztr, ytr)
        dt = time.perf_counter() - t
        a = _acc(m.predict(Zva), yva)
        if best is None or a > best[0]:
            best = (a, C, m, dt)
    out["linear SVM"] = Candidate("linear SVM", best[2], lambda m, Z: m.decision_function(Z),
                                  params=f"C={best[1]:g}", fit_time=best[3])

    best = None
    for C in (1, 10, 100):
        for g in (0.1, 0.3, 1, 3):
            t = time.perf_counter()
            m = SVC(kernel="rbf", C=C, gamma=g).fit(Ztr, ytr)
            dt = time.perf_counter() - t
            a = _acc(m.predict(Zva), yva)
            if best is None or a > best[0]:
                best = (a, (C, g), m, dt)
    out["RBF SVM"] = Candidate("RBF SVM", best[2], lambda m, Z: m.decision_function(Z),
                               params=f"C={best[1][0]:g}, gamma={best[1][1]:g}", fit_time=best[3])

    best = None
    for k in (1, 3, 5, 11, 21, 51):
        t = time.perf_counter()
        m = KNN(k).fit(Ztr, ytr)
        dt = time.perf_counter() - t
        a = _acc(m.predict(Zva), yva)
        if best is None or a > best[0]:
            best = (a, k, m, dt)
    out["KNN"] = Candidate("KNN", best[2], lambda m, Z: clip_logit(m.predict_proba(Z), 0.5 / m.k),
                           lambda m, Z: m.predict_proba(Z), f"k={best[1]}", best[3])

    best = None
    for d in (2, 4, 6, 8, 16, None):
        t = time.perf_counter()
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Ztr, ytr)
        dt = time.perf_counter() - t
        a = _acc(m.predict(Zva), yva)
        if best is None or a > best[0]:
            best = (a, d, m, dt)
    out["tree"] = Candidate("tree", best[2], lambda m, Z: clip_logit(m.predict_proba(Z)[:, 1], 1e-3),
                            lambda m, Z: m.predict_proba(Z)[:, 1], f"max_depth={best[1]}", best[3])

    for c in out.values():
        c.calibrate(Zva, yva)
    if verbose:
        print("    selected hyperparameters (validation): "
              + "; ".join(f"{c.name}: {c.params}" for c in out.values()))
    return out


def prepared_splits(splits, cols=None):
    """Engineered + standardized features (training statistics) for a dict of splits.

    cols selects a subset of the engineered columns (default: all five)."""
    cols = list(range(5)) if cols is None else list(cols)
    E = {k: engineered(v[0])[:, cols] for k, v in splits.items()}
    st = Standardizer().fit(E["train"])
    return st, {k: st.transform(v) for k, v in E.items()}


def predict_time(c, Z, reps=3):
    best = np.inf
    for _ in range(reps):
        t = time.perf_counter()
        c.proba(Z)
        best = min(best, time.perf_counter() - t)
    return best / Z.shape[0] * 1000.0     # seconds per 1000 candidates
