"""
Shared tools for Module 5: feature pipelines, splitters, metrics.

Reuses Module 4 (classifiers.py): LogisticRegressionGD (Module 2 gradient descent inside),
Standardizer, PlattCalibrator and the metric functions. scikit-learn's KNeighborsClassifier is used
inside the heavy cross-validation loops; Module 4 verified it agrees 100% with our own KNN.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

# KFold / StratifiedKFold are called with a groups argument on purpose (same call for every splitter)
warnings.filterwarnings("ignore", message="The groups parameter is ignored")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-4"))

from classifiers import (LogisticRegressionGD, PlattCalibrator, Standardizer, auc, brier,  # noqa: E402,F401
                         clip_logit, confusion, log_loss, rates, reliability, sigmoid)
from grouped_data import INVARIANT, engineered  # noqa: E402

ALL5 = [0, 1, 2, 3, 4]


class Pipeline:
    """engineered features -> column subset -> standardize (fit on the training rows) -> model.

    kind: "logistic" (lam), "knn" (k), "tree" (depth), "svm" (C, gamma)
    score(): continuous score for ranking (logit / vote logit / SVM margin)
    proba(): the model's own probability (SVM: sigmoid of the margin - NOT calibrated)
    """

    def __init__(self, kind, cols=ALL5, **params):
        self.kind, self.cols, self.params = kind, list(cols), params

    def _Z(self, X):
        return self.st.transform(engineered(X)[:, self.cols])

    def fit(self, X, y):
        E = engineered(X)[:, self.cols]
        self.st = Standardizer().fit(E)
        Z = self.st.transform(E)
        p = self.params
        if self.kind == "logistic":
            self.m = LogisticRegressionGD(lam=p.get("lam", 1e-4)).fit(Z, y)
        elif self.kind == "knn":
            self.m = KNeighborsClassifier(n_neighbors=p["k"]).fit(Z, y)
        elif self.kind == "tree":
            self.m = DecisionTreeClassifier(max_depth=p.get("depth"), random_state=0).fit(Z, y)
        elif self.kind == "svm":
            self.m = SVC(C=p["C"], gamma=p["gamma"]).fit(Z, y)
        else:
            raise ValueError(self.kind)
        return self

    def score(self, X):
        Z = self._Z(X)
        if self.kind == "logistic":
            return self.m.decision_function(Z)
        if self.kind == "svm":
            return self.m.decision_function(Z)
        p = self.m.predict_proba(Z)[:, 1]
        # keep vote fractions 0 and 1 finite without merging any distinct values (k=1 included)
        return clip_logit(p, 0.5 / (self.params.get("k", 200) + 1))

    def proba(self, X):
        Z = self._Z(X)
        if self.kind == "logistic":
            return self.m.predict_proba(Z)
        if self.kind == "svm":
            return sigmoid(self.m.decision_function(Z))
        return self.m.predict_proba(Z)[:, 1]


def describe(kind, params):
    return kind + "(" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"


def evaluate(pipe, X, y, threshold=0.5):
    p = pipe.proba(X)
    r = rates(y, (p >= threshold).astype(int))
    return {"accuracy": r["accuracy"], "auc": auc(y, pipe.score(X)), "precision": r["precision"],
            "recall": r["recall"], "fpr": r["fpr"], "brier": brier(y, p)}


def grouped_test_split(groups, rng, test_fraction=0.3):
    """Pick whole images at random until they hold ~test_fraction of the candidates."""
    ids = rng.permutation(np.unique(groups))
    counts = {g: np.sum(groups == g) for g in ids}
    target, chosen, total = test_fraction * groups.size, [], 0
    for g in ids:
        if total >= target:
            break
        chosen.append(g)
        total += counts[g]
    test = np.isin(groups, chosen)
    return np.nonzero(~test)[0], np.nonzero(test)[0]


def random_test_split(n, rng, test_size):
    perm = rng.permutation(n)
    return np.sort(perm[test_size:]), np.sort(perm[:test_size])
