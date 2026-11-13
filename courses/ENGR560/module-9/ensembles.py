"""
From-scratch ensemble tools for Module 9.

bootstrap_indices / group_bootstrap_indices    candidate-level and image-level resampling
BaggingScratch                                 bagged decision trees (sklearn DecisionTreeClassifier
                                               as the base learner), candidate- or image-level bootstrap,
                                               explicit-sample mode for exact comparison with sklearn,
                                               per-member predictions, group-aware out-of-bag scoring
majority_vote / soft_vote                      hard and soft voting
pairwise_diversity                             disagreement, error correlation, joint errors, etc.
platt / expected calibration error helpers     reused from Module 4
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-4"))

from classifiers import PlattCalibrator, auc, brier, clip_logit, log_loss, rates, reliability  # noqa: E402,F401


# --------------------------------------------------------------------------- bootstrap
def bootstrap_indices(n, rng):
    """n draws with replacement from range(n)."""
    return rng.integers(0, n, size=n)


def group_bootstrap_indices(groups, rng):
    """Draw as many IMAGES as there are, with replacement; return all candidate rows of the drawn images
    (an image drawn twice contributes its candidates twice)."""
    ids = np.unique(groups)
    rows = {g: np.nonzero(groups == g)[0] for g in ids}
    drawn = rng.choice(ids, size=ids.size, replace=True)
    return np.concatenate([rows[g] for g in drawn]), drawn


# --------------------------------------------------------------------------- bagging
class BaggingScratch:
    def __init__(self, n_estimators=50, max_depth=None, random_state=0, level="candidate",
                 min_samples_leaf=1, max_features=None):
        self.n_estimators, self.max_depth, self.random_state = n_estimators, max_depth, random_state
        self.level, self.min_samples_leaf, self.max_features = level, min_samples_leaf, max_features

    def fit(self, X, y, groups=None, sample_indices=None, tree_seeds=None):
        """sample_indices / tree_seeds: optional explicit bootstrap samples and tree seeds (used to
        reproduce scikit-learn's members exactly)."""
        X, y = np.asarray(X), np.asarray(y)
        self.estimators_, self.samples_, self.in_bag_groups_ = [], [], []
        for i in range(self.n_estimators):
            rng = np.random.default_rng([self.random_state, i])
            if sample_indices is not None:
                idx = np.asarray(sample_indices[i])
            elif self.level == "image":
                idx, drawn = group_bootstrap_indices(groups, rng)
                self.in_bag_groups_.append(set(drawn.tolist()))
            else:
                idx = bootstrap_indices(len(y), rng)
            seed = int(tree_seeds[i]) if tree_seeds is not None else int(rng.integers(2**31 - 1))
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                          max_features=self.max_features, random_state=seed).fit(X[idx], y[idx])
            self.estimators_.append(tree)
            self.samples_.append(idx)
        return self

    @staticmethod
    def _star_proba(tree, X):
        p = tree.predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1]
        return np.full(len(X), float(tree.classes_[0] == 1))       # a bootstrap sample with one class

    def member_proba(self, X, m=None):
        ests = self.estimators_ if m is None else self.estimators_[:m]
        return np.array([self._star_proba(t, X) for t in ests])

    def predict_proba(self, X, m=None):
        """Average of the member class probabilities (first m members if given)."""
        p1 = self.member_proba(X, m).mean(axis=0)
        return np.column_stack([1 - p1, p1])

    def predict(self, X, threshold=0.5, m=None):
        return (self.predict_proba(X, m)[:, 1] >= threshold).astype(int)

    def oob_proba(self, X, groups=None):
        """Out-of-bag probability per row. level='candidate': trees whose bootstrap missed THAT ROW.
        level='image' with groups: trees whose bootstrap missed THAT ROW'S WHOLE IMAGE."""
        n = len(X)
        s, c = np.zeros(n), np.zeros(n)
        for i, t in enumerate(self.estimators_):
            if self.level == "image" and groups is not None:
                out = ~np.isin(groups, list(self.in_bag_groups_[i]))
            else:
                out = np.ones(n, bool)
                out[self.samples_[i]] = False
            if out.any():
                s[out] += self._star_proba(t, X[out])
                c[out] += 1
        return np.where(c > 0, s / np.maximum(c, 1), np.nan), c


# --------------------------------------------------------------------------- voting
def majority_vote(predictions):
    """predictions: (n_models, n_samples) of 0/1. Ties (even number of models) go to the star class."""
    P = np.asarray(predictions)
    return (P.sum(axis=0) * 2 >= P.shape[0]).astype(int)


def soft_vote(probabilities, weights=None):
    """Course implementation: weighted average of (n_models, n_samples) probabilities."""
    P = np.asarray(probabilities, dtype=float)
    if weights is None:
        weights = np.ones(P.shape[0])
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return np.average(P, axis=0, weights=weights)


# --------------------------------------------------------------------------- diversity
def pairwise_diversity(pred_a, pred_b, y):
    ea, eb = pred_a != y, pred_b != y
    corr = np.corrcoef(ea, eb)[0, 1] if ea.std() > 0 and eb.std() > 0 else np.nan
    return {"disagreement": float(np.mean(pred_a != pred_b)), "error_corr": float(corr),
            "both_wrong": float(np.mean(ea & eb)), "a_right_b_wrong": float(np.mean(~ea & eb)),
            "a_wrong_b_right": float(np.mean(ea & ~eb))}


def ece(y, p):
    return reliability(y, p)[3]
