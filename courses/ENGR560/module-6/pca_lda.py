"""
PCA and binary LDA from scratch (Module 6, Experiments B and E).

PCAFromScratch follows the course interface and also behaves as a scikit-learn transformer
(BaseEstimator / TransformerMixin), so it can sit inside a Pipeline and be refit inside every
cross-validation fold.

    fit:    mean_ = column means of the TRAINING data;  Xc = X - mean_
            Xc = U S V^T  (np.linalg.svd, full_matrices=False)
            components_ = first k rows of V^T            (orthonormal principal directions)
            explained_variance_ = S^2 / (N - 1)          (variance along each direction)
            explained_variance_ratio_ = S^2 / sum(S^2)   (over ALL components, as sklearn does)
    transform:          Z = (X - mean_) V_k
    inverse_transform:  X_hat = Z V_k^T + mean_

LDAFromScratch (two classes):
    class means m0, m1;  S_W = sum_c sum_{i in c} (x_i - m_c)(x_i - m_c)^T
    S_B = (m1 - m0)(m1 - m0)^T  (rank one for two classes)
    discriminant direction w ~ S_W^{-1} (m1 - m0)  (maximises w^T S_B w / w^T S_W w)
    optional shrinkage: S_W/(N-2) -> (1 - a) S_W/(N-2) + a * (trace/D) I  for singular pixel data
    decision: equal-covariance Gaussian model with the training class priors
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PCAFromScratch(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = Vt.shape[0] if self.n_components is None else int(self.n_components)
        self.n_samples_ = X.shape[0]
        self.singular_values_all_ = s
        self.components_ = Vt[:k]
        self.singular_values_ = s[:k]
        var_all = s**2 / (X.shape[0] - 1)
        self.explained_variance_ = var_all[:k]
        self.explained_variance_ratio_ = (s**2 / np.sum(s**2))[:k]
        self.cumulative_explained_variance_ = np.cumsum(s**2)[:k] / np.sum(s**2)
        self.rank_ = int(np.sum(s > s[0] * max(X.shape) * np.finfo(float).eps)) if s.size else 0
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return np.asarray(Z, float) @ self.components_ + self.mean_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class LDAFromScratch(BaseEstimator):
    def __init__(self, shrinkage=0.0):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        X0, X1 = X[y == 0], X[y == 1]
        self.means_ = np.vstack([X0.mean(axis=0), X1.mean(axis=0)])
        D0, D1 = X0 - self.means_[0], X1 - self.means_[1]
        self.S_W_ = D0.T @ D0 + D1.T @ D1
        d = self.means_[1] - self.means_[0]
        self.S_B_ = np.outer(d, d)
        n = X.shape[0]
        cov = self.S_W_ / (n - 2)                       # pooled within-class covariance
        if self.shrinkage:
            mu = np.trace(cov) / cov.shape[0]
            cov = (1 - self.shrinkage) * cov + self.shrinkage * mu * np.eye(cov.shape[0])
        self.covariance_ = cov
        self.w_ = np.linalg.solve(cov, d)              # direction ~ S_W^{-1}(m1 - m0)
        self.priors_ = np.array([np.mean(y == 0), np.mean(y == 1)])
        mid = 0.5 * (self.means_[0] + self.means_[1])
        self.b_ = -self.w_ @ mid + np.log(self.priors_[1] / self.priors_[0])
        self.direction_ = self.w_ / np.linalg.norm(self.w_)
        return self

    def transform(self, X):
        """Projection onto the unit discriminant axis."""
        return np.asarray(X, float) @ self.direction_

    def decision_function(self, X):
        return np.asarray(X, float) @ self.w_ + self.b_

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def fisher_ratio(self, w=None):
        w = self.w_ if w is None else w
        return float((w @ self.S_B_ @ w) / (w @ self.S_W_ @ w))
