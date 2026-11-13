"""
From-scratch clustering for Module 7: K-means (Experiment A) and Gaussian-mixture EM (Experiment B).

KMeansScratch
    init "random" (K distinct observations) or "k-means++" (D^2 seeding), or an explicit array.
    Lloyd iterations: assign to nearest centre -> centre = mean of its members.
    EMPTY CLUSTERS are handled explicitly: an empty cluster's centre is moved to the observation
    that is currently farthest from its own centre (and that observation leaves its old cluster);
    every such event is counted in `n_empty_fixes_`.
    Convergence: squared centre shift <= tol * mean feature variance (scikit-learn's definition), or
    unchanged labels. A final assignment step makes labels_ and inertia_ consistent with centres.
    History of centres / labels / inertia is kept for visualisation.

log_gaussian(X, mean, cov)
    log N(x | mu, Sigma) via a Cholesky factor L (Sigma = L L^T) and triangular solves:
        z = L^-1 (x - mu),  log|Sigma| = 2 sum log diag(L)
        log N = -0.5 (D log 2 pi + log|Sigma| + z^T z)
    Sigma^-1 is never formed. Non-symmetric, non-finite or non-positive-definite input raises
    InvalidCovarianceError.

GaussianMixtureScratch
    covariance_type "diag" or "full"; reg_covar added to every variance / the diagonal.
    init "kmeans" (one-hot responsibilities from K-means++, as sklearn does), "random"
    (Dirichlet responsibilities), or explicit (weights, means, covariances).
    E-step in log space with logsumexp; M-step closed form; stops when the change in mean
    log-likelihood per sample < tol (sklearn's criterion). Keeps the log-likelihood history.
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp


# --------------------------------------------------------------------------- K-means
def sq_dist(X, C):
    return np.maximum(np.sum(X**2, 1)[:, None] + np.sum(C**2, 1)[None, :] - 2 * X @ C.T, 0.0)


def kmeans_plus_plus(X, k, rng):
    n = X.shape[0]
    centres = [X[rng.integers(n)]]
    d2 = sq_dist(X, centres[0][None, :]).ravel()
    for _ in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        centres.append(X[rng.choice(n, p=p)])
        d2 = np.minimum(d2, sq_dist(X, centres[-1][None, :]).ravel())
    return np.array(centres)


class KMeansScratch:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-6, seed=0, init="k-means++", keep_history=False):
        self.n_clusters, self.max_iter, self.tol, self.seed = n_clusters, max_iter, tol, seed
        self.init, self.keep_history = init, keep_history

    def _init_centres(self, X, rng):
        if isinstance(self.init, np.ndarray):
            return self.init.astype(float).copy()
        if self.init == "random":
            return X[rng.choice(X.shape[0], self.n_clusters, replace=False)].astype(float)
        return kmeans_plus_plus(X, self.n_clusters, rng)

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        C = self._init_centres(X, rng)                                 # 1. initialise
        tol_abs = self.tol * np.mean(np.var(X, axis=0))
        self.n_empty_fixes_ = 0
        self.history_ = []
        self.inertia_history_ = []
        labels = None
        self.converged_ = False
        for it in range(self.max_iter):
            d2 = sq_dist(X, C)
            new_labels = d2.argmin(axis=1)                             # 2. assign
            self.inertia_history_.append(float(d2[np.arange(len(X)), new_labels].sum()))   # 4. inertia
            if self.keep_history:
                self.history_.append((C.copy(), new_labels.copy()))
            newC = C.copy()
            for j in range(self.n_clusters):                           # 3. update (+ empty clusters)
                members = new_labels == j
                if members.any():
                    newC[j] = X[members].mean(axis=0)
                else:
                    own = d2[np.arange(len(X)), new_labels]
                    far = int(np.argmax(own))
                    newC[j] = X[far]
                    new_labels[far] = j
                    d2[far] = 0.0
                    self.n_empty_fixes_ += 1
            shift = np.sum((newC - C) ** 2)
            same = labels is not None and np.array_equal(labels, new_labels)
            C, labels = newC, new_labels
            if shift <= tol_abs or same:                               # 5. convergence
                self.converged_ = True
                break
        self.n_iter_ = it + 1
        self.cluster_centers_ = C
        d2 = sq_dist(X, C)
        self.labels_ = d2.argmin(axis=1)
        self.inertia_ = float(d2[np.arange(len(X)), self.labels_].sum())
        return self

    def predict(self, X):
        return sq_dist(np.asarray(X, float), self.cluster_centers_).argmin(axis=1)

    def transform(self, X):
        return np.sqrt(sq_dist(np.asarray(X, float), self.cluster_centers_))

    def fit_predict(self, X):
        return self.fit(X).labels_


def best_of(X, k, n_init=10, init="k-means++", seed=0):
    runs = [KMeansScratch(k, seed=seed + i, init=init).fit(X) for i in range(n_init)]
    return min(runs, key=lambda m: m.inertia_)


# --------------------------------------------------------------------------- Gaussian log density
class InvalidCovarianceError(ValueError):
    pass


def check_covariance(cov, name="covariance"):
    cov = np.asarray(cov, float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise InvalidCovarianceError(f"{name} must be square, got shape {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise InvalidCovarianceError(f"{name} contains non-finite values")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12 * max(1.0, np.abs(cov).max())):
        raise InvalidCovarianceError(f"{name} is not symmetric")
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise InvalidCovarianceError(f"{name} is not positive definite") from None


def log_gaussian(X, mean, cov):
    """log N(x | mean, cov) for every row of X, using Cholesky + triangular solves."""
    X = np.atleast_2d(np.asarray(X, float))
    L = check_covariance(cov)
    Z = solve_triangular(L, (X - mean).T, lower=True)            # D x N
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    D = X.shape[1]
    return -0.5 * (D * np.log(2 * np.pi) + logdet + np.sum(Z**2, axis=0))


def log_gaussian_diag(X, mean, var):
    var = np.asarray(var, float)
    if np.any(~np.isfinite(var)) or np.any(var <= 0):
        raise InvalidCovarianceError("diagonal variances must be finite and > 0")
    D = X.shape[1]
    return -0.5 * (D * np.log(2 * np.pi) + np.sum(np.log(var)) + np.sum((X - mean) ** 2 / var, axis=1))


# --------------------------------------------------------------------------- Gaussian mixture EM
class GaussianMixtureScratch:
    def __init__(self, n_components=3, covariance_type="diag", reg_covar=1e-6, max_iter=300, tol=1e-5,
                 seed=0, init="kmeans"):
        self.n_components, self.covariance_type, self.reg_covar = n_components, covariance_type, reg_covar
        self.max_iter, self.tol, self.seed, self.init = max_iter, tol, seed, init

    # ---- pieces
    def _log_prob(self, X):
        """N x K matrix of log w_k + log N(x | mu_k, Sigma_k)."""
        cols = []
        for k in range(self.n_components):
            if self.covariance_type == "diag":
                lp = log_gaussian_diag(X, self.means_[k], self.covariances_[k])
            else:
                lp = log_gaussian(X, self.means_[k], self.covariances_[k])
            cols.append(np.log(self.weights_[k]) + lp)
        return np.column_stack(cols)

    def _e_step(self, X):
        lp = self._log_prob(X)
        lse = logsumexp(lp, axis=1)
        return np.exp(lp - lse[:, None]), float(np.mean(lse))

    def _m_step(self, X, R):
        n, D = X.shape
        Nk = R.sum(axis=0) + 10 * np.finfo(float).eps
        self.weights_ = Nk / n
        self.means_ = (R.T @ X) / Nk[:, None]
        if self.covariance_type == "diag":
            avg_x2 = (R.T @ (X * X)) / Nk[:, None]
            self.covariances_ = np.maximum(avg_x2 - self.means_**2, 0) + self.reg_covar
        else:
            covs = np.empty((self.n_components, D, D))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                c = (R[:, k][:, None] * diff).T @ diff / Nk[k]
                c = 0.5 * (c + c.T) + self.reg_covar * np.eye(D)
                for attempt in range(6):                     # keep the covariance valid
                    try:
                        check_covariance(c)
                        break
                    except InvalidCovarianceError:
                        c = c + (10.0 ** attempt) * self.reg_covar * np.eye(D)
                        self.n_cov_repairs_ += 1
                covs[k] = c
            self.covariances_ = covs

    # ---- public API
    def fit(self, X, weights_init=None, means_init=None, covariances_init=None):
        X = np.asarray(X, float)
        n, D = X.shape
        rng = np.random.default_rng(self.seed)
        self.n_cov_repairs_ = 0
        if means_init is not None:
            self.weights_ = np.asarray(weights_init, float)
            self.means_ = np.asarray(means_init, float)
            self.covariances_ = np.asarray(covariances_init, float)
        else:
            if self.init == "kmeans":
                lab = KMeansScratch(self.n_components, seed=int(rng.integers(1 << 30))).fit(X).labels_
                R = np.eye(self.n_components)[lab]
            else:
                R = rng.dirichlet(np.ones(self.n_components), size=n)
            self._m_step(X, R)
        self.log_likelihood_history_ = []
        prev = -np.inf
        self.converged_ = False
        for it in range(self.max_iter):
            R, ll = self._e_step(X)                # E-step (+ observed-data log-likelihood)
            self.log_likelihood_history_.append(ll)
            if abs(ll - prev) < self.tol:          # convergence on the mean log-likelihood
                self.converged_ = True
                break
            prev = ll
            self._m_step(X, R)                     # M-step
        self.n_iter_ = it + 1
        self.lower_bound_ = self.log_likelihood_history_[-1]
        return self

    def predict_proba(self, X):
        return self._e_step(np.asarray(X, float))[0]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score_samples(self, X):
        return logsumexp(self._log_prob(np.asarray(X, float)), axis=1)

    def score(self, X):
        return float(np.mean(self.score_samples(X)))

    def n_parameters(self, D=None):
        D = self.means_.shape[1] if D is None else D
        K = self.n_components
        cov = K * D * (D + 1) // 2 if self.covariance_type == "full" else K * D
        return (K - 1) + K * D + cov

    def bic(self, X):
        return -2 * self.score(X) * len(X) + self.n_parameters() * np.log(len(X))

    def aic(self, X):
        return -2 * self.score(X) * len(X) + 2 * self.n_parameters()


def best_gmm(X, k, cov="full", n_init=5, seed=0, **kw):
    fits = [GaussianMixtureScratch(k, cov, seed=seed + i, **kw).fit(X) for i in range(n_init)]
    return max(fits, key=lambda m: m.lower_bound_), fits


def match_permutation(A, B):
    """Permutation p minimising sum ||A[i] - B[p[i]]||^2 (Hungarian algorithm)."""
    from scipy.optimize import linear_sum_assignment
    _, p = linear_sum_assignment(sq_dist(np.asarray(A, float), np.asarray(B, float)))
    return p
