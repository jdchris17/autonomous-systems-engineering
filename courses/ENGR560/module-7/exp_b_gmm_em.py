"""
Experiment B: Gaussian mixture EM from scratch.

Part 1  log_gaussian (Cholesky + triangular solves) vs scipy.stats.multivariate_normal; detection
        of invalid covariances; accuracy on ill-conditioned covariances vs an explicit-inverse formula.
Part 2  EM with diagonal, then full covariances, on data drawn from a known 3-component 2-D mixture.
Part 3  Property tests: weights sum to one, responsibilities sum to one, covariances valid,
        log-likelihood non-decreasing, known parameters recovered, agreement with sklearn
        GaussianMixture given identical initial parameters and convergence settings.
Part 4  Hard (K-means) vs soft (GMM) assignment on overlapping populations; calibration of the
        responsibilities; and a misspecification demo where a single curved population is split
        between Gaussian components.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture

from clustering import (GaussianMixtureScratch, InvalidCovarianceError, KMeansScratch, log_gaussian,
                        match_permutation)
from data7 import OUT

TRUE_W = np.array([0.5, 0.2, 0.3])
TRUE_MU = np.array([[0.0, 0.0], [3.0, 0.5], [1.5, 3.0]])
TRUE_COV = np.array([[[2.0, 1.2], [1.2, 1.0]], [[0.3, 0.0], [0.0, 1.5]], [[1.0, -0.6], [-0.6, 0.8]]])


def sample(n, rng):
    z = rng.choice(3, size=n, p=TRUE_W)
    X = np.array([rng.multivariate_normal(TRUE_MU[k], TRUE_COV[k]) for k in z])
    return X, z


def check(label, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    return ok


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(71)
    print("Experiment B: Gaussian mixture EM from scratch")

    # ---------------------------------------------------------------- Part 1
    print("\n  Part 1 - Gaussian log density")
    errs = []
    for D in (1, 2, 3, 5, 8):
        A = rng.normal(size=(D, D))
        C = A @ A.T + 0.5 * np.eye(D)
        mu = rng.normal(size=D)
        X = rng.normal(size=(200, D)) * 2
        errs.append(np.max(np.abs(log_gaussian(X, mu, C) - multivariate_normal(mu, C).logpdf(X))))
    check("matches scipy multivariate_normal.logpdf for D = 1, 2, 3, 5, 8", max(errs) < 1e-9, f"max error {max(errs):.1e}")
    for name, bad in (("non-symmetric", np.array([[1.0, 0.5], [0.0, 1.0]])),
                      ("indefinite", np.array([[1.0, 2.0], [2.0, 1.0]])),
                      ("contains NaN", np.array([[1.0, np.nan], [np.nan, 1.0]])),
                      ("singular", np.array([[1.0, 1.0], [1.0, 1.0]]))):
        try:
            log_gaussian(np.zeros((1, 2)), np.zeros(2), bad)
            check(f"rejects {name} covariance", False)
        except InvalidCovarianceError as e:
            check(f"rejects {name} covariance", True, str(e))
    print("    ill-conditioned covariances (D = 6): |log density error| vs a long-double eigendecomposition reference")
    print(f"      {'condition number':>18}{'Cholesky (mine)':>18}{'explicit inverse':>18}")
    for cond in (1e2, 1e6, 1e10, 1e13):
        Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        ev = np.logspace(0, -np.log10(cond), 6)
        C = (Q * ev) @ Q.T
        C = 0.5 * (C + C.T)
        x = rng.normal(size=(50, 6)) * np.sqrt(ev[-1]) * 3
        # reference: exact formula with the known eigen-decomposition in extended precision
        zq = (x @ Q).astype(np.longdouble)
        ref = -0.5 * (6 * np.log(2 * np.pi) + np.sum(np.log(ev.astype(np.longdouble))) + np.sum(zq**2 / ev, axis=1))
        mine = log_gaussian(x, np.zeros(6), C)
        Ci = np.linalg.inv(C)
        sign, ld = np.linalg.slogdet(C)
        naive = -0.5 * (6 * np.log(2 * np.pi) + ld + np.einsum("ij,jk,ik->i", x, Ci, x))
        print(f"      {cond:>18.0e}{np.max(np.abs(mine - ref)):>18.2e}{np.max(np.abs(naive - ref)):>18.2e}")

    # ---------------------------------------------------------------- Parts 2-3
    X, z = sample(6000, rng)
    print(f"\n  Parts 2-3 - EM on {len(X)} draws from a known 3-component mixture")
    fits = {}
    for cov in ("diag", "full"):
        g = GaussianMixtureScratch(3, cov, seed=1, tol=1e-8, max_iter=2000).fit(X)
        fits[cov] = g
        R = g.predict_proba(X)
        h = np.array(g.log_likelihood_history_)
        print(f"   {cov} covariance: converged {g.converged_} in {g.n_iter_} iterations, mean log-likelihood {h[-1]:.5f}")
        check("mixture weights sum to one", abs(g.weights_.sum() - 1) < 1e-12, f"sum - 1 = {g.weights_.sum() - 1:.1e}")
        check("every responsibility vector sums to one", np.max(np.abs(R.sum(1) - 1)) < 1e-12,
              f"max deviation {np.max(np.abs(R.sum(1) - 1)):.1e}")
        if cov == "full":
            mins = [np.linalg.eigvalsh(c).min() for c in g.covariances_]
            sym = max(np.abs(c - c.T).max() for c in g.covariances_)
            check("covariances symmetric positive definite", min(mins) > 0 and sym < 1e-12,
                  f"min eigenvalue {min(mins):.3f}, max asymmetry {sym:.1e}, repairs {g.n_cov_repairs_}")
        else:
            check("variances positive", g.covariances_.min() > 0, f"min {g.covariances_.min():.3f}")
        check("log-likelihood never decreases", np.min(np.diff(h)) > -1e-10, f"largest decrease {max(0, -np.min(np.diff(h))):.1e}")
    g = fits["full"]
    p = match_permutation(TRUE_MU, g.means_)
    print("   recovery of the known mixture (full covariance, components matched by the Hungarian algorithm):")
    for k in range(3):
        j = p[k]
        print(f"      component {k}: weight {TRUE_W[k]:.3f} -> {g.weights_[j]:.3f}; mean {TRUE_MU[k]} ->"
              f" {np.round(g.means_[j], 3)}; covariance rel. error"
              f" {np.linalg.norm(g.covariances_[j] - TRUE_COV[k]) / np.linalg.norm(TRUE_COV[k]):.3f}")
    true_ll = np.mean(np.log(sum(TRUE_W[k] * multivariate_normal(TRUE_MU[k], TRUE_COV[k]).pdf(X) for k in range(3))))
    check("fitted log-likelihood >= log-likelihood of the TRUE parameters (MLE)", g.lower_bound_ >= true_ll - 1e-9,
          f"{g.lower_bound_:.4f} vs {true_ll:.4f}")
    print(f"   diagonal model log-likelihood {fits['diag'].lower_bound_:.4f} < full {g.lower_bound_:.4f}"
          " - the true covariances are correlated, so 'diag' is misspecified")

    print("\n   agreement with sklearn GaussianMixture from identical initial parameters (tol 1e-10, reg 1e-6):")
    for cov in ("full", "diag"):
        km = KMeansScratch(3, seed=5).fit(X)
        w0 = np.bincount(km.labels_) / len(X)
        mu0 = km.cluster_centers_
        if cov == "full":
            c0 = np.array([np.cov(X[km.labels_ == k].T) + 1e-6 * np.eye(2) for k in range(3)])
            prec0 = np.linalg.inv(c0)
        else:
            c0 = np.array([X[km.labels_ == k].var(axis=0) + 1e-6 for k in range(3)])
            prec0 = 1 / c0
        mine = GaussianMixtureScratch(3, cov, tol=1e-10, max_iter=5000).fit(X, w0, mu0, c0)
        sk = GaussianMixture(3, covariance_type=cov, tol=1e-10, max_iter=5000, reg_covar=1e-6, weights_init=w0,
                             means_init=mu0, precisions_init=prec0).fit(X)
        dr = np.max(np.abs(mine.predict_proba(X) - sk.predict_proba(X)))
        dl = abs(mine.score(X) - sk.score(X))
        check(f"{cov}: responsibilities agree", dr < 1e-4,
              f"max |diff| {dr:.1e}; mean log-likelihood diff {dl:.1e}; iterations mine {mine.n_iter_}, sklearn {sk.n_iter_}")

    # ---------------------------------------------------------------- Part 4
    print("\n  Part 4 - hard (K-means) vs soft (GMM) assignment on the same overlapping data")
    km = KMeansScratch(3, seed=0).fit(X)
    R = g.predict_proba(X)
    pk = match_permutation(TRUE_MU, km.cluster_centers_)
    inv_k = np.argsort(pk)
    inv_g = np.argsort(p)
    km_true = inv_k[km.labels_]
    g_true = inv_g[R.argmax(1)]
    print(f"    accuracy vs the generating component: K-means {np.mean(km_true == z):.3f}, GMM (argmax) {np.mean(g_true == z):.3f}")
    mx = R.max(1)
    print(f"    observations with max responsibility < 0.6: {np.mean(mx < 0.6):.1%}; < 0.9: {np.mean(mx < 0.9):.1%}")
    print("    calibration of the responsibilities (is a 0.52 really a coin flip?):")
    print(f"      {'max responsibility':>20}{'n':>7}{'argmax is the true component':>30}")
    edges = (0.33, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0001)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mx >= lo) & (mx < hi)
        if m.sum():
            print(f"      {f'{lo:.2f}-{min(hi, 1):.2f}':>20}{m.sum():>7}{np.mean(g_true[m] == z[m]):>30.3f}")
    hard_wrong_conf = np.mean(km_true[mx < 0.6] != z[mx < 0.6])
    print(f"    K-means labels in that uncertain zone are wrong {hard_wrong_conf:.0%} of the time, with no warning")

    # misspecification demo
    Xm, ym = make_moons(1200, noise=0.06, random_state=3)
    blob = np.random.default_rng(4).normal([1.0, -0.9], 0.12, (300, 2))
    Xc = np.vstack([Xm[ym == 0], blob])
    yc = np.r_[np.zeros((ym == 0).sum(), int), np.ones(300, int)]
    print("\n    Misspecification demo: ONE crescent-shaped population + one compact blob")
    print(f"      {'K':>3}{'BIC':>11}{'crescent pts with max resp < 0.8':>35}{'blob pts with max resp < 0.8':>31}")
    crescent_fits = {}
    for K in (2, 3, 4, 5):
        gm = max((GaussianMixtureScratch(K, "full", seed=s).fit(Xc) for s in range(5)), key=lambda m: m.lower_bound_)
        crescent_fits[K] = gm
        mxc = gm.predict_proba(Xc).max(1)
        print(f"      {K:>3}{gm.bic(Xc):>11.1f}{np.mean(mxc[yc == 0] < 0.8):>35.1%}{np.mean(mxc[yc == 1] < 0.8):>31.1%}")

    plots(X, z, km, R, Xc, crescent_fits)


def plots(X, z, km, R, Xc, crescent_fits):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=z, s=3, cmap="tab10", vmin=0, vmax=9)
    ax.set_title("generating component (truth)")
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], c=km.labels_, s=3, cmap="tab10", vmin=0, vmax=9)
    ax.set_title("K-means: hard assignment")
    ax = axes[2]
    ax.scatter(X[:, 0], X[:, 1], c=np.clip(R[:, [0, 1, 2]], 0, 1), s=3)
    ax.set_title("GMM responsibilities as RGB mixture")
    ax = axes[3]
    mx = R.max(1)
    sc = ax.scatter(X[:, 0], X[:, 1], c=mx, s=3, cmap="magma", vmin=0.33, vmax=1)
    fig.colorbar(sc, ax=ax, label="max responsibility")
    ax.set_title("assignment confidence (dark = boundary)")
    for a in axes:
        a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_hard_vs_soft.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, K in zip(axes, (2, 3, 5)):
        gm = crescent_fits[K]
        mx = gm.predict_proba(Xc).max(1)
        sc = ax.scatter(Xc[:, 0], Xc[:, 1], c=mx, s=4, cmap="magma", vmin=0.33, vmax=1)
        for k in range(K):
            w, V = np.linalg.eigh(gm.covariances_[k])
            t = np.linspace(0, 2 * np.pi, 100)
            ell = (V @ (2 * np.sqrt(w)[:, None] * np.vstack([np.cos(t), np.sin(t)]))).T + gm.means_[k]
            ax.plot(ell[:, 0], ell[:, 1], "c-", lw=1)
        ax.set_title(f"K = {K}: dark = uncertain (2-sigma ellipses)")
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes, shrink=0.8, label="max responsibility")
    fig.suptitle("Misspecified mixture: a single curved population is split between Gaussians")
    fig.savefig(OUT / "exp_b_misspecification.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\n  saved", OUT / "exp_b_hard_vs_soft.png")
    print("  saved", OUT / "exp_b_misspecification.png")


if __name__ == "__main__":
    main()
