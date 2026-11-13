"""
Experiment B: PCA from scratch, validated against scikit-learn.

Data: development patches flattened to X in R^{N x 1024} (raw pixels; Experiment C compares
preprocessing). The mean image is learned from the training (development) images only and reused
for the untouched test images.

Checks
  1. explained-variance ratios agree with sklearn PCA
  2. principal directions are orthonormal
  3. reconstructions agree with sklearn's to numerical tolerance (compared through projection
     matrices P_k = V_k^T V_k and reconstructed images, not elementwise eigenvector signs)
  4. reconstruction error decreases as components are added
  5. the complete representation reconstructs the centred data to numerical precision, up to rank
Engineering question: SVD of X vs eigen-decomposition of the covariance X^T X / (N - 1).
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from pca_lda import PCAFromScratch
from patches import SIZE, development_set, untouched_test_set

OUT = __import__("pathlib").Path(__file__).parent / "outputs"
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def main():
    OUT.mkdir(exist_ok=True)
    d, t = development_set(), untouched_test_set()
    X = d["X"].reshape(len(d["y"]), -1)
    Xt = t["X"].reshape(len(t["y"]), -1)
    print(f"Experiment B: PCA from scratch on X ({X.shape[0]} x {X.shape[1]}), held-out {Xt.shape[0]} patches")

    mine = PCAFromScratch().fit(X)
    sk = PCA(svd_solver="full").fit(X)
    V, Vs = mine.components_, sk.components_
    print("\n  1. explained-variance ratio: max |mine - sklearn| ="
          f" {np.max(np.abs(mine.explained_variance_ratio_ - sk.explained_variance_ratio_)):.2e};"
          f" explained variance max relative diff {np.max(np.abs(mine.explained_variance_ / sk.explained_variance_ - 1)):.2e}")
    print(f"  2. orthonormality: max |V V^T - I| = {np.max(np.abs(V @ V.T - np.eye(V.shape[0]))):.2e}")
    cos = np.abs(np.sum(V * Vs, axis=1))
    print(f"     |cos| between matching components: first 20 min {cos[:20].min():.6f};"
          f" components with |cos| < 0.99: {np.sum(cos < 0.99)} of {cos.size}")
    s = mine.singular_values_all_
    gaps = np.nonzero(np.abs(np.diff(s[:21])) / s[1:21] < 0.02)[0]
    print(f"     near-equal singular-value pairs among the first 20 (gap < 2%): {[(int(i), int(i) + 1) for i in gaps]}"
          " -> individual vectors may rotate inside such pairs")
    print("  3. projection matrices and reconstructions vs sklearn (held-out patches):")
    print(f"     {'k':>6}{'|P_mine - P_sk|_F':>20}{'max |Xhat diff|':>18}{'train MSE':>12}{'test MSE':>12}{'cum. EV':>9}")
    tr_err, te_err = [], []
    # one full fit each; the k-component models are the first k rows (an SVD gives all of them at once)
    def recon(Xin, comps, mean):
        return (Xin - mean) @ comps.T @ comps + mean
    m16 = PCAFromScratch(16).fit(X)
    print(f"     (PCAFromScratch(16) vs slicing the full fit: max diff"
          f" {np.max(np.abs(m16.inverse_transform(m16.transform(Xt)) - recon(Xt, V[:16], mine.mean_))):.1e})")
    for k in KS:
        Vk, Vsk = V[:k], Vs[:k]
        dP = np.linalg.norm(Vk.T @ Vk - Vsk.T @ Vsk)
        r_m = recon(Xt, Vk, mine.mean_)
        r_s = recon(Xt, Vsk, sk.mean_)
        e_tr = np.mean((X - recon(X, Vk, mine.mean_)) ** 2)
        e_te = np.mean((Xt - r_m) ** 2)
        tr_err.append(e_tr)
        te_err.append(e_te)
        print(f"     {k:>6}{dP:>20.2e}{np.max(np.abs(r_m - r_s)):>18.2e}{e_tr:>12.3f}{e_te:>12.3f}"
              f"{mine.cumulative_explained_variance_[k - 1]:>9.4f}")
    print(f"  4. error decreases with k: train {all(np.diff(tr_err) <= 1e-9)}, test {all(np.diff(te_err) <= 1e-9)}")
    full = PCAFromScratch().fit(X)
    rec = full.inverse_transform(full.transform(X))
    print(f"  5. full representation (k = {full.components_.shape[0]}, rank {full.rank_}):"
          f" max |X - Xhat| = {np.max(np.abs(X - rec)):.2e} (pixel values up to {X.max():.0f})")
    small = X[:500]
    ps = PCAFromScratch().fit(small)
    rs = ps.inverse_transform(ps.transform(small))
    print(f"     N < D case (500 patches): {ps.components_.shape[0]} components, numerical rank {ps.rank_}"
          f" (= N - 1 after centring); max |X - Xhat| = {np.max(np.abs(small - rs)):.2e}")

    # ---------------------------------------------------------------- engineering question
    print("\n  Engineering question: SVD of X vs eigen-decomposition of the covariance matrix")
    t0 = time.perf_counter()
    PCAFromScratch().fit(X)
    t_svd = time.perf_counter() - t0
    t0 = time.perf_counter()
    Xc = X - X.mean(axis=0)
    C = Xc.T @ Xc / (X.shape[0] - 1)
    w, _ = np.linalg.eigh(C)
    t_cov = time.perf_counter() - t0
    lam_svd = s**2 / (X.shape[0] - 1)
    lam_cov = np.sort(w)[::-1]
    rel = np.abs(lam_cov - lam_svd) / lam_svd
    print(f"    patches (D = 1024): SVD {t_svd:.2f} s, covariance + eigh {t_cov:.2f} s; eigenvalue"
          f" spread {lam_svd[0] / lam_svd[-1]:.1e}; max relative eigenvalue disagreement {rel.max():.1e}")
    rng = np.random.default_rng(0)
    n, D = 400, 200
    Uq, _ = np.linalg.qr(rng.normal(size=(n, D)))
    Vq, _ = np.linalg.qr(rng.normal(size=(D, D)))
    sv = np.logspace(0, -12, D)
    A = (Uq * sv) @ Vq.T
    s_svd = np.linalg.svd(A, compute_uv=False)
    s_cov = np.sqrt(np.clip(np.sort(np.linalg.eigvalsh(A.T @ A))[::-1], 0, None))
    good_svd = np.max(np.nonzero(np.abs(s_svd / sv - 1) < 0.01)[0])
    good_cov = np.max(np.nonzero(np.abs(s_cov / sv - 1) < 0.01)[0])
    print(f"    test matrix with singular values 1 ... 1e-12: SVD recovers them to 1% down to"
          f" {sv[good_svd]:.0e}; covariance route only down to {sv[good_cov]:.0e}"
          f" (~ sqrt(machine eps) = {np.sqrt(np.finfo(float).eps):.0e})")

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    ax = axes[0]
    ax.semilogy(np.arange(1, 129), mine.explained_variance_ratio_[:128], "o-", ms=3, label="mine")
    ax.semilogy(np.arange(1, 129), sk.explained_variance_ratio_[:128], "x", ms=4, label="sklearn")
    ax.set_xlabel("component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title("Scree plot (raw pixels)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    ax.semilogx(np.arange(1, 1025), mine.cumulative_explained_variance_, color="k")
    ax.set_xlabel("components k")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("Cumulative explained variance")
    ax.grid(alpha=0.3, which="both")
    ax = axes[2]
    ax.loglog(KS, tr_err, "s--", label="training patches")
    ax.loglog(KS, te_err, "o-", label="held-out patches")
    ax.set_xlabel("components k")
    ax.set_ylabel("reconstruction MSE [e-^2]")
    ax.set_title("Reconstruction error")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_pca_validation.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(3, 7, figsize=(14, 6.4))
    axes[0, 0].imshow(mine.mean_.reshape(SIZE, SIZE), cmap="gray", origin="lower")
    axes[0, 0].set_title("mean image", fontsize=8)
    for i, ax in enumerate(axes.ravel()[1:]):
        v = V[i].reshape(SIZE, SIZE)
        lim = np.abs(v).max()
        ax.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        ax.set_title(f"PC {i + 1}: {mine.explained_variance_ratio_[i]:.1%}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Mean image and the first 20 principal directions (raw pixels)")
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_components.png", dpi=120)
    plt.close(fig)
    print("\n  saved", OUT / "exp_b_pca_validation.png")
    print("  saved", OUT / "exp_b_components.png")


if __name__ == "__main__":
    main()
