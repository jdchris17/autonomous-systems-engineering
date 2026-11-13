"""
Experiment E: PCA vs LDA vs nonlinear embeddings.

Part 1  Binary LDA from scratch on the 3 hand-engineered features (class means, within- and
        between-class scatter, w ~ S_W^-1 (m1 - m0)), compared with sklearn
        LinearDiscriminantAnalysis; projection onto the discriminant axis.
Part 2  Same training groups (60 development images) and test groups (30 untouched images):
          engineered features: logistic regression on the 3 features vs LDA (1 axis) vs PCA (1, 2 axes)
          pixels (1024, background-subtracted): logistic regression on pixels, LDA with shrinkage
          (own implementation, shrinkage chosen by grouped CV), PCA(k) -> LDA, PCA(k) -> logistic,
          and sklearn's Ledoit-Wolf LDA.
        A binary LDA representation is a single number, so the classifier on top is just a
        threshold; comparing with a directly trained linear classifier separates the value of the
        representation from the behaviour of the final classifier.
Part 3  t-SNE of a stratified subset (PCA to 50 dimensions first), coloured by class, subtype and a
        nuisance variable, at three perplexities.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from exp_c_reconstruction import preprocess
from pca_lda import LDAFromScratch, PCAFromScratch
from patches import SUBTYPES, development_set, engineered_features, untouched_test_set

OUT = __import__("pathlib").Path(__file__).parent / "outputs"
COLS = dict(zip(SUBTYPES, ("tab:blue", "tab:cyan", "tab:red", "tab:orange", "tab:purple", "tab:gray")))


def acc(y, yhat):
    return float(np.mean(y == yhat))


def grouped_cv(fn, X, y, g, seed=0):
    return np.mean([fn(X[tr], y[tr], X[va], y[va])
                    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g)])


def main():
    OUT.mkdir(exist_ok=True)
    d, t = development_set(), untouched_test_set()
    y, g, yt = d["y"], d["group"], t["y"]
    F, Ft = engineered_features(d["X"]), engineered_features(t["X"])
    print("Experiment E: PCA vs LDA vs nonlinear embeddings")

    # ---------------------------------------------------------------- Part 1
    lda = LDAFromScratch().fit(F, y)
    sk = LinearDiscriminantAnalysis(solver="eigen").fit(F, y)
    sk_svd = LinearDiscriminantAnalysis(solver="svd").fit(F, y)
    cos = abs(lda.w_ @ sk.coef_.ravel()) / np.linalg.norm(lda.w_) / np.linalg.norm(sk.coef_)
    print("\n  Part 1 - binary LDA on (log width, roundness, log SNR), 60 development images")
    print(f"    class means: artifact {np.round(lda.means_[0], 3)}, star {np.round(lda.means_[1], 3)}")
    print(f"    S_W = {np.round(lda.S_W_, 1).tolist()}")
    print(f"    S_B = (m1 - m0)(m1 - m0)^T, rank {np.linalg.matrix_rank(lda.S_B_)}")
    print(f"    w = S_W^-1 (m1 - m0) direction: {np.round(lda.direction_, 4)}")
    print(f"    vs sklearn LDA (eigen): |cos| = {cos:.10f}; decision values max diff"
          f" {np.max(np.abs(lda.decision_function(Ft) - sk.decision_function(Ft))):.2e};"
          f" identical test predictions {np.mean(lda.predict(Ft) == sk.predict(Ft)):.4f}")
    corr = np.corrcoef(lda.transform(Ft), sk_svd.transform(Ft).ravel())[0, 1]
    print(f"    vs sklearn LDA (svd) transform(): correlation of the projections {abs(corr):.10f}")
    rng = np.random.default_rng(0)
    rand_ratio = [lda.fisher_ratio(v) for v in rng.normal(size=(2000, 3))]
    J = lda.fisher_ratio()
    pc1 = PCAFromScratch(1).fit(StandardScaler().fit_transform(F)).components_[0] / StandardScaler().fit(F).scale_
    print(f"    Fisher ratio J(w) = w^T S_B w / w^T S_W w relative to the LDA direction: LDA 1.000;"
          f" best of 2000 random directions {max(rand_ratio) / J:.3f}; first PCA direction {lda.fisher_ratio(pc1) / J:.3f}")
    print("    (decision values differ slightly from sklearn because the pooled covariance is divided by N - 2"
          " here; the direction is identical)")

    # ---------------------------------------------------------------- Part 2a: engineered features
    print("\n  Part 2 - same training groups (development) and test groups (untouched); test ROC AUC / accuracy")
    sc = StandardScaler().fit(F)
    A, At = sc.transform(F), sc.transform(Ft)
    rows = []
    lr = LogisticRegression(max_iter=5000).fit(A, y)
    rows.append(("3 features", "logistic regression on all 3 features", roc_auc_score(yt, lr.decision_function(At)),
                 acc(yt, lr.predict(At))))
    l = LDAFromScratch().fit(A, y)
    rows.append(("3 features", "LDA: 1 discriminant axis, LDA threshold", roc_auc_score(yt, l.transform(At)),
                 acc(yt, l.predict(At))))
    for k in (1, 2):
        p = PCAFromScratch(k).fit(A)
        m = LogisticRegression(max_iter=5000).fit(p.transform(A), y)
        rows.append(("3 features", f"PCA {k} axis -> logistic", roc_auc_score(yt, m.decision_function(p.transform(At))),
                     acc(yt, m.predict(p.transform(At)))))

    # ---------------------------------------------------------------- Part 2b: pixels
    B, _ = preprocess(d["X"], "B")
    Bt, _ = preprocess(t["X"], "B")
    scp = StandardScaler().fit(B)
    S, St = scp.transform(B), scp.transform(Bt)
    l0 = LDAFromScratch().fit(S, y)
    cond = np.linalg.cond(l0.S_W_)
    print(f"    pixel within-class scatter S_W: 1024 x 1024, condition number {cond:.2e}"
          f" (N = {len(y)} > D, so invertible but badly conditioned)")

    def lr_cv(C):
        return grouped_cv(lambda a, b, c, e: roc_auc_score(e, LogisticRegression(C=C, max_iter=5000).fit(a, b).decision_function(c)), S, y, g)
    Cs = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    best_C = max(Cs, key=lr_cv)
    m = LogisticRegression(C=best_C, max_iter=5000).fit(S, y)
    rows.append(("1024 pixels", f"logistic regression on pixels (C={best_C} by grouped CV)",
                 roc_auc_score(yt, m.decision_function(St)), acc(yt, m.predict(St))))

    def lda_cv(a):
        return grouped_cv(lambda x, b, c, e: roc_auc_score(e, LDAFromScratch(a).fit(x, b).decision_function(c)), S, y, g)
    shr = (0.0, 0.01, 0.1, 0.3, 0.6, 0.9, 0.97, 0.99, 1.0)
    shr_scores = {a: lda_cv(a) for a in shr}
    a_best = max(shr_scores, key=shr_scores.get)
    print("    own shrinkage LDA, grouped-CV AUC by shrinkage: " + ", ".join(f"{a}: {v:.3f}" for a, v in shr_scores.items()))
    l = LDAFromScratch(a_best).fit(S, y)
    rows.append(("1024 pixels", f"LDA, own shrinkage {a_best} (grouped CV)", roc_auc_score(yt, l.decision_function(St)),
                 acc(yt, l.predict(St))))
    l00 = LDAFromScratch(0.0).fit(S, y)
    rows.append(("1024 pixels", "LDA, no shrinkage", roc_auc_score(yt, l00.decision_function(St)), acc(yt, l00.predict(St))))
    lw = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(S, y)
    rows.append(("1024 pixels", "sklearn LDA, Ledoit-Wolf shrinkage", roc_auc_score(yt, lw.decision_function(St)),
                 acc(yt, lw.predict(St))))
    pca = PCAFromScratch(128).fit(B)
    Z, Zt = pca.transform(B), pca.transform(Bt)
    for k in (8, 16, 32, 64, 128):
        pl = LDAFromScratch().fit(Z[:, :k], y)
        pm = LogisticRegression(max_iter=5000).fit(StandardScaler().fit(Z[:, :k]).transform(Z[:, :k]), y)
        sk_ = StandardScaler().fit(Z[:, :k])
        rows.append(("1024 pixels", f"PCA {k} -> LDA (1 axis)", roc_auc_score(yt, pl.decision_function(Zt[:, :k])),
                     acc(yt, pl.predict(Zt[:, :k]))))
        rows.append(("1024 pixels", f"PCA {k} -> logistic", roc_auc_score(yt, pm.decision_function(sk_.transform(Zt[:, :k]))),
                     acc(yt, pm.predict(sk_.transform(Zt[:, :k])))))
    print(f"    {'inputs':<13}{'method':<56}{'test AUC':>9}{'accuracy':>10}")
    for r in rows:
        print(f"    {r[0]:<13}{r[1]:<56}{r[2]:>9.3f}{r[3]:>10.3f}")

    # projections figure
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, (lab, z) in zip(axes, (("LDA axis, 3 features", LDAFromScratch().fit(A, y).transform(At)),
                                   ("PC 1, 3 features", PCAFromScratch(1).fit(A).transform(At).ravel()),
                                   (f"LDA axis, pixels (shrinkage {a_best})", l.transform(St)))):
        bins = np.linspace(np.percentile(z, 0.5), np.percentile(z, 99.5), 50)
        ax.hist(z[yt == 0], bins=bins, alpha=0.55, density=True, color="tab:red", label="artifact")
        ax.hist(z[yt == 1], bins=bins, alpha=0.55, density=True, color="tab:blue", label="star")
        ax.set_title(f"{lab}: test AUC {max(roc_auc_score(yt, z), 1 - roc_auc_score(yt, z)):.3f}")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_e_projections.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- Part 3: t-SNE
    per = 200
    idx = np.concatenate([rng.choice(np.nonzero(d["subtype"] == s)[0], per, replace=False) for s in SUBTYPES])
    Bs, _ = preprocess(d["X"][idx], "B")
    Cs_, _ = preprocess(d["X"][idx], "C")
    print(f"\n  Part 3 - t-SNE on a stratified subset ({per} patches per subtype from the development images),"
          " PCA to 50 dimensions first")
    embeds = {}
    for name, M in (("B background-subtracted", Bs), ("C flux-normalised", Cs_)):
        Zs = PCAFromScratch(50).fit_transform(M)
        for perp in (5, 30, 100):
            if name.startswith("C") and perp != 30:
                continue
            embeds[(name, perp)] = TSNE(perplexity=perp, init="pca", random_state=0).fit_transform(Zs)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10.5))
    sub = d["subtype"][idx]
    yy = d["y"][idx]
    logB = np.log10(d["B"][idx])
    snr = np.log10(d["snr_true"][idx])
    E = embeds[("B background-subtracted", 30)]
    ax = axes[0, 0]
    ax.scatter(E[yy == 0, 0], E[yy == 0, 1], s=5, c="tab:red", label="artifact")
    ax.scatter(E[yy == 1, 0], E[yy == 1, 1], s=5, c="tab:blue", label="star")
    ax.set_title("B, perplexity 30: coloured by class")
    ax.legend(markerscale=3)
    for ax, (key, title) in zip((axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]),
                                ((("B background-subtracted", 30), "B, perplexity 30: by subtype"),
                                 (("B background-subtracted", 5), "B, perplexity 5: by subtype"),
                                 (("B background-subtracted", 100), "B, perplexity 100: by subtype"),
                                 (("C flux-normalised", 30), "C flux-normalised, perplexity 30: by subtype"))):
        E2 = embeds[key]
        for s in SUBTYPES:
            m = sub == s
            ax.scatter(E2[m, 0], E2[m, 1], s=5, c=COLS[s], label=s)
        ax.set_title(title)
    axes[0, 1].legend(markerscale=3, fontsize=8)
    for ax, (v, lab) in zip((axes[0, 2], axes[0, 3]), ((snr, "log10 true SNR"), (logB, "log10 image background"))):
        sc_ = ax.scatter(E[:, 0], E[:, 1], s=5, c=v, cmap="viridis")
        fig.colorbar(sc_, ax=ax, shrink=0.8, label=lab)
        ax.set_title(f"B, perplexity 30: coloured by {lab}")
    axes[1, 3].axis("off")
    axes[1, 3].text(0.0, 0.5, "Reading t-SNE safely:\n- it preserves NEIGHBOURHOODS, not distances\n"
                    "- cluster sizes and gaps between clusters\n  are not meaningful\n- the picture changes with perplexity\n"
                    "- no transform() for new patches: it is a\n  visualisation, not a feature extractor\n"
                    "- fit on development images only", fontsize=11, va="center")
    for ax in axes.ravel()[:7]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "exp_e_tsne.png", dpi=120)
    plt.close(fig)

    # neighbourhood purity in the embedding vs in PCA space (quantifies what the picture shows)
    from sklearn.neighbors import NearestNeighbors

    def purity(Xe, labels, k=10):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Xe)
        ind = nn.kneighbors(Xe, return_distance=False)[:, 1:]
        return float(np.mean(labels[ind] == labels[:, None]))
    Z50 = PCAFromScratch(50).fit_transform(Bs)
    print(f"    10-nearest-neighbour label agreement (subtype / class):")
    print(f"      PCA-50 space:            {purity(Z50, sub):.3f} / {purity(Z50, yy):.3f}")
    for (name, perp), E2 in embeds.items():
        print(f"      t-SNE {name[:1]}, perplexity {perp:<4}  {purity(E2, sub):.3f} / {purity(E2, yy):.3f}")
    print("  saved", OUT / "exp_e_projections.png")
    print("  saved", OUT / "exp_e_tsne.png")


if __name__ == "__main__":
    main()
