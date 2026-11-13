"""
Experiment D: hierarchical clustering and DBSCAN.

Part 1  Agglomerative clustering (sklearn) and SciPy linkage on an 800-patch random subset of the
        standardised engineered features: single, complete, average, Ward; dendrogram; memberships
        when the tree is cut at different heights.
Part 2  DBSCAN on all development patches (same standardised space): eps x min_samples grid; number
        of clusters, noise fraction, stability under a 10% change of eps, subtype composition.
Part 3  Geometry stress tests: two Gaussian clouds, unequal-variance ellipses, two interlocking
        crescents, dense + sparse cluster, clusters with isolated outliers. K-means (own), GMM (own,
        full), Ward and single linkage, DBSCAN.
Part 4  What DBSCAN calls noise in the astronomical features, judged with the synthetic metadata.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from clustering import GaussianMixtureScratch, KMeansScratch
from data7 import OUT, SUBTYPES, load, standardize

EPS = (0.15, 0.2, 0.3, 0.4, 0.5)
MINS = (5, 10, 20)


def composition(sub, lab, title, limit=12):
    print(f"      {title}")
    print(f"        {'cluster':>8}{'n':>6}" + "".join(f"{s[:11]:>12}" for s in SUBTYPES))
    ids = [c for c in np.unique(lab)]
    ids = sorted(ids, key=lambda c: (c == -1, -np.sum(lab == c)))[:limit]
    for c in ids:
        m = lab == c
        name = "noise" if c == -1 else str(c)
        print(f"        {name:>8}{m.sum():>6}" + "".join(f"{np.mean(sub[m] == s):>12.0%}" for s in SUBTYPES))


def stress_sets(rng):
    sets = {}
    X = np.vstack([rng.normal([0, 0], 0.6, (250, 2)), rng.normal([3, 0], 0.6, (250, 2))])
    sets["two Gaussian clouds"] = (X, np.repeat([0, 1], 250), 2)
    X = np.vstack([rng.multivariate_normal([0, 0], [[4, 0], [0, 0.15]], 300),
                   rng.multivariate_normal([0, 1.6], [[0.1, 0], [0, 0.1]], 150)])
    sets["unequal-variance ellipses"] = (X, np.repeat([0, 1], [300, 150]), 2)
    X, y = make_moons(500, noise=0.06, random_state=1)
    sets["interlocking crescents"] = (X, y, 2)
    X = np.vstack([rng.normal([0, 0], 0.2, (300, 2)), rng.normal([2.5, 0], 1.0, (100, 2))])
    sets["dense + sparse cluster"] = (X, np.repeat([0, 1], [300, 100]), 2)
    X = np.vstack([rng.normal([0, 0], 0.4, (200, 2)), rng.normal([3, 3], 0.4, (200, 2)), rng.uniform(-4, 7, (20, 2))])
    sets["clusters + isolated outliers"] = (X, np.r_[np.repeat([0, 1], 200), np.full(20, -1)], 2)
    return sets


def kdist_eps(X, k=10, q=0.95):
    d = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)[0][:, -1]
    return float(np.quantile(d, q))


def main():
    OUT.mkdir(exist_ok=True)
    z = load()
    S, _ = standardize(z["dev_eng"])
    sub = z["dev_subtype"]
    rng = np.random.default_rng(72)
    print("Experiment D: hierarchical clustering and DBSCAN (standardised log width, roundness, log SNR)")

    # ---------------------------------------------------------------- Part 1
    idx = rng.choice(len(S), 800, replace=False)
    Ss, subs = S[idx], sub[idx]
    print("\n  Part 1 - agglomerative clustering on 800 random development patches, cut at 6 clusters")
    print(f"    {'linkage':<10}{'cluster sizes (largest first)':<42}{'ARI vs subtype':>16}")
    for link in ("single", "complete", "average", "ward"):
        lab = AgglomerativeClustering(6, linkage=link).fit_predict(Ss)
        sizes = sorted((int(v) for v in np.bincount(lab)), reverse=True)
        print(f"    {link:<10}{str(sizes):<42}{adjusted_rand_score(subs, lab):>16.3f}")
    Z = linkage(Ss, "ward")
    print("    Ward dendrogram cut at different heights:")
    for k in (2, 3, 6, 10):
        lab = fcluster(Z, k, criterion="maxclust")
        h = Z[-(k - 1), 2] if k > 1 else Z[-1, 2]
        print(f"      {k:>3} clusters (cut below merge height {h:6.2f}): ARI vs subtype {adjusted_rand_score(subs, lab):.3f}")
    composition(subs, fcluster(Z, 6, criterion="maxclust"), "Ward, 6 clusters - subtype composition")
    Zs = linkage(Ss, "single")
    composition(subs, fcluster(Zs, 6, criterion="maxclust"), "single linkage, 6 clusters - chaining")
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for ax, (Zl, name) in zip(axes, ((Z, "Ward"), (Zs, "single"))):
        dendrogram(Zl, truncate_mode="lastp", p=40, ax=ax, color_threshold=Zl[-5, 2], no_labels=True)
        ax.set_title(f"{name} linkage dendrogram (last 40 merges, 800 patches)")
        ax.set_ylabel("merge height")
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_dendrograms.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- Part 2
    print("\n  Part 2 - DBSCAN on all 2982 development patches")
    print(f"    {'eps':>6}{'min_samples':>13}{'clusters':>10}{'noise':>8}{'ARI vs subtype':>16}{'stability (ARI eps vs 1.1 eps)':>32}")
    db = {}
    for eps in EPS:
        for ms in MINS:
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(S)
            lab2 = DBSCAN(eps=eps * 1.1, min_samples=ms).fit_predict(S)
            db[(eps, ms)] = lab
            nc = len(set(lab)) - (1 if -1 in lab else 0)
            print(f"    {eps:>6}{ms:>13}{nc:>10}{np.mean(lab == -1):>8.1%}{adjusted_rand_score(sub, lab):>16.3f}"
                  f"{adjusted_rand_score(lab, lab2):>32.3f}")
    eps_k = kdist_eps(S, 10)
    lab = DBSCAN(eps=eps_k, min_samples=10).fit_predict(S)
    print(f"    k-distance heuristic (95th percentile of the 10th-neighbour distance) -> eps = {eps_k:.3f}:"
          f" {len(set(lab)) - (-1 in lab)} clusters, {np.mean(lab == -1):.1%} noise")
    composition(sub, db[(0.3, 10)], "DBSCAN eps = 0.3, min_samples = 10 - composition")

    # ---------------------------------------------------------------- Part 3
    sets = stress_sets(rng)
    methods = ("K-means", "GMM (full)", "Ward", "single linkage", "DBSCAN")
    print("\n  Part 3 - geometry stress tests: ARI vs the generating labels (outliers excluded from ARI;"
          " DBSCAN noise counted as its own label)")
    print(f"    {'dataset':<30}" + "".join(f"{m:>16}" for m in methods) + f"{'DBSCAN outlier recall':>23}")
    fig, axes = plt.subplots(len(sets), len(methods) + 1, figsize=(3.4 * (len(methods) + 1), 3.1 * len(sets)))
    for r, (name, (X, y, k)) in enumerate(sets.items()):
        Xs = (X - X.mean(0)) / X.std(0)
        eps = kdist_eps(Xs, 8, 0.9)
        labs = {"K-means": KMeansScratch(k, seed=0).fit(Xs).labels_,
                "GMM (full)": max((GaussianMixtureScratch(k, "full", seed=s).fit(Xs) for s in range(5)),
                                  key=lambda m: m.lower_bound_).predict(Xs),
                "Ward": AgglomerativeClustering(k, linkage="ward").fit_predict(Xs),
                "single linkage": AgglomerativeClustering(k, linkage="single").fit_predict(Xs),
                "DBSCAN": DBSCAN(eps=eps, min_samples=8).fit_predict(Xs)}
        keep = y != -1
        cells = [f"{adjusted_rand_score(y[keep], labs[m][keep]):>16.3f}" for m in methods]
        rec = f"{np.mean(labs['DBSCAN'][~keep] == -1):.0%}" if (~keep).any() else "-"
        print(f"    {name:<30}" + "".join(cells) + f"{rec:>23}")
        axes[r, 0].scatter(X[:, 0], X[:, 1], c=y, s=4, cmap="tab10", vmin=-1, vmax=8)
        axes[r, 0].set_ylabel(name, fontsize=9)
        axes[r, 0].set_title("truth" if r == 0 else "")
        for c, m in enumerate(methods, start=1):
            axes[r, c].scatter(X[:, 0], X[:, 1], c=labs[m], s=4, cmap="tab10", vmin=-1, vmax=8)
            axes[r, c].set_title(f"{m}: ARI {adjusted_rand_score(y[keep], labs[m][keep]):.2f}", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_stress_tests.png", dpi=110)
    plt.close(fig)

    # ---------------------------------------------------------------- Part 4
    lab = db[(0.3, 10)]
    noise = lab == -1
    print(f"\n  Part 4 - what DBSCAN (eps 0.3, min_samples 10) calls noise: {noise.sum()} of {len(S)} patches")
    print(f"    {'subtype':<14}{'share of all':>14}{'share of noise':>16}{'enrichment':>12}{'noise rate':>12}")
    for s in SUBTYPES:
        a, b = np.mean(sub == s), np.mean(sub[noise] == s)
        print(f"    {s:<14}{a:>14.1%}{b:>16.1%}{b / a:>12.2f}{np.mean(noise[sub == s]):>12.1%}")
    snr = z["dev_snr_true"]
    q = np.quantile(snr, [0.5, 0.9, 0.99])
    print(f"    true SNR of noise points: median {np.median(snr[noise]):.1f} vs {np.median(snr):.1f} overall;"
          f" {np.mean(snr[noise] > q[1]):.0%} are above the 90th SNR percentile, {np.mean(snr[noise] > q[2]):.0%} above the 99th")
    bright_noise = noise & (sub == "bright star")
    print(f"    bright stars flagged as noise: {bright_noise.sum()}, median true SNR {np.median(snr[bright_noise]):.0f}")
    # are they rare-but-valid, or badly measured? compare with the measurement flags and their own subtype
    m_obs = z["dev_eng"]
    floor = np.isclose(m_obs[:, 0], np.log(0.2))
    print(f"    noise points on the width floor (unresolved measurement): {np.mean(floor[noise]):.0%} vs"
          f" {np.mean(floor):.0%} overall; failed measurements among noise: {np.sum(noise & np.isclose(m_obs[:, 2], np.log(0.1)))}")
    for s in ("faint star", "hot pixel", "elongated"):
        m1, m2 = noise & (sub == s), ~noise & (sub == s)
        if m1.sum() >= 3:
            print(f"    {s:<11} noise vs non-noise medians (log w, roundness, log SNR):"
                  f" {np.round(np.median(m_obs[m1], 0), 2)} vs {np.round(np.median(m_obs[m2], 0), 2)}")
    print("  saved", OUT / "exp_d_dendrograms.png")
    print("  saved", OUT / "exp_d_stress_tests.png")


if __name__ == "__main__":
    main()
