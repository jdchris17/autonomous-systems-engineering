"""
Experiment A: K-means from scratch.

Part 1  2-D datasets with three Gaussian populations: (i) well separated, equal size and variance;
        (ii) overlapping, unequal sizes and different covariance shapes. Iteration visualisation,
        random vs K-means++ initialisation, an explicit empty-cluster test.
Part 2  Same initial centres given to KMeansScratch and sklearn KMeans(algorithm="lloyd"); centres
        matched with the Hungarian algorithm before comparing.
Part 3  200 initialisations per strategy: distribution of final inertia, how often the best
        solution is found, how many distinct solutions appear.
Part 4  Module 6 engineered features (log width, roundness, log SNR), standardised with development
        statistics; K in {2, 3, 4, 5, 6, 8, 10}; inertia and silhouette curves, initialisation
        stability (pairwise ARI), and cluster-by-subtype tables. Labels are used only afterwards.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from clustering import KMeansScratch, best_of, match_permutation
from data7 import OUT, SUBTYPES, load, standardize

KS = (2, 3, 4, 5, 6, 8, 10)


def three_gaussians(rng, kind):
    if kind == "separated":
        means = [(0, 0), (6, 0), (3, 5)]
        covs = [np.eye(2)] * 3
        n = [300, 300, 300]
    else:
        means = [(0, 0), (3, 0.5), (1.5, 3)]
        covs = [np.array([[2.0, 1.2], [1.2, 1.0]]), np.array([[0.3, 0.0], [0.0, 1.5]]),
                np.array([[1.0, -0.6], [-0.6, 0.8]])]
        n = [500, 200, 300]
    X = np.vstack([rng.multivariate_normal(m, c, k) for m, c, k in zip(means, covs, n)])
    y = np.repeat(np.arange(3), n)
    return X, y


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(70)
    data = {kind: three_gaussians(rng, kind) for kind in ("separated", "overlapping")}
    print("Experiment A: K-means from scratch")

    # ---------------------------------------------------------------- Part 1 + 2
    print("\n  Parts 1-2 - scratch vs sklearn with identical initial centres (lloyd, tol 1e-6)")
    print(f"    {'dataset':<12}{'init':<11}{'iters (mine/sk)':>16}{'inertia mine':>14}{'inertia sk':>14}"
          f"{'max |centre diff|':>19}{'label agreement':>17}{'ARI vs truth':>14}")
    for kind, (X, y) in data.items():
        for init in ("random", "k-means++"):
            m0 = KMeansScratch(3, init=init, seed=3)
            C0 = m0._init_centres(X, np.random.default_rng(3))
            mine = KMeansScratch(3, init=C0).fit(X)
            sk = KMeans(3, init=C0, n_init=1, algorithm="lloyd", tol=1e-6).fit(X)
            p = match_permutation(mine.cluster_centers_, sk.cluster_centers_)
            diff = np.abs(mine.cluster_centers_ - sk.cluster_centers_[p]).max()
            agree = np.mean(p[mine.labels_] == sk.labels_)
            print(f"    {kind:<12}{init:<11}{f'{mine.n_iter_} / {sk.n_iter_}':>16}{mine.inertia_:>14.4f}"
                  f"{sk.inertia_:>14.4f}{diff:>19.2e}{agree:>17.4f}{adjusted_rand_score(y, mine.labels_):>14.3f}")
    X, y = data["separated"]
    bad = np.array([[0.0, 0.0], [6.0, 0.0], [100.0, 100.0]])
    m = KMeansScratch(3, init=bad).fit(X)
    print(f"    empty-cluster test: third centre started at (100, 100), far from all data -> {m.n_empty_fixes_}"
          f" empty-cluster repair(s); final cluster sizes {np.bincount(m.labels_, minlength=3).tolist()},"
          f" ARI {adjusted_rand_score(y, m.labels_):.3f}")

    # iteration visualisation (overlapping data, random init)
    X, y = data["overlapping"]
    m = KMeansScratch(3, init="random", seed=11, keep_history=True).fit(X)
    steps = [0, 1, 2, min(4, m.n_iter_ - 1), m.n_iter_ - 1]
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(4 * (len(steps) + 1), 4))
    for ax, s in zip(axes, steps):
        C, lab = m.history_[s]
        ax.scatter(X[:, 0], X[:, 1], c=lab, s=5, cmap="tab10", vmin=0, vmax=9)
        traj = np.array([h[0] for h in m.history_[: s + 1]])
        for j in range(3):
            ax.plot(traj[:, j, 0], traj[:, j, 1], "k.-", lw=1)
        ax.scatter(C[:, 0], C[:, 1], marker="X", s=180, c="k")
        ax.set_title(f"iteration {s}: inertia {m.inertia_history_[s]:.0f}", fontsize=9)
    axes[-1].plot(m.inertia_history_, "o-")
    axes[-1].set_xlabel("iteration")
    axes[-1].set_ylabel("inertia")
    axes[-1].set_title("inertia never increases")
    axes[-1].grid(alpha=0.3)
    fig.suptitle("K-means iterations (overlapping data, random initialisation); black = centre trajectories")
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_iterations.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- Part 3
    z = load()
    E, _ = standardize(z["dev_eng"])
    sub = z["dev_subtype"]
    print("\n  Part 3 - initialisation sensitivity (200 initialisations each; 'found best' = within 0.01% of the"
          " best inertia, 'distinct' = solutions differing by more than 0.1%)")
    print(f"    {'dataset':<22}{'K':>3} {'init':<11}{'best inertia':>14}{'found best':>12}{'distinct sols':>15}"
          f"{'worst / best':>14}{'mean iters':>12}")
    hists = {}
    for name, Xd, K in (("separated (2-D)", data["separated"][0], 3), ("overlapping (2-D)", data["overlapping"][0], 3),
                        ("astro features (3-D)", E, 6)):
        for init in ("random", "k-means++"):
            fits = [KMeansScratch(K, init=init, seed=s).fit(Xd) for s in range(200)]
            iner = np.array([f.inertia_ for f in fits])
            best = iner.min()
            distinct = len(np.unique(np.round(iner / best, 3)))      # solutions within 0.1% grouped
            hists[(name, init)] = iner / best
            print(f"    {name:<22}{K:>3} {init:<11}{best:>14.3f}{np.mean(iner <= best * (1 + 1e-4)):>12.1%}"
                  f"{distinct:>15}{iner.max() / best:>14.3f}{np.mean([f.n_iter_ for f in fits]):>12.1f}")

    # ---------------------------------------------------------------- Part 4
    print("\n  Part 4 - astronomical source features (standardised, development images, labels unused in fitting)")
    print(f"    {'K':>3}{'inertia':>11}{'silhouette':>12}{'mean pairwise ARI (20 inits)':>30}{'ARI vs subtype':>16}{'ARI vs star/artifact':>22}")
    rows = {}
    for K in KS:
        fits = [KMeansScratch(K, init="k-means++", seed=s).fit(E) for s in range(20)]
        best = min(fits, key=lambda f: f.inertia_)
        pair = [adjusted_rand_score(fits[i].labels_, fits[j].labels_) for i in range(20) for j in range(i + 1, 20)]
        sil = silhouette_score(E, best.labels_)
        rows[K] = (best, sil, np.mean(pair))
        print(f"    {K:>3}{best.inertia_:>11.1f}{sil:>12.3f}{np.mean(pair):>30.3f}"
              f"{adjusted_rand_score(sub, best.labels_):>16.3f}{adjusted_rand_score(z['dev_y'], best.labels_):>22.3f}")
    kbest = max(KS, key=lambda k: rows[k][1])
    for K in sorted({kbest, 3, 6}):
        best = rows[K][0]
        print(f"\n    Cluster x subtype table, K = {K}" + (" (best silhouette)" if K == kbest else ""))
        print(f"      {'cluster':>8}{'n':>6}" + "".join(f"{s[:11]:>13}" for s in SUBTYPES)
              + f"{'centre (log w, round, log SNR), std. units':>46}")
        for j in np.argsort(best.cluster_centers_[:, 2]):
            m = best.labels_ == j
            print(f"      {j:>8}{m.sum():>6}" + "".join(f"{np.mean(sub[m] == s):>13.0%}" for s in SUBTYPES)
                  + f"{np.array2string(best.cluster_centers_[j], precision=2):>46}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    ax = axes[0]
    ax.plot(KS, [rows[k][0].inertia_ for k in KS], "o-")
    ax.set_xlabel("K")
    ax.set_ylabel("inertia")
    ax.set_title("Inertia vs K (astro features)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(KS, [rows[k][1] for k in KS], "o-", label="silhouette")
    ax.plot(KS, [rows[k][2] for k in KS], "s--", label="init stability (mean pairwise ARI)")
    ax.set_xlabel("K")
    ax.set_title("Silhouette and stability vs K")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[2]
    for (name, init), v in hists.items():
        if name.startswith("astro") or name.startswith("separated"):
            ax.hist(v, bins=40, histtype="step", lw=1.5, label=f"{name}, {init}")
    ax.set_yscale("log")
    ax.set_xlabel("final inertia / best inertia")
    ax.set_title("Initialisation sensitivity (200 runs)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_curves.png", dpi=130)
    plt.close(fig)
    print("\n  saved", OUT / "exp_a_iterations.png")
    print("  saved", OUT / "exp_a_curves.png")


if __name__ == "__main__":
    main()
