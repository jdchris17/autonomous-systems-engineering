"""
Experiment C: learning curves, validation curves and hyperparameter-selection stability.

Realistic grouped development set (50 images), all 5 engineered features, metric ROC AUC.

Part 1  Learning curves in INDEPENDENT IMAGES: 10 repeats; each repeat holds out 10 images for
        evaluation and trains on N in {5, 10, 20, 30, 40} of the other 40 images.
        Plus a groups-vs-candidates check: the same number of training candidates taken either
        from 10 whole images or spread over 40 images.
Part 2  Validation curves with StratifiedGroupKFold(5): KNN k, RBF SVM C x gamma, tree depth;
        mean and fold-to-fold standard deviation.
Part 3  Selection stability: the whole selection procedure (StratifiedGroupKFold(3) on a fresh
        50-image development set) repeated on 10 independently generated development sets;
        each selected model is scored on the common untouched test set (30 new images), both
        overall and per image (each image is a different imaging environment).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from grouped_data import (N_DEV_IMAGES, OUT, PER_IMAGE, generate_grouped_candidates,
                          development_set, untouched_test_set)
from ml_tools import ALL5, Pipeline, auc

FAMILIES = {"logistic": ("logistic", {"lam": 1e-4}), "KNN": ("knn", {"k": 21}),
            "RBF SVM": ("svm", {"C": 1, "gamma": 1}), "tree": ("tree", {"depth": 6})}
N_IMAGES = (5, 10, 20, 30, 40)
REPEATS = 10
VC = {"KNN": [("knn", {"k": k}) for k in (1, 3, 5, 11, 21, 51, 101, 201)],
      "tree": [("tree", {"depth": d}) for d in (2, 3, 4, 6, 8, 10, 15, None)],
      "RBF SVM": [("svm", {"C": C, "gamma": gm}) for C in (0.1, 1, 10, 100) for gm in (0.03, 0.1, 0.3, 1, 3)]}
SEL = {"KNN": [("knn", {"k": k}) for k in (5, 11, 21, 51, 101)],
       "RBF SVM": [("svm", {"C": C, "gamma": gm}) for C in (1, 10, 100) for gm in (0.1, 0.3, 1)],
       "tree": [("tree", {"depth": d}) for d in (2, 4, 6, 8, 10, None)]}


# --------------------------------------------------------------------------- Part 1
def lc_repeat(rep, X, y, g):
    rng = np.random.default_rng([rep, 31])
    ids = rng.permutation(np.unique(g))
    ev, pool = ids[:10], ids[10:]
    ev_m = np.isin(g, ev)
    out = {}
    for n in N_IMAGES:
        tr_m = np.isin(g, pool[:n])
        for fam, (kind, params) in FAMILIES.items():
            p = Pipeline(kind, ALL5, **params).fit(X[tr_m], y[tr_m])
            out[(fam, n)] = (auc(y[tr_m], p.score(X[tr_m])), auc(y[ev_m], p.score(X[ev_m])), tr_m.sum())
    # same candidate budget: 10 whole images vs the same number spread over 40 images
    few = np.nonzero(np.isin(g, pool[:10]))[0]
    many = rng.choice(np.nonzero(np.isin(g, pool[:40]))[0], size=few.size, replace=False)
    for fam, (kind, params) in FAMILIES.items():
        for lab, idx in (("10 images", few), ("40 images", many)):
            p = Pipeline(kind, ALL5, **params).fit(X[idx], y[idx])
            out[(fam, lab)] = auc(y[ev_m], p.score(X[ev_m]))
    out["budget"] = few.size
    return out


# --------------------------------------------------------------------------- Part 2
def fold_score(kind, params, tr, va, X, y):
    return auc(y[va], Pipeline(kind, ALL5, **params).fit(X[tr], y[tr]).score(X[va]))


def validation_curves(X, y, g):
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    jobs = [(fam, i, kind, params, tr, va) for fam, grid in VC.items()
            for i, (kind, params) in enumerate(grid) for tr, va in folds]
    scores = Parallel(n_jobs=-1)(delayed(fold_score)(k, p, tr, va, X, y) for _, _, k, p, tr, va in jobs)
    res = {}
    for (fam, i, *_), s in zip(jobs, scores):
        res.setdefault((fam, i), []).append(s)
    return {k: np.array(v) for k, v in res.items()}


# --------------------------------------------------------------------------- Part 3
def select_on(dataset_seed, Xt, yt, gt):
    X, y, g, _, _ = generate_grouped_candidates(N_DEV_IMAGES, PER_IMAGE, np.random.default_rng(dataset_seed))
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=dataset_seed).split(X, y, g))
    out = {}
    for fam, grid in SEL.items():
        means = [np.mean([fold_score(k, p, tr, va, X, y) for tr, va in folds]) for k, p in grid]
        kind, params = grid[int(np.argmax(means))]
        m = Pipeline(kind, ALL5, **params).fit(X, y)
        s = m.score(Xt)
        per_image = [auc(yt[gt == i], s[gt == i]) for i in np.unique(gt)
                     if 0 < yt[gt == i].sum() < (gt == i).sum()]
        out[fam] = {"params": params, "cv": max(means), "test": auc(yt, s),
                    "img_mean": float(np.mean(per_image)), "img_sd": float(np.std(per_image)),
                    "img_p10": float(np.percentile(per_image, 10))}
    return out


def main():
    OUT.mkdir(exist_ok=True)
    X, y, g, _, _ = development_set()
    Xt, yt, gt, _, _ = untouched_test_set()
    print("Experiment C: learning curves, validation curves, selection stability (metric: ROC AUC)")

    # ---------------------------------------------------------------- Part 1
    lc = Parallel(n_jobs=REPEATS)(delayed(lc_repeat)(r, X, y, g) for r in range(REPEATS))
    print(f"\n  Part 1 - learning curves: training images N, 10 held-out images, {REPEATS} repeats"
          " (mean train AUC / held-out AUC +- sd)")
    print(f"    {'N images':>9}{'~cands':>8}" + "".join(f"{f:>22}" for f in FAMILIES))
    for n in N_IMAGES:
        cells = []
        for f in FAMILIES:
            tr = np.array([r[(f, n)][0] for r in lc])
            ev = np.array([r[(f, n)][1] for r in lc])
            cells.append(f"{tr.mean():.3f} / {ev.mean():.3f}+-{ev.std():.3f}")
        print(f"    {n:>9}{np.mean([r[('logistic', n)][2] for r in lc]):>8.0f}" + "".join(f"{c:>22}" for c in cells))
    budget = np.mean([r["budget"] for r in lc])
    print(f"\n    Same candidate budget (~{budget:.0f} candidates): from 10 whole images vs spread over 40 images"
          " (held-out AUC, mean +- sd)")
    for f in FAMILIES:
        a = np.array([r[(f, "10 images")] for r in lc])
        b = np.array([r[(f, "40 images")] for r in lc])
        print(f"      {f:<10} 10 images {a.mean():.3f} +- {a.std():.3f}   40 images {b.mean():.3f} +- {b.std():.3f}"
              f"   gain {np.mean(b - a):+.3f} (40 better in {np.mean(b > a):.0%} of repeats)")

    # ---------------------------------------------------------------- Part 2
    vc = validation_curves(X, y, g)
    print("\n  Part 2 - validation curves, StratifiedGroupKFold(5): mean AUC +- fold sd")
    for fam in ("KNN", "tree"):
        print(f"    {fam}: " + ", ".join(f"{list(VC[fam][i][1].values())[0]}: {vc[(fam, i)].mean():.3f}"
                                         f"+-{vc[(fam, i)].std():.3f}" for i in range(len(VC[fam]))))
    print("    RBF SVM (rows C, columns gamma):")
    print("      " + f"{'':>8}" + "".join(f"{gm:>16}" for gm in (0.03, 0.1, 0.3, 1, 3)))
    for ci, C in enumerate((0.1, 1, 10, 100)):
        print(f"      {f'C={C}':>8}" + "".join(f"{vc[('RBF SVM', ci * 5 + j)].mean():>9.3f}+-{vc[('RBF SVM', ci * 5 + j)].std():.3f}"
                                         for j in range(5)))

    # ---------------------------------------------------------------- Part 3
    sel = Parallel(n_jobs=10)(delayed(select_on)(7000 + r, Xt, yt, gt) for r in range(10))
    print("\n  Part 3 - selection repeated on 10 independent development sets (50 images each);"
          " selected model scored on the untouched test set")
    print(f"    {'dataset':>8}" + "".join(f"{f:>34}" for f in SEL))
    for r, s in enumerate(sel):
        print(f"    {r:>8}" + "".join(f"{str(s[f]['params']):>22} {s[f]['test']:.3f}".rjust(34) for f in SEL))
    print("    selection frequency:")
    for f in SEL:
        ps = [str(s[f]["params"]) for s in sel]
        u, c = np.unique(ps, return_counts=True)
        print(f"      {f:<8}" + ", ".join(f"{a} x{b}" for a, b in sorted(zip(u, c), key=lambda t: -t[1])))
    print(f"\n    {'family':<10}{'CV best':>9}{'test AUC':>10}{'sd over datasets':>18}{'per-image AUC':>15}"
          f"{'per-image sd':>14}{'worst-10% image':>17}")
    for f in SEL:
        cv = np.array([s[f]["cv"] for s in sel])
        te = np.array([s[f]["test"] for s in sel])
        print(f"    {f:<10}{cv.mean():>9.3f}{te.mean():>10.3f}{te.std():>18.4f}"
              f"{np.mean([s[f]['img_mean'] for s in sel]):>15.3f}{np.mean([s[f]['img_sd'] for s in sel]):>14.3f}"
              f"{np.mean([s[f]['img_p10'] for s in sel]):>17.3f}")

    plot(lc, vc)


def plot(lc, vc):
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8))
    ax = axes[0]
    for f, col in zip(FAMILIES, ("tab:gray", "tab:blue", "tab:red", "tab:green")):
        tr = np.array([[r[(f, n)][0] for n in N_IMAGES] for r in lc])
        ev = np.array([[r[(f, n)][1] for n in N_IMAGES] for r in lc])
        ax.plot(N_IMAGES, tr.mean(0), "--", color=col, alpha=0.7)
        ax.errorbar(N_IMAGES, ev.mean(0), yerr=ev.std(0), fmt="o-", color=col, capsize=3, label=f)
    ax.set_xlabel("independent training images")
    ax.set_ylabel("ROC AUC (dashed = training)")
    ax.set_title("Learning curves (10 held-out images)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ks = [p["k"] for _, p in VC["KNN"]]
    m = np.array([vc[("KNN", i)].mean() for i in range(len(ks))])
    s = np.array([vc[("KNN", i)].std() for i in range(len(ks))])
    ax.errorbar(ks, m, yerr=s, fmt="o-", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("k")
    ax.set_title("KNN validation curve (grouped 5-fold, +-1 fold sd)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ds = [str(p["depth"]) for _, p in VC["tree"]]
    m = np.array([vc[("tree", i)].mean() for i in range(len(ds))])
    s = np.array([vc[("tree", i)].std() for i in range(len(ds))])
    ax.errorbar(ds, m, yerr=s, fmt="o-", capsize=3, color="tab:green")
    ax.set_xlabel("max_depth")
    ax.set_title("Tree validation curve")
    ax.grid(alpha=0.3)

    ax = axes[3]
    M = np.array([[vc[("RBF SVM", ci * 5 + j)].mean() for j in range(5)] for ci in range(4)])
    im = ax.imshow(M, cmap="viridis", origin="lower")
    for ci in range(4):
        for j in range(5):
            ax.text(j, ci, f"{M[ci, j]:.3f}\n+-{vc[('RBF SVM', ci * 5 + j)].std():.3f}", ha="center",
                    va="center", fontsize=7, color="w" if M[ci, j] < M.mean() else "k")
    ax.set_xticks(range(5), [0.03, 0.1, 0.3, 1, 3])
    ax.set_yticks(range(4), [0.1, 1, 10, 100])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    ax.set_title("RBF SVM validation surface")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_learning_validation_curves.png", dpi=130)
    plt.close(fig)
    print(f"\n  saved {OUT / 'exp_c_learning_validation_curves.png'}")


if __name__ == "__main__":
    main()
