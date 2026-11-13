"""
Experiment B: K-fold and nested cross-validation.

Part 1  Own K-fold index generator (kfold_indices) with unit tests, compared against sklearn KFold.
Part 2  KFold / StratifiedKFold / GroupKFold / StratifiedGroupKFold on the realistic grouped
        development set (all 5 features): fold-to-fold variability vs the truth on new images.
Part 3  Nested cross-validation, 5 outer x 3 inner folds, for
          KNN       k in {5, 11, 21, 51, 101}
          RBF SVM   C in {1, 10, 100} x gamma in {0.1, 0.3, 1}
        Inside every outer fold: fit the standardizer on the outer-training rows only (the
        Pipeline does this inside every fit), inner CV for selection, refit the winner on the
        whole outer-training portion, score once on the outer fold. Run twice: with group-aware
        splits (StratifiedGroupKFold, inner and outer) and with ordinary StratifiedKFold.
        Metric: ROC AUC.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from grouped_data import OUT, development_set, untouched_test_set
from ml_tools import ALL5, Pipeline, auc

GRIDS = {"KNN": [("knn", {"k": k}) for k in (5, 11, 21, 51, 101)],
         "RBF SVM": [("svm", {"C": C, "gamma": g}) for C in (1, 10, 100) for g in (0.1, 0.3, 1)]}
FIXED = {"KNN": ("knn", {"k": 21}), "RBF SVM": ("svm", {"C": 1, "gamma": 1})}


# --------------------------------------------------------------------------- Part 1
def kfold_indices(n, k, seed=None, shuffle=False):
    """Yield (train_idx, val_idx) for k folds over n observations.

    Fold sizes differ by at most one (the first n % k folds get one extra), matching sklearn.
    With shuffle=True the order is permuted with np.random.RandomState(seed), which is the
    generator sklearn's KFold uses, so the folds can be compared exactly.
    """
    if not 2 <= k <= n:
        raise ValueError("need 2 <= k <= n")
    idx = np.arange(n)
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1
    start = 0
    for s in sizes:
        val = idx[start:start + s]
        train = np.concatenate([idx[:start], idx[start + s:]])
        yield np.sort(train), np.sort(val)
        start += s


def test_kfold():
    results = []
    for n, k, shuffle, seed in ((10, 3, False, None), (101, 5, True, 0), (7616, 5, True, 42),
                                (50, 50, False, None), (13, 4, True, 7)):
        folds = list(kfold_indices(n, k, seed, shuffle))
        val_all = np.concatenate([v for _, v in folds])
        once = np.array_equal(np.sort(val_all), np.arange(n))
        disjoint = all(np.intersect1d(t, v).size == 0 and t.size + v.size == n for t, v in folds)
        sizes = [v.size for _, v in folds]
        balanced = max(sizes) - min(sizes) <= 1
        sk = list(KFold(k, shuffle=shuffle, random_state=seed if shuffle else None).split(np.zeros(n)))
        same = all(np.array_equal(np.sort(a[1]), b[1]) and np.array_equal(np.sort(a[0]), b[0])
                   for a, b in zip(sk, folds))
        results.append((n, k, shuffle, once, disjoint, balanced, same))
    return results


# --------------------------------------------------------------------------- Part 2
def cv_scores(splitter, X, y, g, kind, params):
    scores, comps = [], []
    for tr, va in splitter.split(X, y, g):
        p = Pipeline(kind, ALL5, **params).fit(X[tr], y[tr])
        scores.append(auc(y[va], p.score(X[va])))
        comps.append((y[va].mean(), len(np.unique(g[va])), len(set(np.unique(g[va])) & set(np.unique(g[tr])))))
    return np.array(scores), comps


# --------------------------------------------------------------------------- Part 3
def inner_select(X, y, g, grid, inner):
    means = []
    for kind, params in grid:
        s = [auc(y[va], Pipeline(kind, ALL5, **params).fit(X[tr], y[tr]).score(X[va]))
             for tr, va in inner.split(X, y, g)]
        means.append(np.mean(s))
    best = int(np.argmax(means))
    return grid[best], means[best], means


def outer_fold(tr, te, X, y, g, grid, fixed, grouped, seed):
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed) if grouped else \
        StratifiedKFold(3, shuffle=True, random_state=seed)
    (kind, params), best_inner, _ = inner_select(X[tr], y[tr], g[tr], grid, inner)
    sel = Pipeline(kind, ALL5, **params).fit(X[tr], y[tr])
    fix = Pipeline(fixed[0], ALL5, **fixed[1]).fit(X[tr], y[tr])
    return {"params": params, "best_inner": best_inner, "outer": auc(y[te], sel.score(X[te])),
            "fixed": auc(y[te], fix.score(X[te]))}


def nested(X, y, g, family, grouped, seed=0):
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed) if grouped else \
        StratifiedKFold(5, shuffle=True, random_state=seed)
    return Parallel(n_jobs=5)(delayed(outer_fold)(tr, te, X, y, g, GRIDS[family], FIXED[family],
                                                  grouped, seed + 1)
                              for tr, te in outer.split(X, y, g))


def main():
    OUT.mkdir(exist_ok=True)
    print("Experiment B: K-fold and nested cross-validation")
    print("\n  Part 1 - own kfold_indices vs sklearn KFold")
    print(f"    {'n':>6}{'k':>4}{'shuffle':>9}{'each once in val':>18}{'train/val disjoint':>20}"
          f"{'sizes +-1':>11}{'== sklearn':>12}")
    for n, k, sh, once, disj, bal, same in test_kfold():
        print(f"    {n:>6}{k:>4}{str(sh):>9}{str(once):>18}{str(disj):>20}{str(bal):>11}{str(same):>12}")
    try:
        list(kfold_indices(3, 5))
        print("    k > n not rejected  [FAIL]")
    except ValueError:
        print("    k > n correctly rejected with ValueError")

    X, y, g, _, _ = development_set()
    Xt, yt, gt, _, _ = untouched_test_set()

    # ---------------------------------------------------------------- Part 2
    kind, params = "knn", {"k": 51}
    truth = auc(yt, Pipeline(kind, ALL5, **params).fit(X, y).score(Xt))
    print(f"\n  Part 2 - 5-fold CV of KNN k=51 (all 5 features); truth on 30 new images: AUC {truth:.3f}")
    print(f"    {'splitter':<22}{'mean AUC':>9}{'fold sd':>9}{'min':>7}{'max':>7}{'val star frac sd':>18}"
          f"{'val images':>12}{'val imgs also in train':>24}")
    splitters = {"KFold (shuffled)": KFold(5, shuffle=True, random_state=0),
                 "StratifiedKFold": StratifiedKFold(5, shuffle=True, random_state=0),
                 "GroupKFold": GroupKFold(5, shuffle=True, random_state=0),
                 "StratifiedGroupKFold": StratifiedGroupKFold(5, shuffle=True, random_state=0)}
    part2 = {}
    for name, sp in splitters.items():
        rep_scores = []
        for rep in range(5):                        # 5 different shuffles of the same scheme
            sp.random_state = rep
            s, comps = cv_scores(sp, X, y, g, kind, params)
            rep_scores.append(s)
            if rep == 0:
                c0 = comps
        S = np.array(rep_scores)
        part2[name] = S
        print(f"    {name:<22}{S.mean():>9.3f}{S.std(axis=1).mean():>9.3f}{S.min():>7.3f}{S.max():>7.3f}"
              f"{np.std([c[0] for c in c0]):>18.3f}{np.mean([c[1] for c in c0]):>12.1f}"
              f"{np.mean([c[2] for c in c0]):>24.1f}")
    print("    (fold sd = sd of the 5 fold scores, averaged over 5 reshuffles; min/max over all 25 folds)")

    # ---------------------------------------------------------------- Part 3
    print("\n  Part 3 - nested cross-validation (5 outer x 3 inner folds), metric ROC AUC")
    part3 = {}
    for family in GRIDS:
        for grouped in (True, False):
            t = time.perf_counter()
            res = nested(X, y, g, family, grouped)
            part3[(family, grouped)] = res
            # final model: same selection procedure on the whole development set, then the test set
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=1) if grouped else \
                StratifiedKFold(3, shuffle=True, random_state=1)
            (k_, p_), best_full, _ = inner_select(X, y, g, GRIDS[family], inner)
            final = auc(yt, Pipeline(k_, ALL5, **p_).fit(X, y).score(Xt))
            fixed_test = auc(yt, Pipeline(FIXED[family][0], ALL5, **FIXED[family][1]).fit(X, y).score(Xt))
            label = "grouped" if grouped else "ungrouped"
            print(f"\n    {family}, {label} splits ({time.perf_counter() - t:.0f} s)")
            print(f"      {'outer fold':>10}{'selected':>26}{'best inner':>12}{'outer score':>13}{'fixed cfg':>11}")
            for i, r in enumerate(res):
                print(f"      {i:>10}{str(r['params']):>26}{r['best_inner']:>12.3f}{r['outer']:>13.3f}{r['fixed']:>11.3f}")
            bi = np.mean([r["best_inner"] for r in res])
            oo = np.array([r["outer"] for r in res])
            ff = np.array([r["fixed"] for r in res])
            print(f"      fixed configuration {FIXED[family][1]}: outer CV {ff.mean():.3f} +- {ff.std():.3f};"
                  f" untouched test {fixed_test:.3f}")
            print(f"      best inner-CV score (mean of the 5 maxima):  {bi:.3f}")
            print(f"      nested outer-fold estimate:                {oo.mean():.3f} +- {oo.std():.3f}")
            print(f"      final model (selected on all dev data: {p_}, inner best {best_full:.3f}):"
                  f" untouched test {final:.3f}")
            part3[(family, grouped, "summary")] = (ff.mean(), bi, oo.mean(), final, fixed_test)

    winners_curse(X, y, g)
    plot(part2, part3, truth)


def wc_rep(r, X, y, g, grid, n_images):
    """Select k by grouped 3-fold CV on a random subset of n_images development images, then score
    the very same fold models on the development images OUTSIDE the subset (independent images,
    identical training size). Returns, per config, (CV score, external score)."""
    sub = np.random.default_rng([r, 99]).choice(np.unique(g), n_images, replace=False)
    inn = np.isin(g, sub)
    Xs, ys, gs, Xo, yo = X[inn], y[inn], g[inn], X[~inn], y[~inn]
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=r).split(Xs, ys, gs))
    cv, ext = [], []
    for k_, p in grid:
        c, e = [], []
        for tr, va in folds:
            m = Pipeline(k_, ALL5, **p).fit(Xs[tr], ys[tr])
            c.append(auc(ys[va], m.score(Xs[va])))
            e.append(auc(yo, m.score(Xo)))
        cv.append(np.mean(c))
        ext.append(np.mean(e))
    return np.array(cv), np.array(ext)


def winners_curse(X, y, g):
    """Isolate selection optimism from the training-size effect: same fold sizes, fresh folds."""
    grid = [("knn", {"k": k}) for k in (3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 71, 101, 151, 201)]
    print("\n  Selection optimism ('winner's curse'): KNN over 14 values of k, grouped 3-fold CV on a random"
          " subset of development images;\n  the same fold models are then scored on the development images"
          " outside the subset (independent images, same training size)")
    print(f"    {'subset':>8}{'repeats':>9}{'best CV score':>15}{'its external':>14}{'gap, chosen k':>15}"
          f"{'gap, avg over all k':>21}{'selection optimism':>20}")
    for n_img, reps in ((10, 60), (25, 40)):
        res = Parallel(n_jobs=-1)(delayed(wc_rep)(r, X, y, g, grid, n_img) for r in range(reps))
        best = np.array([cv.max() for cv, _ in res])
        best_ext = np.array([ext[np.argmax(cv)] for cv, ext in res])
        gap_all = np.array([np.mean(cv - ext) for cv, ext in res])
        sel = (best - best_ext) - gap_all
        print(f"    {n_img:>5} im{reps:>9}{best.mean():>15.4f}{best_ext.mean():>14.4f}{np.mean(best - best_ext):>+15.4f}"
              f"{gap_all.mean():>+21.4f}{f'{sel.mean():+.4f} +- {sel.std() / np.sqrt(reps):.4f}':>20}")
    print("    gap = CV score - external score. 'avg over all k' is the CV bias with no selection involved;"
          " the selection optimism is the extra gap of the chosen k (+- standard error).")
    print("    Training-size effect, for comparison: in nested CV the inner models see ~53% of the"
          " development images, outer models ~80%.")


def plot(part2, part3, truth):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    names = list(part2)
    ax.boxplot([part2[n].ravel() for n in names], tick_labels=names)
    ax.axhline(truth, color="k", ls="--", label=f"truth on new images {truth:.3f}")
    ax.set_ylabel("fold AUC (5 folds x 5 reshuffles)")
    ax.set_title("KNN k=51: fold-to-fold variability by CV scheme")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    labels = ["fixed config\n(outer CV)", "best inner\nCV score", "nested outer\nestimate", "untouched\ntest"]
    x = np.arange(4)
    for k, (family, grouped) in enumerate([(f, gr) for f in GRIDS for gr in (True, False)]):
        vals = part3[(family, grouped, "summary")][:4]
        ax.plot(x + 0.06 * (k - 1.5), vals, "o-" if grouped else "s--",
                label=f"{family}, {'grouped' if grouped else 'ungrouped'} splits")
    ax.set_xticks(x, labels)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Nested cross-validation: what each number estimates")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_cross_validation.png", dpi=130)
    plt.close(fig)
    print(f"\n  saved {OUT / 'exp_b_cross_validation.png'}")


if __name__ == "__main__":
    main()
