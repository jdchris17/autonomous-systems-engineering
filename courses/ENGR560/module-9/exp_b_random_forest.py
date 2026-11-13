"""
Experiment B: random forests and feature randomisation (scikit-learn RandomForestClassifier).

Representations: eng (3 engineered features) and hybrid (eng + 8 PCA-C components = 11 features).
Grouped validation: StratifiedGroupKFold(5) over the 60 development images; final checks on the 30
untouched images.

Part 1  single tree vs random forest: training vs grouped-validation AUC
Part 2  max_features in {1, 2, 3} (eng) and {1, 2, 3, 5, 8, 11} (hybrid)
Part 3  M in {10, 25, 50, 100, 200} x max_depth x min_samples_leaf, selected by grouped CV
Part 4  tree-error correlation on validation images: bagged trees (max_features = all) vs random
        forest (max_features = 1 or 3); Breiman's strength / correlation
Part 5  OOB vs grouped CV vs test, and a group-aware OOB from image-level bagging
Feature importance: impurity (MDI) vs permutation importance on held-out images
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from data9 import OUT, load
from ensembles import BaggingScratch

ENG = ["log width", "roundness", "log SNR"]
HYB = ENG + [f"PC-C {i + 1}" for i in range(8)]


def cv_auc(make, X, y, folds, train=False):
    va, tr = [], []
    for a, b in folds:
        m = make().fit(X[a], y[a])
        va.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
        if train:
            tr.append(roc_auc_score(y[a], m.predict_proba(X[a])[:, 1]))
    return (np.mean(va), np.std(va)) if not train else (np.mean(tr), np.mean(va), np.std(va))


def grid_cell(M, depth, leaf, X, y, folds):
    return cv_auc(lambda: RandomForestClassifier(M, max_depth=depth, min_samples_leaf=leaf, n_jobs=1, random_state=0), X, y, folds)


def tree_errors(model_trees, X, y, threshold=0.5):
    E = np.array([(t.predict_proba(X)[:, -1] >= threshold) != y for t in model_trees])
    return E


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    y, g = d["dev"]["y"], d["dev"]["group"]
    yt = d["test"]["y"]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(d["dev"]["eng"], y, g))
    print("Experiment B: random forests and feature randomisation")

    # ---------------------------------------------------------------- Part 1
    print("\n  Part 1 - baseline (grouped 5-fold CV; train AUC = on the training folds)")
    print(f"    {'representation':<16}{'model':<26}{'train AUC':>10}{'val AUC':>9}{'sd':>7}")
    for rep in ("eng", "hybrid"):
        X = d["dev"][rep]
        for name, make in (("decision tree (unrestricted)", lambda: DecisionTreeClassifier(random_state=0)),
                           ("decision tree (depth 6)", lambda: DecisionTreeClassifier(max_depth=6, random_state=0)),
                           ("random forest (200 trees)", lambda: RandomForestClassifier(200, n_jobs=-1, random_state=0))):
            tr, va, sd = cv_auc(make, X, y, folds, train=True)
            print(f"    {rep:<16}{name:<26}{tr:>10.3f}{va:>9.3f}{sd:>7.3f}")

    # ---------------------------------------------------------------- Part 2
    print("\n  Part 2 - feature randomisation (200 trees, unrestricted depth), grouped-CV AUC")
    mf = {"eng": (1, 2, 3), "hybrid": (1, 2, 3, 5, 8, 11)}
    mf_res = {}
    for rep, opts in mf.items():
        X = d["dev"][rep]
        r = {k: cv_auc(lambda k=k: RandomForestClassifier(200, max_features=k, n_jobs=-1, random_state=0), X, y, folds) for k in opts}
        mf_res[rep] = r
        print(f"    {rep:<8}" + ", ".join(f"max_features={k}: {v[0]:.3f}+-{v[1]:.3f}" for k, v in r.items())
              + f"   ({len(opts) and opts[-1]} = all features = bagged trees)")

    # ---------------------------------------------------------------- Part 3
    print("\n  Part 3 - ensemble size and tree complexity (hybrid features), grouped-CV AUC")
    Xh = d["dev"]["hybrid"]
    Ms, depths, leaves = (10, 25, 50, 100, 200), (4, 8, 16, None), (1, 5, 20)
    cells = [(M, dp, lf) for M in Ms for dp in depths for lf in leaves]
    res = Parallel(n_jobs=-1)(delayed(grid_cell)(M, dp, lf, Xh, y, folds) for M, dp, lf in cells)
    grid = dict(zip(cells, res))
    print(f"    {'M':>5}" + "".join(f"{f'd={dp},leaf={lf}':>16}" for dp in depths for lf in leaves))
    for M in Ms:
        print(f"    {M:>5}" + "".join(f"{grid[(M, dp, lf)][0]:>16.3f}" for dp in depths for lf in leaves))
    best = max(cells, key=lambda c: grid[c][0])
    print(f"    best: M = {best[0]}, max_depth = {best[1]}, min_samples_leaf = {best[2]} -> grouped-CV AUC {grid[best][0]:.3f}"
          f" +- {grid[best][1]:.3f}")

    # ---------------------------------------------------------------- Part 4
    print("\n  Part 4 - tree-error correlation on independent validation images (fold 0; 100 trees each)")
    tr, va = folds[0]
    print(f"    {'representation':<16}{'ensemble':<30}{'mean tree acc':>14}{'mean error corr':>17}{'ens. acc':>10}{'ens. AUC':>10}")
    corr_dist = {}
    for rep in ("eng", "hybrid"):
        X = d["dev"][rep]
        nfeat = X.shape[1]
        variants = {f"bagged trees (max_features={nfeat})": nfeat, "random forest (max_features=1)": 1}
        if rep == "hybrid":
            variants["random forest (max_features=3)"] = 3
        for name, k in variants.items():
            rf = RandomForestClassifier(100, max_features=k, random_state=1, n_jobs=-1).fit(X[tr], y[tr])
            E = tree_errors(rf.estimators_, X[va], y[va])
            C = np.corrcoef(E)
            iu = np.triu_indices(len(E), 1)
            corr_dist[(rep, name)] = C[iu]
            p = rf.predict_proba(X[va])[:, 1]
            print(f"    {rep:<16}{name:<30}{1 - E.mean():>14.3f}{np.nanmean(C[iu]):>17.3f}"
                  f"{np.mean((p >= 0.5) == y[va]):>10.3f}{roc_auc_score(y[va], p):>10.3f}")

    # ---------------------------------------------------------------- Part 5
    print("\n  Part 5 - out-of-bag vs grouped CV vs the untouched test images (hybrid features)")
    M, dp, lf = best
    rf = RandomForestClassifier(M, max_depth=dp, min_samples_leaf=lf, oob_score=True, random_state=0, n_jobs=-1).fit(Xh, y)
    oob_auc = roc_auc_score(y, rf.oob_decision_function_[:, 1])
    test_auc = roc_auc_score(yt, rf.predict_proba(d["test"]["hybrid"])[:, 1])
    print(f"    selected forest (M={M}, depth={dp}, leaf={lf}): OOB AUC {oob_auc:.3f} (OOB accuracy {rf.oob_score_:.3f});"
          f" grouped-CV AUC {grid[best][0]:.3f}; test AUC on 30 unseen images {test_auc:.3f}")
    # same comparison with deep trees where same-image siblings matter more
    rf2 = RandomForestClassifier(200, oob_score=True, random_state=0, n_jobs=-1, max_features=None).fit(Xh, y)
    print(f"    bagged deep trees (max_features=all): OOB AUC {roc_auc_score(y, rf2.oob_decision_function_[:, 1]):.3f},"
          f" grouped-CV AUC {mf_res['hybrid'][11][0]:.3f}, test AUC {roc_auc_score(yt, rf2.predict_proba(d['test']['hybrid'])[:, 1]):.3f}")
    for lvl in ("candidate", "image"):
        b = BaggingScratch(200, random_state=0, level=lvl).fit(Xh, y, groups=g)
        po, cnt = b.oob_proba(Xh, groups=g)
        ok = ~np.isnan(po)
        print(f"    own bagging, {lvl}-level bootstrap: {'OOB' if lvl == 'candidate' else 'group-OOB (trees that never saw the IMAGE)'}"
              f" AUC {roc_auc_score(y[ok], po[ok]):.3f} (rows scored {ok.mean():.0%}, mean trees per row {cnt[ok].mean():.0f});"
              f" test AUC {roc_auc_score(yt, b.predict_proba(d['test']['hybrid'])[:, 1]):.3f}")

    # ---------------------------------------------------------------- feature importance
    print("\n  Feature importance (selected forest, hybrid features): impurity (MDI, training data) vs permutation"
          " (AUC drop on the untouched images, 10 repeats)")
    pi = permutation_importance(rf, d["test"]["hybrid"], yt, scoring="roc_auc", n_repeats=10, random_state=0, n_jobs=-1)
    order = np.argsort(-pi.importances_mean)
    print(f"    {'feature':<12}{'MDI':>8}{'permutation':>13}")
    for i in order:
        print(f"    {HYB[i]:<12}{rf.feature_importances_[i]:>8.3f}{pi.importances_mean[i]:>8.3f} +- {pi.importances_std[i]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(19, 4.8))
    ax = axes[0]
    for dp, lf, mk in ((None, 1, "o-"), (16, 5, "s-"), (8, 5, "^-"), (4, 1, "v-")):
        ax.plot(Ms, [grid[(M_, dp, lf)][0] for M_ in Ms], mk, label=f"max_depth={dp}, leaf={lf}")
    ax.set_xscale("log")
    ax.set_xlabel("trees M")
    ax.set_ylabel("grouped-CV AUC")
    ax.set_title("Random forest: ensemble size and complexity (hybrid)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for key, v in corr_dist.items():
        ax.hist(v, bins=40, histtype="step", lw=1.5, density=True, label=f"{key[0]}: {key[1]}")
    ax.set_xlabel("pairwise correlation of tree error indicators")
    ax.set_title("Tree-error correlation on validation images")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[2]
    x = np.arange(len(HYB))
    ax.bar(x - 0.2, rf.feature_importances_[np.arange(len(HYB))], 0.4, label="impurity (MDI)")
    ax.bar(x + 0.2, pi.importances_mean, 0.4, yerr=pi.importances_std, label="permutation (test AUC drop)")
    ax.set_xticks(x, HYB, rotation=40)
    ax.set_title("Feature importance (selected forest)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_random_forest.png", dpi=130)
    plt.close(fig)
    print("\n  saved", OUT / "exp_b_random_forest.png")


if __name__ == "__main__":
    main()
