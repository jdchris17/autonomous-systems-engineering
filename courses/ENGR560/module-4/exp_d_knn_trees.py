"""
Experiment D: nearest neighbours, decision trees, and the full model comparison.

Part 1  Own NumPy KNN (Euclidean, majority vote) for k in {1, 3, 5, 11, 21}, checked against
        scikit-learn KNeighborsClassifier(algorithm="brute"); effect of feature scaling.
Part 2  scikit-learn DecisionTreeClassifier, depth in {2, 4, 8, 16, unrestricted}; a shallow tree
        visualised; which features the splits use.
Part 3  All six classifiers (validation-selected hyperparameters, model_zoo) on the SAME held-out
        test set.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from classifiers import KNN, OUT, Standardizer, auc, brier, log_loss, rates, roc_curve
from model_zoo import build_candidates, predict_time, prepared_splits
from star_data import ENG_NAMES, FEATURE_NAMES, engineered, standard_splits

K_LIST = (1, 3, 5, 11, 21)
DEPTHS = (2, 4, 8, 16, None)


def main():
    OUT.mkdir(exist_ok=True)
    sp = standard_splits()
    y = {k: v[1] for k, v in sp.items()}
    st, Z = prepared_splits(sp)
    print("Experiment D: nearest neighbours, decision trees, model comparison")

    # ----------------------------------------------------------------- Part 1
    print("\n  Part 1 - KNN (own NumPy implementation vs scikit-learn, standardized engineered features)")
    print(f"    {'k':>3}{'train acc':>11}{'val acc':>9}{'agree w/ sklearn':>18}{'own s':>8}{'sk s':>7}")
    knn_val = {}
    for k in K_LIST:
        t = time.perf_counter()
        mine = KNN(k).fit(Z["train"], y["train"])
        pm = mine.predict(Z["val"])
        t_m = time.perf_counter() - t
        t = time.perf_counter()
        sk = KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(Z["train"], y["train"])
        ps = sk.predict(Z["val"])
        t_s = time.perf_counter() - t
        tr = np.mean(mine.predict(Z["train"]) == y["train"])
        knn_val[k] = np.mean(pm == y["val"])
        print(f"    {k:>3}{tr:>11.3f}{knn_val[k]:>9.3f}{np.mean(pm == ps):>18.4f}{t_m:>8.2f}{t_s:>7.2f}")
    print("    (k = 1 training accuracy is 1 by construction: each point is its own neighbour)")

    print("\n    Feature scaling, validation accuracy:")
    E = {k: engineered(v[0]) for k, v in sp.items()}
    raw = {k: v[0] for k, v in sp.items()}
    st_raw = Standardizer().fit(raw["train"])
    variants = {
        "raw measured features (no scaling)": raw,
        "raw measured, standardized": {k: st_raw.transform(v) for k, v in raw.items()},
        "engineered (logs), no scaling": E,
        "engineered, standardized": Z,
    }
    print(f"      {'inputs':<38}" + "".join(f"{f'k={k}':>8}" for k in K_LIST))
    for name, D in variants.items():
        accs = [np.mean(KNN(k).fit(D["train"], y["train"]).predict(D["val"]) == y["val"]) for k in K_LIST]
        print(f"      {name:<38}" + "".join(f"{a:>8.3f}" for a in accs))
    print("      raw feature standard deviations: " + ", ".join(
        f"{n} {s:.2f}" for n, s in zip(FEATURE_NAMES, raw["train"].std(axis=0))))

    # ----------------------------------------------------------------- Part 2
    print("\n  Part 2 - decision trees")
    print(f"    {'max_depth':>10}{'leaves':>8}{'train acc':>11}{'val acc':>9}")
    trees = {}
    for d in DEPTHS:
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Z["train"], y["train"])
        trees[d] = m
        print(f"    {str(d):>10}{m.get_n_leaves():>8}{np.mean(m.predict(Z['train']) == y['train']):>11.3f}"
              f"{np.mean(m.predict(Z['val']) == y['val']):>9.3f}")
    imp = trees[4].feature_importances_
    print("    depth-4 feature importances: " + ", ".join(
        f"{n} {v:.2f}" for n, v in sorted(zip(ENG_NAMES, imp), key=lambda t: -t[1])))
    shallow = DecisionTreeClassifier(max_depth=3, random_state=0).fit(E["train"], y["train"])
    print("    depth-3 tree on UNSTANDARDIZED engineered features (readable thresholds):")
    for line in export_text(shallow, feature_names=list(ENG_NAMES), decimals=2).splitlines():
        print("      " + line)
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(shallow, feature_names=list(ENG_NAMES), class_names=["artifact", "star"],
              filled=True, rounded=True, fontsize=9, ax=ax, impurity=False, proportion=True)
    ax.set_title("Depth-3 decision tree (features: log values; width in log pixels)")
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_shallow_tree.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.plot(K_LIST, [np.mean(KNN(k).fit(Z["train"], y["train"]).predict(Z["train"]) == y["train"])
                     for k in K_LIST], "s--", label="train")
    ax.plot(K_LIST, [knn_val[k] for k in K_LIST], "o-", label="validation")
    ax.set_xscale("log")
    ax.set_xticks(K_LIST, [str(k) for k in K_LIST])
    ax.set_xlabel("k")
    ax.set_ylabel("accuracy")
    ax.set_title("KNN: small k memorises, large k smooths")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    labs = [str(d) for d in DEPTHS]
    ax.plot(labs, [np.mean(trees[d].predict(Z["train"]) == y["train"]) for d in DEPTHS], "s--", label="train")
    ax.plot(labs, [np.mean(trees[d].predict(Z["val"]) == y["val"]) for d in DEPTHS], "o-", label="validation")
    ax.set_xlabel("max_depth")
    ax.set_title("Decision tree: depth vs accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_knn_tree_curves.png", dpi=140)
    plt.close(fig)

    # ----------------------------------------------------------------- Part 3
    print("\n  Part 3 - all classifiers on the same held-out test set")
    cands = build_candidates(Z["train"], y["train"], Z["val"], y["val"])
    print(f"    {'model':<12}{'acc':>7}{'bal acc':>9}{'prec':>7}{'recall':>8}{'F1':>7}{'AUC':>7}"
          f"{'logloss':>9}{'Brier':>7}{'fit s':>8}{'ms/1k pred':>12}")
    rows = {}
    for name, c in cands.items():
        yhat = c.predict(Z["test"])
        r = rates(y["test"], yhat)
        p = c.proba(Z["test"])
        s = c.score(Z["test"])
        rows[name] = (s, r)
        print(f"    {name:<12}{r['accuracy']:>7.3f}{r['balanced_acc']:>9.3f}{r['precision']:>7.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>7.3f}{auc(y['test'], s):>7.3f}{log_loss(y['test'], p):>9.3f}"
              f"{brier(y['test'], p):>7.3f}{c.fit_time:>8.2f}{1000 * predict_time(c, Z['test']):>12.2f}")
    print("    (log loss and Brier use Platt-calibrated probabilities fitted on validation;"
          " accuracy etc. use each model's own decision rule)")
    print(f"    always-artifact baseline accuracy: {1 - y['test'].mean():.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, (s, _) in rows.items():
        f, t = roc_curve(y["test"], s)
        ax.plot(f, t, label=f"{name} (AUC {auc(y['test'], s):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate (recall)")
    ax.set_title("ROC curves on the shared test set")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_roc_comparison.png", dpi=140)
    plt.close(fig)
    for f in ("exp_d_shallow_tree.png", "exp_d_knn_tree_curves.png", "exp_d_roc_comparison.png"):
        print(f"  saved {OUT / f}")


if __name__ == "__main__":
    main()
