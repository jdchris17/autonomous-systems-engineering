"""
Experiment C: AdaBoost and gradient boosting (hybrid features: 3 engineered + 8 PCA-C).

Grouped validation: StratifiedGroupKFold(5) over the 60 development images. Learning curves use
staged predictions averaged over the 5 folds. The untouched test images are used for the
label-contamination study only.

Part 1  AdaBoost (SAMME) with depth-1 / depth-2 trees: number of learners x learning rate; vs a
        single stump and a shallow (depth-3) tree
Part 2  GradientBoostingClassifier: trees, learning rate, depth, min samples per leaf, subsampling;
        training and validation log loss per boosting iteration
Part 3  which physical subtypes carry the largest errors (out-of-fold), and where AdaBoost puts weight
Part 4  label contamination q in {0, 0.02, 0.05, 0.10, 0.20} of TRAINING labels (test labels clean)
Part 5  boosting as gradient descent: the negative gradient of log loss w.r.t. the score is y - p
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import scipy.special as sps
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from data9 import OUT, SUBTYPES, load

N_ADA, N_GB = 500, 400
ADA_GRID = [(d, lr) for d in (1, 2) for lr in (0.1, 0.5, 1.0)]
GB_GRID = [(lr, d, leaf, ss) for lr in (0.03, 0.1, 0.3) for d in (1, 2, 3, 5) for leaf in (1, 20) for ss in (1.0, 0.5)]


def sigmoid(z):
    return sps.expit(z)          # numerically stable logistic function (module attribute: picklable)


def ada_fold(depth, lr, tr, va, X, y):
    m = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth), n_estimators=N_ADA, learning_rate=lr,
                           random_state=0).fit(X[tr], y[tr])
    return [roc_auc_score(y[va], s) for s in m.staged_decision_function(X[va])]


def gb_fold(lr, d, leaf, ss, tr, va, X, y):
    m = GradientBoostingClassifier(n_estimators=N_GB, learning_rate=lr, max_depth=d, min_samples_leaf=leaf,
                                   subsample=ss, random_state=0).fit(X[tr], y[tr])
    trl = [log_loss(y[tr], sigmoid(s)) for s in m.staged_decision_function(X[tr])]
    val = [log_loss(y[va], sigmoid(s)) for s in m.staged_decision_function(X[va])]
    vau = [roc_auc_score(y[va], s) for s in m.staged_decision_function(X[va])]
    return np.array(trl), np.array(val), np.array(vau)


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    X, y, g, sub = d["dev"]["hybrid"], d["dev"]["y"], d["dev"]["group"], d["dev"]["subtype"]
    Xt, yt = d["test"]["hybrid"], d["test"]["y"]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    print("Experiment C: AdaBoost and gradient boosting (hybrid features, grouped 5-fold CV)")

    # ---------------------------------------------------------------- Part 1
    jobs = [(dp, lr, tr, va) for dp, lr in ADA_GRID for tr, va in folds]
    res = Parallel(n_jobs=-1)(delayed(ada_fold)(dp, lr, tr, va, X, y) for dp, lr, tr, va in jobs)
    ada = {}
    for (dp, lr, *_), r in zip(jobs, res):
        ada.setdefault((dp, lr), []).append(r)
    ada = {k: np.mean(v, axis=0) for k, v in ada.items()}
    base = {}
    for name, make in (("single stump", lambda: DecisionTreeClassifier(max_depth=1)),
                       ("shallow tree (depth 3)", lambda: DecisionTreeClassifier(max_depth=3))):
        base[name] = np.mean([roc_auc_score(y[va], make().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]) for tr, va in folds])
    print("\n  Part 1 - AdaBoost, grouped-CV AUC at selected numbers of weak learners")
    print(f"    {'learner':<14}{'lr':>5}" + "".join(f"{f'n={n}':>9}" for n in (1, 10, 50, 100, 200, 500)) + f"{'best (n)':>14}")
    for (dp, lr), curve in ada.items():
        print(f"    {f'depth-{dp} tree':<14}{lr:>5}" + "".join(f"{curve[n - 1]:>9.3f}" for n in (1, 10, 50, 100, 200, 500))
              + f"{f'{curve.max():.3f} ({curve.argmax() + 1})':>14}")
    for name, v in base.items():
        print(f"    {name}: {v:.3f}")
    ada_best = max(ada, key=lambda k: ada[k].max())
    ada_n = int(ada[ada_best].argmax() + 1)
    print(f"    selected AdaBoost: depth-{ada_best[0]} trees, learning rate {ada_best[1]}, {ada_n} learners")

    # ---------------------------------------------------------------- Part 2
    jobs = [(c, tr, va) for c in GB_GRID for tr, va in folds]
    res = Parallel(n_jobs=-1)(delayed(gb_fold)(*c, tr, va, X, y) for c, tr, va in jobs)
    gb = {}
    for (c, *_), r in zip(jobs, res):
        gb.setdefault(c, []).append(r)
    gb = {c: tuple(np.mean([r[i] for r in v], axis=0) for i in range(3)) for c, v in gb.items()}
    print("\n  Part 2 - gradient boosting: best iteration by validation log loss (mean over grouped folds)")
    print(f"    {'lr':>5}{'depth':>6}{'leaf':>5}{'subsample':>10}{'best iter':>10}{'val log loss':>13}{'val AUC there':>14}"
          f"{'val LL @400':>12}{'train LL @400':>14}")
    for c in sorted(gb, key=lambda c: gb[c][1].min())[:12]:
        trl, val, vau = gb[c]
        i = int(val.argmin())
        print(f"    {c[0]:>5}{c[1]:>6}{c[2]:>5}{c[3]:>10}{i + 1:>10}{val[i]:>13.4f}{vau[i]:>14.3f}{val[-1]:>12.4f}{trl[-1]:>14.4f}")
    print("    ... worst configurations (overfitting):")
    for c in sorted(gb, key=lambda c: gb[c][1][-1])[-3:]:
        trl, val, vau = gb[c]
        print(f"    {c[0]:>5}{c[1]:>6}{c[2]:>5}{c[3]:>10}{int(val.argmin()) + 1:>10}{val.min():>13.4f}{vau[val.argmin()]:>14.3f}"
              f"{val[-1]:>12.4f}{trl[-1]:>14.4f}")
    gb_best = min(gb, key=lambda c: gb[c][1].min())
    gb_n = int(gb[gb_best][1].argmin() + 1)
    print(f"    selected GB: learning rate {gb_best[0]}, depth {gb_best[1]}, min leaf {gb_best[2]}, subsample {gb_best[3]},"
          f" {gb_n} trees")

    # ---------------------------------------------------------------- Part 3
    def make_models():
        return {"tree (depth 6)": DecisionTreeClassifier(max_depth=6, random_state=0),
                "random forest": RandomForestClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=0),
                "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=ada_best[0]), n_estimators=ada_n,
                                               learning_rate=ada_best[1], random_state=0),
                "gradient boosting": GradientBoostingClassifier(n_estimators=gb_n, learning_rate=gb_best[0], max_depth=gb_best[1],
                                                                min_samples_leaf=gb_best[2], subsample=gb_best[3], random_state=0)}
    oof = {k: np.zeros(len(y)) for k in make_models()}
    for tr, va in folds:
        for k, m in make_models().items():
            oof[k][va] = m.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    floor = np.isclose(d["dev"]["eng"][:, 0], np.log(0.2))
    print("\n  Part 3 - out-of-fold errors by physical subtype (error rate at p = 0.5 / mean log loss)")
    print(f"    {'subtype':<22}" + "".join(f"{k:>22}" for k in oof))
    groups = [(s, sub == s) for s in SUBTYPES] + [("unresolved (width floor)", floor), ("ALL", np.ones(len(y), bool))]
    for s, m in groups:
        cells = []
        for k, p in oof.items():
            pc = np.clip(p[m], 1e-6, 1 - 1e-6)
            ll = -np.mean(y[m] * np.log(pc) + (1 - y[m]) * np.log(1 - pc))
            cells.append(f"{np.mean((p[m] >= 0.5) != y[m]):.3f} / {ll:.3f}")
        print(f"    {s:<22}" + "".join(f"{c:>22}" for c in cells))
    print("    grouped-CV AUC: " + ", ".join(f"{k} {roc_auc_score(y, p):.3f}" for k, p in oof.items()))
    ab = make_models()["AdaBoost"].fit(X, y)
    yy = np.where(y == 1, 1, -1)
    F = np.zeros(len(y))
    for est, a in zip(ab.estimators_, ab.estimator_weights_):
        F += a * (est.predict(X) != y)
    w = np.exp(F - F.max())
    w /= w.sum()
    print("    AdaBoost final training-sample weights (SAMME, reconstructed from the fitted learners):")
    for s, m in groups[:-1]:
        top = w >= np.quantile(w, 0.99)
        print(f"      {s:<26} mean weight x n = {w[m].mean() * len(y):6.2f}   share of the top-1% weights {np.mean(m[top]):6.1%}"
              f"   (share of data {m.mean():5.1%})")

    # ---------------------------------------------------------------- Part 4
    print("\n  Part 4 - label contamination of the TRAINING labels (all 60 development images); AUC on the 30"
          " untouched test images (clean labels), mean of 3 seeds")
    sc = StandardScaler().fit(d["dev"]["eng"])
    Et, Ed = sc.transform(d["test"]["eng"]), sc.transform(d["dev"]["eng"])
    qs = (0.0, 0.02, 0.05, 0.10, 0.20)

    def contam(q, seed):
        r = np.random.default_rng([seed, int(q * 100)])
        yc = y.copy()
        flip = r.random(len(y)) < q
        yc[flip] = 1 - yc[flip]
        out = {}
        for k, m in make_models().items():
            if k == "tree (depth 6)":
                continue
            out[k] = roc_auc_score(yt, m.fit(X, yc).predict_proba(Xt)[:, 1])
        out["KNN (k=45, eng)"] = roc_auc_score(yt, KNeighborsClassifier(45).fit(Ed, yc).predict_proba(Et)[:, 1])
        return out
    cres = Parallel(n_jobs=-1)(delayed(contam)(q, s) for q in qs for s in range(3))
    tab = {}
    for (q, s), r in zip([(q, s) for q in qs for s in range(3)], cres):
        for k, v in r.items():
            tab.setdefault((k, q), []).append(v)
    names = list(cres[0])
    print(f"    {'q':>6}" + "".join(f"{k:>20}" for k in names))
    for q in qs:
        print(f"    {q:>6}" + "".join(f"{np.mean(tab[(k, q)]):>20.3f}" for k in names))
    print("    drop from q=0 to q=0.20: " + ", ".join(f"{k} {np.mean(tab[(k, 0.2)]) - np.mean(tab[(k, 0.0)]):+.3f}" for k in names))

    # ---------------------------------------------------------------- Part 5
    print("\n  Part 5 - boosting as gradient descent in function space (6 observations)")
    ys = np.array([1, 0, 1, 1, 0, 0], float)
    Fs = np.array([0.4, -1.2, -0.3, 2.0, 0.7, 0.1])
    p = sigmoid(Fs)

    def L(F):
        q = sigmoid(F)
        return -(ys * np.log(q) + (1 - ys) * np.log(1 - q))
    h = 1e-6
    num = -(L(Fs + h) - L(Fs - h)) / (2 * h)
    print(f"    scores F            {np.round(Fs, 3)}")
    print(f"    labels y            {ys}")
    print(f"    -dL/dF (numerical)  {np.round(num, 6)}")
    print(f"    y - p (analytic)    {np.round(ys - p, 6)}   max |difference| {np.max(np.abs(num - (ys - p))):.1e}")
    # one gradient-boosting step on real data, reproduced by hand
    gb1 = GradientBoostingClassifier(n_estimators=1, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
    p0 = y.mean()
    F0 = np.log(p0 / (1 - p0))
    r = y - p0
    reg = DecisionTreeRegressor(max_depth=1, criterion="squared_error").fit(X, r)   # == "friedman_mse" (deprecated alias)
    leaf = reg.apply(X)
    newton = {l: r[leaf == l].sum() / (p0 * (1 - p0) * np.sum(leaf == l)) for l in np.unique(leaf)}
    F1 = F0 + np.array([newton[l] for l in leaf])
    print(f"    real data, first boosting step: initial score F0 = log-odds of the star fraction = {F0:.4f};"
          f" pseudo-residuals y - p0 fitted by a depth-1 regression tree (split on feature {reg.tree_.feature[0]} at"
          f" {reg.tree_.threshold[0]:.3f}); Newton leaf values {np.round(list(newton.values()), 4)}")
    print(f"    reproduces sklearn's first iteration: max |F1_mine - F1_sklearn| = {np.max(np.abs(F1 - gb1.decision_function(X))):.1e};"
          f" log loss {log_loss(y, sigmoid(np.full(len(y), F0))):.4f} -> {log_loss(y, sigmoid(F1)):.4f}")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.8))
    ax = axes[0]
    for (dp, lr), curve in ada.items():
        ax.plot(np.arange(1, N_ADA + 1), curve, label=f"depth {dp}, lr {lr}")
    for name, v in base.items():
        ax.axhline(v, ls=":", color="k" if "stump" in name else "gray", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("weak learners")
    ax.set_ylabel("grouped-CV AUC")
    ax.set_title("AdaBoost learning curves")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for c, col in ((gb_best, "tab:blue"), ((0.3, 5, 1, 1.0), "tab:red"), ((0.03, 1, 1, 1.0), "tab:green")):
        trl, val, _ = gb[c]
        ax.plot(trl, "--", color=col)
        ax.plot(val, "-", color=col, label=f"lr {c[0]}, depth {c[1]}, leaf {c[2]}, ss {c[3]}")
    ax.set_xlabel("boosting iteration")
    ax.set_ylabel("log loss (dashed train, solid validation)")
    ax.set_title("Gradient boosting: training vs validation loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[2]
    for k in names:
        ax.plot(qs, [np.mean(tab[(k, q)]) for q in qs], "o-", label=k)
    ax.set_xlabel("fraction of corrupted training labels q")
    ax.set_ylabel("test AUC (clean labels)")
    ax.set_title("Robustness to label contamination")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_boosting.png", dpi=130)
    plt.close(fig)
    np.save(OUT / "selected_boosting.npy", np.array([ada_best[0], ada_best[1], ada_n, *gb_best, gb_n], dtype=float))
    print("\n  saved", OUT / "exp_c_boosting.png")


if __name__ == "__main__":
    main()
