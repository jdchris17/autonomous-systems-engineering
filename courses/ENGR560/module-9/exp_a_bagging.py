"""
Experiment A: bagging from first principles.

Features: the three engineered features (log width, roundness, log SNR). Base learner: an unrestricted
DecisionTreeClassifier (the unstable classifier). Grouped validation: StratifiedGroupKFold(5) over the
60 development images; the 30 untouched images only for the final unseen-group comparison.

Part 1  bootstrap sampler: duplicates, exclusions, unique fraction vs 1 - (1 - 1/n)^n -> 1 - 1/e
Part 2  BaggingScratch (ensembles.py)
Part 3  vs sklearn BaggingClassifier: exact member-by-member reproduction from sklearn's own samples
Part 4  ensemble size M in {1, 5, 10, 25, 50, 100, 200}, 5 seeds, training and grouped-validation AUC
Part 5  candidate-level vs image-level bootstrap vs a single tree on unseen image groups
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from data9 import OUT, load
from ensembles import BaggingScratch, bootstrap_indices, group_bootstrap_indices

MS = (1, 5, 10, 25, 50, 100, 200)
SEEDS = range(5)


def size_curve(seed, X, y, g, folds):
    tr_auc = {m: [] for m in MS}
    va_auc = {m: [] for m in MS}
    va_pred = {m: [] for m in MS}
    for tr, va in folds:
        b = BaggingScratch(max(MS), random_state=seed).fit(X[tr], y[tr])
        Ptr, Pva = b.member_proba(X[tr]), b.member_proba(X[va])
        for m in MS:
            tr_auc[m].append(roc_auc_score(y[tr], Ptr[:m].mean(0)))
            va_auc[m].append(roc_auc_score(y[va], Pva[:m].mean(0)))
            va_pred[m].append(Pva[:m].mean(0))
    return {m: (np.mean(tr_auc[m]), np.mean(va_auc[m]), np.concatenate(va_pred[m])) for m in MS}


def strategy_eval(seed, X, y, g, Xt, yt, gt):
    out = {}
    models = {"single tree": DecisionTreeClassifier(random_state=seed).fit(X, y),
              "candidate bagging": BaggingScratch(100, random_state=seed).fit(X, y),
              "image bagging": BaggingScratch(100, random_state=seed, level="image").fit(X, y, groups=g)}
    for name, m in models.items():
        p = m.predict_proba(Xt)[:, 1]
        per = [roc_auc_score(yt[gt == i], p[gt == i]) for i in np.unique(gt) if 0 < yt[gt == i].sum() < (gt == i).sum()]
        out[name] = {"auc": roc_auc_score(yt, p), "worst": np.min(per), "p10": np.percentile(per, 10), "p": p,
                     "acc": np.mean((p >= 0.5) == yt)}
        if name != "single tree":
            members = m.member_proba(Xt)
            out[name]["member_auc"] = np.mean([roc_auc_score(yt, q) for q in members[:20]])
            out[name]["member_spread"] = float(np.mean(members.std(axis=0)))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    X, y, g = d["dev"]["eng"], d["dev"]["y"], d["dev"]["group"]
    Xt, yt, gt = d["test"]["eng"], d["test"]["y"], d["test"]["group"]
    n = len(y)
    print("Experiment A: bagging from first principles (3 engineered features)")

    # ---------------------------------------------------------------- Part 1
    rng = np.random.default_rng(0)
    idx = bootstrap_indices(n, rng)
    counts = np.bincount(idx, minlength=n)
    print(f"\n  Part 1 - one bootstrap sample of n = {n}: max multiplicity {counts.max()}, observations drawn"
          f" more than once {np.mean(counts > 1):.1%}, never drawn {np.mean(counts == 0):.1%}")
    uniq = np.array([np.unique(bootstrap_indices(n, rng)).size / n for _ in range(2000)])
    theory = 1 - (1 - 1 / n) ** n
    print(f"    unique fraction over 2000 samples: {uniq.mean():.4f} +- {uniq.std():.4f};"
          f" theory 1 - (1 - 1/n)^n = {theory:.4f}  (limit 1 - 1/e = {1 - np.exp(-1):.4f})")
    ug = []
    for _ in range(2000):
        rows, drawn = group_bootstrap_indices(g, rng)
        ug.append((np.unique(drawn).size / 60, np.unique(rows).size / n, rows.size / n))
    ug = np.array(ug)
    print(f"    image-level bootstrap: unique images {ug[:, 0].mean():.4f} (theory {1 - (1 - 1 / 60) ** 60:.4f}),"
          f" unique candidates {ug[:, 1].mean():.4f}, sample size / n = {ug[:, 2].mean():.3f} +- {ug[:, 2].std():.3f}"
          " (varies, because images differ in size)")

    # ---------------------------------------------------------------- Part 3
    sk = BaggingClassifier(DecisionTreeClassifier(), n_estimators=25, bootstrap=True, max_samples=1.0,
                           random_state=7).fit(X, y)
    samples = sk.estimators_samples_
    seeds = [e.random_state for e in sk.estimators_]
    mine = BaggingScratch(25).fit(X, y, sample_indices=samples, tree_seeds=seeds)
    member_eq = all(np.array_equal(a.predict_proba(Xt), b.predict_proba(Xt)) for a, b in zip(mine.estimators_, sk.estimators_))
    diff = np.abs(mine.predict_proba(Xt) - sk.predict_proba(Xt)).max()
    print(f"\n  Part 3 - vs sklearn BaggingClassifier(25 unrestricted trees): rebuilding every member from sklearn's"
          f" own bootstrap samples and tree seeds -> all 25 members identical: {member_eq};"
          f" ensemble max |proba diff| {diff:.1e}")
    free = BaggingScratch(25, random_state=7).fit(X, y)
    print(f"    with independent random samples: test AUC mine {roc_auc_score(yt, free.predict_proba(Xt)[:, 1]):.3f}"
          f" vs sklearn {roc_auc_score(yt, sk.predict_proba(Xt)[:, 1]):.3f} (same algorithm, different draws)")

    # ---------------------------------------------------------------- Part 4
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    curves = Parallel(n_jobs=len(SEEDS))(delayed(size_curve)(s, X, y, g, folds) for s in SEEDS)
    print("\n  Part 4 - ensemble size: AUC mean over 5 grouped folds; spread across 5 seeds")
    print(f"    {'M':>5}{'train AUC':>11}{'val AUC':>9}{'sd over seeds':>15}{'prediction sd across seeds':>28}")
    rows = {}
    for m in MS:
        tr = np.array([c[m][0] for c in curves])
        va = np.array([c[m][1] for c in curves])
        P = np.array([c[m][2] for c in curves])
        rows[m] = (tr.mean(), va.mean(), va.std(), P.std(axis=0).mean())
        print(f"    {m:>5}{tr.mean():>11.3f}{va.mean():>9.3f}{va.std():>15.4f}{P.std(axis=0).mean():>28.4f}")

    # ---------------------------------------------------------------- Part 5
    res = Parallel(n_jobs=len(SEEDS))(delayed(strategy_eval)(s, X, y, g, Xt, yt, gt) for s in SEEDS)
    cvres = {}
    for name in ("single tree", "candidate bagging", "image bagging"):
        a = []
        for tr, va in folds:
            if name == "single tree":
                m = DecisionTreeClassifier(random_state=0).fit(X[tr], y[tr])
            else:
                m = BaggingScratch(100, random_state=0, level="image" if "image" in name else "candidate").fit(X[tr], y[tr], groups=g[tr])
            a.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        cvres[name] = np.array(a)
    print("\n  Part 5 - candidate vs image bootstrap vs single tree (100 trees; 5 seeds)")
    print(f"    {'model':<20}{'grouped-CV AUC':>16}{'test AUC':>10}{'test acc':>10}{'worst image':>13}{'10th pct image':>16}"
          f"{'member AUC':>12}{'member spread':>15}")
    for name in ("single tree", "candidate bagging", "image bagging"):
        a = np.array([r[name]["auc"] for r in res])
        print(f"    {name:<20}{cvres[name].mean():>10.3f}+-{cvres[name].std():.3f}{a.mean():>10.3f}"
              f"{np.mean([r[name]['acc'] for r in res]):>10.3f}{np.mean([r[name]['worst'] for r in res]):>13.3f}"
              f"{np.mean([r[name]['p10'] for r in res]):>16.3f}"
              + (f"{np.mean([r[name]['member_auc'] for r in res]):>12.3f}{np.mean([r[name]['member_spread'] for r in res]):>15.3f}"
                 if name != "single tree" else f"{'-':>12}{'-':>15}"))
    ab = np.array([r["image bagging"]["auc"] - r["candidate bagging"]["auc"] for r in res])
    print(f"    image minus candidate bagging, test AUC: {ab.mean():+.4f} +- {ab.std():.4f} over seeds")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.plot(MS, [rows[m][0] for m in MS], "s--", label="training AUC")
    ax.errorbar(MS, [rows[m][1] for m in MS], yerr=[rows[m][2] for m in MS], fmt="o-", capsize=3, label="grouped-validation AUC (+-sd over seeds)")
    ax.set_xscale("log")
    ax.set_xlabel("number of bagged trees M")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Bagging: performance vs ensemble size")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(MS, [rows[m][3] for m in MS], "o-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of bagged trees M")
    ax.set_ylabel("mean sd of a candidate's probability across seeds")
    ax.set_title("Bagging reduces prediction variance (~1/sqrt(M) at first)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_bagging.png", dpi=130)
    plt.close(fig)
    print("\n  saved", OUT / "exp_a_bagging.png")


if __name__ == "__main__":
    main()
