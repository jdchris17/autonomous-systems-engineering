"""
Experiment A: data leakage and grouped evaluation.

Datasets (grouped_data.py, 50 development images x 100-200 candidates):
  control    group_effects=False  images are arbitrary batches of independent candidates
  realistic  group_effects=True   shared focus / sky / noise / PSF shape / measurement bias /
                                  star fraction per image
  stress     realistic + an ARTIFICIAL random per-image detector "batch code" on the reported
             background (meaningful only for images the model has seen)
Feature sets: all 5 engineered features, and Module 4's invariant 3 (no peak, no background).

Strategy A  random candidate split, 30% of candidates in test
Strategy B  grouped image split, whole images until ~30% of candidates are in test
Classifiers: logistic regression (own), KNN k=5, KNN k=51, decision tree depth 10.  20 seeds.

Two reference measurements separate leakage from composition/difficulty effects:
  external  train on all 50 development images, test on 30 brand-new images (untouched_test_set):
            the generalization that both strategies are trying to estimate
  siblings  same held-out images T, their candidates split 50/50 into T_a and T_b. Model 1 trains
            on (other images, subsampled) + T_a; model 2 on the other images only, same size.
            Both are scored on the SAME T_b candidates, so composition and difficulty are
            identical; the difference is purely what seeing siblings from the same images buys.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from grouped_data import INVARIANT, OUT, development_set, untouched_test_set
from ml_tools import ALL5, Pipeline, evaluate, grouped_test_split, random_test_split

MODELS = {"logistic": ("logistic", {"lam": 1e-4}), "KNN k=5": ("knn", {"k": 5}),
          "KNN k=51": ("knn", {"k": 51}), "tree d=10": ("tree", {"depth": 10})}
CONFIGS = {"control": dict(group_effects=False), "realistic": dict(group_effects=True),
           "stress": dict(group_effects=True, stress=True)}
FEATS = {"all 5": ALL5, "invariant 3": INVARIANT}
SEEDS = range(20)
METRICS = ("accuracy", "auc", "precision", "recall", "brier")


def one_seed(seed, X, y, g, cols):
    rng = np.random.default_rng([seed, 7])
    tr_g, te_g = grouped_test_split(g, rng)
    tr_r, te_r = random_test_split(y.size, rng, te_g.size)       # same test size as grouped
    out = {}
    for mname, (kind, params) in MODELS.items():
        row = {}
        for sname, (tr, te) in (("random", (tr_r, te_r)), ("grouped", (tr_g, te_g))):
            p = Pipeline(kind, cols, **params).fit(X[tr], y[tr])
            e = evaluate(p, X[te], y[te])
            e["train_accuracy"] = evaluate(p, X[tr], y[tr])["accuracy"]
            e["test_star_frac"] = y[te].mean()
            e["test_images"] = len(np.unique(g[te]))
            e["unseen_test_images"] = len(set(np.unique(g[te])) - set(np.unique(g[tr])))
            row[sname] = e
        # sibling control on the grouped test images
        T = te_g
        Tperm = rng.permutation(T)
        Ta, Tb = Tperm[: T.size // 2], Tperm[T.size // 2:]
        other = rng.permutation(tr_g)
        with_sib = np.concatenate([other[: other.size - Ta.size], Ta])
        p1 = Pipeline(kind, cols, **params).fit(X[with_sib], y[with_sib])
        p2 = Pipeline(kind, cols, **params).fit(X[other], y[other])
        row["sib_with"] = evaluate(p1, X[Tb], y[Tb])["auc"]
        row["sib_without"] = evaluate(p2, X[Tb], y[Tb])["auc"]
        out[mname] = row
    return out


def main():
    OUT.mkdir(exist_ok=True)
    print("Experiment A: data leakage and grouped evaluation (20 seeds per cell)")
    summary = {}
    for cname, ckw in CONFIGS.items():
        X, y, g, sub, table = development_set(**ckw)
        Xt, yt, gt, _, _ = untouched_test_set(**ckw)
        for fname, cols in FEATS.items():
            if cname == "control" and fname == "invariant 3":
                continue
            res = Parallel(n_jobs=-1)(delayed(one_seed)(s, X, y, g, cols) for s in SEEDS)
            ext = {m: evaluate(Pipeline(k, cols, **p).fit(X, y), Xt, yt) for m, (k, p) in MODELS.items()}
            summary[(cname, fname)] = (res, ext)
    X, y, g, sub, table = development_set()
    print(f"  realistic development set: {len(table)} images, {y.size} candidates,"
          f" per-image star fraction {min(t['pstar'] for t in table):.2f}-{max(t['pstar'] for t in table):.2f}")

    # ------------------------------------------------ required table (realistic, all 5 features)
    res, ext = summary[("realistic", "all 5")]
    print("\n  Required table - realistic data, all 5 features: mean (sd) over 20 seeds;"
          " 'new images' = model trained on all 50 images, tested on 30 brand-new images")
    for m in MODELS:
        print(f"\n    {m}")
        print(f"      {'quantity':<28}{'random split':>18}{'grouped split':>18}{'new images':>12}")
        rows = [("training accuracy", "train_accuracy"), ("test accuracy", "accuracy"),
                ("ROC AUC", "auc"), ("precision", "precision"), ("recall", "recall"),
                ("Brier score", "brier"), ("test star fraction", "test_star_frac"),
                ("images in test set", "test_images"),
                ("independent (unseen) test images", "unseen_test_images")]
        for lab, key in rows:
            cells = []
            for sname in ("random", "grouped"):
                v = np.array([r[m][sname][key] for r in res], dtype=float)
                cells.append(f"{v.mean():.3f} ({v.std():.3f})" if key not in ("test_images", "unseen_test_images")
                             else f"{v.mean():.1f}")
            e = ext[m].get({"accuracy": "accuracy", "auc": "auc", "precision": "precision",
                            "recall": "recall", "brier": "brier"}.get(key, ""), None)
            et = f"{e:.3f}" if e is not None else ("30" if key == "unseen_test_images" else "-")
            print(f"      {lab:<28}{cells[0]:>18}{cells[1]:>18}{et:>12}")

    # ------------------------------------------------ summary over configurations
    print("\n  AUC summary: random-split estimate, grouped-split estimate, truth on new images, and the"
          " sibling control (same test candidates)")
    print(f"    {'data':<11}{'features':<13}{'model':<11}{'random':>8}{'grouped':>9}{'new imgs':>10}"
          f"{'rand-new':>10}{'grp-new':>9}{'sib: with':>11}{'without':>9}{'leak':>8}")
    plot_rows = []
    for (cname, fname), (res, ext) in summary.items():
        for m in MODELS:
            r = np.mean([x[m]["random"]["auc"] for x in res])
            gg = np.mean([x[m]["grouped"]["auc"] for x in res])
            sw = np.mean([x[m]["sib_with"] for x in res])
            so = np.mean([x[m]["sib_without"] for x in res])
            e = ext[m]["auc"]
            plot_rows.append((cname, fname, m, r, gg, e, sw - so))
            print(f"    {cname:<11}{fname:<13}{m:<11}{r:>8.3f}{gg:>9.3f}{e:>10.3f}{r - e:>+10.3f}{gg - e:>+9.3f}"
                  f"{sw:>11.3f}{so:>9.3f}{sw - so:>+8.3f}")
    res, _ = summary[("realistic", "all 5")]
    sf_r = np.array([x["KNN k=5"]["random"]["test_star_frac"] for x in res])
    sf_g = np.array([x["KNN k=5"]["grouped"]["test_star_frac"] for x in res])
    print(f"\n  Test-set composition (realistic): star fraction random {sf_r.mean():.3f} +- {sf_r.std():.3f},"
          f" grouped {sf_g.mean():.3f} +- {sf_g.std():.3f}")
    a_r = np.array([x["KNN k=5"]["random"]["auc"] for x in res])
    a_g = np.array([x["KNN k=5"]["grouped"]["auc"] for x in res])
    print(f"  KNN k=5 AUC spread across seeds: random sd {a_r.std():.3f}, grouped sd {a_g.std():.3f};"
          f" grouped < random in {np.mean(a_g < a_r):.0%} of seeds")

    plot(plot_rows)


def plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    labels = [f"{c}\n{f}\n{m}" for c, f, m, *_ in rows]
    x = np.arange(len(rows))
    ax = axes[0]
    ax.plot(x, [r[3] for r in rows], "o", color="tab:red", label="random-split estimate")
    ax.plot(x, [r[4] for r in rows], "s", color="tab:blue", label="grouped-split estimate")
    ax.plot(x, [r[5] for r in rows], "k_", ms=18, mew=2, label="truth: 30 brand-new images")
    ax.set_xticks(x, labels, fontsize=6)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Estimated vs actual generalization")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar(x, [r[6] for r in rows], color=["tab:gray" if r[0] == "control" else
                                           ("tab:orange" if r[0] == "stress" else "tab:green") for r in rows])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, labels, fontsize=6)
    ax.set_ylabel("AUC gain from seeing same-image siblings")
    ax.set_title("Sibling control: same test candidates, only the training set differs\n"
                 "(grey = control, green = realistic, orange = artificial stress)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_leakage.png", dpi=130)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_a_leakage.png'}")


if __name__ == "__main__":
    main()
