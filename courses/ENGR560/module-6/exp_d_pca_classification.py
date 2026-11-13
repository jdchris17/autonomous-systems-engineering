"""
Experiment D: PCA as a classification representation.

Representations (everything learned is fit inside each training fold only):
  hand       log width, roundness, log SNR measured from the pixels (Modules 4-5 feature set)
  raw        1024 background-subtracted pixels (approach B), standardized per pixel
  PCA-B(k)   first k PCA scores of the background-subtracted pixels, standardized
  PCA-C(k)   first k PCA scores of background-subtracted, flux-normalised pixels (approach C)
             k in {2, 4, 8, 16, 32, 64, 128}
Classifiers: logistic regression (C in {0.01, 0.1, 1, 10}), KNN (k in {5, 15, 45}),
             RBF SVM (C in {1, 10} x gamma in {0.3, 1, 3} / n_features).
Model selection: StratifiedGroupKFold(5) over the 60 development images, metric ROC AUC; the
component count is treated as a hyperparameter. Final numbers: refit on all development images,
evaluated once on the 30 untouched test images (plus a realistic-prevalence variant at 15% stars).
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from exp_c_reconstruction import preprocess
from pca_lda import PCAFromScratch
from patches import development_set, engineered_features, untouched_test_set

OUT = __import__("pathlib").Path(__file__).parent / "outputs"
KS = (2, 4, 8, 16, 32, 64, 128)
CLASSIFIERS = {
    "logistic": [("logistic", {"C": c}) for c in (0.01, 0.1, 1, 10)],
    "KNN": [("knn", {"k": k}) for k in (5, 15, 45)],
    "RBF SVM": [("svm", {"C": c, "g": g}) for c in (1, 10) for g in (0.3, 1, 3)],
}


def make(kind, p, n_features):
    if kind == "logistic":
        return LogisticRegression(C=p["C"], max_iter=5000)
    if kind == "knn":
        return KNeighborsClassifier(p["k"])
    return SVC(C=p["C"], gamma=p["g"] / n_features)


def score(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


def representations(Ptr, Pte):
    """All representations for one train/eval pair; PCA fit once per approach, then sliced."""
    reps = {}
    Ftr, Fte = engineered_features(Ptr), engineered_features(Pte)
    reps["hand"] = (Ftr, Fte)
    Btr, _ = preprocess(Ptr, "B")
    Bte, _ = preprocess(Pte, "B")
    reps["raw"] = (Btr, Bte)
    for ap in ("B", "C"):
        Ar, _ = preprocess(Ptr, ap)
        Ae, _ = preprocess(Pte, ap)
        pca = PCAFromScratch(max(KS)).fit(Ar)
        Ztr, Zte = pca.transform(Ar), pca.transform(Ae)
        for k in KS:
            reps[f"PCA-{ap}({k})"] = (Ztr[:, :k], Zte[:, :k])
    return reps


def fit_eval(rep, Xtr, ytr, Xte, yte, kind, p):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    t0 = time.perf_counter()
    m = make(kind, p, A.shape[1]).fit(A, ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = score(m, B)
    t_pred = time.perf_counter() - t0
    yhat = m.predict(B)
    tp, fp, fn = np.sum((yhat == 1) & (yte == 1)), np.sum((yhat == 1) & (yte == 0)), np.sum((yhat == 0) & (yte == 1))
    return {"auc": roc_auc_score(yte, s), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "t_fit": t_fit, "t_pred_per_1000": 1000 * t_pred / len(yte)}


def fold_reps(tr, va, P):
    return representations(P[tr], P[va])


def main():
    OUT.mkdir(exist_ok=True)
    d, t = development_set(), untouched_test_set()
    P, y, g = d["X"], d["y"], d["group"]
    print("Experiment D: PCA as a classification representation (grouped 5-fold CV on 60 development images)")
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(P, y, g))
    fold_data = Parallel(n_jobs=5)(delayed(fold_reps)(tr, va, P) for tr, va in folds)
    rep_names = list(fold_data[0])

    jobs = [(f, rep, cname, i, kind, p) for f in range(5) for rep in rep_names
            for cname, grid in CLASSIFIERS.items() for i, (kind, p) in enumerate(grid)]
    res = Parallel(n_jobs=-1)(delayed(fit_eval)(rep, fold_data[f][rep][0], y[folds[f][0]], fold_data[f][rep][1],
                                                y[folds[f][1]], kind, p) for f, rep, cname, i, kind, p in jobs)
    cv = {}
    for (f, rep, cname, i, *_), r in zip(jobs, res):
        cv.setdefault((rep, cname, i), []).append(r["auc"])
    best = {}
    for rep in rep_names:
        for cname, grid in CLASSIFIERS.items():
            i = int(np.argmax([np.mean(cv[(rep, cname, j)]) for j in range(len(grid))]))
            best[(rep, cname)] = (i, np.mean(cv[(rep, cname, i)]), np.std(cv[(rep, cname, i)]))

    print("\n  Part 2 - grouped-CV ROC AUC (mean +- fold sd, best hyperparameters for each cell)")
    print(f"    {'representation':<14}" + "".join(f"{c:>22}" for c in CLASSIFIERS))
    for rep in rep_names:
        print(f"    {rep:<14}" + "".join(f"{best[(rep, c)][1]:>14.3f} +- {best[(rep, c)][2]:.3f}" for c in CLASSIFIERS))

    # selection: for every classifier, the best representation family member chosen by CV
    families = {"hand": ["hand"], "raw": ["raw"], "PCA-B": [r for r in rep_names if r.startswith("PCA-B")],
                "PCA-C": [r for r in rep_names if r.startswith("PCA-C")]}
    print("\n  Final evaluation: hyperparameters and k chosen by grouped CV, refit on all development images,"
          " scored once on the 30 untouched test images")
    final_reps = representations(P, t["X"])
    t15 = untouched_test_set(star_fraction=0.15)
    reps15 = representations(P, t15["X"])
    print(f"    {'classifier':<11}{'representation':<15}{'hyperparameters':<22}{'CV AUC':>8}{'test AUC':>10}"
          f"{'precision':>10}{'recall':>8}{'fit s':>8}{'ms/1000':>9}{'AUC @15%':>10}{'prec @15%':>11}")
    final = {}
    for cname, grid in CLASSIFIERS.items():
        for fam, members in families.items():
            rep = max(members, key=lambda r: best[(r, cname)][1])
            i = best[(rep, cname)][0]
            kind, p = grid[i]
            r = fit_eval(rep, final_reps[rep][0], y, final_reps[rep][1], t["y"], kind, p)
            r15 = fit_eval(rep, reps15[rep][0], y, reps15[rep][1], t15["y"], kind, p)
            final[(cname, fam)] = (rep, r)
            print(f"    {cname:<11}{rep:<15}{str(p):<22}{best[(rep, cname)][1]:>8.3f}{r['auc']:>10.3f}"
                  f"{r['precision']:>10.3f}{r['recall']:>8.3f}{r['t_fit']:>8.2f}{r['t_pred_per_1000']:>9.2f}"
                  f"{r15['auc']:>10.3f}{r15['precision']:>11.3f}")
    print(f"    (realistic-prevalence variant: same 30 test images' conditions regenerated with 15% stars;"
          f" {t15['y'].mean():.1%} stars)")

    # ---------------------------------------------------------------- Part 3: variance vs performance
    evB = PCAFromScratch().fit(preprocess(P, "B")[0]).cumulative_explained_variance_
    evC = PCAFromScratch().fit(preprocess(P, "C")[0]).cumulative_explained_variance_
    print("\n  Part 3 - cumulative explained variance vs grouped-CV AUC (best classifier per k)")
    print(f"    {'k':>5}{'cum EV (B)':>12}{'AUC PCA-B':>11}{'cum EV (C)':>12}{'AUC PCA-C':>11}")
    for k in KS:
        aB = max(best[(f"PCA-B({k})", c)][1] for c in CLASSIFIERS)
        aC = max(best[(f"PCA-C({k})", c)][1] for c in CLASSIFIERS)
        print(f"    {k:>5}{evB[k - 1]:>12.3f}{aB:>11.3f}{evC[k - 1]:>12.3f}{aC:>11.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, ap, ev in ((axes[0], "B", evB), (axes[1], "C", evC)):
        for cname, col in zip(CLASSIFIERS, ("tab:blue", "tab:green", "tab:red")):
            m = [best[(f"PCA-{ap}({k})", cname)][1] for k in KS]
            s = [best[(f"PCA-{ap}({k})", cname)][2] for k in KS]
            ax.errorbar(KS, m, yerr=s, fmt="o-", color=col, capsize=3, label=f"{cname}: grouped-CV AUC")
            ax.axhline(best[("hand", cname)][1], color=col, ls=":", lw=1)
        ax.axhline(np.nan, color="k", ls=":", label="dotted: same classifier on 3 hand features")
        ax2 = ax.twinx()
        ax2.plot(KS, ev[np.array(KS) - 1], "k--", label="cumulative explained variance")
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("cumulative explained variance")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("principal components k")
        ax.set_ylabel("ROC AUC")
        ax.set_title(f"PCA-{ap}: variance kept vs classification performance")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_variance_vs_auc.png", dpi=130)
    plt.close(fig)
    print("\n  saved", OUT / "exp_d_variance_vs_auc.png")


if __name__ == "__main__":
    main()
