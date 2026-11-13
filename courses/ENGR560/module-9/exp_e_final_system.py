"""
Experiment E: final system comparison and deployment trade-offs.

Part 1  Finalists, FROZEN from development-data results (Experiments B-D; no test data involved):
          best single classifier    KNN, k = 45, engineered features         (highest development OOF AUC
                                                                               among non-ensemble models, Exp D)
          best random forest        200 trees, min_samples_leaf 5, hybrid   (Exp B grouped-CV grid)
          best boosting model       gradient boosting: lr 0.03, depth 5, min leaf 20, 252 trees, hybrid
                                                                              (Exp C; beat AdaBoost on CV AUC and log loss)
          best heterogeneous        stacking of KNN + RBF SVM + RF          (Exp D nested evaluation)
Calibration and thresholds (Part 3) come from grouped out-of-fold predictions on the development
images only: Platt calibration, then
          Policy A (high recall)     largest threshold with recall >= 0.95
          Policy B (high precision)  smallest threshold with precision >= 0.90
Part 2  untouched Module 6 test images: AUC, precision/recall/FPR, log loss, Brier, group-bootstrap
        intervals, fit time, latency, serialized size, worst-image performance; paired differences
Part 4  shift scenarios (30 new images each): measurement noise, defocus, bright background, rare
        stars, unseen ghosts; finalists and the stacking ensemble's own members
Part 5  latency at batch sizes 10 / 100 / 1000 / 5000 (warm-up, repeated timings, median and p95)
"""

import os
import pickle
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from data9 import OUT, SCENARIOS, SUBTYPES_SHIFT, features_from_patches, generate_shifted, load
from ensembles import PlattCalibrator, brier, clip_logit, log_loss, rates
from exp_d_voting_stacking import EnsembleProcedure

THR = np.round(np.arange(0.01, 1.0, 0.01), 2)


class Single:
    def __init__(self, kind):
        self.kind = kind
        self.rep = "eng" if kind == "KNN" else "hybrid"

    def fit(self, R, y, g=None, seed=0):
        X = R[self.rep]
        if self.kind == "KNN":
            self.sc = StandardScaler().fit(X)
            self.m = KNeighborsClassifier(45).fit(self.sc.transform(X), y)
        elif self.kind == "random forest":
            self.m = RandomForestClassifier(200, min_samples_leaf=5, random_state=0, n_jobs=1).fit(X, y)
        else:
            self.m = GradientBoostingClassifier(n_estimators=252, learning_rate=0.03, max_depth=5, min_samples_leaf=20,
                                                subsample=1.0, random_state=0).fit(X, y)
        return self

    def score(self, R):
        X = R[self.rep]
        if self.kind == "KNN":
            return clip_logit(self.m.predict_proba(self.sc.transform(X))[:, 1], 0.005)
        if self.kind == "random forest":
            return clip_logit(self.m.predict_proba(X)[:, 1], 0.0025)
        return self.m.decision_function(X)


class Stack:
    rep = "both"

    def fit(self, R, y, g, seed=0):
        self.p = EnsembleProcedure().fit(R, y, g, seed=seed)
        return self

    def score(self, R):
        return clip_logit(self.p.predict_all(R)["stacking"], 1e-4)

    def members(self, R):
        out = self.p.predict_all(R)
        return {n: out[n] for n in self.p.names}


FINALISTS = {"KNN (best single)": lambda: Single("KNN"), "random forest": lambda: Single("random forest"),
             "gradient boosting": lambda: Single("gradient boosting"), "stacking ensemble": lambda: Stack()}


def subset(R, idx):
    return {k: v[idx] for k, v in R.items()}


def oof_scores(name, R, y, g):
    s = np.zeros(len(y))
    for k, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=11).split(np.zeros(len(y)), y, g)):
        s[va] = FINALISTS[name]().fit(subset(R, tr), y[tr], g[tr], seed=k).score(subset(R, va))
    return s


def thresholds(y, p):
    r = {t: rates(y, (p >= t).astype(int)) for t in THR}
    a = max(t for t in THR if r[t]["recall"] >= 0.95)
    ok = [t for t in THR if r[t]["precision"] >= 0.90 and r[t]["accepted"] >= 30]
    b = min(ok) if ok else max((t for t in THR if r[t]["accepted"] >= 30), key=lambda t: r[t]["precision"])
    return a, b, bool(ok)


def policy_cell(r):
    return f"{r['recall']:.3f} / {r['precision']:.3f} / {r['fpr']:.3f}"


def group_boot(y, g, fn, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(g)
    rows = {i: np.nonzero(g == i)[0] for i in ids}
    vals = []
    for _ in range(B):
        ix = np.concatenate([rows[i] for i in rng.choice(ids, ids.size)])
        vals.append(fn(ix))
    return np.array(vals)


def time_calls(fn, reps):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.median(ts), np.percentile(ts, 95)


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    R = {"eng": d["dev"]["eng"], "hybrid": d["dev"]["hybrid"]}
    Rt = {"eng": d["test"]["eng"], "hybrid": d["test"]["hybrid"]}
    y, g, yt, gt = d["dev"]["y"], d["dev"]["group"], d["test"]["y"], d["test"]["group"]
    print("Experiment E: final system comparison and deployment trade-offs")
    print("  Part 1 - finalists frozen from development-data results: " + ", ".join(FINALISTS))

    # ---------------------------------------------------------------- calibration + thresholds (development only)
    oofs = dict(zip(FINALISTS, Parallel(n_jobs=4)(delayed(oof_scores)(n, R, y, g) for n in FINALISTS)))
    cal, pol = {}, {}
    print("\n  Part 3 - calibration and thresholds from grouped out-of-fold development predictions")
    for n, s in oofs.items():
        cal[n] = PlattCalibrator().fit(s, y)
        pol[n] = thresholds(y, cal[n].transform(s))
        print(f"    {n:<20} OOF AUC {roc_auc_score(y, s):.3f}; Policy A threshold {pol[n][0]:.2f}, Policy B threshold"
              f" {pol[n][1]:.2f}" + ("" if pol[n][2] else " (precision 0.90 not reachable on development data)"))

    # ---------------------------------------------------------------- final fits
    models, fit_t, size = {}, {}, {}
    for n in FINALISTS:
        t0 = time.perf_counter()
        models[n] = FINALISTS[n]().fit(R, y, g)
        fit_t[n] = time.perf_counter() - t0
        size[n] = len(pickle.dumps(models[n]))

    # ---------------------------------------------------------------- Part 2
    print("\n  Part 2 - untouched test images (30 images, 1499 candidates); 1000 group-bootstrap resamples of images")
    print(f"    {'model':<20}{'AUC [95% CI]':>22}{'log loss':>9}{'Brier':>7}{'Policy A: recall / prec / FPR':>32}"
          f"{'Policy B: recall / prec / FPR':>32}{'worst img AUC':>14}{'10th pct':>9}{'fit s':>7}{'size KB':>9}")
    P = {}
    boot = {}
    for n, m in models.items():
        p = cal[n].transform(m.score(Rt))
        P[n] = p
        ra = rates(yt, (p >= pol[n][0]).astype(int))
        rb = rates(yt, (p >= pol[n][1]).astype(int))
        b = group_boot(yt, gt, lambda ix: roc_auc_score(yt[ix], p[ix]))
        boot[n] = b
        per = [roc_auc_score(yt[gt == i], p[gt == i]) for i in np.unique(gt) if 0 < yt[gt == i].sum() < (gt == i).sum()]
        print(f"    {n:<20}{f'{roc_auc_score(yt, p):.3f} [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]':>22}"
              f"{log_loss(yt, p):>9.3f}{brier(yt, p):>7.3f}"
              f"{policy_cell(ra):>32}{policy_cell(rb):>32}"
              f"{np.min(per):>14.3f}{np.percentile(per, 10):>9.3f}{fit_t[n]:>7.1f}{size[n] / 1024:>9.0f}")
    print("    group-bootstrap intervals for the frozen policies:")
    for n, p in P.items():
        ta, tb = pol[n][0], pol[n][1]
        rec = group_boot(yt, gt, lambda ix: rates(yt[ix], (p[ix] >= ta).astype(int))["recall"])
        pre = group_boot(yt, gt, lambda ix: rates(yt[ix], (p[ix] >= tb).astype(int))["precision"])
        print(f"      {n:<20} Policy A recall [{np.percentile(rec, 2.5):.3f}, {np.percentile(rec, 97.5):.3f}] "
              f"(P(>=0.95) = {np.mean(rec >= 0.95):.0%});  Policy B precision [{np.nanpercentile(pre, 2.5):.3f},"
              f" {np.nanpercentile(pre, 97.5):.3f}] (P(>=0.90) = {np.mean(pre >= 0.90):.0%})")
    print("    paired AUC differences vs the stacking ensemble (same resampled images):")
    for n in FINALISTS:
        if n == "stacking ensemble":
            continue
        pn, ps = P[n], P["stacking ensemble"]
        diff = group_boot(yt, gt, lambda ix: roc_auc_score(yt[ix], ps[ix]) - roc_auc_score(yt[ix], pn[ix]))
        print(f"      stacking - {n:<20} {diff.mean():+.4f}  95% CI [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]")

    # ---------------------------------------------------------------- Part 4
    print("\n  Part 4 - distribution shift (30 newly generated images per scenario; frozen models, calibration and thresholds)")
    shift = {}
    for sc in SCENARIOS:
        X, ys, subs, gs = generate_shifted(sc)
        eng, hyb = features_from_patches(X, d["pca_mean"], d["pca_comps"])
        shift[sc] = ({"eng": eng, "hybrid": hyb}, ys, subs)
    names = list(FINALISTS) + ["  member: KNN", "  member: RBF SVM", "  member: random forest"]
    print(f"    {'model':<26}" + "".join(f"{sc[:24]:>26}" for sc in SCENARIOS))
    auc_tab = {}
    for n in names:
        cells = []
        for sc, (Rs, ys, subs) in shift.items():
            if n.startswith("  member"):
                p = models["stacking ensemble"].members(Rs)[n.split(": ")[1]]
                a = roc_auc_score(ys, p)
                cells.append(f"{a:.3f}")
            else:
                p = cal[n].transform(models[n].score(Rs))
                a = roc_auc_score(ys, p)
                r = rates(ys, (p >= pol[n][0]).astype(int))
                cells.append(f"{a:.3f} (A: {r['recall']:.2f}/{r['fpr']:.2f})")
            auc_tab[(n, sc)] = a
        print(f"    {n:<26}" + "".join(f"{c:>26}" for c in cells))
    print("    cells: AUC (Policy A recall / FPR at the frozen threshold); members: AUC of the stacking ensemble's own base models")
    print("    AUC drop vs nominal:")
    for n in names:
        print(f"      {n:<26}" + "".join(f"{auc_tab[(n, sc)] - auc_tab[(n, 'nominal')]:>+10.3f}" for sc in list(SCENARIOS)[1:]))
    Rs, ys, subs = shift["unseen ghosts (35% of artifacts)"]
    print("    ghosts accepted at Policy A: " + ", ".join(
        f"{n} {np.mean(cal[n].transform(models[n].score(Rs))[subs == 'ghost'] >= pol[n][0]):.0%}" for n in FINALISTS))
    Rs, ys, subs = shift["rare stars (5%)"]
    print("    rare stars (5%), Policy B precision: " + ", ".join(
        f"{n} {rates(ys, (cal[n].transform(models[n].score(Rs)) >= pol[n][1]).astype(int))['precision']:.2f}" for n in FINALISTS))

    # ---------------------------------------------------------------- Part 5
    import threadpoolctl
    blas = threadpoolctl.threadpool_info()
    print("\n  Part 5 - prediction latency (CPU only; no GPU used by any model). Single process; random forest n_jobs=1.")
    print(f"    hardware: {platform.processor()} | {os.cpu_count()} logical CPUs | {platform.platform()}")
    print(f"    software: Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__},"
          f" scikit-learn {sklearn.__version__}, BLAS {blas[0]['internal_api'] if blas else 'n/a'}"
          f" ({blas[0].get('num_threads', '?') if blas else '?'} threads)")
    Xp = np.load(OUT.parent.parent / "module-7" / "outputs" / "cache_module6_data.npz")["test_X"].astype(float)
    print(f"    {'model':<20}" + "".join(f"{f'N={b}: med / p95 ms':>26}" for b in (10, 100, 1000, 5000)) + f"{'us/candidate @5000':>20}")
    lat = {}
    for n, m in models.items():
        cells = []
        for bsz in (10, 100, 1000, 5000):
            idx = np.arange(bsz) % len(yt)
            Rb = subset(Rt, idx)
            med, p95 = time_calls(lambda: cal[n].transform(m.score(Rb)), 30 if bsz <= 1000 else 10)
            lat[(n, bsz)] = med
            cells.append(f"{1000 * med:8.2f} / {1000 * p95:8.2f}")
        print(f"    {n:<20}" + "".join(f"{c:>26}" for c in cells) + f"{1e6 * lat[(n, 5000)] / 5000:>20.1f}")
    cells = []
    for bsz in (10, 100, 1000, 5000):
        Pb = Xp[np.arange(bsz) % len(Xp)]
        med, p95 = time_calls(lambda: features_from_patches(Pb, d["pca_mean"], d["pca_comps"]), 10 if bsz <= 1000 else 3)
        cells.append(f"{1000 * med:8.2f} / {1000 * p95:8.2f}")
    print(f"    {'feature extraction':<20}" + "".join(f"{c:>26}" for c in cells) + "   (moments + PCA-C from 32x32 patches)")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    sc_names = list(SCENARIOS)
    for n, mk in zip(names, ("o-", "s-", "^-", "D-", "o:", "s:", "^:")):
        ax.plot(range(len(sc_names)), [auc_tab[(n, sc)] for sc in sc_names], mk, label=n.strip())
    ax.set_xticks(range(len(sc_names)), [s.split(" (")[0] for s in sc_names], rotation=20)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Finalists and stacking members under distribution shift")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for n in FINALISTS:
        ax.loglog((10, 100, 1000, 5000), [1000 * lat[(n, b)] for b in (10, 100, 1000, 5000)], "o-", label=n)
    ax.set_xlabel("batch size (candidates)")
    ax.set_ylabel("median prediction latency [ms]")
    ax.set_title("Prediction latency (features precomputed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_e_final_system.png", dpi=130)
    plt.close(fig)
    print("\n  saved", OUT / "exp_e_final_system.png")


if __name__ == "__main__":
    main()
