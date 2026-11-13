"""
Experiment D: operational thresholds, calibration and uncertainty, with image-level independence.

Classifiers: Module 4's selected KNN (k = 51, invariant features: log width, roundness, log SNR)
and an alternative, a depth-6 decision tree on the same features.

Data roles (all splits by IMAGE):
  fit images          30 development images: train the classifier
  calibration images  the other 20 development images: fit calibrators, choose thresholds
  test images         30 untouched images: evaluate once
Policy A (high recall):    largest threshold with calibration recall >= 0.95
Policy B (high precision): smallest threshold with calibration precision >= 0.90
(instructional targets, not validated operational requirements)

Part 2 compares native / Platt / isotonic probabilities, overall and by imaging condition, and
repeats calibration + threshold selection strictly inside each outer fold of a grouped 5-fold CV.
Part 3 puts group-bootstrap confidence intervals (resampling whole test images) on recall,
precision and FPR, and compares them with a naive candidate-level bootstrap.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

from grouped_data import INVARIANT, OUT, development_set, untouched_test_set
from ml_tools import Pipeline, PlattCalibrator, brier, log_loss, rates, reliability

MODELS = {"KNN k=51": ("knn", {"k": 51}), "tree d=6": ("tree", {"depth": 6})}
RECALL_T, PREC_T = 0.95, 0.90
THR = np.round(np.arange(0.01, 1.0, 0.01), 2)
B = 2000


class Calibrated:
    def __init__(self, pipe, method, Xc, yc):
        self.pipe, self.method = pipe, method
        if method == "platt":
            self.cal = PlattCalibrator().fit(pipe.score(Xc), yc)
        elif method == "isotonic":
            self.cal = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(pipe.proba(Xc), yc)

    def proba(self, X):
        if self.method == "native":
            return self.pipe.proba(X)
        if self.method == "platt":
            return self.cal.transform(self.pipe.score(X))
        return self.cal.predict(self.pipe.proba(X))


def choose_thresholds(y, p):
    r = {t: rates(y, (p >= t).astype(int)) for t in THR}
    a = max([t for t in THR if r[t]["recall"] >= RECALL_T], default=0.01)
    ok = [t for t in THR if r[t]["precision"] >= PREC_T and r[t]["accepted"] >= 30]
    b = min(ok) if ok else max((t for t in THR if r[t]["accepted"] >= 30), key=lambda t: r[t]["precision"])
    return {"A": float(a), "B": float(b), "B_feasible": bool(ok)}


def per_image(y, p, g, t):
    out = []
    for i in np.unique(g):
        m = g == i
        out.append(rates(y[m], (p[m] >= t).astype(int)))
    return out


def counts(y, p, g, t):
    """Per-image TP, FP, FN, TN arrays (for fast bootstrapping)."""
    ids = np.unique(g)
    C = np.zeros((ids.size, 4))
    yhat = p >= t
    for k, i in enumerate(ids):
        m = g == i
        C[k] = [np.sum(yhat[m] & (y[m] == 1)), np.sum(yhat[m] & (y[m] == 0)),
                np.sum(~yhat[m] & (y[m] == 1)), np.sum(~yhat[m] & (y[m] == 0))]
    return C


def metrics_from_counts(C):
    tp, fp, fn, tn = C[..., 0], C[..., 1], C[..., 2], C[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / (tp + fn), tp / (tp + fp), fp / (fp + tn)


def main():
    OUT.mkdir(exist_ok=True)
    X, y, g, sub, table = development_set()
    Xt, yt, gt, subt, tabt = untouched_test_set()
    rng = np.random.default_rng(55)
    ids = rng.permutation(np.unique(g))
    fit_m, cal_m = np.isin(g, ids[:30]), np.isin(g, ids[30:])
    print("Experiment D: thresholds, calibration, uncertainty (invariant features)")
    print(f"  fit images 30 ({fit_m.sum()} candidates), calibration images 20 ({cal_m.sum()}),"
          f" untouched test images 30 ({yt.size})")

    store = {}
    for mname, (kind, params) in MODELS.items():
        pipe = Pipeline(kind, INVARIANT, **params).fit(X[fit_m], y[fit_m])
        cals = {m: Calibrated(pipe, m, X[cal_m], y[cal_m]) for m in ("native", "platt", "isotonic")}
        store[mname] = cals

    # ---------------------------------------------------------------- Part 1
    print("\n  Part 1 - thresholds chosen on the calibration images (Platt probabilities), evaluated on"
          " the untouched test images")
    thr_all = {}
    for mname, cals in store.items():
        c = cals["platt"]
        th = choose_thresholds(y[cal_m], c.proba(X[cal_m]))
        thr_all[mname] = th
        pt = c.proba(Xt)
        th_cheat = choose_thresholds(yt, pt)
        print(f"\n    {mname}: Policy A threshold {th['A']:.2f}, Policy B threshold {th['B']:.2f}"
              + ("" if th["B_feasible"] else " (precision target not reachable on calibration images)"))
        print(f"      {'policy':<8}{'split':<14}{'recall':>8}{'precision':>11}{'FPR':>7}{'target met':>12}"
              f"{'images meeting target':>23}")
        for pol, key, tgt in (("A", "recall", RECALL_T), ("B", "precision", PREC_T)):
            for lab, yy, pp, gg in (("calibration", y[cal_m], c.proba(X[cal_m]), g[cal_m]), ("test", yt, pt, gt)):
                r = rates(yy, (pp >= th[pol]).astype(int))
                pim = [d[key] for d in per_image(yy, pp, gg, th[pol]) if not np.isnan(d[key])]
                print(f"      {pol:<8}{lab:<14}{r['recall']:>8.3f}{r['precision']:>11.3f}{r['fpr']:>7.3f}"
                      f"{str(r[key] >= tgt):>12}{np.mean(np.array(pim) >= tgt):>22.0%}")
        print(f"      (if thresholds were tuned ON the test set: A {th_cheat['A']:.2f}, B {th_cheat['B']:.2f}"
              " - the targets would be met by construction, which says nothing about new images)")

    # ---------------------------------------------------------------- Part 2
    print("\n  Part 2 - native vs calibrated probabilities on the untouched test images")
    print(f"    {'model':<10}{'probabilities':<14}{'Brier':>8}{'log loss':>10}{'ECE':>7}"
          f"{'ECE by condition (low / mid / high defocus)':>46}{'ECE by sky (low / mid / high)':>32}")
    def_t = np.array([t["defocus"] for t in tabt])
    sky_t = np.array([t["sky"] for t in tabt])
    img_ids = np.array([t["image"] for t in tabt])
    def bins(v):
        q = np.quantile(v, [1 / 3, 2 / 3])
        return [img_ids[v <= q[0]], img_ids[(v > q[0]) & (v <= q[1])], img_ids[v > q[1]]]
    rel = {}
    for mname, cals in store.items():
        for meth, c in cals.items():
            p = c.proba(Xt)
            e = reliability(yt, p)
            rel[(mname, meth)] = e
            ece_def = [reliability(yt[np.isin(gt, b)], p[np.isin(gt, b)])[3] for b in bins(def_t)]
            ece_sky = [reliability(yt[np.isin(gt, b)], p[np.isin(gt, b)])[3] for b in bins(sky_t)]
            print(f"    {mname:<10}{meth:<14}{brier(yt, p):>8.4f}{log_loss(yt, p):>10.4f}{e[3]:>7.3f}"
                  f"{' / '.join(f'{v:.3f}' for v in ece_def):>46}{' / '.join(f'{v:.3f}' for v in ece_sky):>32}")

    print("\n    Strict nested evaluation (grouped 5-fold): inside each outer-training portion, 70% of"
          " its images fit the KNN, 30% calibrate it and choose thresholds; outer fold scored once")
    print(f"      {'fold':>5}{'ECE native':>12}{'ECE Platt':>11}{'thr A':>7}{'recall@A':>10}{'thr B':>7}"
          f"{'prec@B':>8}")
    nested_rows = []
    for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=3).split(X, y, g)):
        r_ = np.random.default_rng(k)
        tr_ids = r_.permutation(np.unique(g[tr]))
        n_fit = int(round(0.7 * tr_ids.size))
        fm, cm = np.isin(g, tr_ids[:n_fit]), np.isin(g, tr_ids[n_fit:])
        pipe = Pipeline("knn", INVARIANT, k=51).fit(X[fm], y[fm])
        c = Calibrated(pipe, "platt", X[cm], y[cm])
        th = choose_thresholds(y[cm], c.proba(X[cm]))
        pn, pc = pipe.proba(X[te]), c.proba(X[te])
        ra = rates(y[te], (pc >= th["A"]).astype(int))
        rb = rates(y[te], (pc >= th["B"]).astype(int))
        nested_rows.append((reliability(y[te], pn)[3], reliability(y[te], pc)[3], ra["recall"], rb["precision"]))
        print(f"      {k:>5}{nested_rows[-1][0]:>12.3f}{nested_rows[-1][1]:>11.3f}{th['A']:>7.2f}{ra['recall']:>10.3f}"
              f"{th['B']:>7.2f}{rb['precision']:>8.3f}")
    nr = np.array(nested_rows)
    print(f"      mean{nr[:, 0].mean():>13.3f}{nr[:, 1].mean():>11.3f}{'':>7}{nr[:, 2].mean():>10.3f}{'':>7}{nr[:, 3].mean():>8.3f}"
          f"   (recall >= 0.95 in {np.mean(nr[:, 2] >= RECALL_T):.0%} of folds, precision >= 0.90 in"
          f" {np.mean(nr[:, 3] >= PREC_T):.0%})")

    # ---------------------------------------------------------------- Part 3
    print(f"\n  Part 3 - bootstrap 95% intervals on the untouched test set ({B} resamples), Platt"
          " probabilities, frozen thresholds")
    print(f"    {'model':<10}{'policy':<7}{'metric':<10}{'point':>7}{'group bootstrap CI':>22}{'width':>7}"
          f"{'naive (candidate) CI':>24}{'width':>7}{'P(target met)':>15}")
    boot_store = {}
    for mname, cals in store.items():
        p = cals["platt"].proba(Xt)
        for pol in ("A", "B"):
            t = thr_all[mname][pol]
            C = counts(yt, p, gt, t)
            rb = np.random.default_rng(9)
            idx = rb.integers(0, C.shape[0], size=(B, C.shape[0]))
            Cb = C[idx].sum(axis=1)
            rec_b, prec_b, fpr_b = metrics_from_counts(Cb)
            # naive: resample candidates ignoring images
            yhat = p >= t
            cand = np.column_stack([yhat & (yt == 1), yhat & (yt == 0), ~yhat & (yt == 1), ~yhat & (yt == 0)]).astype(float)
            nb = np.random.default_rng(10).integers(0, yt.size, size=(B, yt.size))
            Cn = np.zeros((B, 4))
            for b0 in range(0, B, 200):
                Cn[b0:b0 + 200] = cand[nb[b0:b0 + 200]].sum(axis=1)
            rec_n, prec_n, fpr_n = metrics_from_counts(Cn)
            point = metrics_from_counts(C.sum(axis=0))
            boot_store[(mname, pol)] = (rec_b, prec_b)
            for lab, pt_, vb, vn, tgt in (("recall", point[0], rec_b, rec_n, RECALL_T if pol == "A" else None),
                                          ("precision", point[1], prec_b, prec_n, PREC_T if pol == "B" else None),
                                          ("FPR", point[2], fpr_b, fpr_n, None)):
                lo, hi = np.nanpercentile(vb, [2.5, 97.5])
                nlo, nhi = np.nanpercentile(vn, [2.5, 97.5])
                pm = f"{np.mean(vb >= tgt):.0%}" if tgt is not None else "-"
                print(f"    {mname:<10}{pol:<7}{lab:<10}{pt_:>7.3f}{f'[{lo:.3f}, {hi:.3f}]':>22}{hi - lo:>7.3f}"
                      f"{f'[{nlo:.3f}, {nhi:.3f}]':>24}{nhi - nlo:>7.3f}{pm:>15}")

    plot(rel, boot_store, thr_all)


def plot(rel, boot_store, thr_all):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    for ax, mname in zip(axes[:2], MODELS):
        for meth, mk in (("native", "o-"), ("platt", "s-"), ("isotonic", "^-")):
            conf, freq, cnt, ece = rel[(mname, meth)]
            ok = cnt >= 10
            ax.plot(conf[ok], freq[ok], mk, label=f"{meth} (ECE {ece:.3f})")
        ax.plot([0, 1], [0, 1], "k:")
        ax.set_aspect("equal")
        ax.set_xlabel("predicted P(star)")
        ax.set_ylabel("observed star fraction")
        ax.set_title(f"{mname}: reliability on 30 untouched images")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax = axes[2]
    for mname, col in zip(MODELS, ("tab:blue", "tab:green")):
        rec_b, _ = boot_store[(mname, "A")]
        _, prec_b = boot_store[(mname, "B")]
        ax.hist(rec_b, bins=40, alpha=0.5, color=col, label=f"{mname}: recall, Policy A")
        ax.hist(prec_b, bins=40, alpha=0.5, color=col, hatch="//", histtype="stepfilled",
                label=f"{mname}: precision, Policy B")
    ax.axvline(RECALL_T, color="k", ls="--", lw=1)
    ax.axvline(PREC_T, color="k", ls=":", lw=1)
    ax.set_xlabel("metric value (group bootstrap over test images)")
    ax.set_title("Is the target met? Group-bootstrap distributions")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_calibration_bootstrap.png", dpi=130)
    plt.close(fig)
    print(f"\n  saved {OUT / 'exp_d_calibration_bootstrap.png'}")


if __name__ == "__main__":
    main()
