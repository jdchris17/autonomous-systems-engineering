"""
Experiment E: classification thresholds, class imbalance, and probability calibration.

Model: logistic regression (validation-selected L2), trained at P(star) = 0.4 on the standard
training split. Thresholds for operational policies are chosen on VALIDATION and evaluated once
on the untouched TEST split.

Class imbalance: new test sets with P(star) in {0.5, 0.1, 0.01}, generated with the SAME
class-conditional feature distributions (only the label prior changes).

Calibration: reliability diagrams, log loss and Brier score for logistic regression and three
other probabilistic classifiers, native vs Platt-calibrated on the validation split.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classifiers import (OUT, auc, brier, confusion, log_loss, rates, reliability, roc_curve,
                         sigmoid)
from model_zoo import build_candidates, prepared_splits
from star_data import engineered, generate_candidates, standard_splits

THRESHOLDS = np.round(np.arange(0.02, 0.99, 0.01), 2)
PRIORS = (0.5, 0.1, 0.01)
TRAIN_PRIOR = 0.4
RECALL_TARGET = 0.95      # Policy A
PRECISION_TARGET = 0.90   # Policy B


def sweep(y, p):
    return {t: rates(y, (p >= t).astype(int)) for t in THRESHOLDS}


def pr_curve(y, p):
    ts = np.unique(p)[::-1]
    prec, rec = [], []
    for t in ts[:: max(1, ts.size // 400)]:
        r = rates(y, (p >= t).astype(int))
        prec.append(r["precision"])
        rec.append(r["recall"])
    return np.array(rec), np.array(prec)


def prior_shift(p, new_prior, old_prior=TRAIN_PRIOR):
    """Bayes correction of a calibrated probability for a new class prior (likelihoods unchanged)."""
    logit = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
    shift = np.log(new_prior / (1 - new_prior)) - np.log(old_prior / (1 - old_prior))
    return sigmoid(logit + shift)


def policy_table(mname, y, p_val, p_te):
    swv = sweep(y["val"], p_val)
    thr_a = max(t for t in THRESHOLDS if swv[t]["recall"] >= RECALL_TARGET)
    ok_b = [t for t in THRESHOLDS if swv[t]["precision"] >= PRECISION_TARGET and swv[t]["accepted"] >= 50]
    if ok_b:
        thr_b, note = min(ok_b), ""
    else:
        cand = [t for t in THRESHOLDS if swv[t]["accepted"] >= 50]
        thr_b = max(cand, key=lambda t: swv[t]["precision"])
        note = f"  (target unreachable: best val precision {swv[thr_b]['precision']:.3f})"
    print(f"\n    {mname}: Policy A threshold {thr_a:.2f}, Policy B threshold {thr_b:.2f}{note}")
    print(f"      {'policy':<14}{'thr':>6}{'precision':>11}{'recall':>8}{'FPR':>7}{'accepted':>10}"
          f"{'stars lost':>12}{'artifacts passed':>18}")
    for name, t in (("A", thr_a), ("B", thr_b), ("p >= 0.5", 0.5)):
        for split, yy, pp in (("val", y["val"], p_val), ("test", y["test"], p_te)):
            yhat = (pp >= t).astype(int)
            r = rates(yy, yhat)
            C = confusion(yy, yhat)
            print(f"      {name + ' ' + split:<14}{t:>6.2f}{r['precision']:>11.3f}{r['recall']:>8.3f}"
                  f"{r['fpr']:>7.3f}{r['accepted']:>10}{C[1, 0]:>12}{C[0, 1]:>18}")
    return thr_a, thr_b


def main():
    OUT.mkdir(exist_ok=True)
    sp = standard_splits()
    y = {k: v[1] for k, v in sp.items()}
    st, Z = prepared_splits(sp)
    print("Experiment E: thresholds, imbalance, calibration")
    cands = build_candidates(Z["train"], y["train"], Z["val"], y["val"])
    lr = cands["logistic"]
    p_val = lr.native_proba(Z["val"])
    p_te = lr.native_proba(Z["test"])

    # ----------------------------------------------------------------- threshold sweep
    sw = sweep(y["test"], p_te)
    print(f"\n  Logistic regression threshold sweep on TEST (n = {y['test'].size},"
          f" {y['test'].sum()} stars)")
    print(f"    {'thr':>5}{'precision':>11}{'recall':>8}{'FPR':>7}{'F1':>7}{'accepted':>10}")
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        r = sw[t]
        print(f"    {t:>5.1f}{r['precision']:>11.3f}{r['recall']:>8.3f}{r['fpr']:>7.3f}{r['f1']:>7.3f}"
              f"{r['accepted']:>10}")
    print("  Confusion matrices [[TN, FP], [FN, TP]]:")
    for t in (0.2, 0.5, 0.8):
        print(f"    threshold {t}: {confusion(y['test'], (p_te >= t).astype(int)).tolist()}")

    # ----------------------------------------------------------------- policies (val -> test)
    print("\n  Operational policies (thresholds chosen on VALIDATION, evaluated once on TEST)")
    print(f"    Policy A - high recall:    largest threshold with val recall >= {RECALL_TARGET}")
    print(f"    Policy B - high precision: smallest threshold with val precision >= {PRECISION_TARGET}"
          " (if no threshold reaches it: the threshold with the best val precision, >= 50 accepted)")
    policies = {}
    for mname, pv, pt in (("logistic", p_val, p_te),
                          ("RBF SVM", cands["RBF SVM"].proba(Z["val"]), cands["RBF SVM"].proba(Z["test"]))):
        policies[mname] = policy_table(mname, y, pv, pt)
    thr_a, thr_b = policies["logistic"]

    # ----------------------------------------------------------------- class imbalance
    print("\n  Class imbalance: same class-conditional features, different P(star);"
          " model trained at P(star) = 0.4, threshold 0.5")
    print(f"    {'P(star)':>8}{'n':>8}{'stars':>7}{'recall':>8}{'FPR':>7}{'precision':>11}"
          f"{'theory prec':>13}{'accepted':>10}{'AUC':>7}{'mean p':>8}{'ECE':>7}"
          f"{'mean p corr.':>14}{'ECE corr.':>11}")
    imb = {}
    for k, pi in enumerate(PRIORS):
        n = int(min(200_000, 1500 / pi))
        X, yy = generate_candidates(n, np.random.default_rng([500, k]), star_fraction=pi)
        pp = lr.native_proba(st.transform(engineered(X)))
        r = rates(yy, (pp >= 0.5).astype(int))
        tpr, fpr = r["recall"], r["fpr"]
        theory = tpr * pi / (tpr * pi + fpr * (1 - pi))
        pc = prior_shift(lr.proba(st.transform(engineered(X))), pi)
        imb[pi] = (yy, pp, pc)
        print(f"    {pi:>8}{n:>8}{yy.sum():>7}{tpr:>8.3f}{fpr:>7.3f}{r['precision']:>11.3f}{theory:>13.3f}"
              f"{r['accepted']:>10}{auc(yy, pp):>7.3f}{pp.mean():>8.3f}{reliability(yy, pp)[3]:>7.3f}"
              f"{pc.mean():>14.4f}{reliability(yy, pc)[3]:>11.4f}")
    print("    theory precision = recall*pi / (recall*pi + FPR*(1 - pi)).  'corr.' = Platt-calibrated"
          " probability with its log-odds shifted by log[pi/(1-pi)] - log[0.4/0.6]: the Bayes prior"
          " correction, which needs the new prior to be known.")

    # ----------------------------------------------------------------- calibration
    print("\n  Probability calibration on TEST (Platt calibrators fitted on VALIDATION)")
    print(f"    {'model':<26}{'log loss':>9}{'Brier':>8}{'ECE':>7}{'stars among p in [0.85, 0.95]':>32}")
    rel = {}
    for name, label, getp in (
            ("logistic", "logistic (native)", lambda c: c.native_proba(Z["test"])),
            ("logistic", "logistic (Platt)", lambda c: c.proba(Z["test"])),
            ("RBF SVM", "RBF SVM (Platt on margin)", lambda c: c.proba(Z["test"])),
            ("KNN", "KNN k=21 (native votes)", lambda c: c.native_proba(Z["test"])),
            ("KNN", "KNN (Platt)", lambda c: c.proba(Z["test"])),
            ("tree", "tree (native leaf freq.)", lambda c: c.native_proba(Z["test"])),
            ("tree", "tree (Platt)", lambda c: c.proba(Z["test"]))):
        p = getp(cands[name])
        conf, freq, cnt, ece = reliability(y["test"], p)
        m = (p >= 0.85) & (p <= 0.95)
        hit = f"{y['test'][m].mean():.2f} of {m.sum()}" if m.any() else "none"
        rel[label] = (conf, freq, cnt)
        print(f"    {label:<26}{log_loss(y['test'], p):>9.4f}{brier(y['test'], p):>8.4f}{ece:>7.3f}{hit:>32}")

    plots(y, p_te, sw, imb, rel, thr_a, thr_b)


def plots(y, p_te, sw, imb, rel, thr_a, thr_b):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    ax = axes[0]
    for pi, (yy, pp, _) in imb.items():
        rec, prec = pr_curve(yy, pp)
        ax.plot(rec, prec, label=f"P(star) = {pi}")
    rec, prec = pr_curve(y["test"], p_te)
    ax.plot(rec, prec, "k", lw=2, label="standard test (P = 0.4)")
    for t, mk, lab in ((thr_a, "o", "Policy A"), (thr_b, "s", "Policy B"), (0.5, "^", "p >= 0.5")):
        if t is not None:
            r = sw[round(t, 2)]
            ax.plot(r["recall"], r["precision"], mk, ms=9, mfc="none", mec="k", mew=2, label=f"{lab} ({t:.2f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall: the curve depends on the prior")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for pi, (yy, pp, _) in imb.items():
        f, t = roc_curve(yy, pp)
        ax.plot(f, t, label=f"P(star) = {pi}")
    f, t = roc_curve(y["test"], p_te)
    ax.plot(f, t, "k", lw=2, label="standard test")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("recall")
    ax.set_title("ROC: (almost) unchanged by the prior")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for k in ("precision", "recall", "fpr", "f1"):
        ax.plot(THRESHOLDS, [sw[t][k] for t in THRESHOLDS], label=k)
    ax2 = ax.twinx()
    ax2.plot(THRESHOLDS, [sw[t]["accepted"] for t in THRESHOLDS], "k:", label="accepted")
    ax2.set_ylabel("accepted candidates (dotted)")
    for t in (thr_a, thr_b):
        if t is not None:
            ax.axvline(t, color="gray", lw=1, ls="--")
    ax.set_xlabel("probability threshold")
    ax.set_title("Logistic regression: metrics vs threshold (test)")
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_e_pr_roc_thresholds.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, keys in zip(axes, (("logistic (native)", "RBF SVM (Platt on margin)", "KNN k=21 (native votes)",
                                "tree (native leaf freq.)"),
                               ("logistic (Platt)", "RBF SVM (Platt on margin)", "KNN (Platt)", "tree (Platt)"))):
        for k in keys:
            conf, freq, cnt = rel[k]
            ok = cnt >= 10
            ax.plot(conf[ok], freq[ok], "o-", label=k)
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect calibration")
        ax.set_xlabel("mean predicted P(star) in bin")
        ax.set_ylabel("observed fraction of stars")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title("Reliability: native probabilities")
    axes[1].set_title("Reliability: after Platt calibration on validation")
    fig.tight_layout()
    fig.savefig(OUT / "exp_e_reliability.png", dpi=130)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_e_pr_roc_thresholds.png'}")
    print(f"  saved {OUT / 'exp_e_reliability.png'}")


if __name__ == "__main__":
    main()
