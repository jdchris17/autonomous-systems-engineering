"""
Experiment F: robustness and distribution shift, plus the final integration.

1. Build the system (star_classifier.build_system): all candidates trained on nominal data,
   calibrated and selected WITHOUT the final test sets.
2. Evaluate every candidate, without retraining, on the FINAL TEST draw of every scenario.
3. Separate prior-probability shift from feature-distribution shift:
     prior shift  -> recall and FPR unchanged, precision and calibration change (fixable with
                     the Bayes prior correction if the new prior is known)
     feature shift -> recall / FPR / AUC themselves change; no relabelling of probabilities fixes it
4. An unlabelled shift monitor: per-feature standardized mean shift and the fraction of candidates
   outside the training feature range - computable on real data, where labels do not exist.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classifiers import OUT, auc, log_loss, rates, reliability
from star_classifier import FEATURE_SETS, PRIOR_ONLY, SCENARIOS, build_system, scenario_sets
from star_data import ENG_NAMES, engineered

MODELS = ("perceptron", "logistic", "linear SVM", "RBF SVM", "KNN", "tree")


def main():
    OUT.mkdir(exist_ok=True)
    print("Experiment F: robustness and distribution shift")
    clf, prepared, y = build_system()
    card = clf.card

    print("\n  Model selection on the SELECTION sets (never the final test sets):")
    print(f"    {'pipeline':<26}{'nominal AUC':>12}{'nominal LL':>11}{'ECE':>7}{'shift LL':>10}"
          f"{'shift AUC':>11}{'worst AUC':>11}{'robust LL':>11}{'ms/1000':>9}")
    for n, r in card["selection_table"].items():
        flag = "  <- selected" if n == card["selected_model"] else ("" if n in card["eligible"] else "  (ineligible)")
        print(f"    {n:<26}{r['nominal_auc']:>12.3f}{r['nominal_logloss']:>11.3f}{r['nominal_ece']:>7.3f}"
              f"{r['shift_logloss_mean']:>10.3f}{r['shift_auc_mean']:>11.3f}{r['shift_auc_worst']:>11.3f}"
              f"{r['robust_logloss']:>11.3f}{r['cost_ms_per_1000']:>9.1f}{flag}")
    print(f"    selected: {card['selected_model']} ({card['hyperparameters']}),"
          f" thresholds: Policy A {card['thresholds']['A']:.2f}, Policy B {card['thresholds']['B']:.2f}")

    # ----------------------------------------------------------------- final test sets
    test = scenario_sets("final test")
    auc_tab = np.zeros((len(SCENARIOS), len(MODELS) * len(FEATURE_SETS)))
    col_labels = []
    for f_i, (fs, cols) in enumerate(FEATURE_SETS.items()):
        st_f, _, cands = prepared[fs]
        print(f"\n  All candidates, feature set '{fs}', on the FINAL TEST sets (no retraining):"
              " AUC / balanced accuracy at calibrated p >= 0.5 / log loss")
        print(f"    {'scenario':<25}" + "".join(f"{m:>20}" for m in MODELS))
        for i, (sname, (X, yy, _)) in enumerate(test.items()):
            ZZ = st_f.transform(engineered(X)[:, cols])
            cells = []
            for j, m in enumerate(MODELS):
                c = cands[m]
                p = c.proba(ZZ)
                a = auc(yy, c.score(ZZ))
                auc_tab[i, f_i * len(MODELS) + j] = a
                ba = rates(yy, (p >= 0.5).astype(int))["balanced_acc"]
                cells.append(f"{a:.3f}/{ba:.3f}/{log_loss(yy, p):.2f}")
            print(f"    {sname:<25}" + "".join(f"{c:>20}" for c in cells))
        col_labels += [f"{m}\n[{fs}]" for m in MODELS]
    # the selected system and the unlabelled monitor work on raw features / all five columns
    st_all, Z, _ = prepared["all 5"]
    TZ = {k: (st_all.transform(engineered(v[0])), v[1], v[2], v[0]) for k, v in test.items()}

    # ----------------------------------------------------------------- selected system in detail
    print(f"\n  Selected system ({card['selected_model']}) under each scenario, both policies:")
    print(f"    {'scenario':<25}{'P(star)':>8}{'policy':>7}{'recall':>8}{'FPR':>7}{'precision':>11}"
          f"{'accepted':>10}{'ECE':>7}{'mean p':>8}")
    detail = {}
    for sname, (ZZ, yy, sub, Xraw) in TZ.items():
        p = clf.proba(Xraw)
        ece = reliability(yy, p)[3]
        for pol in ("A", "B"):
            r = rates(yy, (p >= clf.thresholds[pol]).astype(int))
            detail[(sname, pol)] = r
            print(f"    {sname:<25}{yy.mean():>8.3f}{pol:>7}{r['recall']:>8.3f}{r['fpr']:>7.3f}"
                  f"{r['precision']:>11.3f}{r['accepted']:>10}{ece:>7.3f}{p.mean():>8.3f}")

    # prior shift: precision and calibration fix
    ZZ, yy, sub, Xraw = TZ["different prevalence"]
    pc = clf.proba_for_prior(Xraw, 0.05)
    print(f"\n  Prior shift (P(star) = 0.05): ECE {reliability(yy, clf.proba(Xraw))[3]:.3f} ->"
          f" {reliability(yy, pc)[3]:.3f} after the Bayes prior correction; mean p"
          f" {clf.proba(Xraw).mean():.3f} -> {pc.mean():.3f} (true star fraction {yy.mean():.3f})")

    # acceptance rate by subtype
    print("\n  Fraction ACCEPTED by subtype, Policy A (stars: higher is better; artifacts: lower):")
    subtypes = ("bright star", "faint star", "hot pixel", "diffuse", "noise peak", "compact irregular", "ghost")
    print(f"    {'scenario':<25}" + "".join(f"{t:>12}" for t in subtypes))
    for sname, (ZZ, yy, sub, Xraw) in TZ.items():
        acc_ = clf.decide(Xraw, "A")
        cells = [f"{acc_[sub == t].mean():>12.2f}" if np.any(sub == t) else f"{'-':>12}" for t in subtypes]
        print(f"    {sname:<25}" + "".join(cells))

    # ----------------------------------------------------------------- unlabelled shift monitor
    print("\n  Unlabelled shift monitor (no labels needed): standardized mean shift of each feature"
          " vs training, and % of candidates outside the training 0.5-99.5% range on any feature")
    lo, hi = np.percentile(Z["train"], [0.5, 99.5], axis=0)
    print(f"    {'scenario':<25}" + "".join(f"{n:>15}" for n in ENG_NAMES) + f"{'outside':>10}")
    for sname, (ZZ, yy, sub, Xraw) in TZ.items():
        d = ZZ.mean(axis=0)
        out_ = np.mean(np.any((ZZ < lo) | (ZZ > hi), axis=1))
        print(f"    {sname:<25}" + "".join(f"{v:>+15.2f}" for v in d) + f"{out_:>10.1%}")

    plots(auc_tab, col_labels, detail, TZ, clf)


def plots(auc_tab, col_labels, detail, TZ, clf):
    names = list(SCENARIOS)
    fig, axes = plt.subplots(1, 2, figsize=(24, 6.5), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    im = ax.imshow(auc_tab, cmap="viridis", vmin=0.5, vmax=0.95)
    for i in range(auc_tab.shape[0]):
        for j in range(auc_tab.shape[1]):
            ax.text(j, i, f"{auc_tab[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="w" if auc_tab[i, j] < 0.75 else "k")
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=7)
    ax.set_yticks(range(len(names)), [n + (" (prior only)" if n in PRIOR_ONLY else "") for n in names])
    ax.set_title("AUC on the final test sets (no retraining)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    x = np.arange(len(names))
    for k, (pol, mk) in enumerate((("A", "o"), ("B", "s"))):
        ax.plot(x, [detail[(n, pol)]["recall"] for n in names], mk + "-", color="tab:blue",
                alpha=1 - 0.4 * k, label=f"recall, Policy {pol}")
        ax.plot(x, [detail[(n, pol)]["fpr"] for n in names], mk + "--", color="tab:red",
                alpha=1 - 0.4 * k, label=f"FPR, Policy {pol}")
        ax.plot(x, [detail[(n, pol)]["precision"] for n in names], mk + ":", color="tab:green",
                alpha=1 - 0.4 * k, label=f"precision, Policy {pol}")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Selected system ({clf.card['selected_model']}) per scenario")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_f_shift.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, sname in zip(axes, ("nominal", "different prevalence")):
        ZZ, yy, sub, Xraw = TZ[sname]
        for lab, p in (("calibrated (training prior)", clf.proba(Xraw)),
                       ("prior-corrected", clf.proba_for_prior(Xraw, yy.mean()))):
            conf, freq, cnt, ece = reliability(yy, p)
            ok = cnt >= 10
            ax.plot(conf[ok], freq[ok], "o-", label=f"{lab}, ECE {ece:.3f}")
        ax.plot([0, 1], [0, 1], "k:")
        ax.set_aspect("equal")
        ax.set_xlabel("predicted P(star)")
        ax.set_ylabel("observed star fraction")
        ax.set_title(f"Reliability, {sname} (P(star) = {yy.mean():.2f})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_f_prior_shift_calibration.png", dpi=130)
    plt.close(fig)
    print(f"\n  saved {OUT / 'exp_f_shift.png'}")
    print(f"  saved {OUT / 'exp_f_prior_shift_calibration.png'}")


if __name__ == "__main__":
    main()
