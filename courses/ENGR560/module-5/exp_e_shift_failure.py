"""
Experiment E: noise, shift and failure analysis.

Models are trained and frozen exactly as in Experiment D (30 fit images, 20 calibration images of the
development set; Platt calibration; Policy A / B thresholds chosen on the calibration images):
  KNN k=51 and a depth-6 tree, both on the invariant features (log width, roundness, log SNR),
  plus KNN k=51 on all 5 features as a contrast for the background scenario.

Every scenario is a fresh set of 30 images (own seed) with the per-image variation of the
development data plus the scenario perturbation:
  nominal                         familiar camera and noise
  noise x1.5 / x2 / x3            pixel noise AND feature-measurement scatter (moderate ... severe)
  pixel noise x3 only             pixel noise alone: shows the detection-threshold selection effect
  defocus x1.2 / x1.4 / x1.7 / x2 broadened stellar PSF
  background x3 / x10             brighter sky
  rare stars 5% / 1%              reduced prevalence (prior shift only)
  unseen artifacts                35% of artifacts are ghosts (never in training)
Recorded: ROC AUC, recall / precision / FPR at the FROZEN thresholds, ECE, acceptance rate per
source subtype, % of candidates outside the training feature range, and an error decomposition
that attributes the change in false positives / false negatives to source subtypes.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp_d_thresholds_calibration import Calibrated, choose_thresholds
from grouped_data import (INVARIANT, OUT, PER_IMAGE, development_set, engineered,
                          generate_grouped_candidates, scenario)
from ml_tools import ALL5, Pipeline, auc, rates, reliability

SCENARIOS = {
    "nominal": scenario(),
    "noise x1.5": scenario(noise_mult=1.5, feature_noise_mult=1.5),
    "noise x2": scenario(noise_mult=2.0, feature_noise_mult=2.0),
    "noise x3": scenario(noise_mult=3.0, feature_noise_mult=3.0),
    "pixel noise x3 only": scenario(noise_mult=3.0),
    "defocus x1.2": scenario(defocus_mult=1.2),
    "defocus x1.4": scenario(defocus_mult=1.4),
    "defocus x1.7": scenario(defocus_mult=1.7),
    "defocus x2": scenario(defocus_mult=2.0),
    "background x3": scenario(background_mult=3.0),
    "background x10": scenario(background_mult=10.0),
    "rare stars 5%": scenario(star_fraction=0.05),
    "rare stars 1%": scenario(star_fraction=0.01),
    "unseen artifacts": scenario(ghost_fraction=0.35),
}
SUBTYPES = ("bright star", "faint star", "hot pixel", "diffuse", "noise peak", "compact irregular", "ghost")
MODELS = {"KNN k=51 [inv 3]": ("knn", {"k": 51}, INVARIANT), "tree d=6 [inv 3]": ("tree", {"depth": 6}, INVARIANT),
          "KNN k=51 [all 5]": ("knn", {"k": 51}, ALL5)}


def build():
    X, y, g, _, _ = development_set()
    ids = np.random.default_rng(55).permutation(np.unique(g))      # identical split to Experiment D
    fit_m, cal_m = np.isin(g, ids[:30]), np.isin(g, ids[30:])
    frozen = {}
    for name, (kind, params, cols) in MODELS.items():
        pipe = Pipeline(kind, cols, **params).fit(X[fit_m], y[fit_m])
        c = Calibrated(pipe, "platt", X[cal_m], y[cal_m])
        frozen[name] = (c, choose_thresholds(y[cal_m], c.proba(X[cal_m])))
    E = engineered(X[fit_m])
    lo, hi = np.percentile(E, [0.5, 99.5], axis=0)
    return frozen, (lo, hi)


def main():
    OUT.mkdir(exist_ok=True)
    frozen, (lo, hi) = build()
    print("Experiment E: noise, shift and failure analysis (30 new images per scenario; thresholds frozen)")
    for name, (_, th) in frozen.items():
        print(f"  {name}: Policy A threshold {th['A']:.2f}, Policy B threshold {th['B']:.2f}")

    data = {}
    for k, (sname, sc) in enumerate(SCENARIOS.items()):
        data[sname] = generate_grouped_candidates(30, PER_IMAGE, np.random.default_rng([900, k]), sc=sc,
                                                  first_id=10_000 * (k + 1))

    results = {}
    for mname, (c, th) in frozen.items():
        print(f"\n  {mname}")
        print(f"    {'scenario':<18}{'P(star)':>8}{'AUC':>7}{'recall A':>9}{'prec A':>8}{'FPR A':>7}"
              f"{'recall B':>9}{'prec B':>8}{'FPR B':>7}{'ECE':>7}{'outside inv':>12}{'outside all':>12}")
        for sname, (X, y, g, sub, _) in data.items():
            p = c.proba(X)
            ra = rates(y, (p >= th["A"]).astype(int))
            rb = rates(y, (p >= th["B"]).astype(int))
            E = engineered(X)
            out_inv = np.mean(np.any((E[:, INVARIANT] < lo[INVARIANT]) | (E[:, INVARIANT] > hi[INVARIANT]), axis=1))
            out_all = np.mean(np.any((E < lo) | (E > hi), axis=1))
            accA = {t: (np.mean(p[sub == t] >= th["A"]) if np.any(sub == t) else np.nan) for t in SUBTYPES}
            results[(mname, sname)] = {"auc": auc(y, c.pipe.score(X)), "ra": ra, "rb": rb,
                                       "ece": reliability(y, p)[3], "accA": accA, "sub": sub, "y": y,
                                       "pA": p >= th["A"], "out": out_inv}
            print(f"    {sname:<18}{y.mean():>8.3f}{results[(mname, sname)]['auc']:>7.3f}{ra['recall']:>9.3f}"
                  f"{ra['precision']:>8.3f}{ra['fpr']:>7.3f}{rb['recall']:>9.3f}{rb['precision']:>8.3f}"
                  f"{rb['fpr']:>7.3f}{results[(mname, sname)]['ece']:>7.3f}{out_inv:>12.1%}{out_all:>12.1%}")

    main_model = "KNN k=51 [inv 3]"
    print(f"\n  Acceptance rate by subtype at Policy A ({main_model}); stars: want high, artifacts: want low")
    print(f"    {'scenario':<18}" + "".join(f"{t:>14}" for t in SUBTYPES))
    for sname in SCENARIOS:
        a = results[(main_model, sname)]["accA"]
        print(f"    {sname:<18}" + "".join(f"{a[t]:>14.2f}" if not np.isnan(a[t]) else f"{'-':>14}" for t in SUBTYPES))

    # ---------------------------------------------------------------- failure report
    print(f"\n  Failure report ({main_model}, Policy A): errors per 1000 candidates by source subtype,"
          " and the change vs nominal")
    nom = results[(main_model, "nominal")]

    def per1000(r):
        sub, y, acc = r["sub"], r["y"], r["pA"]
        n = y.size
        fp = {t: 1000 * np.sum(acc & (sub == t)) / n for t in SUBTYPES[2:]}
        fn = {t: 1000 * np.sum(~acc & (sub == t)) / n for t in SUBTYPES[:2]}
        return fp, fn

    fp0, fn0 = per1000(nom)
    print(f"    {'scenario':<18}{'FP/1000':>9}{'FN/1000':>9}   largest contributors to the change vs nominal")
    report = []
    for sname in SCENARIOS:
        fp, fn = per1000(results[(main_model, sname)])
        delta = {f"+FP {t}": fp[t] - fp0[t] for t in fp}
        delta.update({f"+FN {t}": fn[t] - fn0[t] for t in fn})
        top = sorted(delta.items(), key=lambda kv: -abs(kv[1]))[:3]
        report.append((sname, sum(fp.values()), sum(fn.values()), top))
        print(f"    {sname:<18}{sum(fp.values()):>9.1f}{sum(fn.values()):>9.1f}   "
              + ", ".join(f"{k} {v:+.0f}" for k, v in top))
    print("    (FP per 1000 candidates depends on the subtype mix of the scenario; for rare stars the drop"
          " in FN is just fewer stars, and the precision collapse appears in the table above)")

    plot(results)


def plot(results):
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.6))
    ax = axes[0]
    for mname, mk in zip(MODELS, ("o-", "s--", "^:")):
        for fam, col, keys, xs in (("noise", "tab:red", ["nominal", "noise x1.5", "noise x2", "noise x3"], [1, 1.5, 2, 3]),
                                   ("defocus", "tab:blue", ["nominal", "defocus x1.2", "defocus x1.4", "defocus x1.7",
                                                            "defocus x2"], [1, 1.2, 1.4, 1.7, 2])):
            ax.plot(xs, [results[(mname, k)]["auc"] for k in keys], mk, color=col,
                    label=f"{fam}: {mname}")
    ax.set_xlabel("severity multiplier")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Degradation with noise and defocus severity")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = list(SCENARIOS)
    m = "KNN k=51 [inv 3]"
    x = np.arange(len(names))
    ax.plot(x, [results[(m, n)]["ra"]["recall"] for n in names], "o-", label="recall (Policy A)")
    ax.plot(x, [results[(m, n)]["ra"]["precision"] for n in names], "s-", label="precision (Policy A)")
    ax.plot(x, [results[(m, n)]["rb"]["precision"] for n in names], "s--", label="precision (Policy B)")
    ax.plot(x, [results[(m, n)]["ra"]["fpr"] for n in names], "^-", label="FPR (Policy A)")
    ax.plot(x, [results[(m, n)]["ece"] for n in names], "d:", label="ECE")
    ax.axhline(0.95, color="k", lw=0.6, ls="--")
    ax.axhline(0.90, color="k", lw=0.6, ls=":")
    ax.set_xticks(x, names, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{m}: frozen-threshold metrics per scenario")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    M = np.array([[results[(m, n)]["accA"][t] for t in SUBTYPES] for n in names])
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(SUBTYPES)), SUBTYPES, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_title("Acceptance rate by subtype (Policy A)\nstars should be green-high, artifacts low")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "exp_e_shift_failure.png", dpi=130)
    plt.close(fig)
    print(f"\n  saved {OUT / 'exp_e_shift_failure.png'}")


if __name__ == "__main__":
    main()
