"""
Experiment C: selecting mixture models.

Data: Module 6 engineered features (log width, roundness, log SNR) of the 60 development images;
the 30 untouched images are used once, for the final held-out log-likelihood.
Candidates: K in {1, 2, 3, 4, 5, 6, 8, 10} x {diag, full}, own EM, 5 random initialisations each.

Part 1  training log-likelihood, AIC, BIC, free parameters, convergence status, init spread
Part 2  parameter counts:  full  p = (K-1) + K D + K D(D+1)/2
                           diag  p = (K-1) + K D + K D          (D variances per component)
Part 3  selection by GROUPED cross-validation (GroupKFold over images): held-out log-likelihood per
        observation on unseen images; standardisation is refit inside each training fold.
        No subtype labels are used for selection.
Part 4  what the selected components correspond to (labels used only here).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold

from clustering import GaussianMixtureScratch
from data7 import OUT, SUBTYPES, load, standardize

KS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)
COVS = ("diag", "full")
N_INIT = 5
REGS = {"no floor": 1e-6, "resolution floor": 0.05}   # reg_covar in standardised units (variance)
SEL_REG = "resolution floor"


def fit_best(X, K, cov, seed=0, reg=1e-6):
    fits = [GaussianMixtureScratch(K, cov, seed=seed + i, max_iter=500, tol=1e-6, reg_covar=reg).fit(X)
            for i in range(N_INIT)]
    return max(fits, key=lambda m: m.lower_bound_), fits


def cv_cell(K, cov, folds, E, reg):
    out = []
    for tr, va in folds:
        Xtr, Xva = standardize(E[tr], E[va])
        m, _ = fit_best(Xtr, K, cov, reg=reg)
        out.append(m.score(Xva))
    return out


def eta2(labels, v):
    """Correlation ratio: share of the variance of v explained by cluster membership."""
    v = np.asarray(v, float)
    tot = np.sum((v - v.mean()) ** 2)
    return float(sum(np.sum(labels == k) * (v[labels == k].mean() - v.mean()) ** 2 for k in np.unique(labels)) / tot)


def main():
    OUT.mkdir(exist_ok=True)
    z = load()
    E, Et = z["dev_eng"], z["test_eng"]
    S, St = standardize(E, Et)
    n, D = S.shape
    print(f"Experiment C: mixture model selection on {n} development patches ({len(np.unique(z['dev_group']))} images), D = {D}")
    floor_w = np.isclose(E[:, 0], np.log(0.2))
    print(f"  Point masses in the measured features: {floor_w.sum()} patches ({floor_w.mean():.1%}) sit exactly on the"
          f" width floor (unresolved), {np.sum(E[:, 1] >= 0.9999)} at roundness = 1, {np.sum(np.isclose(E[:, 2], np.log(0.1)))}"
          " failed measurements. A Gaussian component can collapse onto such a spike and gain unbounded likelihood,"
          " so selection is run with and without a covariance floor (reg_covar).")

    cells = [(K, c) for K in KS for c in COVS]
    folds = list(GroupKFold(5).split(E, groups=z["dev_group"]))
    full_fits, cvs = {}, {}
    for rname, reg in REGS.items():
        res = Parallel(n_jobs=-1)(delayed(fit_best)(S, K, c, reg=reg) for K, c in cells)
        full_fits[rname] = dict(zip(cells, res))
        cvres = Parallel(n_jobs=-1)(delayed(cv_cell)(K, c, folds, E, reg) for K, c in cells)
        cvs[rname] = dict(zip(cells, [np.array(v) for v in cvres]))

    # ---------------------------------------------------------------- Part 1 + 2
    for rname in REGS:
        full_fit = full_fits[rname]
        print(f"\n  Parts 1-2 - {rname} (reg_covar {REGS[rname]}): fit on all development patches (best of {N_INIT} inits)")
        print(f"    {'K':>3}{'cov':>6}{'params':>8}{'formula':>9}{'sklearn':>9}{'train LL/obs':>14}{'init spread':>13}"
              f"{'AIC':>11}{'BIC':>11}{'converged':>11}{'min comp. weight':>18}")
        for (K, c), (m, fits) in full_fit.items():
            p_formula = (K - 1) + K * D + (K * D * (D + 1) // 2 if c == "full" else K * D)
            skp = GaussianMixture(K, covariance_type=c).fit(S[:200])._n_parameters()
            lls = [f.lower_bound_ for f in fits]
            print(f"    {K:>3}{c:>6}{m.n_parameters():>8}{p_formula:>9}{skp:>9}{m.score(S):>14.4f}{max(lls) - min(lls):>13.4f}"
                  f"{m.aic(S):>11.1f}{m.bic(S):>11.1f}{f'{sum(f.converged_ for f in fits)}/{N_INIT}':>11}{m.weights_.min():>18.4f}")

    # ---------------------------------------------------------------- Part 3
    print("\n  Part 3 - grouped 5-fold CV over images: held-out log-likelihood per observation (mean +- sd over folds)")
    print(f"    {'K':>3}" + "".join(f"{f'{c}, {r}':>30}" for r in REGS for c in COVS))
    for K in KS:
        print(f"    {K:>3}" + "".join(f"{cvs[r][(K, c)].mean():>22.4f} +- {cvs[r][(K, c)].std():.3f}" for r in REGS for c in COVS))
    for r in REGS:
        bb = max(cells, key=lambda kc: cvs[r][kc].mean())
        print(f"    {r}: highest CV log-likelihood at K = {bb[0]}, {bb[1]}")
    full_fit, cv = full_fits[SEL_REG], cvs[SEL_REG]
    print(f"  Selection uses the {SEL_REG} models:")
    best = max(cells, key=lambda kc: cv[kc].mean())
    se = cv[best].std() / np.sqrt(len(folds))
    simple = min([kc for kc in cells if cv[kc].mean() >= cv[best].mean() - se],
                 key=lambda kc: full_fit[kc][0].n_parameters())
    bic_best = min(cells, key=lambda kc: full_fit[kc][0].bic(S))
    aic_best = min(cells, key=lambda kc: full_fit[kc][0].aic(S))
    print(f"    highest CV log-likelihood: K = {best[0]}, {best[1]};  one-standard-error rule (fewest parameters within"
          f" 1 SE = {se:.4f}): K = {simple[0]}, {simple[1]}")
    print(f"    for comparison - BIC picks K = {bic_best[0]}, {bic_best[1]}; AIC picks K = {aic_best[0]}, {aic_best[1]}")
    sel = simple
    model = full_fit[sel][0]
    print(f"    SELECTED: K = {sel[0]}, {sel[1]} covariance -> held-out log-likelihood on the 30 untouched images:"
          f" {model.score(St):.4f} per observation (development fit {model.score(S):.4f})")

    # ---------------------------------------------------------------- Part 4
    lab = model.predict(S)
    sub = z["dev_subtype"]
    print(f"\n  Part 4 - what the {sel[0]} selected components represent (development patches; labels used only now)")
    print(f"    {'comp':>5}{'weight':>8}{'n':>6}" + "".join(f"{s[:11]:>12}" for s in SUBTYPES)
          + f"{'log10 SNR':>11}{'size px':>9}{'axis r.':>8}{'log10 B':>9}{'PSF sig':>9}")
    size = np.sqrt(z["dev_s_major"] * z["dev_s_minor"])
    for k in np.argsort(-model.weights_):
        m = lab == k
        if not m.any():
            continue
        print(f"    {k:>5}{model.weights_[k]:>8.3f}{m.sum():>6}" + "".join(f"{np.mean(sub[m] == s):>12.0%}" for s in SUBTYPES)
              + f"{np.median(np.log10(z['dev_snr_true'][m])):>11.2f}{np.median(size[m]):>9.2f}"
              f"{np.median(z['dev_s_minor'][m] / z['dev_s_major'][m]):>8.2f}{np.median(np.log10(z['dev_B'][m])):>9.2f}"
              f"{np.median(z['dev_img_sigma'][m]):>9.2f}")
    print("    association of component membership with each variable (eta^2 = variance explained; AMI for labels):")
    print(f"      AMI with subtype {adjusted_mutual_info_score(sub, lab):.3f}, with star/artifact"
          f" {adjusted_mutual_info_score(z['dev_y'], lab):.3f}, with image id {adjusted_mutual_info_score(z['dev_group'], lab):.3f}")
    for name, v in (("log true SNR (bright vs faint)", np.log10(z["dev_snr_true"])), ("log source size (compact vs diffuse)", np.log(size)),
                    ("axis ratio (elongation)", z["dev_s_minor"] / z["dev_s_major"]), ("log image background", np.log(z["dev_B"])),
                    ("image PSF sigma (focus)", z["dev_img_sigma"])):
        print(f"      eta^2 {name:<38}{eta2(lab, v):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, c in zip(axes[:2], COVS):
        ax.plot(KS, [full_fit[(K, c)][0].score(S) for K in KS], "s--", label="training LL/obs")
        ax.errorbar(KS, [cv[(K, c)].mean() for K in KS], yerr=[cv[(K, c)].std() for K in KS], fmt="o-", capsize=3,
                    label="grouped-CV held-out LL/obs")
        ax.set_xlabel("K")
        ax.set_title(f"{c} covariance")
        ax.legend()
        ax.grid(alpha=0.3)
    ax = axes[2]
    for c, mk in zip(COVS, ("o-", "s-")):
        ax.plot(KS, [full_fit[(K, c)][0].bic(S) for K in KS], mk, label=f"BIC, {c}")
        ax.plot(KS, [full_fit[(K, c)][0].aic(S) for K in KS], mk.replace("-", ":"), label=f"AIC, {c}")
    ax.set_xlabel("K")
    ax.set_title("Information criteria (lower is better)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_model_selection.png", dpi=130)
    plt.close(fig)
    print("\n  saved", OUT / "exp_c_model_selection.png")


if __name__ == "__main__":
    main()
