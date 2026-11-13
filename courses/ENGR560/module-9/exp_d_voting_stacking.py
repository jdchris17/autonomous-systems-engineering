"""
Experiment D: voting, stacking and error diversity.

Base models (three different principles):
  KNN       similarity          engineered features, standardised; k in {15, 45, 101}
  RBF SVM   margin / kernel     engineered features, standardised; C in {1, 10} x gamma in {0.3, 1}
  RF        bagged trees        hybrid features; 200 trees, min_samples_leaf in {1, 5}

`EnsembleProcedure.fit` uses ONLY the groups it is given:
  1. hyperparameters per base model by grouped 3-fold CV (AUC)
  2. grouped 5-fold out-of-fold (OOF) scores for every base model
  3. Platt calibration of each base model on its OOF scores           -> calibrated probabilities
  4. soft-vote weights minimising OOF log loss on a simplex grid       -> weighted soft voting
  5. logistic-regression meta-classifier on OOF logits                 -> stacking
  6. base models refit on all given groups
Hard voting is our own majority vote of the calibrated 0.5 decisions.

Part 4 runs the whole procedure inside each outer fold of a grouped 5-fold CV over the development
images (nested evaluation, as in Module 5). Parts 1-3 and 5 fit it once on all development images
and evaluate once on the 30 untouched test images.
"""

import itertools
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data9 import OUT, SUBTYPES, load
from ensembles import PlattCalibrator, brier, clip_logit, ece, log_loss, majority_vote, pairwise_diversity, soft_vote

warnings.filterwarnings("ignore", message="Mean of empty slice")     # hard voting has no log loss / ECE

BASE = {
    "KNN": ("eng", [{"k": k} for k in (15, 45, 101)]),
    "RBF SVM": ("eng", [{"C": c, "gamma": gm} for c in (1, 10) for gm in (0.3, 1.0)]),
    "random forest": ("hybrid", [{"leaf": lf} for lf in (1, 5)]),
}


class Base:
    def __init__(self, name, params):
        self.name, self.params = name, params
        self.rep = BASE[name][0]

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        p = self.params
        if self.name == "KNN":
            self.m = KNeighborsClassifier(p["k"]).fit(Z, y)
        elif self.name == "RBF SVM":
            self.m = SVC(C=p["C"], gamma=p["gamma"]).fit(Z, y)
        else:
            self.m = RandomForestClassifier(200, min_samples_leaf=p["leaf"], n_jobs=1, random_state=0).fit(X, y)
        return self

    def score(self, X):
        if self.name == "RBF SVM":
            return self.m.decision_function(self.sc.transform(X))
        Z = self.sc.transform(X) if self.name == "KNN" else X
        return clip_logit(self.m.predict_proba(Z)[:, 1], 0.005)


def grouped_folds(y, g, k, seed):
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g))


class EnsembleProcedure:
    def fit(self, R, y, g, seed=0):
        """R: dict representation -> feature matrix (rows = candidates of the given groups)."""
        self.params, oof = {}, {}
        sel_folds = grouped_folds(y, g, 3, seed)
        oof_folds = grouped_folds(y, g, 5, seed + 1)
        for name, (rep, grid) in BASE.items():
            X = R[rep]
            cv = [np.mean([roc_auc_score(y[va], Base(name, p).fit(X[tr], y[tr]).score(X[va])) for tr, va in sel_folds])
                  for p in grid]
            self.params[name] = grid[int(np.argmax(cv))]
            s = np.zeros(len(y))
            for tr, va in oof_folds:
                s[va] = Base(name, self.params[name]).fit(X[tr], y[tr]).score(X[va])
            oof[name] = s
        self.names = list(BASE)
        self.cal = {n: PlattCalibrator().fit(oof[n], y) for n in self.names}
        P = np.array([self.cal[n].transform(oof[n]) for n in self.names])
        grid = [w for w in itertools.product(np.arange(0, 1.01, 0.05), repeat=3) if abs(sum(w) - 1) < 1e-9]
        self.weights = np.array(min(grid, key=lambda w: log_loss(y, np.clip(soft_vote(P, w), 1e-6, 1 - 1e-6))))
        self.meta = LogisticRegression(C=1.0).fit(np.column_stack([clip_logit(p, 1e-4) for p in P]), y)
        self.oof_auc = {n: roc_auc_score(y, oof[n]) for n in self.names}
        self.models = {n: Base(n, self.params[n]).fit(R[BASE[n][0]], y) for n in self.names}
        return self

    def predict_all(self, R):
        P = np.array([self.cal[n].transform(self.models[n].score(R[BASE[n][0]])) for n in self.names])
        out = {n: P[i] for i, n in enumerate(self.names)}
        out["hard vote"] = majority_vote((P >= 0.5).astype(int)).astype(float)
        out["soft vote (equal)"] = soft_vote(P)
        out["soft vote (weighted)"] = soft_vote(P, self.weights)
        out["stacking"] = self.meta.predict_proba(np.column_stack([clip_logit(p, 1e-4) for p in P]))[:, 1]
        return out


def metrics(y, p):
    hard = set(np.unique(p)) <= {0.0, 1.0}
    out = {"auc": roc_auc_score(y, p), "acc": np.mean((p >= 0.5) == y)}
    out["logloss"] = np.nan if hard else log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    out["brier"] = brier(y, p)
    out["ece"] = np.nan if hard else ece(y, p)
    return out


def outer_fold(tr, te, R, y, g, k):
    proc = EnsembleProcedure().fit({r: X[tr] for r, X in R.items()}, y[tr], g[tr], seed=k)
    preds = proc.predict_all({r: X[te] for r, X in R.items()})
    best_name = max(proc.oof_auc, key=proc.oof_auc.get)        # "best individual", chosen inside the fold
    res = {n: metrics(y[te], p) for n, p in preds.items()}
    res["best individual (chosen in fold)"] = metrics(y[te], preds[best_name])
    return res, proc.params, proc.weights


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    R = {"eng": d["dev"]["eng"], "hybrid": d["dev"]["hybrid"]}
    Rt = {"eng": d["test"]["eng"], "hybrid": d["test"]["hybrid"]}
    y, g, yt, gt, subt = d["dev"]["y"], d["dev"]["group"], d["test"]["y"], d["test"]["group"], d["test"]["subtype"]
    print("Experiment D: voting, stacking and error diversity")

    # ---------------------------------------------------------------- Part 4 (nested)
    outer = grouped_folds(y, g, 5, 100)
    res = Parallel(n_jobs=5)(delayed(outer_fold)(tr, te, R, y, g, k) for k, (tr, te) in enumerate(outer))
    print("\n  Part 4 - nested grouped evaluation: the WHOLE procedure (selection, calibration, weights, meta-model)"
          " refit inside each of 5 outer folds")
    for k, (_, params, w) in enumerate(res):
        print(f"    fold {k}: params {params}; soft-vote weights (KNN, SVM, RF) {np.round(w, 2)}")
    names = list(res[0][0])
    print(f"    {'method':<34}{'outer AUC':>16}{'accuracy':>10}{'log loss':>10}{'ECE':>8}")
    for n in names:
        a = np.array([r[0][n]["auc"] for r in res])
        print(f"    {n:<34}{a.mean():>10.3f}+-{a.std():.3f}{np.mean([r[0][n]['acc'] for r in res]):>10.3f}"
              f"{np.nanmean([r[0][n]['logloss'] for r in res]):>10.3f}{np.nanmean([r[0][n]['ece'] for r in res]):>8.3f}")

    # ---------------------------------------------------------------- final fit
    proc = EnsembleProcedure().fit(R, y, g, seed=0)
    preds = proc.predict_all(Rt)
    print(f"\n  Final procedure fit on all 60 development images: params {proc.params}")
    print(f"    OOF AUC on development: " + ", ".join(f"{n} {v:.3f}" for n, v in proc.oof_auc.items()))
    print(f"    soft-vote weights (KNN, SVM, RF): {np.round(proc.weights, 2)};  meta-classifier coefficients on logits:"
          f" {np.round(proc.meta.coef_.ravel(), 3)}, intercept {proc.meta.intercept_[0]:.3f}")

    # ---------------------------------------------------------------- Parts 1, 2, 5
    best_name = max(proc.oof_auc, key=proc.oof_auc.get)
    print(f"\n  Parts 1, 2, 5 - untouched test images (30 images, {len(yt)} candidates). Best individual by"
          f" development OOF AUC: {best_name}")
    print(f"    {'method':<26}{'AUC':>7}{'accuracy':>10}{'log loss':>10}{'Brier':>8}{'ECE':>8}{'AUC 95% group-bootstrap CI':>30}"
          f"{'AUC - best indiv. (CI)':>28}")
    rng = np.random.default_rng(0)
    ids = np.unique(gt)
    rows = {i: np.nonzero(gt == i)[0] for i in ids}
    boots = [np.concatenate([rows[i] for i in rng.choice(ids, ids.size)]) for _ in range(1000)]
    for n, p in preds.items():
        m = metrics(yt, p)
        b = np.array([roc_auc_score(yt[ix], p[ix]) for ix in boots])
        bd = np.array([roc_auc_score(yt[ix], p[ix]) - roc_auc_score(yt[ix], preds[best_name][ix]) for ix in boots])
        ll = "-" if np.isnan(m["logloss"]) else f"{m['logloss']:.3f}"
        ec = "-" if np.isnan(m["ece"]) else f"{m['ece']:.3f}"
        print(f"    {n:<26}{m['auc']:>7.3f}{m['acc']:>10.3f}{ll:>10}{m['brier']:>8.3f}{ec:>8}"
              f"{f'[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]':>30}"
              f"{f'{bd.mean():+.3f} [{np.percentile(bd, 2.5):+.3f}, {np.percentile(bd, 97.5):+.3f}]':>28}")
    print("    (hard voting outputs 0/1, so its AUC is that of a single operating point and it has no log loss / ECE)")

    # ---------------------------------------------------------------- Part 3
    hard = {n: (preds[n] >= 0.5).astype(int) for n in proc.names}
    print("\n  Part 3 - pairwise diversity on the test images (calibrated 0.5 decisions)")
    print(f"    {'pair':<28}{'disagree':>10}{'error corr':>12}{'both wrong':>12}{'A right, B wrong':>18}{'A wrong, B right':>18}")
    for a, b in itertools.combinations(proc.names, 2):
        dv = pairwise_diversity(hard[a], hard[b], yt)
        print(f"    {a + ' / ' + b:<28}{dv['disagreement']:>10.3f}{dv['error_corr']:>12.3f}{dv['both_wrong']:>12.3f}"
              f"{dv['a_right_b_wrong']:>18.3f}{dv['a_wrong_b_right']:>18.3f}")
    print("    error rate by subtype (and how often ALL three are wrong / exactly one is right):")
    print(f"    {'subtype':<14}" + "".join(f"{n:>15}" for n in proc.names) + f"{'all 3 wrong':>13}{'only 1 right':>14}{'hard vote':>11}{'stacking':>10}")
    E = np.array([hard[n] != yt for n in proc.names])
    for s in SUBTYPES:
        m = subt == s
        print(f"    {s:<14}" + "".join(f"{E[i, m].mean():>15.3f}" for i in range(3))
              + f"{np.mean(E[:, m].all(0)):>13.3f}{np.mean(E[:, m].sum(0) == 2):>14.3f}"
              f"{np.mean((preds['hard vote'][m] >= 0.5) != yt[m]):>11.3f}{np.mean((preds['stacking'][m] >= 0.5) != yt[m]):>10.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    from ensembles import reliability
    for n in ("KNN", "RBF SVM", "random forest", "soft vote (weighted)", "stacking"):
        conf, freq, cnt, e = reliability(yt, preds[n])
        ok = cnt >= 10
        ax.plot(conf[ok], freq[ok], "o-", label=f"{n} (ECE {e:.3f})")
    ax.plot([0, 1], [0, 1], "k:")
    ax.set_xlabel("predicted P(star)")
    ax.set_ylabel("observed star fraction")
    ax.set_title("Reliability on the untouched test images")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax = axes[1]
    M = np.array([[pairwise_diversity(hard[a], hard[b], yt)["error_corr"] if a != b else 1 for b in proc.names] for a in proc.names])
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.6 else "k")
    ax.set_xticks(range(3), proc.names)
    ax.set_yticks(range(3), proc.names)
    ax.set_title("Error-indicator correlation (test)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_voting_stacking.png", dpi=130)
    plt.close(fig)
    print("\n  saved", OUT / "exp_d_voting_stacking.png")


if __name__ == "__main__":
    main()
