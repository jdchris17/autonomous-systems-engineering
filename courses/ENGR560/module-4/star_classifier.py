"""
Reproducible candidate-star classification system (Module 4 deliverable).

    python star_classifier.py        builds the system, prints its card, writes outputs/star_classifier.json

Pipeline (every step deterministic given the seeds in star_data.SPLIT_SEEDS / SHIFT_SEEDS):
  1. features      engineered(): log peak, log width, roundness, log background, log SNR
  2. scaling       Standardizer fitted on the TRAINING split only
  3. candidates    six classifiers x two feature sets, hyperparameters chosen on the VALIDATION
                   split (model_zoo). Feature sets: "all 5", and "invariant 3" = log width,
                   roundness, log SNR (drops log peak and log background, whose absolute levels
                   depend on sky brightness and exposure rather than on what the source is)
  4. calibration   Platt scaling of each candidate's score, fitted on the VALIDATION split
  5. selection     on SELECTION sets that no model or calibrator has seen: a fresh draw under
                   nominal conditions plus one draw per feature-shift scenario (shift-val seeds).
                   robust log loss = 0.5 * nominal log loss + 0.5 * mean log loss over the
                   feature-shift scenarios. Eligible: nominal ECE <= 0.05 and prediction cost
                   <= 1000 ms per 1000 candidates. Lowest robust log loss wins; any candidate within
                   1% of it that is cheaper to evaluate is preferred.
  6. thresholds    Policy A (high recall):    largest threshold with validation recall >= 0.95
                   Policy B (high precision): smallest threshold with validation precision >= 0.90
The final shifted TEST sets (shift-test seeds) are never used by any of these steps.
"""

import json

import numpy as np

from classifiers import OUT, log_loss, rates, reliability, auc
from model_zoo import build_candidates, predict_time, prepared_splits
from star_data import conditions, engineered, generate_candidates, standard_splits

# Scenarios for Experiment F. "prior" changes only P(star); all others change the features.
SCENARIOS = {
    "nominal": conditions(),
    "higher sensor noise": conditions(noise_scale=1.8),
    "optical defocus": conditions(defocus=1.6),
    "different prevalence": conditions(star_fraction=0.05),
    "brighter background": conditions(background_scale=6.0),
    "hot-pixel contamination": conditions(artifact_mix={"hot pixel": 0.60, "diffuse": 0.10,
                                                        "noise peak": 0.20, "compact irregular": 0.10}),
    "unseen artifact type": conditions(artifact_mix={"hot pixel": 0.13, "diffuse": 0.13,
                                                     "noise peak": 0.26, "compact irregular": 0.13,
                                                     "ghost": 0.35}),
}
PRIOR_ONLY = {"different prevalence"}
SHIFT_SEEDS = {"selection": 610, "final test": 620}
FEATURE_SETS = {"all 5": [0, 1, 2, 3, 4], "invariant 3": [1, 2, 4]}
SHIFT_SIZES = {"selection": 3000, "final test": 6000}
RECALL_TARGET, PRECISION_TARGET = 0.95, 0.90
ECE_MAX, COST_MAX_MS, TIE = 0.05, 1000.0, 1.01


def scenario_sets(kind):
    out = {}
    for k, (name, cond) in enumerate(SCENARIOS.items()):
        rng = np.random.default_rng([SHIFT_SEEDS[kind], k])
        out[name] = generate_candidates(SHIFT_SIZES[kind], rng, cond=cond, return_subtypes=True)
    return out


def choose_thresholds(y, p):
    ts = np.round(np.arange(0.01, 1.0, 0.01), 2)
    r = {t: rates(y, (p >= t).astype(int)) for t in ts}
    thr_a = max(t for t in ts if r[t]["recall"] >= RECALL_TARGET)
    ok = [t for t in ts if r[t]["precision"] >= PRECISION_TARGET and r[t]["accepted"] >= 50]
    thr_b = min(ok) if ok else max((t for t in ts if r[t]["accepted"] >= 50),
                                   key=lambda t: r[t]["precision"])
    return {"A": float(thr_a), "B": float(thr_b)}


class StarCandidateClassifier:
    """Selected, calibrated classifier with two operating policies."""

    def __init__(self, standardizer, cols, candidate, thresholds, card):
        self.st, self.cols, self.c = standardizer, cols, candidate
        self.thresholds, self.card = thresholds, card

    def features(self, X_raw):
        return self.st.transform(engineered(X_raw)[:, self.cols])

    def proba(self, X_raw):
        """Calibrated P(star | features), valid for the TRAINING prior P(star) = 0.4."""
        return self.c.proba(self.features(X_raw))

    def proba_for_prior(self, X_raw, prior):
        """Bayes prior correction when the expected star fraction differs from 0.4."""
        p = np.clip(self.proba(X_raw), 1e-12, 1 - 1e-12)
        z = np.log(p / (1 - p)) + np.log(prior / (1 - prior)) - np.log(0.4 / 0.6)
        return 1 / (1 + np.exp(-z))

    def decide(self, X_raw, policy="A"):
        return (self.proba(X_raw) >= self.thresholds[policy]).astype(int)


def selection_table(pipelines, sel_sets):
    """pipelines: {name: (candidate, standardizer, cols)}."""
    rows = {}
    for name, (c, st, cols) in pipelines.items():
        Zs = {k: (st.transform(engineered(v[0])[:, cols]), v[1]) for k, v in sel_sets.items()}
        nom_Z, nom_y = Zs["nominal"]
        p = c.proba(nom_Z)
        shift_ll = [log_loss(yy, c.proba(ZZ)) for k, (ZZ, yy) in Zs.items()
                    if k != "nominal" and k not in PRIOR_ONLY]
        shift_auc = [auc(yy, c.score(ZZ)) for k, (ZZ, yy) in Zs.items()
                     if k != "nominal" and k not in PRIOR_ONLY]
        nom_ll = log_loss(nom_y, p)
        rows[name] = {
            "nominal_auc": auc(nom_y, c.score(nom_Z)),
            "nominal_logloss": nom_ll,
            "nominal_ece": reliability(nom_y, p)[3],
            "shift_logloss_mean": float(np.mean(shift_ll)),
            "shift_auc_mean": float(np.mean(shift_auc)),
            "shift_auc_worst": float(np.min(shift_auc)),
            "robust_logloss": 0.5 * nom_ll + 0.5 * float(np.mean(shift_ll)),
            "cost_ms_per_1000": 1000.0 * predict_time(c, nom_Z),
        }
    eligible = [n for n, r in rows.items() if r["nominal_ece"] <= ECE_MAX and r["cost_ms_per_1000"] <= COST_MAX_MS]
    best = min(eligible, key=lambda n: rows[n]["robust_logloss"])
    near = [n for n in eligible if rows[n]["robust_logloss"] <= TIE * rows[best]["robust_logloss"]]
    chosen = min(near, key=lambda n: rows[n]["cost_ms_per_1000"])
    return rows, chosen, eligible


def build_system(verbose=True):
    sp = standard_splits()
    y = {k: v[1] for k, v in sp.items()}
    pipelines, prepared = {}, {}
    for fs, cols in FEATURE_SETS.items():
        st, Z = prepared_splits(sp, cols)
        if verbose:
            print(f"  feature set '{fs}':")
        cands = build_candidates(Z["train"], y["train"], Z["val"], y["val"], verbose=verbose)
        prepared[fs] = (st, Z, cands)
        for m, c in cands.items():
            pipelines[f"{m} [{fs}]"] = (c, st, cols)
    rows, chosen, eligible = selection_table(pipelines, scenario_sets("selection"))
    c, st, cols = pipelines[chosen]
    Z = prepared[[fs for fs, cl in FEATURE_SETS.items() if cl == cols][0]][1]
    thresholds = choose_thresholds(y["val"], c.proba(Z["val"]))
    names = ["log peak", "log width", "roundness", "log background", "log SNR"]
    card = {
        "selected_model": chosen,
        "hyperparameters": c.params,
        "features": [names[i] for i in cols],
        "standardizer_mean": st.mu.tolist(),
        "standardizer_std": st.sd.tolist(),
        "calibration": "Platt scaling on the validation split",
        "training_prior": 0.4,
        "thresholds": thresholds,
        "policies": {"A": f"high recall: val recall >= {RECALL_TARGET}",
                     "B": f"high precision: val precision >= {PRECISION_TARGET}"},
        "selection_rule": "min 0.5*nominal + 0.5*mean feature-shift log loss on unseen selection sets;"
                          f" nominal ECE <= {ECE_MAX}; cost <= {COST_MAX_MS} ms/1000; within"
                          f" {TIE - 1:.0%} prefer cheaper",
        "selection_table": rows,
        "eligible": eligible,
        "seeds": {"splits": {"train": 401, "val": 402, "test": 403}, "shift": SHIFT_SEEDS},
    }
    return StarCandidateClassifier(st, cols, c, thresholds, card), prepared, y


def main():
    OUT.mkdir(exist_ok=True)
    clf, *_ = build_system()
    print(json.dumps({k: v for k, v in clf.card.items() if k != "selection_table"}, indent=2))
    with open(OUT / "star_classifier.json", "w") as f:
        json.dump(clf.card, f, indent=2)
    # reproducibility check: a second build gives identical probabilities
    clf2, *_ = build_system(verbose=False)
    X, _ = generate_candidates(1000, np.random.default_rng(1))
    print(f"rebuild max |p1 - p2| = {np.max(np.abs(clf.proba(X) - clf2.proba(X))):.2e}")


if __name__ == "__main__":
    main()
