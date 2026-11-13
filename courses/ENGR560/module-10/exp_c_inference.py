"""
Experiment C: final inference pipeline and the one-time test evaluation.

The best model from Experiment B (selected on validation loss) is loaded from its checkpoint through
StarInference. Thresholds are chosen on the VALIDATION images only (Policy A: largest threshold with
recall >= 0.95; Policy B: smallest with precision >= 0.90); the old NumPy threshold is not reused.
Reproducibility checks: (1) a fresh StarInference from the checkpoint gives identical outputs;
(2) retraining the selected configuration from scratch with the same seed reproduces the checkpoint.
"""

import json
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data10 import OUT, splits
from exp_b_optimizers import run
from inference import StarInference
from torch_mlp import count_trainable

NUMPY_MLP = {"auc": 0.888, "brier": 0.134, "recall_A": 0.917, "params": 81, "train": "~23 s, selected run",
             "latency": "~0.6-1.7 us/candidate", "threshold_A": 0.40}


def rates(y, yhat):
    tp, fp = np.sum(yhat & (y == 1)), np.sum(yhat & (y == 0))
    fn, tn = np.sum(~yhat & (y == 1)), np.sum(~yhat & (y == 0))
    return {"recall": tp / (tp + fn), "precision": tp / max(tp + fp, 1), "fpr": fp / (fp + tn)}


def main():
    torch.set_num_threads(1)
    S = splits()
    sel = json.load(open(OUT / "selection.json"))
    ck = OUT / f"best_{sel['best']}.pt"
    clf = StarInference(ck)
    print(f"Experiment C: inference pipeline. Checkpoint {ck.name}: {sel['best']} lr {sel['lr']}, epoch {clf.epoch},"
          f" best validation loss {clf.best_val_loss:.4f}, device {clf.device}")

    # ---------------------------------------------------------------- thresholds from validation only
    pv = clf.predict_proba(S["Xva"])
    ts = np.round(np.arange(0.01, 1.0, 0.01), 2)
    tA = max(t for t in ts if rates(S["yva"], pv >= t)["recall"] >= 0.95)
    okB = [t for t in ts if rates(S["yva"], pv >= t)["precision"] >= 0.90 and np.sum(pv >= t) >= 30]
    tB = min(okB) if okB else None
    print(f"  validation-only thresholds: Policy A {tA:.2f} (the NumPy MLP's was {NUMPY_MLP['threshold_A']:.2f}),"
          f" Policy B {tB}")
    print(f"  validation probabilities: mean {pv.mean():.3f}, 10/50/90th percentile"
          f" {np.round(np.percentile(pv, [10, 50, 90]), 3).tolist()} - a different network gives differently scaled"
          " outputs, so the threshold must be re-derived rather than copied")

    # ---------------------------------------------------------------- reproducibility
    clf2 = StarInference(ck)
    same_load = np.max(np.abs(clf2.predict_proba(S["Xte"]) - clf.predict_proba(S["Xte"])))
    model_re, _ = run(sel["best"], sel["lr"], S)
    sd_ck = torch.load(ck, weights_only=True)["model_state_dict"]
    max_param_diff = max(float((model_re.state_dict()[k] - sd_ck[k]).abs().max()) for k in sd_ck)
    print(f"  reproducibility: reloading the checkpoint -> max output difference {same_load:.1e};"
          f" retraining from scratch with the same seed -> max parameter difference {max_param_diff:.1e}")

    # ---------------------------------------------------------------- the single test evaluation
    yte, gte = S["yte"], S["gte"]
    p = clf.predict_proba(S["Xte"])
    auc = roc_auc_score(yte, p)
    brier = float(np.mean((p - yte) ** 2))
    ra = rates(yte, p >= tA)
    rb = rates(yte, p >= tB) if tB is not None else None
    rng = np.random.default_rng(0)
    rows = {i: np.nonzero(gte == i)[0] for i in np.unique(gte)}
    boot = np.array([(roc_auc_score(yte[ix], p[ix]), rates(yte[ix], p[ix] >= tA)["recall"])
                     for ix in (np.concatenate([rows[i] for i in rng.choice(list(rows), len(rows))]) for _ in range(1000))])
    for _ in range(20):
        clf.predict_proba(S["Xte"])
    ts_ = []
    for _ in range(200):
        t0 = time.perf_counter()
        clf.predict_proba(S["Xte"])
        ts_.append(time.perf_counter() - t0)
    lat_med = np.median(ts_) / len(yte) * 1e6
    lat_p95 = np.percentile(ts_, 95) / len(yte) * 1e6
    single = []
    x1 = S["Xte"][:1]
    for _ in range(500):
        t0 = time.perf_counter()
        clf.predict_proba(x1)
        single.append(time.perf_counter() - t0)

    print(f"\n  Untouched test images (30 images, {len(yte)} candidates), evaluated once")
    print(f"    {'metric':<30}{'NumPy MLP (Module 8 folder)':>30}{'PyTorch MLP':>30}")
    print(f"    {'test ROC AUC':<30}{NUMPY_MLP['auc']:>30.3f}"
          f"{f'{auc:.3f} [{np.percentile(boot[:, 0], 2.5):.3f}, {np.percentile(boot[:, 0], 97.5):.3f}]':>30}")
    print(f"    {'Brier score':<30}{NUMPY_MLP['brier']:>30.3f}{brier:>30.3f}")
    recall_cell = f"{ra['recall']:.3f}"
    pa_cell = f"{ra['precision']:.3f} / {ra['fpr']:.3f}"
    pb_cell = "-" if rb is None else f"{rb['precision']:.3f} (recall {rb['recall']:.3f})"
    train_cell = f"{sel['train_time_s']:.1f} s ({sel['steps']} steps)"
    lat_cell = f"{lat_med:.2f} us/cand. (p95 {lat_p95:.2f})"
    print(f"    {'high-recall policy recall':<30}{NUMPY_MLP['recall_A']:>30.3f}{recall_cell:>30}")
    print(f"    {'  precision / FPR at Policy A':<30}{'0.726 / 0.409':>30}{pa_cell:>30}")
    print(f"    {'high-precision policy prec.':<30}{'0.904':>30}{pb_cell:>30}")
    print(f"    {'trainable parameters':<30}{NUMPY_MLP['params']:>30}{count_trainable(clf.model):>30}")
    print(f"    {'training time':<30}{NUMPY_MLP['train']:>30}{train_cell:>30}")
    print(f"    {'inference latency (batch 1499)':<30}{NUMPY_MLP['latency']:>30}{lat_cell:>30}")
    print(f"    single-candidate call latency (fixed PyTorch overhead): median {1e6 * np.median(single):.0f} us")
    print(f"    Policy A recall group-bootstrap 95% interval [{np.percentile(boot[:, 1], 2.5):.3f},"
          f" {np.percentile(boot[:, 1], 97.5):.3f}]; P(recall >= 0.95) = {np.mean(boot[:, 1] >= 0.95):.0%}")
    print(f"    hardware: CPU only ({torch.get_num_threads()} thread), torch {torch.__version__}")


if __name__ == "__main__":
    main()
