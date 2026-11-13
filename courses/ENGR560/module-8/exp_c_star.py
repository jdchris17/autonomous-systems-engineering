"""
Experiment C: the NumPy network on star classification.

Data: Module 6 grouped patches (via Module 9's data loader): 60 development images, 30 untouched test
images. Features: log width, roundness, log SNR.
Split of the development images (by image): 45 training images, 15 validation images.
  - standardisation fitted on the training images only
  - 3 -> 16 -> 1 network, tanh hidden, full-batch gradient descent, learning rate 0.5
  - up to 40,000 iterations; early stopping when the validation loss has not improved for 3,000
    iterations; the parameters at the best validation iteration are kept
  - 5 initialisation seeds; the seed with the lowest validation loss is kept (a development choice)
  - Policy A threshold (largest with validation recall >= 0.95) and Policy B threshold (smallest with
    validation precision >= 0.90) are chosen on the validation images
The test images are used exactly once, at the end.
Gradient-boosting numbers are those of the finalist frozen in the Module 9 folder (hybrid features);
for a like-for-like check, the same gradient-boosting configuration is also refit here on the same
3 features and the same 45 training images.
"""

import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-9"))

from data9 import load  # noqa: E402

from mlp import MLP, train  # noqa: E402

OUT = HERE / "outputs"
LR, MAX_IT, PATIENCE, EVERY = 0.5, 40_000, 3_000, 10
GB_MODULE9 = {"auc": 0.922, "brier": 0.113, "recall_A": 0.952, "size_kb": 744}


def rates(y, yhat):
    tp, fp = np.sum(yhat & (y == 1)), np.sum(yhat & (y == 0))
    fn, tn = np.sum(~yhat & (y == 1)), np.sum(~yhat & (y == 0))
    return {"recall": tp / (tp + fn), "precision": tp / max(tp + fp, 1), "fpr": fp / (fp + tn)}


def thresholds(y, p):
    ts = np.round(np.arange(0.01, 1.0, 0.01), 2)
    a = max(t for t in ts if rates(y, p >= t)["recall"] >= 0.95)
    ok = [t for t in ts if rates(y, p >= t)["precision"] >= 0.90 and np.sum(p >= t) >= 30]
    b = min(ok) if ok else None
    return a, b


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    X, y, g = d["dev"]["eng"], d["dev"]["y"].astype(float), d["dev"]["group"]
    Xt, yt, gt = d["test"]["eng"], d["test"]["y"].astype(float), d["test"]["group"]
    ids = np.random.default_rng(8).permutation(np.unique(g))
    tr, va = np.isin(g, ids[:45]), np.isin(g, ids[45:])
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Z = (X - mu) / sd
    Zt = (Xt - mu) / sd
    print("Experiment C: NumPy MLP 3 -> 16 -> 1 on the star / artifact features")
    print(f"  development: {tr.sum()} training candidates (45 images), {va.sum()} validation candidates (15 images);"
          f" test: {len(yt)} candidates (30 untouched images)")

    runs = []
    for seed in range(5):
        m = MLP(3, 16, seed=seed)
        t0 = time.perf_counter()
        h, best_it = train(m, Z[tr], y[tr], LR, MAX_IT, Z[va], y[va], patience=PATIENCE, eval_every=EVERY)
        runs.append((min(h["val"]), seed, m, h, best_it, time.perf_counter() - t0))
    print(f"\n  Training (learning rate {LR}, early stopping patience {PATIENCE} iterations):")
    print(f"    {'seed':>5}{'best iter':>11}{'stopped at':>12}{'train loss':>12}{'val loss':>10}{'val AUC':>9}{'fit s':>7}")
    for vl, seed, m, h, bi, t in runs:
        print(f"    {seed:>5}{bi:>11}{h['iter'][-1]:>12}{h['train'][list(h['iter']).index(bi)]:>12.4f}{vl:>10.4f}"
              f"{roc_auc_score(y[va], m.predict_proba(Z[va])):>9.3f}{t:>7.1f}")
    vl, seed, model, h, best_it, fit_t = min(runs, key=lambda r: r[0])
    print(f"    selected on validation loss: seed {seed}, iteration {best_it}")

    pv = model.predict_proba(Z[va])
    tA, tB = thresholds(y[va], pv)
    print(f"  thresholds from the validation images: Policy A (recall >= 0.95) {tA:.2f};"
          f" Policy B (precision >= 0.90) {tB if tB is None else f'{tB:.2f}'}")

    # ---------------------------------------------------------------- the single test evaluation
    p = model.predict_proba(Zt)
    auc = roc_auc_score(yt, p)
    brier = np.mean((p - yt) ** 2)
    ll = np.mean(-(yt * np.log(np.clip(p, 1e-12, 1)) + (1 - yt) * np.log(np.clip(1 - p, 1e-12, 1))))
    ra = rates(yt, p >= tA)
    rb = rates(yt, p >= tB) if tB is not None else None
    rng = np.random.default_rng(0)
    rows = {i: np.nonzero(gt == i)[0] for i in np.unique(gt)}
    boot = []
    for _ in range(1000):
        ix = np.concatenate([rows[i] for i in rng.choice(list(rows), len(rows))])
        boot.append((roc_auc_score(yt[ix], p[ix]), rates(yt[ix], p[ix] >= tA)["recall"]))
    boot = np.array(boot)
    model.cache = None
    size = len(pickle.dumps({"W1": model.W1, "b1": model.b1, "W2": model.W2, "b2": model.b2, "mu": mu, "sd": sd,
                             "threshold_A": tA, "threshold_B": tB}))
    t0 = time.perf_counter()
    for _ in range(200):
        model.predict_proba(Zt)
    lat = (time.perf_counter() - t0) / 200 / len(yt) * 1e6

    gb = GradientBoostingClassifier(n_estimators=252, learning_rate=0.03, max_depth=5, min_samples_leaf=20,
                                    random_state=0).fit(X[tr], y[tr])
    pg = gb.predict_proba(Xt)[:, 1]
    print("\n  Untouched test images (evaluated once)")
    print(f"    {'metric':<34}{'gradient boosting (Module 9 finalist)':>40}{'NumPy neural network':>24}{'GB, same 3 features':>22}")
    print(f"    {'test ROC AUC':<34}{GB_MODULE9['auc']:>40.3f}{auc:>24.3f}{roc_auc_score(yt, pg):>22.3f}")
    print(f"    {'  95% group-bootstrap CI':<34}{'[0.907, 0.936]':>40}{f'[{np.percentile(boot[:, 0], 2.5):.3f}, {np.percentile(boot[:, 0], 97.5):.3f}]':>24}")
    print(f"    {'Brier score':<34}{GB_MODULE9['brier']:>40.3f}{brier:>24.3f}{np.mean((pg - yt) ** 2):>22.3f}")
    print(f"    {'log loss':<34}{0.346:>40.3f}{ll:>24.3f}")
    print(f"    {'high-recall policy (A) recall':<34}{GB_MODULE9['recall_A']:>40.3f}{ra['recall']:>24.3f}")
    pa = f"{ra['precision']:.3f} / {ra['fpr']:.3f}"
    print(f"    {'  precision / FPR at Policy A':<34}{'0.771 / 0.333':>40}{pa:>24}")
    if rb is not None:
        print(f"    {'high-precision policy (B) prec.':<34}{0.874:>40.3f}{rb['precision']:>24.3f}")
    gb_size = f"{GB_MODULE9['size_kb']} KB"
    mlp_size = f"{size / 1024:.1f} KB ({model.n_params()} parameters)"
    print(f"    {'serialized model size':<34}{gb_size:>40}{mlp_size:>24}{f'{len(pickle.dumps(gb)) / 1024:.0f} KB':>22}")
    print(f"    {'prediction latency':<34}{'3.0 us / candidate':>40}{f'{lat:.2f} us / candidate':>24}")
    print(f"    {'training time':<34}{'3.0 s':>40}{f'{fit_t:.1f} s (one seed)':>24}")
    print(f"    Policy A recall group-bootstrap 95% interval [{np.percentile(boot[:, 1], 2.5):.3f},"
          f" {np.percentile(boot[:, 1], 97.5):.3f}], P(recall >= 0.95) = {np.mean(boot[:, 1] >= 0.95):.0%}")
    seeds_auc = [roc_auc_score(yt, r[2].predict_proba(Zt)) for r in runs]
    print(f"    (for information only, not used for any choice: test AUC of all 5 seeds {np.round(seeds_auc, 3).tolist()})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    ax = axes[0]
    ax.plot(h["iter"], h["train"], label="training images")
    ax.plot(h["iter"], h["val"], label="validation images")
    ax.axvline(best_it, color="k", ls=":", label=f"selected iteration {best_it}")
    ax.set_xlabel("gradient-descent iteration")
    ax.set_ylabel("binary cross-entropy")
    ax.set_title(f"Training curve (seed {seed}) with early stopping")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    gw, gr = np.meshgrid(np.linspace(-3, 3, 250), np.linspace(-2.6, 1.3, 250))
    snr_med = np.median(Z[:, 2])
    G = np.column_stack([gw.ravel(), gr.ravel(), np.full(gw.size, snr_med)])
    P = model.predict_proba(G).reshape(gw.shape)
    cs = ax.contourf(gw, gr, P, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.7)
    ax.contour(gw, gr, P, levels=[tA], colors="k", linestyles="--")
    near = np.abs(Zt[:, 2] - snr_med) < 0.3
    ax.scatter(Zt[near & (yt == 0), 0], Zt[near & (yt == 0), 1], s=6, c="tab:red", label="artifact (test)")
    ax.scatter(Zt[near & (yt == 1), 0], Zt[near & (yt == 1), 1], s=6, c="tab:blue", label="star (test)")
    fig.colorbar(cs, ax=ax, label="P(star)")
    ax.set_xlabel("log width (standardised)")
    ax.set_ylabel("roundness (standardised)")
    ax.set_title("Learned boundary at median SNR (dashed: Policy A threshold)")
    ax.legend(fontsize=8)
    ax = axes[2]
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    conf = [p[idx == b].mean() for b in range(10) if np.sum(idx == b) >= 10]
    freq = [yt[idx == b].mean() for b in range(10) if np.sum(idx == b) >= 10]
    ax.plot(conf, freq, "o-", label="NumPy MLP (test)")
    ax.plot([0, 1], [0, 1], "k:")
    ax.set_xlabel("predicted P(star)")
    ax.set_ylabel("observed star fraction")
    ax.set_title("Reliability on the untouched test images")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_star.png", dpi=130)
    plt.close(fig)
    print("\n  saved", OUT / "exp_c_star.png")


if __name__ == "__main__":
    main()
