"""
Experiment A: train a CNN on detector-proposed 32 x 32 patches.

Images (independent seeds, split by image): 100 training images (mixed nominal..low-SNR conditions),
30 validation images (same mix), 40 nominal test images. Batch 64, Adam, lr 0.001, BCEWithLogitsLoss,
early stopping on validation loss (patience 10 epochs, best weights restored), float32 throughout.
Augmentation on training patches only: rot90, flips, +-2 px translations.
The operating threshold is chosen on the VALIDATION images: the largest threshold keeping >= 98% of
the detector-proposed true stars (the Experiment C objective), then frozen.
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from cnn import StarCNN, augment, center_crop, count_params, transform
from data12 import OUT, SPLITS, SUBTYPES, build

SEED, MAX_EPOCHS, PATIENCE, BATCH = 0, 60, 10, 64


def scores(model, X, mean, std, batch=1024):
    model.eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(X), batch):
            out.append(model(torch.from_numpy(transform(center_crop(X[s:s + batch]), mean, std)[:, None])).numpy())
    return np.concatenate(out)


def recall_threshold(y, p, target=0.98):
    ps = np.sort(p[y == 1])
    k = int(np.floor((1 - target) * len(ps)))
    return float(ps[k])          # largest threshold with >= target of the true stars at or above it


def main():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    tr, va, te = build(SPLITS["train"]), build(SPLITS["val"]), build(SPLITS["test"], "nominal")
    print(f"Experiment A: CNN on detector-proposed patches (data built in {time.perf_counter() - t0:.1f} s)")
    for name, d in (("train (100 images, mixed)", tr), ("validation (30 images, mixed)", va), ("test (40 nominal images)", te)):
        print(f"  {name:<31} {len(d['y']):>5} patches, star fraction {d['y'].mean():.2f}: "
              + ", ".join(f"{s} {np.sum(d['subtype'] == s)}" for s in SUBTYPES))
    mean = float(np.mean(np.arcsinh(center_crop(tr["X"]) / 3.0)))
    std = float(np.std(np.arcsinh(center_crop(tr["X"]) / 3.0)))
    model = StarCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    Xv = torch.from_numpy(transform(center_crop(va["X"]), mean, std)[:, None])
    yv = torch.from_numpy(va["y"].astype(np.float32))
    hist = {"epoch": [], "train": [], "val": []}
    best, best_state, since = np.inf, None, 0
    t_train = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(tr["y"]))
        losses = []
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            xb = torch.from_numpy(transform(augment(tr["X"][idx], rng), mean, std)[:, None])
            yb = torch.from_numpy(tr["y"][idx].astype(np.float32))
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item() * len(idx))
        model.eval()
        with torch.inference_mode():
            vl = loss_fn(model(Xv), yv).item()
        hist["epoch"].append(epoch)
        hist["train"].append(sum(losses) / len(order))
        hist["val"].append(vl)
        if vl < best - 1e-5:
            best, since, best_epoch = vl, 0, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    train_time = time.perf_counter() - t_train
    model.load_state_dict(best_state)
    pv = scores(model, va["X"], mean, std)
    thr = recall_threshold(va["y"], pv, 0.98)
    prob_thr = 1 / (1 + np.exp(-thr))
    torch.save({"model_state_dict": model.state_dict(),
                "preprocessing": {"input": "(pixel - image background) / image background RMS, 32x32 centred on the"
                                           " detector peak pixel", "asinh_a": 3.0, "mean": mean, "std": std,
                                  "patch": 32, "detector": "Module 11 detect(fwhm_assumed=3, k=5)"},
                "threshold_logit": thr, "best_epoch": best_epoch, "best_val_loss": best,
                "train_images": [SPLITS["train"].start, SPLITS["train"].stop]}, OUT / "cnn_best.pt")
    json.dump({"threshold_logit": thr, "threshold_prob": prob_thr, "train_time_s": train_time}, open(OUT / "cnn_meta.json", "w"))
    print(f"\n  {count_params(model)} parameters; trained {hist['epoch'][-1]} epochs in {train_time:.0f} s;"
          f" best epoch {best_epoch}, validation loss {best:.4f}")
    print(f"  validation-selected threshold (>= 98% of proposed true stars kept): P(star) >= {prob_thr:.3f}")

    p = scores(model, te["X"], mean, std)
    yt, st = te["y"], te["subtype"]
    acc = p >= thr
    tp, fp = np.sum(acc & (yt == 1)), np.sum(acc & (yt == 0))
    print(f"\n  Test (40 nominal images, {len(yt)} detector candidates):")
    print(f"    ROC AUC {roc_auc_score(yt, p):.4f}; at the frozen threshold: precision {tp / (tp + fp):.4f},"
          f" recall (of proposed true stars) {tp / yt.sum():.4f}, false-positive rate {fp / np.sum(yt == 0):.4f}")
    print(f"    {'subtype':<16}{'candidates':>11}{'accepted as star':>18}{'median P(star)':>16}")
    for s in SUBTYPES:
        m = st == s
        if m.any():
            print(f"    {s:<16}{m.sum():>11}{np.mean(acc[m]):>18.1%}{np.median(1 / (1 + np.exp(-p[m]))):>16.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hist["epoch"], hist["train"], "o-", ms=3, label="training (augmented batches)")
    ax.plot(hist["epoch"], hist["val"], "s-", ms=3, label="validation images")
    ax.axvline(best_epoch, color="k", ls=":", label=f"best epoch {best_epoch} (restored)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("CNN training (Adam 1e-3, batch 64, early stopping)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_training.png", dpi=130)
    plt.close(fig)
    print("  saved", OUT / "exp_a_training.png", "and", OUT / "cnn_best.pt")


if __name__ == "__main__":
    main()
