"""
Experiment B: mini-batch training and optimizers.

3 -> 16 -> 1 network, batch size 64, BCEWithLogitsLoss, early stopping on validation loss (patience 40
epochs, best parameters restored), max 500 epochs, same image groups as Module 8.
  Configuration A: SGD,  starting learning rate 0.05
  Configuration B: Adam, starting learning rate 0.001
Learning rates are adjusted ONLY on validation results: each optimizer also tries 2x ... 32x larger (doubling until validation loss turns over)
rate, and the one with the lowest validation loss is kept. The selected configurations' best checkpoints
are saved to outputs/ for Experiment C.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data10 import OUT, splits
from torch_mlp import TorchMLP
from trainer import Trainer

GRID = {"SGD": (0.05, 0.1, 0.2, 0.4, 0.8, 1.6), "Adam": (0.001, 0.002, 0.004, 0.008, 0.016, 0.032)}
SEED = 0


def run(opt_name, lr, S, checkpoint=None):
    torch.manual_seed(SEED)
    model = TorchMLP(3, 16)
    opt = torch.optim.SGD(model.parameters(), lr=lr) if opt_name == "SGD" else torch.optim.Adam(model.parameters(), lr=lr)
    Z = lambda X: (X - S["mean"]) / S["std"]  # noqa: E731
    res = Trainer(model, opt).fit(Z(S["Xtr"]), S["ytr"], Z(S["Xva"]), S["yva"], batch_size=64, max_epochs=500,
                                  patience=40, seed=SEED, checkpoint=checkpoint, config={**model.config, "optimizer": opt_name,
                                                                                         "lr": lr, "batch_size": 64},
                                  mean=S["mean"], std=S["std"])
    return model, res


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    S = splits()
    print(f"Experiment B: mini-batch training (torch {torch.__version__}, CPU, 1 thread);"
          f" train {len(S['ytr'])} / validation {len(S['yva'])} candidates; batch size 64")
    print(f"    {'optimizer':<10}{'lr':>7}{'best val loss':>15}{'best epoch':>12}{'steps to best':>15}{'stopped':>9}"
          f"{'total steps':>13}{'elapsed s':>11}")
    results = {}
    for name, lrs in GRID.items():
        for lr in lrs:
            _, res = run(name, lr, S)
            results[(name, lr)] = res
            print(f"    {name:<10}{lr:>7g}{res['best_val_loss']:>15.4f}{res['best_epoch']:>12}{res['best_steps']:>15}"
                  f"{res['stopped_epoch']:>9}{res['total_steps']:>13}{res['elapsed_s']:>11.1f}")
    chosen = {}
    for name, lrs in GRID.items():
        lr = min(lrs, key=lambda l: results[(name, l)]["best_val_loss"])
        chosen[name] = lr
        model, res = run(name, lr, S, checkpoint=OUT / f"best_{name}.pt")
        results[(name, "final")] = res
    print(f"\n  selected on validation loss: SGD lr {chosen['SGD']}, Adam lr {chosen['Adam']}")
    target = max(results[(n, 'final')]["best_val_loss"] for n in GRID) * 1.01
    print(f"  steps and time to first reach a validation loss within 1% of the worse optimizer's best ({target:.4f}):")
    for name in GRID:
        h = results[(name, "final")]["history"]
        i = next(k for k, v in enumerate(h["val_loss"]) if v <= target)
        print(f"    {name:<5} {h['steps'][i]:>6} optimizer steps, epoch {h['epoch'][i]:>4}, {h['elapsed_s'][i]:6.2f} s"
              f"   (best {results[(name, 'final')]['best_val_loss']:.4f} at {results[(name, 'final')]['best_steps']} steps,"
              f" {results[(name, 'final')]['elapsed_s']:.1f} s total incl. patience)")
    best_name = min(GRID, key=lambda n: results[(n, "final")]["best_val_loss"])
    print(f"  best model for Experiment C: {best_name} (lr {chosen[best_name]}),"
          f" checkpoint {OUT / f'best_{best_name}.pt'}")
    json.dump({"best": best_name, "lr": chosen[best_name], "lrs": chosen,
               "train_time_s": results[(best_name, "final")]["elapsed_s"],
               "steps": results[(best_name, "final")]["best_steps"]}, open(OUT / "selection.json", "w"), indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, xkey, xl in ((axes[0], "steps", "optimizer steps"), (axes[1], "elapsed_s", "elapsed time [s]")):
        for name, col in (("SGD", "tab:blue"), ("Adam", "tab:red")):
            h = results[(name, "final")]["history"]
            ax.plot(h[xkey], h["train_loss"], "--", color=col, label=f"{name} (lr {chosen[name]}): training")
            ax.plot(h[xkey], h["val_loss"], "-", color=col, label=f"{name} (lr {chosen[name]}): validation")
            r = results[(name, "final")]
            bx = r["best_steps"] if xkey == "steps" else h["elapsed_s"][h["epoch"].index(r["best_epoch"])]
            ax.plot(bx, r["best_val_loss"], "o", color=col, ms=8)
        ax.set_xlabel(xl)
        ax.set_ylabel("binary cross-entropy")
        ax.set_ylim(0.3, 0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("SGD vs Adam, batch 64 (dots: restored best model)")
    axes[1].set_title("Same curves against wall-clock time")
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_sgd_vs_adam.png", dpi=130)
    plt.close(fig)
    print("  saved", OUT / "exp_b_sgd_vs_adam.png")


if __name__ == "__main__":
    main()
