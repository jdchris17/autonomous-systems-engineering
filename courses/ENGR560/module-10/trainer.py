"""
Reusable PyTorch training loop (Module 10).

Trainer.fit(...)  mini-batch training with a DataLoader, validation evaluation every epoch, early
                  stopping on validation loss (patience in epochs), best-model restoration, metric
                  logging (per epoch: train loss, validation loss, optimizer steps, elapsed time) and
                  checkpointing of the best model.
save_checkpoint / load_checkpoint
                  checkpoint = {"model_state_dict", "optimizer_state_dict", "epoch", "best_val_loss",
                                "config", "normalization": {"mean", "std"} (training-image statistics),
                                "steps"}. Tensors and plain Python types only, so it loads with
                  torch.load(weights_only=True) - no arbitrary pickled objects are executed.
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch_size, seed):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, config, mean, std, steps):
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "epoch": int(epoch), "best_val_loss": float(best_val_loss), "config": dict(config),
                "normalization": {"mean": torch.as_tensor(mean, dtype=torch.float32),
                                  "std": torch.as_tensor(std, dtype=torch.float32)},
                "steps": int(steps)}, path)


def load_checkpoint(path, model_class, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = ck["config"]
    model = model_class(cfg["input_dim"], cfg["hidden_dim"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    return model, ck


class Trainer:
    def __init__(self, model, optimizer, loss_fn=None, device="cpu"):
        self.model, self.optimizer = model.to(device), optimizer
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()
        self.device = device

    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        return float(self.loss_fn(self.model(Xt), yt))

    def fit(self, X, y, Xv, yv, batch_size=64, max_epochs=500, patience=40, seed=0, checkpoint=None,
            config=None, mean=None, std=None, log_path=None):
        loader = make_loader(X, y, batch_size, seed)
        hist = {"epoch": [], "steps": [], "train_loss": [], "val_loss": [], "elapsed_s": []}
        best, best_state, best_epoch, best_steps, since, steps = np.inf, None, 0, 0, 0, 0
        t0 = time.perf_counter()
        hist["epoch"].append(0)
        hist["steps"].append(0)
        hist["train_loss"].append(self.evaluate(X, y))
        hist["val_loss"].append(self.evaluate(Xv, yv))
        hist["elapsed_s"].append(0.0)
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                steps += 1
            tr, va = self.evaluate(X, y), self.evaluate(Xv, yv)
            for k, v in zip(hist, (epoch, steps, tr, va, time.perf_counter() - t0)):
                hist[k].append(v)
            if va < best - 1e-6:
                best, best_epoch, best_steps, since = va, epoch, steps, 0
                best_state = copy.deepcopy(self.model.state_dict())
                best_opt = copy.deepcopy(self.optimizer.state_dict())
                if checkpoint is not None:
                    save_checkpoint(checkpoint, self.model, self.optimizer, epoch, best, config, mean, std, steps)
            else:
                since += 1
                if since >= patience:
                    break
        self.model.load_state_dict(best_state)                     # restore the best parameters
        self.optimizer.load_state_dict(best_opt)
        result = {"best_val_loss": best, "best_epoch": best_epoch, "best_steps": best_steps,
                  "stopped_epoch": hist["epoch"][-1], "total_steps": steps,
                  "elapsed_s": hist["elapsed_s"][-1], "history": hist}
        if log_path is not None:
            Path(log_path).write_text(json.dumps({k: v for k, v in result.items()}, indent=1))
        return result
