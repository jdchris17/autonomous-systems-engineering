"""
Reproducible inference component for the trained star-candidate network (Module 10, Experiment C).

    clf = StarInference("outputs/best_Adam.pt")          # model + training-set normalisation
    p   = clf.predict_proba(features)                     # (N, 3) engineered features -> (N,) NumPy

Steps inside predict_proba:
  1. apply the SAVED training-image normalisation (mean, std from the checkpoint)
  2. move the input tensor to the model's device
  3. run the model in eval mode under torch.inference_mode() (no gradient tracking)
  4. convert logits to probabilities with a sigmoid
  5. return an ordinary NumPy array
The checkpoint is loaded with weights_only=True: only tensors and plain Python values are read.
"""

import numpy as np
import torch

from torch_mlp import TorchMLP
from trainer import load_checkpoint


class StarInference:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ck = load_checkpoint(checkpoint_path, TorchMLP, self.device)
        self.model.eval()
        self.mean = ck["normalization"]["mean"].to(self.device)
        self.std = ck["normalization"]["std"].to(self.device)
        self.config, self.epoch, self.best_val_loss = ck["config"], ck["epoch"], ck["best_val_loss"]

    def predict_proba(self, features):
        x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=self.device)   # 2. device
        x = (x - self.mean) / self.std                                                      # 1. normalise
        self.model.eval()
        with torch.inference_mode():                                                        # 3. no autograd
            p = torch.sigmoid(self.model(x))                                                # 4. probabilities
        return p.cpu().numpy().astype(np.float64)                                           # 5. NumPy
