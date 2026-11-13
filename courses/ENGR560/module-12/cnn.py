"""
Small CNN for 32 x 32 candidate patches, plus its preprocessing (Module 12).

Preprocessing (all fixed or fit on TRAINING images only, stored in the checkpoint):
  1. patch already in units of the image's measured background noise ((pixel - bkg) / rms)
  2. t = asinh(x / a), a = 3        compresses bright cores without destroying faint-source contrast;
                                   monotonic, so relative brightness ordering is kept
  3. (t - mean) / std              one scalar mean / std computed on the training patches
Architecture (about 20k parameters):
  conv 5x5, 1->8, ReLU, maxpool 2   -> 16 x 16
  conv 3x3, 8->16, ReLU, maxpool 2  -> 8 x 8
  conv 3x3, 16->32, ReLU, maxpool 2 -> 4 x 4
  flatten 512 -> linear 32, ReLU -> linear 1 (logit)
"""

import numpy as np
import torch
from torch import nn

ASINH_A = 3.0


class StarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def transform(P, mean=0.0, std=1.0):
    return ((np.arcsinh(P / ASINH_A) - mean) / std).astype(np.float32)


def center_crop(P, size=32):
    o = (P.shape[-1] - size) // 2
    return P[..., o:o + size, o:o + size]


def augment(batch, rng):
    """Physically appropriate for a circular PSF: rot90, flips, +-2 px translation (crop of 36 x 36)."""
    out = np.empty((len(batch), 32, 32), np.float32)
    for n, p in enumerate(batch):
        dy, dx = rng.integers(0, 5, 2)
        q = p[dy:dy + 32, dx:dx + 32]
        q = np.rot90(q, rng.integers(4))
        if rng.random() < 0.5:
            q = q[:, ::-1]
        if rng.random() < 0.5:
            q = q[::-1, :]
        out[n] = q
    return out


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def load(path):
    ck = torch.load(path, weights_only=True)
    m = StarCNN()
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, ck["preprocessing"], ck
