"""
VERBATIM copy of the Module 12 CNN (source: module-12/cnn.py, sha256 50715bba939313de...).
Copied: ASINH_A, StarCNN, transform, center_crop; plus extract() from module-12/data12.py below.
Do not edit: tests/test_regression.py checks these against the original module.
"""

import numpy as np
from torch import nn

PATCH = 32
PAD = 2

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


def extract(image, xy, bkg, rms, size=PATCH + 2 * PAD):
    """Patches centred on the candidate pixel, background-subtracted and divided by the image's measured
    background RMS (so pixel values are in units of the detector's own noise - no learned parameters;
    relative brightness / SNR information is preserved). Out-of-image pixels are filled with 0 (= bkg)."""
    h = size // 2
    P = np.zeros((len(xy), size, size), np.float32)
    ny, nx = image.shape
    for n, (x, y) in enumerate(xy):
        i, j = int(round(x)), int(round(y))
        y0, y1, x0, x1 = j - h, j - h + size, i - h, i - h + size
        sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, ny), max(x0, 0), min(x1, nx)
        P[n, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = (image[sy0:sy1, sx0:sx1] - bkg) / rms
    return P
