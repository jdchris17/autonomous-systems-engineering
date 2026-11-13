"""
Module 10 data: the same star-candidate features and image-group split as Module 8's Experiment C.
  45 training images / 15 validation images (development, same seed as Module 8) / 30 untouched test images
  standardisation statistics from the TRAINING images only
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-9"))

from data9 import load  # noqa: E402

OUT = HERE / "outputs"


def splits():
    d = load()
    X, y, g = d["dev"]["eng"], d["dev"]["y"].astype(np.float64), d["dev"]["group"]
    ids = np.random.default_rng(8).permutation(np.unique(g))          # identical to Module 8
    tr, va = np.isin(g, ids[:45]), np.isin(g, ids[45:])
    mean, std = X[tr].mean(0), X[tr].std(0)
    return {"Xtr": X[tr], "ytr": y[tr], "Xva": X[va], "yva": y[va],
            "Xte": d["test"]["eng"], "yte": d["test"]["y"].astype(np.float64), "gte": d["test"]["group"],
            "mean": mean, "std": std}
