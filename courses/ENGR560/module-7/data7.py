"""
Module 7 data access: Module 6's grouped patch datasets and representations, computed once and
cached on disk (outputs/cache_*.npz) so every experiment uses identical numbers.

  engineered   log width, roundness, log SNR measured from the pixels (Module 6)
  PCA-B(k)     PCA scores of background-subtracted pixels       (Module 6 approach B)
  PCA-C(k)     PCA scores of flux-normalised pixels             (Module 6 approach C)
PCA is fit on DEVELOPMENT images only; standardisation statistics come from development data only.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-6"))

from exp_c_reconstruction import preprocess  # noqa: E402
from pca_lda import PCAFromScratch  # noqa: E402
from patches import (SUBTYPES, development_set, engineered_features, measure,  # noqa: E402,F401
                     untouched_test_set)

OUT = HERE / "outputs"
ENG_NAMES = ("log width", "roundness", "log SNR")
K_MAX = 16


def _build():
    OUT.mkdir(exist_ok=True)
    d, t = development_set(), untouched_test_set()
    out = {}
    for tag, ds in (("dev", d), ("test", t)):
        out[f"{tag}_eng"] = engineered_features(ds["X"])
        for key in ("y", "group", "snr_true", "B", "img_sigma", "x0", "y0", "flux", "s_major", "s_minor"):
            out[f"{tag}_{key}"] = ds[key]
        out[f"{tag}_subtype"] = ds["subtype"].astype(str)
        out[f"{tag}_X"] = ds["X"].astype(np.float32)
    for ap in ("B", "C"):
        Fd, _ = preprocess(d["X"], ap)
        Ft, _ = preprocess(t["X"], ap)
        pca = PCAFromScratch(K_MAX).fit(Fd)
        out[f"dev_pca{ap}"] = pca.transform(Fd)
        out[f"test_pca{ap}"] = pca.transform(Ft)
        out[f"evr{ap}"] = pca.explained_variance_ratio_
    np.savez_compressed(OUT / "cache_module6_data.npz", **out)


def load():
    f = OUT / "cache_module6_data.npz"
    if not f.exists():
        _build()
    z = np.load(f, allow_pickle=False)
    return {k: z[k] for k in z.files}


def standardize(dev, test=None):
    mu, sd = dev.mean(axis=0), dev.std(axis=0)
    sd[sd == 0] = 1
    return ((dev - mu) / sd, None if test is None else (test - mu) / sd)
