"""Star catalogue (reference data). Stored as .npz with the unit vectors themselves so that results do not
depend on re-deriving vectors from RA/Dec (bit-for-bit reproducibility with Module 13)."""

from dataclasses import dataclass

import numpy as np

from .geometry import radec_to_vec


@dataclass
class StarCatalog:
    id: np.ndarray            # integer star IDs
    ra: np.ndarray            # deg
    dec: np.ndarray           # deg
    mag: np.ndarray
    vec: np.ndarray           # N x 3 inertial unit vectors
    name: str = "catalog"

    @classmethod
    def from_arrays(cls, ids, ra, dec, mag, vec=None, name="catalog"):
        ra, dec = np.asarray(ra, float), np.asarray(dec, float)
        return cls(np.asarray(ids), ra, dec, np.asarray(mag, float),
                   radec_to_vec(ra, dec) if vec is None else np.asarray(vec, float), name)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["id"], z["ra"], z["dec"], z["mag"], z["vec"], str(z["name"]) if "name" in z else "catalog")

    def save(self, path):
        np.savez(path, id=self.id, ra=self.ra, dec=self.dec, mag=self.mag, vec=self.vec, name=self.name)

    def subset(self, mask, name=None):
        return StarCatalog(self.id[mask], self.ra[mask], self.dec[mask], self.mag[mask], self.vec[mask],
                           name or self.name + "-subset")

    def __len__(self):
        return len(self.id)
