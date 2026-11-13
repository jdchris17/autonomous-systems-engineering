"""
Stages 5-6 core: constrained triangle-signature identification + weighted-SVD attitude (Module 13
solver13.TriangleSolver).

Changes from Module 13 (numerically neutral - tests/test_regression.py proves bit-identical results):
  * camera geometry comes from a CameraModel instead of module constants;
  * every refined hypothesis is recorded (rotation, matches, RMS) so that verification can look for
    COMPETING hypotheses; the module-13 acceptance rule is no longer applied here - it moved, with the
    other criteria, into verification.py / AcceptanceConfig.
The solver only sees pixel coordinates, an optional brightness order and per-source sigmas; it does not
know whether they came from classical or CNN-assisted detection.
"""

import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .geometry import radec_to_vec, wahba_svd

MIN_SIDE = np.radians(0.15)
MIN_C = np.radians(0.6)


class TriangleSolver:
    def __init__(self, catalog, camera, center=(150.0, 20.0), search_radius=2.5, f_tol=None, sigma_px=0.2,
                 k_det=14, mag_limit=None, max_hyp=20000, precheck_min=6):
        self.cam = camera
        self.f0 = camera.focal_length_px
        self.f_tol = camera.focal_length_tolerance if f_tol is None else f_tol
        self.sigma_px, self.k_det, self.max_hyp = sigma_px, k_det, max_hyp
        self.precheck_min = precheck_min      # module-13: 3 + max(2, min_verify // 2) = 6 nearby detections
        self.center_vec = radec_to_vec(*center)
        self.search_radius = np.radians(search_radius)
        half_diag = np.arctan(np.hypot(camera.cx + 0.5, camera.cy + 0.5) / self.f0) * (1 + self.f_tol)
        sep_c = np.arccos(np.clip(catalog.vec @ self.center_vec, -1, 1))
        keep = sep_c <= self.search_radius + half_diag
        if mag_limit is not None:
            keep &= catalog.mag <= mag_limit
        self.idx = np.nonzero(keep)[0]
        self.vec = catalog.vec[self.idx]
        t0 = time.perf_counter()
        S = np.arccos(np.clip(self.vec @ self.vec.T, -1, 1))
        n, max_side = len(self.idx), 2 * half_diag
        tri, sides = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if not (MIN_SIDE <= S[i, j] <= max_side):
                    continue
                k = np.arange(j + 1, n)
                ok = (S[i, k] <= max_side) & (S[j, k] <= max_side) & (S[i, k] >= MIN_SIDE) & (S[j, k] >= MIN_SIDE)
                for kk in k[ok]:
                    tri.append((i, j, kk))
                    sides.append((S[j, kk], S[i, kk], S[i, j]))
        tri = np.array(tri, dtype=int).reshape(-1, 3)
        sides = np.array(sides, dtype=float).reshape(-1, 3)
        order = np.argsort(sides, axis=1)
        self.tri_v = np.take_along_axis(tri, order, 1)
        self.tri_s = np.take_along_axis(sides, order, 1)
        big = self.tri_s[:, 2] >= MIN_C
        self.tri_v, self.tri_s = self.tri_v[big], self.tri_s[big]
        self.tree = cKDTree(self.tri_s[:, :2] / self.tri_s[:, 2:3]) if len(self.tri_s) else None
        self.build_time = time.perf_counter() - t0

    def _assign(self, R, f, u, v, radius):
        pu, pv, front = self.cam.project(R, self.vec, f)
        inf = self.cam.in_frame(pu, pv, front, margin=-radius)
        cand = np.nonzero(inf)[0]
        if len(cand) == 0 or len(u) == 0:
            return np.zeros((0, 2), int), np.zeros(0), int(np.sum(self.cam.in_frame(pu, pv, front)))
        D = np.hypot(pu[cand, None] - u[None, :], pv[cand, None] - v[None, :])
        big = 1e6
        r, c = linear_sum_assignment(np.where(D <= radius, D, big))
        ok = D[r, c] <= radius
        pairs = np.column_stack([c[ok], cand[r[ok]]])
        return pairs, D[r[ok], c[ok]], int(np.sum(self.cam.in_frame(pu, pv, front)))

    def _refine(self, R, f, u, v, w_pix, iters=4):
        radius = max(6.0, 6 * self.sigma_px)
        for it in range(iters):
            pairs, d, n_pred = self._assign(R, f, u, v, radius)
            if len(pairs) < 3:
                return R, f, pairs, d, n_pred
            di, ci = pairs[:, 0], pairs[:, 1]
            c = self.vec[ci] @ R.T
            ix, iy = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
            ou, ov = u[di] - self.cam.cx, v[di] - self.cam.cy
            if len(pairs) >= 6:
                f = float(np.clip(np.sum(w_pix[di] * (ou * ix + ov * iy)) / np.sum(w_pix[di] * (ix**2 + iy**2)),
                                  self.f0 * (1 - self.f_tol), self.f0 * (1 + self.f_tol)))
            b = self.cam.pixel_to_vec(u[di], v[di], f)
            R = wahba_svd(b, self.vec[ci], w_pix[di])
            radius = max(2.5, 4 * self.sigma_px) if it >= 1 else radius
        pairs, d, n_pred = self._assign(R, f, u, v, radius)
        return R, f, pairs, d, n_pred

    def polish(self, R, f, u, v, w_pix, max_iter=50, tol=1e-12):
        """NEW (capstone): iterate the rotation / focal-length alternation of _refine to convergence at the
        final match radius. Module 13 stopped after 4 alternations, which leaves a residual scale error
        that grows with the initial focal-length error (0.08 % for a 2 % error -> 0.25 px at the edge)."""
        radius = max(2.5, 4 * self.sigma_px)
        it = 0
        for it in range(1, max_iter + 1):
            pairs, d, n_pred = self._assign(R, f, u, v, radius)
            if len(pairs) < 6:
                break
            di, ci = pairs[:, 0], pairs[:, 1]
            c = self.vec[ci] @ R.T
            ix, iy = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
            ou, ov = u[di] - self.cam.cx, v[di] - self.cam.cy
            f_new = float(np.clip(np.sum(w_pix[di] * (ou * ix + ov * iy)) / np.sum(w_pix[di] * (ix**2 + iy**2)),
                                  self.f0 * (1 - self.f_tol), self.f0 * (1 + self.f_tol)))
            R = wahba_svd(self.cam.pixel_to_vec(u[di], v[di], f_new), self.vec[ci], w_pix[di])
            done = abs(f_new - f) <= tol * f
            f = f_new
            if done:
                break
        pairs, d, n_pred = self._assign(R, f, u, v, radius)
        return R, f, pairs, d, n_pred, it

    def solve(self, u, v, order=None, sigma_px=None):
        """Returns a dict with the best hypothesis (or None) and every refined hypothesis.
        Coordinates: zero-based pixels. sigma_px: per-source 1-sigma per-axis pixel uncertainty (weights)."""
        t0 = time.perf_counter()
        u, v = np.asarray(u, float), np.asarray(v, float)
        sig = np.full(len(u), self.sigma_px) if sigma_px is None else np.asarray(sigma_px, float)
        w_pix = 1 / np.maximum(sig, 1e-3) ** 2
        order = np.arange(len(u)) if order is None else np.asarray(order)
        top = order[: self.k_det]
        b_all = self.cam.pixel_to_vec(u, v, self.f0) if len(u) else np.zeros((0, 3))
        sth = 3 * np.sqrt(2) * self.sigma_px / self.f0
        hyps = 0
        best, max_quick = None, 0
        tried_arr, n_tried = np.zeros((256, 3, 3)), 0
        dup_trace = 1 + 2 * np.cos(np.radians(0.5))
        det_tree = cKDTree(np.column_stack([u, v])) if len(u) else None
        pre_r = max(6.0, 6 * self.sigma_px)
        hypotheses = []
        ntop = len(top) if self.tree is not None else 0
        for a_ in range(ntop):
            for b_ in range(a_ + 1, ntop):
                for c_ in range(b_ + 1, ntop):
                    t = np.array([top[a_], top[b_], top[c_]])
                    B = b_all[t]
                    S = np.arccos(np.clip(B @ B.T, -1, 1))
                    sd = np.array([S[1, 2], S[0, 2], S[0, 1]])
                    o = np.argsort(sd)
                    sd, tv = sd[o], t[o]
                    if sd[2] < MIN_C or sd[0] < MIN_SIDE:
                        continue
                    tol_r = 2 * sth / sd[2] + 1e-4
                    tol_c = sth + self.f_tol * sd[2]
                    cand = self.tree.query_ball_point(sd[:2] / sd[2], tol_r, p=np.inf)
                    cand = [k for k in cand if abs(self.tri_s[k, 2] - sd[2]) <= tol_c]
                    perms = [(0, 1, 2)]
                    if sd[1] - sd[0] < 2 * sth + self.f_tol * sd[1]:
                        perms.append((1, 0, 2))
                    if sd[2] - sd[1] < 2 * sth + self.f_tol * sd[2]:
                        perms.append((0, 2, 1))
                    for k in cand:
                        for p in perms:
                            cv = self.tri_v[k][list(p)]
                            hyps += 1
                            if hyps > self.max_hyp:
                                break
                            R = wahba_svd(b_all[tv], self.vec[cv])
                            res = np.arccos(np.clip(np.sum((self.vec[cv] @ R.T) * b_all[tv], 1), -1, 1))
                            if res.max() > 4 * sth + self.f_tol * sd[2]:
                                continue
                            if np.arccos(np.clip(R[2] @ self.center_vec, -1, 1)) > self.search_radius * 1.05:
                                continue
                            if n_tried and np.any(np.einsum("ij,nij->n", R, tried_arr[:n_tried]) > dup_trace):
                                continue
                            pu, pv, front = self.cam.project(R, self.vec, self.f0)
                            vis = self.cam.in_frame(pu, pv, front)
                            dist, _ = det_tree.query(np.column_stack([pu[vis], pv[vis]]), distance_upper_bound=pre_r)
                            nq = int(np.sum(np.isfinite(dist)))
                            max_quick = max(max_quick, nq)
                            if nq < self.precheck_min:
                                continue
                            if n_tried == len(tried_arr):
                                tried_arr = np.concatenate([tried_arr, np.zeros_like(tried_arr)])
                            tried_arr[n_tried] = R
                            n_tried += 1
                            Rr, f, pairs, d, n_pred = self._refine(R, self.f0, u, v, w_pix)
                            hyp_cat = set(cv.tolist())
                            n_ver = sum(1 for _, ci in pairs if ci not in hyp_cat)
                            rms = float(np.sqrt(np.mean(d**2))) if len(d) else np.inf
                            hypotheses.append({"R": Rr, "f": f, "n_match": len(pairs), "n_verify": n_ver, "rms_px": rms,
                                               "n_pred": n_pred})
                            sc = (len(pairs), -rms)
                            if best is None or sc > best["score"]:
                                best = {"score": sc, "R": Rr, "f": f, "pairs": pairs, "resid": d, "n_pred": n_pred,
                                        "n_verify": n_ver, "hyp": (tv, cv)}
        out = {"time": time.perf_counter() - t0, "n_hypotheses": hyps, "n_verified": n_tried, "max_quick": max_quick,
               "hypotheses": hypotheses, "best": None}
        if best is not None:
            out["best"] = {"R": best["R"], "f": best["f"], "n_match": len(best["pairs"]), "n_verify": best["n_verify"],
                           "n_pred": best["n_pred"], "det_idx": best["pairs"][:, 0], "cat_idx": self.idx[best["pairs"][:, 1]],
                           "resid_px": best["resid"], "hyp_det": best["hyp"][0], "hyp_cat": self.idx[best["hyp"][1]],
                           "rms_px": -best["score"][1]}
        return out
