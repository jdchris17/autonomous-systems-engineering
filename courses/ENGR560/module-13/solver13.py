"""
Constrained star-pattern solver (Experiment B): triangle signatures -> hypotheses -> verified attitude.

Inputs: detected pixel coordinates only (optionally a brightness ordering), APPROXIMATE intrinsics
(focal length f_approx, principal point), a bounded sky-search region (centre + radius for the boresight)
and an assumed centroid uncertainty. No catalogue IDs, no roll.

1. Catalogue side (precomputed once): stars that can fall in the field for any allowed boresight;
   every triangle whose sides are all <= the field diagonal and >= MIN_SIDE; signature
   (a/c, b/c) with a <= b <= c (scale-free, so a focal-length error does not move it) plus c itself;
   2-D KD-tree on the ratios.
2. Image side: triangles from the brightest K detections; tolerances from the centroid uncertainty
   (3 sigma, both ends of each side) and a focal-length (scale) uncertainty on c.
3. Each compatible pair gives 3 correspondences (vertices identified by their opposite sides; near-
   isosceles cases try the ambiguous orderings) -> weighted SVD -> quick 3-star residual check
   (rejects mirror images) -> boresight must lie in the search region.
4. Verification uses the WHOLE catalogue subset: project with the hypothesis, one-to-one nearest-
   neighbour match to all detections, re-fit R (weighted SVD) and f (linear LSQ) on the inliers, repeat.
   Only matches beyond the 3 hypothesis stars count as verification.
5. Every hypothesis is verified (near-duplicate rotations skipped); the best (most inliers, then lowest
   RMS) is accepted only if it has >= MIN_VERIFY extra matches AND matches >= MIN_FRAC of the catalogue
   stars it predicts in the frame. Otherwise the solver reports "no solution".
"""

import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import sky13 as s

MIN_SIDE = np.radians(0.15)
MIN_C = np.radians(0.6)


class TriangleSolver:
    def __init__(self, cat, center=s.CENTER, search_radius=2.5, f_approx=s.F_TRUE, f_tol=0.03,
                 sigma_px=0.2, k_det=14, min_verify=6, min_frac=0.5, mag_limit=None):
        self.f0, self.f_tol, self.sigma_px, self.k_det = f_approx, f_tol, sigma_px, k_det
        self.min_verify, self.min_frac = min_verify, min_frac
        self.center_vec = s.radec_to_vec(*center)
        self.search_radius = np.radians(search_radius)
        half_diag = np.arctan(np.hypot(s.CX + 0.5, s.CY + 0.5) / f_approx) * (1 + f_tol)
        sep_c = np.arccos(np.clip(cat["vec"] @ self.center_vec, -1, 1))
        keep = sep_c <= self.search_radius + half_diag
        if mag_limit is not None:
            keep &= cat["mag"] <= mag_limit
        self.idx = np.nonzero(keep)[0]
        self.vec = cat["vec"][self.idx]
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
                    sides.append((S[j, kk], S[i, kk], S[i, j]))        # opposite vertex i, j, k
        tri, sides = np.array(tri), np.array(sides)
        order = np.argsort(sides, axis=1)                               # a <= b <= c
        self.tri_v = np.take_along_axis(tri, order, 1)                  # vertex opposite a, b, c
        self.tri_s = np.take_along_axis(sides, order, 1)
        big = self.tri_s[:, 2] >= MIN_C
        self.tri_v, self.tri_s = self.tri_v[big], self.tri_s[big]
        self.tree = cKDTree(self.tri_s[:, :2] / self.tri_s[:, 2:3])
        self.build_time = time.perf_counter() - t0

    # --- helpers -------------------------------------------------------------------------------
    def _assign(self, R, f, u, v, radius):
        pu, pv, front = s.project(R, self.vec, f)
        inf = s.in_frame(pu, pv, front, margin=-radius)
        cand = np.nonzero(inf)[0]
        if len(cand) == 0 or len(u) == 0:
            return np.zeros((0, 2), int), np.zeros(0), int(np.sum(s.in_frame(pu, pv, front)))
        D = np.hypot(pu[cand, None] - u[None, :], pv[cand, None] - v[None, :])
        big = 1e6
        r, c = linear_sum_assignment(np.where(D <= radius, D, big))
        ok = D[r, c] <= radius
        pairs = np.column_stack([c[ok], cand[r[ok]]])                   # (detection, catalogue-subset)
        return pairs, D[r[ok], c[ok]], int(np.sum(s.in_frame(pu, pv, front)))

    def _refine(self, R, f, u, v, w_pix, iters=4):
        radius = max(6.0, 6 * self.sigma_px)
        for it in range(iters):
            pairs, d, n_pred = self._assign(R, f, u, v, radius)
            if len(pairs) < 3:
                return R, f, pairs, d, n_pred
            di, ci = pairs[:, 0], pairs[:, 1]
            c = self.vec[ci] @ R.T
            ix, iy = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
            ou, ov = u[di] - s.CX, v[di] - s.CY
            if len(pairs) >= 6:
                f = float(np.clip(np.sum(w_pix[di] * (ou * ix + ov * iy)) / np.sum(w_pix[di] * (ix**2 + iy**2)),
                                  self.f0 * (1 - self.f_tol), self.f0 * (1 + self.f_tol)))
            b = s.pixel_to_vec(u[di], v[di], f)
            R = s.wahba_svd(b, self.vec[ci], w_pix[di])
            radius = max(2.5, 4 * self.sigma_px) if it >= 1 else radius
        pairs, d, n_pred = self._assign(R, f, u, v, radius)
        return R, f, pairs, d, n_pred

    # --- main ----------------------------------------------------------------------------------
    def solve(self, u, v, order=None, sigma_px=None, max_hyp=20000):
        t0 = time.perf_counter()
        u, v = np.asarray(u, float), np.asarray(v, float)
        sig = np.full(len(u), self.sigma_px) if sigma_px is None else np.asarray(sigma_px, float)
        w_pix = 1 / np.maximum(sig, 1e-3) ** 2
        order = np.arange(len(u)) if order is None else np.asarray(order)
        top = order[: self.k_det]
        b_all = s.pixel_to_vec(u, v, self.f0)
        sth = 3 * np.sqrt(2) * self.sigma_px / self.f0              # 3-sigma side tolerance (rad)
        hyps = 0
        best, max_quick = None, 0
        tried_arr, n_tried = np.zeros((256, 3, 3)), 0
        dup_trace = 1 + 2 * np.cos(np.radians(0.5))                  # tr(R1 R2^T) above this: < 0.5 deg apart
        det_tree = cKDTree(np.column_stack([u, v]))
        pre_r = max(6.0, 6 * self.sigma_px)
        ntop = len(top)
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
                            if hyps > max_hyp:
                                break
                            R = s.wahba_svd(b_all[tv], self.vec[cv])
                            res = np.arccos(np.clip(np.sum((self.vec[cv] @ R.T) * b_all[tv], 1), -1, 1))
                            if res.max() > 4 * sth + self.f_tol * sd[2]:
                                continue                                      # mirror / inconsistent
                            if np.arccos(np.clip(R[2] @ self.center_vec, -1, 1)) > self.search_radius * 1.05:
                                continue                                      # outside the search region
                            if n_tried and np.any(np.einsum("ij,nij->n", R, tried_arr[:n_tried]) > dup_trace):
                                continue                                      # near-duplicate (< 0.5 deg)
                            # cheap pre-check: detections near projected catalogue stars (KD-tree count)
                            pu, pv, front = s.project(R, self.vec, self.f0)
                            vis = s.in_frame(pu, pv, front)
                            dist, _ = det_tree.query(np.column_stack([pu[vis], pv[vis]]), distance_upper_bound=pre_r)
                            nq = int(np.sum(np.isfinite(dist)))
                            max_quick = max(max_quick, nq)
                            if nq < 3 + max(2, self.min_verify // 2):
                                continue
                            if n_tried == len(tried_arr):
                                tried_arr = np.concatenate([tried_arr, np.zeros_like(tried_arr)])
                            tried_arr[n_tried] = R
                            n_tried += 1
                            Rr, f, pairs, d, n_pred = self._refine(R, self.f0, u, v, w_pix)
                            hyp_cat = set(cv.tolist())
                            n_ver = sum(1 for _, ci in pairs if ci not in hyp_cat)
                            rms = float(np.sqrt(np.mean(d**2))) if len(d) else np.inf
                            sc = (len(pairs), -rms)
                            if best is None or sc > best["score"]:
                                best = {"score": sc, "R": Rr, "f": f, "pairs": pairs, "resid": d, "n_pred": n_pred,
                                        "n_verify": n_ver, "hyp": (tv, cv)}
        dt = time.perf_counter() - t0
        out = {"time": dt, "n_hypotheses": hyps, "n_verified": n_tried, "max_quick": max_quick}
        if best is None:
            return {**out, "success": False, "n_verify": max(max_quick - 3, 0), "n_match": max_quick,
                    "reason": f"no hypothesis passed the pre-check (best: {max_quick} stars near predictions)"}
        frac = len(best["pairs"]) / max(best["n_pred"], 1)
        ok = best["n_verify"] >= self.min_verify and frac >= self.min_frac
        out.update({"success": ok, "R": best["R"], "f": best["f"], "n_match": len(best["pairs"]),
                    "n_verify": best["n_verify"], "n_pred": best["n_pred"], "match_frac": frac,
                    "det_idx": best["pairs"][:, 0], "cat_idx": self.idx[best["pairs"][:, 1]],
                    "resid_px": best["resid"], "hyp_det": best["hyp"][0], "hyp_cat": self.idx[best["hyp"][1]]})
        if not ok:
            out["reason"] = (f"best hypothesis failed verification: {best['n_verify']} extra matches,"
                             f" {frac:.0%} of predicted stars")
        return out
