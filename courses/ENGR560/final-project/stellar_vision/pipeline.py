"""
StellarVisionPipeline: image -> calibration -> candidates -> screening -> centroids -> identification ->
attitude -> verification -> AstrometricResult.

    pipeline = StellarVisionPipeline.from_config(VisionConfig(detection_mode="hybrid"))
    result = pipeline.solve(image="night_sky.fits", camera="camera_config.json")

Inputs: a 2-D NumPy array or a FITS path (primary HDU or first image HDU). No truth data, network access or
external service is used. The classical mode never imports torch (machine learning can be disabled without
disabling astrometric solving).
"""

import time
from pathlib import Path

import numpy as np

from . import stages
from .catalog import StarCatalog
from .config import VisionConfig, resolve
from .geometry import CameraModel, pointing_from_attitude, quaternion_from_rotation, tan_wcs
from .result import AstrometricResult, FailureCode
from .solver import TriangleSolver
from .verification import verify


def load_image(image):
    if isinstance(image, (str, Path)):
        from astropy.io import fits
        with fits.open(image) as hdul:
            for h in hdul:
                if h.data is not None and h.data.ndim == 2:
                    return np.asarray(h.data), dict(h.header)
        raise ValueError(f"no 2-D image HDU in {image}")
    return np.asarray(image), {}


class StellarVisionPipeline:
    def __init__(self, config: VisionConfig, catalog: StarCatalog | None = None):
        self.config = config
        self.catalog = catalog if catalog is not None else StarCatalog.load(resolve(config.solver.catalog))
        self.screen = (stages.CNNScreen(config.detection.cnn_checkpoint, config.detection.cnn_threshold, config.deterministic)
                       if config.detection_mode == "hybrid" else stages.ClassicalScreen())
        self._solvers = {}

    @classmethod
    def from_config(cls, config: VisionConfig, catalog: StarCatalog | None = None):
        return cls(config, catalog)

    def _solver(self, camera):
        key = (camera, id(self.catalog))
        if key not in self._solvers:
            sc = self.config.solver
            self._solvers[key] = TriangleSolver(self.catalog, camera, sc.search_center_radec_deg, sc.search_radius_deg,
                                                camera.focal_length_tolerance, sc.tolerance_sigma_px, sc.k_det,
                                                sc.mag_limit, sc.max_hypotheses, sc.precheck_min_matches)
        return self._solvers[key]

    def _camera(self, camera):
        if camera is None:
            return self.config.camera
        return camera if isinstance(camera, CameraModel) else CameraModel.from_json(camera)

    # ------------------------------------------------------------------------------------------
    def solve(self, image, camera=None):
        t = {}
        t0 = time.perf_counter()
        try:
            cam = self._camera(camera)
            img, header = load_image(image)
            if img.shape != (cam.height, cam.width):
                raise ValueError(f"image shape {img.shape} does not match camera {cam.height} x {cam.width}")
            cal = stages.calibrate(img, self.config.calibration)
        except (ValueError, OSError, TypeError) as e:
            return self._fail(FailureCode.INVALID_INPUT, str(e), {"timing_s": {}})
        t["calibration"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        cand = stages.detect_candidates(cal, self.config.detection, cam)
        t["detection"] = time.perf_counter() - t1
        t1 = time.perf_counter()
        scr = self.screen(cal, cand)
        t["classification"] = time.perf_counter() - t1
        t1 = time.perf_counter()
        table = stages.fit_centroids(cal, cand, scr, self.config.detection, self.config.calibration, cam)
        t["centroiding"] = time.perf_counter() - t1
        res = self.solve_sources(stages.to_source_list(table), cam, _timing=t, _t0=t0)
        res.n_detected = len(cand.xy)
        res.sources = table.as_dict()
        res.diagnostics.update({"detection_mode": self.config.detection_mode, "n_screen_rejected": int(np.sum(~scr.keep)),
                                "n_quality_rejected": int(np.sum(~table.accepted)), "background_e": cal.background,
                                "background_rms_e": cal.noise_rms, "calibration_version": cal.version,
                                "candidates": {"x": cand.xy[:, 0], "y": cand.xy[:, 1], "kept_by_screen": scr.keep,
                                               "cnn_score": scr.score}})
        if res.success:      # map matched sources back to rows of the full source table
            res.matched_source_index = np.asarray(res.diagnostics["matched_source_rows"])
        return res

    def solve_sources(self, src, camera=None, _timing=None, _t0=None):
        """Solve from a source list (dict with x, y, flux, sigma_px; zero-based pixels). The solver does not
        know - or need to know - which detector produced it."""
        cam = self._camera(camera)
        acc = self.config.acceptance
        t = _timing if _timing is not None else {}
        t0 = _t0 if _t0 is not None else time.perf_counter()
        n_src = len(src["x"])
        diag = {"timing_s": t, "config_fingerprint": self.config.fingerprint(), "catalog": self.catalog.name,
                "camera": cam.name}
        if n_src == 0:
            return self._finish(self._fail(FailureCode.NO_SOURCES, "no sources detected", diag), t, t0)
        if n_src < acc.min_sources:
            return self._finish(self._fail(FailureCode.INSUFFICIENT_SOURCES,
                                           f"{n_src} usable sources (need {acc.min_sources})", diag, n_src), t, t0)
        t1 = time.perf_counter()
        solver = self._solver(cam)
        sol = solver.solve(src["x"], src["y"], order=np.arange(n_src), sigma_px=src.get("sigma_px"))
        diag.update({"n_hypotheses": sol["n_hypotheses"], "n_hypotheses_verified": sol["n_verified"],
                     "catalog_triangles": len(solver.tri_s)})
        best = sol["best"]
        if best is None:
            return self._finish(self._fail(FailureCode.NO_HYPOTHESIS, "no geometrically compatible catalogue triangle in the"
                                           " search region", diag, n_src), t, t0)
        if self.config.solver.polish_iterations > 0:
            sig = np.asarray(src.get("sigma_px", np.full(n_src, self.config.solver.tolerance_sigma_px)), float)
            R, f, pairs, d, n_pred, it = solver.polish(best["R"], best["f"], np.asarray(src["x"], float),
                                                       np.asarray(src["y"], float), 1 / np.maximum(sig, 1e-3) ** 2,
                                                       self.config.solver.polish_iterations)
            if len(pairs) >= 3:
                hyp = set(best["hyp_cat"].tolist())
                cat_idx = solver.idx[pairs[:, 1]]
                best.update({"R": R, "f": f, "det_idx": pairs[:, 0], "cat_idx": cat_idx, "resid_px": d, "n_pred": n_pred,
                             "n_match": len(pairs), "n_verify": int(sum(1 for c in cat_idx if c not in hyp))})
            diag["polish_iterations"] = it
        t["identification"] = time.perf_counter() - t1
        t1 = time.perf_counter()
        pu, pv, _ = cam.project(best["R"], self.catalog.vec[best["cat_idx"]], best["f"])
        best["pred_u"], best["pred_v"] = pu, pv
        passed, code, reason, info = verify(best, sol["hypotheses"], src, cam, acc)
        t["verification"] = time.perf_counter() - t1
        diag.update({"criteria": info["criteria"], "reduced_chi2": info["chi2_red"], "runner_up": info["runner_up"],
                     "n_distinct_hypotheses": info["n_distinct_hypotheses"],
                     "fraction_d2_above_99.9pct": info["frac_d2_over_13.8"], "hypothesis_star_ids":
                         self.catalog.id[best["hyp_cat"]].tolist(), "matched_source_rows": np.asarray(src.get("source_row", np.arange(n_src)))[best["det_idx"]].tolist()})
        ok = passed or not self.config.verify_solution
        ra, dec, roll = pointing_from_attitude(best["R"])
        res = AstrometricResult(
            success=bool(ok), verified=bool(passed and self.config.verify_solution),
            failure_code=None if ok else code, failure_reason=None if ok else reason,
            attitude=best["R"], quaternion=quaternion_from_rotation(best["R"]), boresight_radec_deg=(ra, dec), roll_deg=roll,
            attitude_covariance=info["covariance"], attitude_sigma_arcsec=info["sigma_arcsec"], focal_length_px=best["f"],
            matched_star_ids=self.catalog.id[best["cat_idx"]], matched_source_index=best["det_idx"],
            measured_centroids=np.column_stack([src["x"][best["det_idx"]], src["y"][best["det_idx"]]]),
            residuals_px=info["residuals"], normalized_residuals_sq=info["d2"], n_sources=n_src,
            n_matched=best["n_match"], n_verify=best["n_verify"], reprojection_rms_px=info["rms"],
            catalog_match_fraction=info["frac"], wcs=tan_wcs(best["R"], cam, best["f"]) if ok else None, diagnostics=diag)
        if not ok:          # keep the rejected hypothesis for diagnostics, but never present it as a solution
            res.diagnostics["rejected_hypothesis"] = {"boresight_radec_deg": (ra, dec), "roll_deg": roll,
                                                      "n_matched": best["n_match"], "n_verify": best["n_verify"]}
            res.attitude = res.quaternion = res.boresight_radec_deg = res.roll_deg = None
            res.attitude_covariance = res.attitude_sigma_arcsec = None
        return self._finish(res, t, t0)

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _fail(code, reason, diag, n_src=0):
        return AstrometricResult(success=False, verified=False, failure_code=code, failure_reason=reason, n_sources=n_src,
                                 diagnostics=diag)

    @staticmethod
    def _finish(res, t, t0):
        t["total"] = time.perf_counter() - t0
        res.diagnostics["timing_s"] = t
        return res
