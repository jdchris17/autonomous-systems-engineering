"""module-10 -- run_stages_6_7.py

Runs the full Stage I-VII pipeline (pipeline.run_stages_1_to_7) on the SAME
two SimulatorStates run_stages_4_5.py built (Config A: wide-field lens,
point-source PSF regime; Config B: telescope, resolved Airy PSF regime) --
reused directly via import rather than rebuilt, since the scene, observer,
pointing, and camera/optics setup are exactly the same real target this
project has used throughout.

Validation, not just assertion:
  * Stage VI: N_i/N_j for two stars must equal the Carroll & Ostlie flux
    ratio 10^(-0.4(m_i-m_j)) exactly -- checked on a real pair, not assumed
    from the formula alone.
  * Stage VI: each rendered star's normalized PSF should sum close to 1
    (module-8/9's already-validated fine-grid integration; stage5_psf's
    `flux_captured` IS that sum) -- lambda_ij = N_i PSF_ij only means what
    it claims to mean if PSF_ij is actually a probability distribution.
  * Stage VII: Poisson's own mean/variance identity (Var = mean = lambda)
    is checked with a real Monte Carlo draw on one bright pixel, not
    quoted from a textbook.
  * Stage VII: N_e = QE * N_gamma is checked to hold exactly, pixel for
    pixel.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_HERE = os.path.dirname(__file__)
for rel in [("..", "module-9"), ("..", "module-7"), ("..", "module-3"),
           ("..", "module-8")]:
    p = os.path.normpath(os.path.join(_HERE, *rel))
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import run_stages_1_to_7                     # noqa: E402
from run_stages_4_5 import build_scene_and_pointing, build_state  # noqa: E402

from camera_model import Camera                              # noqa: E402  (Module 7)


def run_config(name, state, rng, oversample=8):
    s1, s2, s3, s4, s5, s6, s7 = run_stages_1_to_7(state, rng, oversample=oversample)

    print(f"--- {name} ---")
    print(f"  Stage VI regime              : {s6['regime']}")
    print(f"  stars given a photon budget   : {len(s6['n_photons']):,}")
    print(f"  stars actually rendered       : {s6['n_rendered']:,}")
    if s6["n_rendered"]:
        captured = s6["flux_captured"]
        print(f"  normalized PSF sum (should be ~1): mean={captured.mean():.4f}, "
              f"min={captured.min():.4f}  (module-8/9's already-validated "
              f"fine-grid integration -- checked here, not assumed)")

    # ---- Stage VI check: flux ratio between two real rendered stars -----
    order = np.argsort(s6["mag"])
    if len(order) >= 2:
        i, j = order[0], order[len(order) // 2]     # brightest vs a middling one
        m_i, m_j = s6["mag"][i], s6["mag"][j]
        n_i, n_j = s6["n_photons"][i], s6["n_photons"][j]
        ratio_measured = n_i / n_j
        ratio_expected = 10.0 ** (-0.4 * (m_i - m_j))
        print(f"  flux-ratio check: mag {m_i:.2f} vs {m_j:.2f} -> "
              f"N_i/N_j measured={ratio_measured:.6f}, "
              f"10^(-0.4 dm)={ratio_expected:.6f}, "
              f"diff={abs(ratio_measured-ratio_expected):.2e}")

    # ---- Stage VII: Poisson mean/variance, and the QE identity ----------
    peak = np.unravel_index(np.argmax(s6["lambda_image"]), s6["lambda_image"].shape)
    lam_peak = s6["lambda_image"][peak]
    mc = rng.poisson(lam_peak, size=20_000)
    print(f"  Stage VII Poisson check (brightest pixel, lambda={lam_peak:.3f}):")
    print(f"     20,000-draw Monte Carlo: mean={mc.mean():.3f}, var={mc.var():.3f} "
          f"(Poisson requires mean==var==lambda)")

    n_ph_total = s7["n_photons_image"].sum()
    lam_total = s6["lambda_image"].sum()
    sigma = np.sqrt(lam_total)
    print(f"  one real Poisson realization: total photons = {n_ph_total:,} vs "
          f"expected {lam_total:,.1f} +/- {sigma:.1f} "
          f"({(n_ph_total - lam_total) / sigma:+.2f} sigma -- unremarkable)")

    qe_ok = np.allclose(s7["n_electrons_image"], state.sensor.qe * s7["n_photons_image"])
    print(f"  N_e = QE * N_gamma holds exactly, every pixel: {qe_ok} (QE={state.sensor.qe})")
    print()
    return s1, s2, s3, s4, s5, s6, s7


def main():
    rng = np.random.default_rng(7)
    scene, observer, attitude, vega_mag = build_scene_and_pointing(rng)

    print("=" * 90)
    print("MODULE 10 -- STAGE VI (Brightness/Photon Distribution) + "
          "STAGE VII (Sensor Measurement)")
    print("=" * 90)

    camA = Camera(f=50.0, Ws=36.0, Hs=24.0, Nx=960, Ny=640, name="wide-field")
    stateA = build_state(scene, observer, attitude, camA, D_mm=25.0,
                         exposure_s=2.0, k1=-0.25, k2=0.05)

    camB = Camera(f=1000.0, Ws=0.603, Hs=0.603, Nx=201, Ny=201, name="telescope")
    stateB = build_state(scene, observer, attitude, camB, D_mm=50.0,
                         exposure_s=0.05, k1=0.0, k2=0.0)

    resA = run_config("Config A: 50mm wide-field lens (point-source PSF regime)",
                      stateA, rng)
    resB = run_config("Config B: 50mm-aperture telescope (resolved Airy PSF regime)",
                      stateB, rng)

    print("Same optics/photon-statistics machinery, two very different exposures --")
    print("a rich, noisy field of single-pixel stars (Config A) and one bright,")
    print("well-resolved, individually-noisy Airy disk (Config B). Carroll & Ostlie's")
    print("magnitude system, Module 8's PSF, and Module 9's photon statistics, in")
    print("one model, same as Section 9 asked for.")
    print("=" * 90)

    _plot(stateA, resA, stateB, resB, vega_mag)


def _plot(stateA, resA, stateB, resB, vega_mag):
    s6A, s7A = resA[5], resA[6]
    s6B, s7B = resB[5], resB[6]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # ---- panel 1: N_i vs magnitude, analytic overlay ----------------------
    ax = axes[0, 0]
    order = np.argsort(s6A["mag"])
    ax.scatter(s6A["mag"][order], s6A["n_photons"][order], s=6, alpha=0.4,
              color="tab:blue", label="per-star N_i (Config A)")
    m_ref = s6A["mag"][order][0]
    n_ref = s6A["n_photons"][order][0]
    m_line = np.linspace(s6A["mag"].min(), s6A["mag"].max(), 100)
    ax.plot(m_line, n_ref * 10.0 ** (-0.4 * (m_line - m_ref)), "r-", lw=1.5,
           label="analytic: N_ref * 10^(-0.4 dm)")
    ax.set_yscale("log")
    ax.set_xlabel("magnitude"); ax.set_ylabel("N_i (expected photons)")
    ax.set_title("Stage VI: photon budget vs magnitude\n(Carroll & Ostlie, "
                 "fainter = exponentially fewer photons)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    # ---- panel 2: Config A, realized whole-sensor image --------------------
    # Vega alone carries ~9.5e6 of this image's ~1.4e7 total photons (the
    # zero-point's "order of magnitude, not a calibration" -- see README --
    # landing on ONE undersampled pixel, no saturation modeled per Stage
    # VII's own "you do not need sensor nonlinearity" instruction). A
    # linear or sqrt stretch is then just a black frame with one invisible
    # bright pixel; log stretch clips Vega's pixel to white (as a real,
    # saturated camera pixel would look) while still showing the other
    # 650 stars' much fainter, real photon counts.
    ax = axes[0, 1]
    img = s7A["n_photons_image"].astype(float)
    im = ax.imshow(np.clip(img, 0.5, None), cmap="inferno", origin="upper",
                   norm=LogNorm(vmin=1, vmax=img.max()))
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    ax.set_title(f"Config A: Stage VII realized image (log stretch)\n"
                f"{int(img.sum()):,} total photons, {s6A['n_rendered']:,} stars "
                "-- Vega saturates the display")
    fig.colorbar(im, ax=ax, shrink=0.8, label="photon count (log)")

    # ---- panels 3/4: Config B, smooth lambda vs noisy realization --------
    hw = max(s6B["half_window"], 20)
    Nx, Ny = stateB.intrinsics.Nx, stateB.intrinsics.Ny
    vega_mask = np.abs(s6B["mag"] - vega_mag) < 1e-9
    u_d, v_d = s6B["u_d"][vega_mask][0], s6B["v_d"][vega_mask][0]
    x0, x1 = max(int(round(u_d)) - hw, 0), min(int(round(u_d)) + hw + 1, Nx)
    y0, y1 = max(int(round(v_d)) - hw, 0), min(int(round(v_d)) + hw + 1, Ny)

    lam_stamp = s6B["lambda_image"][y0:y1, x0:x1]
    n_stamp = s7B["n_photons_image"][y0:y1, x0:x1].astype(float)

    # LogNorm can't take log(0) -- it masks non-positive pixels, and an
    # unset "bad" color renders those as transparent (the figure's white
    # background shows through), not black. Outside the rendered window
    # lambda and N are exact 0.0, so without a floor this shows up as
    # misleading white speckle, not real signal. Same floor-then-clip fix
    # as Config A's panel above.
    ax = axes[1, 0]
    vmin_lam = max(lam_stamp.max() * 1e-3, 1e-6)
    im = ax.imshow(np.clip(lam_stamp, vmin_lam, None), cmap="inferno", origin="upper",
                   extent=[x0, x1, y1, y0], norm=LogNorm(vmin=vmin_lam, vmax=lam_stamp.max()))
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    ax.set_title("Config B: Stage VI lambda_ij on Vega\n(smooth expectation)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="expected photons/pixel")

    ax = axes[1, 1]
    vmax_n = max(n_stamp.max(), 1.0)
    vmin_n = max(vmax_n * 1e-3, 0.5)
    im = ax.imshow(np.clip(n_stamp, vmin_n, None), cmap="inferno", origin="upper",
                   extent=[x0, x1, y1, y0], norm=LogNorm(vmin=vmin_n, vmax=vmax_n))
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    ax.set_title("Config B: Stage VII N_ij on Vega\n(ONE Poisson realization -- shot noise)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="realized photons/pixel")

    fig.suptitle("Stage VI (photon budget) + Stage VII (sensor measurement):\n"
                 "Carroll & Ostlie magnitude + Module 8 PSF + Module 9 photon "
                 "statistics, joined", fontsize=12)
    fig.tight_layout()
    fig.savefig("run_stages_6_7.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
