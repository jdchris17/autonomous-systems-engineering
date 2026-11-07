"""
Module 11, Parts 2 and 4: Bode plot of L(e^{jwh}), margins, and the sensitivity functions.

Prediction before running (delay):
    z^{-m} = e^{-j w m h} = e^{-j w tau} has magnitude 1 and phase -w tau. So delay leaves |L|, and
    hence the crossover frequency w_c, unchanged, and subtracts w_c tau (in rad) from the phase
    margin: PM_delayed = PM_0 - (180/pi) w_c tau.
    The phase lost grows linearly with frequency, so what matters is w_c tau: the delay measured in
    units of the loop's own time scale 1/w_c. A slow loop (w_c ~ 1 rad/s) loses ~23 deg to 0.4 s; a
    loop three times faster loses three times as much, ~69 deg. A faster controller reacts within a
    shorter time, so the same 0.4 s of stale information is a larger fraction of its reaction time.
    Delay also makes the phase fall without bound, so new -180 deg crossings (and gain margins)
    appear where there were none.

Margins:
    gain crossover w_c: |L| = 1 (all crossings found, interpolated in log w); PM = 180 + angle L(w_c)
    phase crossings: unwrapped phase = -180 - 360 k; GM = -20 log10 |L| there (dB). The relevant GM is
    the smallest one. "no crossing found" when the phase never reaches -180 below Nyquist.

Part 4: S = 1/(1+L), T = L/(1+L), S + T = 1. Measurement noise n enters as y = v + n:
    v = -T n (noise reaching the physical speed), u = -C S n (noise reaching the command).
    For white noise of SD sigma, Parseval gives SD(u) = sigma sqrt((1/pi) int_0^pi |CS(e^{j theta})|^2 dtheta),
    which is compared with the simulated value in robustness_sim.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from loop_models import CONTROLLERS, H_STEP, NYQUIST, C_z, L_z, delay_samples

OUT_DIR = Path(__file__).resolve().parent
W = np.logspace(-3, np.log10(NYQUIST * 0.9999), 40000)
DELAYS = (0.0, 0.4)
COLORS = {"baseline": "tab:blue", "faster": "tab:red"}


def freq_response(ctrl, m):
    L = L_z(np.exp(1j * W * H_STEP), ctrl, m)
    mag_db = 20 * np.log10(np.abs(L))
    phase = np.degrees(np.unwrap(np.angle(L)))
    return L, mag_db, phase


def crossings(x, y, level):
    """All w where y crosses level, by linear interpolation in log10(w). Returns list of (w, idx, frac)."""
    s = y - level
    idx = np.nonzero(np.sign(s[:-1]) * np.sign(s[1:]) < 0)[0]
    out = []
    for i in idx:
        f = s[i] / (s[i] - s[i + 1])
        lw = np.log10(x[i]) + f * (np.log10(x[i + 1]) - np.log10(x[i]))
        out.append((10**lw, i, f))
    return out


def interp(y, i, f):
    return y[i] + f * (y[i + 1] - y[i])


def margins(ctrl, m):
    _, mag, ph = freq_response(ctrl, m)
    gc = [(w, interp(ph, i, f)) for w, i, f in crossings(W, mag, 0.0)]
    pcs = []
    k = 0
    while -180 - 360 * k >= ph.min():
        for w, i, f in crossings(W, ph, -180 - 360 * k):
            pcs.append((w, -interp(mag, i, f), -180 - 360 * k))
        k += 1
    pcs.sort()
    return {"gc": [(w, 180 + p) for w, p in gc], "pc": pcs,
            "pm": min((180 + p for _, p in gc), default=None),
            "gm": min((g for _, g, _ in pcs), default=None)}


def margin_report():
    print("Part 2: loop margins (discrete L(z) = C(z) P(z) z^-m, nominal plant)")
    rows = {}
    for name, ctrl in CONTROLLERS.items():
        for tau in DELAYS:
            m = delay_samples(tau)
            mg = margins(ctrl, m)
            rows[(name, tau)] = mg
            gcs = ", ".join(f"{w:.3f} rad/s (PM {pm:.1f} deg)" for w, pm in mg["gc"]) or "no crossing found"
            pcs = ", ".join(f"{w:.3f} rad/s @ {lvl} deg (GM {g:.1f} dB)" for w, g, lvl in mg["pc"][:3]) \
                or "no crossing found"
            if len(mg["pc"]) > 3:
                pcs += f", ... {len(mg['pc']) - 3} more, every {2 * np.pi / 0.4:.1f} rad/s, GM rising " \
                       f"to {mg['pc'][-1][1]:.1f} dB"
            print(f"  {name:<9} Kp={ctrl.Kp:g} Ki={ctrl.Ki:g}  delay {tau:.1f} s (m = {m:2d})")
            print(f"     gain crossover(s):  {gcs}")
            print(f"     phase crossing(s):  {pcs}")
        w_c, pm0 = rows[(name, 0.0)]["gc"][0]
        print(f"     prediction: PM(0.4 s) = {pm0:.1f} - {np.degrees(w_c * 0.4):.1f} = {pm0 - np.degrees(w_c * 0.4):.1f} deg"
              f"   (measured {rows[(name, 0.4)]['pm']:.1f})")
    L_nyq = {n: abs(L_z(-1.0, c)) for n, c in CONTROLLERS.items()}
    print("  With no delay the phase only reaches -180 deg AT Nyquist (z = -1), where |L| = "
          + ", ".join(f"{n} {20 * np.log10(v):.1f} dB" for n, v in L_nyq.items()) + "\n")
    return rows


def plot_bode(path):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 8))
    for name, ctrl in CONTROLLERS.items():
        for tau, ls in zip(DELAYS, ("-", "--")):
            _, mag, ph = freq_response(ctrl, delay_samples(tau))
            label = f"{name} (Kp={ctrl.Kp:g}, Ki={ctrl.Ki:g}), delay {tau:g} s"
            axes[0].semilogx(W, mag, color=COLORS[name], ls=ls, lw=1.5, label=label)
            axes[1].semilogx(W, ph, color=COLORS[name], ls=ls, lw=1.5)
            mg = margins(ctrl, delay_samples(tau))
            for w, pm in mg["gc"][:1]:
                axes[0].plot(w, 0, "o", color=COLORS[name], ms=5)
                axes[1].plot(w, pm - 180, "o", color=COLORS[name], ms=5)
                axes[1].annotate(f"PM {pm:.0f}°", (w, pm - 180), textcoords="offset points",
                                 xytext=(6, 6 if tau == 0 else -14), fontsize=8, color=COLORS[name])
    axes[0].axhline(0, color="k", lw=0.8)
    axes[1].axhline(-180, color="k", lw=0.8)
    axes[1].set_ylim(-720, 0)
    axes[1].set_yticks(np.arange(-720, 1, 90))
    for ax in axes:
        ax.axvline(NYQUIST, color="0.5", lw=0.8, ls=":")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylim(-80, 80)
    axes[0].set_ylabel("20 log10 |L| (dB)")
    axes[1].set_ylabel("phase of L (deg, unwrapped)")
    axes[1].set_xlabel("ω (rad/s)   [Nyquist π/h = 157 rad/s]")
    axes[0].legend(fontsize=9, loc="lower left")
    axes[0].set_title("(a) magnitude: delay leaves it unchanged")
    axes[1].set_title("(b) phase: both controllers share one curve (×3 gains, same zero); delay subtracts ωτ")
    fig.suptitle("Module 11: loop transfer function L(e^{jωh}) = C(z) P(z) z^{-m}, nominal plant")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def sensitivity(ctrl, w=W):
    z = np.exp(1j * w * H_STEP)
    L = L_z(z, ctrl)
    S = 1 / (1 + L)
    return S, L * S, C_z(z, ctrl) * S


def predicted_command_sd(ctrl, sigma):
    theta = np.linspace(1e-6, np.pi, 200001)
    _, _, CS = sensitivity(ctrl, theta / H_STEP)
    return sigma * np.sqrt(np.trapezoid(np.abs(CS)**2, theta) / np.pi)


def plot_sensitivity(path):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
    print("Part 4: |T| and |CS| (nominal plant, no delay)")
    for name, ctrl in CONTROLLERS.items():
        S, T, CS = sensitivity(ctrl)
        assert np.allclose(S + T, 1.0)
        bw = W[np.nonzero(np.abs(T) < 1 / np.sqrt(2))[0][0]]
        print(f"  {name:<9} peak |T| = {np.abs(T).max():.3f} ({20 * np.log10(np.abs(T).max()):+.2f} dB),"
              f" |T| bandwidth (-3 dB) = {bw:.2f} rad/s,  |CS| at high frequency -> {np.abs(CS[-1]):.3f}"
              f" (~Kp = {ctrl.Kp:g})")
        axes[0].semilogx(W, 20 * np.log10(np.abs(T)), color=COLORS[name], lw=1.6,
                         label=f"{name} (Kp={ctrl.Kp:g}, Ki={ctrl.Ki:g})")
        axes[1].semilogx(W, 20 * np.log10(np.abs(CS)), color=COLORS[name], lw=1.6)
    for ax in axes:
        ax.axvline(NYQUIST, color="0.5", lw=0.8, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylim(-50, 5)
    axes[0].set_ylabel("|T| (dB)")
    axes[0].set_title("(a) |T|: measurement noise → physical speed")
    axes[1].set_ylabel("|CS| (dB)")
    axes[1].set_title("(b) |CS|: measurement noise → command (flat at ≈ Kp above crossover)")
    axes[1].set_xlabel("ω (rad/s)")
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("Module 11: noise sensitivity, nominal plant, no delay")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print()


def main():
    margin_report()
    plot_bode(OUT_DIR / "bode.png")
    plot_sensitivity(OUT_DIR / "sensitivity.png")
    print("Figures saved: bode.png, sensitivity.png")


if __name__ == "__main__":
    main()
