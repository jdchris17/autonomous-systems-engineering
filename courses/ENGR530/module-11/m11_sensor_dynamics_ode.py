"""Engineering challenge: a first-order sensor from its differential equation.
    tau * dy/dt + y = x(t)      ->      H(w) = 1 / (1 + j w tau)

Derivation: for an input x = e^{jwt} the steady-state output has the same form,
y = H e^{jwt}. Then dy/dt = jw H e^{jwt}, so tau (jw H) + H = 1, i.e.
H(w) = 1 / (1 + j w tau). Its magnitude is 1 / sqrt(1 + (w tau)^2) and its phase
is -atan(w tau).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TAU = 0.1
FREQS = [0.1, 1.0, 10.0, 100.0]
DT = 1e-4
SETTLE = 12 * TAU             # let the start-up transient (e^{-t/tau}) die: 6e-6 of its size


def H(freq_hz, tau=TAU):
    return 1.0 / (1.0 + 1j * 2 * np.pi * freq_hz * tau)


def simulate(freq_hz, tau=TAU, dt=DT):
    """Integrate tau y' + y = sin(2 pi f t) from y(0) = 0 with 4th-order Runge-Kutta.

    Runs for the settling time plus two full input periods. Returns (t, x, y).
    """
    T = SETTLE + 2.0 / freq_hz
    n = int(round(T / dt))
    t = np.arange(n + 1) * dt
    w = 2 * np.pi * freq_hz
    y = np.zeros(n + 1)

    def dydt(tt, yy):
        return (np.sin(w * tt) - yy) / tau

    yc = 0.0
    for i in range(n):
        ti = t[i]
        k1 = dydt(ti, yc)
        k2 = dydt(ti + dt / 2, yc + dt / 2 * k1)
        k3 = dydt(ti + dt / 2, yc + dt / 2 * k2)
        k4 = dydt(ti + dt, yc + dt * k3)
        yc += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y[i + 1] = yc
    return t, np.sin(w * t), y


def steady_state_fit(t, y, freq_hz):
    """Least-squares fit y ~ a sin(wt) + b cos(wt) over the last two whole periods.

    Returns amplitude sqrt(a^2 + b^2) and phase atan2(b, a), i.e. y = A sin(wt + phase).
    """
    w = 2 * np.pi * freq_hz
    sel = t >= t[-1] - 2.0 / freq_hz - 1e-12
    A = np.column_stack([np.sin(w * t[sel]), np.cos(w * t[sel])])
    (a, b), *_ = np.linalg.lstsq(A, y[sel], rcond=None)
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


if __name__ == "__main__":
    print(f"tau y' + y = x, tau = {TAU} s   (cutoff 1/(2 pi tau) = {1 / (2 * np.pi * TAU):.3f} Hz)")
    print(f"Input: unit-amplitude sinusoids. Time-domain simulation: RK4, dt = {DT:g} s\n")
    print(f"{'f (Hz)':>7}{'w tau':>9}{'|H| predicted':>15}{'amplitude simulated':>21}{'difference':>12}{'phase pred (deg)':>18}{'phase sim (deg)':>17}")
    sims = {}
    rows = []
    for f in FREQS:
        t, x, y = simulate(f)
        amp, ph = steady_state_fit(t, y, f)
        Hf = H(f)
        sims[f] = (t, x, y)
        rows.append((f, abs(Hf), amp, np.degrees(np.angle(Hf)), np.degrees(ph)))
        print(f"{f:>7g}{2 * np.pi * f * TAU:>9.3f}{abs(Hf):>15.5f}{amp:>21.5f}{abs(amp - abs(Hf)):>12.1e}{np.degrees(np.angle(Hf)):>18.2f}{np.degrees(ph):>17.2f}")

    # Figure 1: time-domain runs.
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, f in zip(axes.flatten(), FREQS):
        t, x, y = sims[f]
        lo = t[-1] - 2.0 / f
        sel = t >= lo
        ax.plot(t[sel], x[sel], color="lightgray", linewidth=2.5, label="input x(t)")
        ax.plot(t[sel], y[sel], color="steelblue", linewidth=1.8, label="simulated y(t)")
        A = abs(H(f))
        ax.axhline(A, color="firebrick", linestyle="--", linewidth=1, label=f"predicted amplitude |H| = {A:.3f}")
        ax.axhline(-A, color="firebrick", linestyle="--", linewidth=1)
        ax.set_title(f"{f:g} Hz input (last two periods, after the transient)")
        ax.set_xlabel("t (s)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle(f"ODE simulation of tau y' + y = x, tau = {TAU} s")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sensor_dynamics_ode.png"), dpi=150)

    # Figure 2: simulated points on the theoretical frequency response.
    ff = np.logspace(-2, 3, 500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].loglog(ff, np.abs(H(ff)), color="steelblue", linewidth=2, label="H(w) = 1/(1 + j w tau)")
    axes[0].loglog([r[0] for r in rows], [r[2] for r in rows], "o", color="firebrick", markersize=9, label="simulated amplitude")
    axes[0].axvline(1 / (2 * np.pi * TAU), color="gray", linestyle=":", label="cutoff 1/(2 pi tau)")
    axes[0].set_title("Magnitude")
    axes[0].set_ylabel("|H|")
    axes[1].semilogx(ff, np.degrees(np.angle(H(ff))), color="steelblue", linewidth=2, label="theory")
    axes[1].semilogx([r[0] for r in rows], [r[4] for r in rows], "o", color="firebrick", markersize=9, label="simulated phase")
    axes[1].axvline(1 / (2 * np.pi * TAU), color="gray", linestyle=":")
    axes[1].set_title("Phase")
    axes[1].set_ylabel("degrees")
    for ax in axes:
        ax.set_xlabel("frequency (Hz)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sensor_dynamics_ode_bode.png"), dpi=150)
    print("\nSaved sensor_dynamics_ode.png and sensor_dynamics_ode_bode.png")

    print(
        """
What this means
---------------
Two completely different routes give the same numbers. One is algebra on the
model: substitute a complex sinusoid into tau y' + y = x and read off
H = 1/(1 + j w tau). The other is brute force: step the differential equation
forward in time, feed it a sinusoid, and measure what comes out. The simulated
steady-state amplitudes match |H| to better than 1e-6 and the phases match
-atan(w tau), so the frequency-domain description IS the physical model.
What it says about this sensor (tau = 0.1 s, cutoff about 1.6 Hz):
  - 0.1 Hz: reproduced almost perfectly (0.998).
  - 1 Hz: already attenuated to 0.85 and lagging by 32 degrees.
  - 10 Hz: only 0.16 of the true amplitude survives.
  - 100 Hz: essentially invisible (0.016).
So the sensor faithfully reports motion below about 1 Hz and strongly smooths
anything faster. The physical modeling (the ODE and its time constant) and the
frequency analysis (the response H) are the same object seen two ways."""
    )
