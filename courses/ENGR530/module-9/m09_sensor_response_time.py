"""Engineering challenge: first-order sensor  h(t) = (1/tau) exp(-t/tau) u(t).
How does tau set response speed, lag, and smoothing?"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DT = 0.001
DURATION = 10.0
TAUS = [0.01, 0.1, 1.0]
COLORS = ["firebrick", "darkorange", "steelblue"]
F_FAST = 5.0


def sensor_h(tau, t):
    """Sampled impulse response (1/tau) exp(-t/tau) for t >= 0."""
    return np.exp(-t / tau) / tau


def continuous_convolve(x, h, dt):
    """Approximate y(t) = integral x(s) h(t - s) ds with the trapezoid rule.

    dt * conv(x, h) alone overweights the endpoints of the sum, which is a
    visible error when dt is not much smaller than tau.
    """
    n = len(x)
    full = np.convolve(x, h)[:n]
    return dt * (full - 0.5 * x * h[0] - 0.5 * x[0] * h[:n])


if __name__ == "__main__":
    t = np.arange(int(round(DURATION / DT)) + 1) * DT
    step = np.ones_like(t)
    fast = np.sin(2 * np.pi * F_FAST * t)

    step_out, fast_out = {}, {}
    for tau in TAUS:
        h = sensor_h(tau, t)
        step_out[tau] = continuous_convolve(step, h, DT)
        fast_out[tau] = continuous_convolve(fast, h, DT)

    def t_at(tau, level):
        return np.interp(level, step_out[tau], t)

    print("STEP RESPONSE (analytic answer: y(t) = 1 - exp(-t/tau))")
    print(f"{'tau (s)':>8}{'max err vs analytic':>21}{'t at 63.2% (s)':>16}{'10-90% rise (s)':>17}{'2% settle (s)':>15}")
    for tau in TAUS:
        err = np.max(np.abs(step_out[tau] - (1 - np.exp(-t / tau))))
        print(f"{tau:>8g}{err:>21.2e}{t_at(tau, 0.632):>16.4f}{t_at(tau, 0.9) - t_at(tau, 0.1):>17.4f}{t_at(tau, 0.98):>15.4f}")
    print(f"{'theory':>8}{'':>21}{'tau':>16}{'2.197 tau':>17}{'3.912 tau':>15}")

    print(f"\n{F_FAST:g} Hz SINUSOID (steady state, last 2 s)")
    print(f"{'tau (s)':>8}{'amplitude out':>15}{'theory':>9}{'lag (s)':>10}{'theory':>9}")
    w = 2 * np.pi * F_FAST
    steady = t >= DURATION - 2.0
    win = (t >= 8.0) & (t < 8.2)  # one input period; input peaks at t = 8.05
    for tau in TAUS:
        y = fast_out[tau]
        amp = (y[steady].max() - y[steady].min()) / 2
        lag = t[win][np.argmax(y[win])] - 8.05
        print(f"{tau:>8g}{amp:>15.3f}{1 / np.sqrt(1 + (w * tau) ** 2):>9.3f}{lag:>10.4f}{np.arctan(w * tau) / w:>9.4f}")

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2)
    ax_lin = fig.add_subplot(gs[0, 0])
    ax_log = fig.add_subplot(gs[0, 1])
    ax_fast = fig.add_subplot(gs[1, :])
    for tau, c in zip(TAUS, COLORS):
        ax_lin.plot(t, step_out[tau], color=c, linewidth=2, label=f"$\\tau$ = {tau:g} s")
        ax_log.plot(t, step_out[tau], color=c, linewidth=2, label=f"$\\tau$ = {tau:g} s")
        ax_lin.plot(tau, 0.632, "o", color=c)
        ax_log.plot(tau, 0.632, "o", color=c)
    ax_lin.plot(t, step, ":", color="gray", label="step input")
    ax_lin.set_xlim(0, 4)
    ax_lin.set_title("Step response, linear time (dots: 63.2% at t = $\\tau$)")
    ax_log.set_xscale("log")
    ax_log.set_xlim(1e-3, 10)
    ax_log.set_title("Same responses, log time: identical shape, shifted by $\\tau$")
    ax_fast.plot(t, fast, color="lightgray", linewidth=2, label=f"input: {F_FAST:g} Hz sinusoid")
    for tau, c in zip(TAUS, COLORS):
        ax_fast.plot(t, fast_out[tau], color=c, linewidth=2, label=f"$\\tau$ = {tau:g} s")
    ax_fast.set_xlim(0, 1.0)
    ax_fast.set_title("Rapidly varying input through each sensor")
    for ax in (ax_lin, ax_log, ax_fast):
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("First-order sensor dynamics: h(t) = (1/$\\tau$) e$^{-t/\\tau}$ u(t)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sensor_response_time.png"), dpi=150)
    print("\nSaved sensor_response_time.png")

    print(
        """
What this means
---------------
tau is the sensor's memory: h says each instant of the input is spread over
roughly the next tau seconds. Everything else follows from that one number.
  - Response speed: to a step the output reaches 63% at t = tau, 90% at about
    2.3 tau, and settles (2%) at about 4 tau. tau = 0.01 s is essentially
    instant, 0.1 s is visibly slower, 1.0 s takes several seconds to arrive.
  - Lag: the output trails the true value. The slower the sensor, the further
    behind it reads while the quantity is changing.
  - Smoothing: the same averaging that delays the response also blurs fast
    changes. The 5 Hz signal passes nearly intact at tau = 0.01 s (amplitude
    ~0.95), is cut to ~0.30 at 0.1 s, and is flattened to ~0.03 at 1.0 s, with
    the survivor also delayed.
It is one trade-off seen three ways: a slow sensor is a smoother sensor and a
laggier sensor. The impulse response h is the physical time constant."""
    )
