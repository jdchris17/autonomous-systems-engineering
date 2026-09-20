"""Stage 9: calculate the error.

    e[n] = x_hat[n] - x_true[n]

Bias, variance, standard deviation, RMSE and SNR for the raw measurement, the filtered
estimate and the parametric estimate, plus an error BUDGET that says where the remaining
error comes from (possible because the whole pipeline is linear).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import project_signal as P
import stage4_antialias as S4
import stage5_sample as S5
import stage7_filter as S7
import stage8_estimate as S8

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = S5.FS
EDGE = int(S8.EDGE * FS)


def error_stats(e, x_true):
    """Bias, variance, std, RMSE of an error sequence, and SNR = signal power / error power in dB."""
    e = np.asarray(e)
    return {"bias": float(np.mean(e)), "variance": float(np.var(e)), "std": float(np.std(e)),
            "rmse": float(np.sqrt(np.mean(e**2))), "snr": float(10 * np.log10(np.mean(x_true**2) / np.mean(e**2)))}


def through_pipeline(analog_component):
    """Push one ANALOG component through the whole chain: analog filter -> ADC -> digital filter -> equalizer."""
    z_n = S4.analog_filter(analog_component)[:: P.FS_ANALOG // FS]
    return S7.estimate_waveform(z_n)


def print_block(title, s):
    print(title)
    print(f"  Bias:               {s['bias']:>9.4f}")
    print(f"  Variance:           {s['variance']:>9.4f}")
    print(f"  Standard deviation: {s['std']:>9.4f}")
    print(f"  RMSE:               {s['rmse']:>9.4f}")
    print(f"  SNR:                {s['snr']:>9.2f} dB")


if __name__ == "__main__":
    a = S5.acquire()
    z, x_true, t_n, comp = a["z_n"], a["x_true_n"], a["t_n"], a["components"]
    x_hat = S7.estimate_waveform(z)
    par = S8.estimate_parameters(x_hat, method="ls")
    x_par = S8.parametric_waveform(par, t_n)

    e_raw, e_hat, e_par = z - x_true, x_hat - x_true, x_par - x_true
    print(f"e[n] = estimate[n] - x_true[n],   {len(z)} samples;  signal power mean(x_true^2) = {np.mean(x_true ** 2):.2f}\n")
    print("WHOLE RECORD")
    print_block("RAW MEASUREMENT  (z[n] - x_true[n])", error_stats(e_raw, x_true))
    print()
    print_block("FILTERED ESTIMATE  (x_hat[n] - x_true[n])", error_stats(e_hat, x_true))
    print()
    print_block("PARAMETRIC ESTIMATE  (x_p[n] - x_true[n], from the fitted A, f, phase)", error_stats(e_par, x_true))

    inner = slice(EDGE, len(z) - EDGE)
    s_raw_i, s_hat_i, s_par_i = (error_stats(e[inner], x_true[inner]) for e in (e_raw, e_hat, e_par))
    s_raw, s_hat, s_par = (error_stats(e, x_true) for e in (e_raw, e_hat, e_par))
    print("\nAWAY FROM THE RECORD ENDS (excluding 2 s at each end)")
    print(f"  {'':<22}{'bias':>9}{'std':>9}{'RMSE':>9}{'SNR (dB)':>11}")
    for name, s in [("raw measurement", s_raw_i), ("filtered estimate", s_hat_i), ("parametric estimate", s_par_i)]:
        print(f"  {name:<22}{s['bias']:>9.4f}{s['std']:>9.4f}{s['rmse']:>9.4f}{s['snr']:>11.2f}")

    print("\nDID THE PROCESSING HELP?")
    print(f"  filtered estimate : RMSE {s_raw['rmse']:.3f} -> {s_hat['rmse']:.3f}  ({s_raw['rmse'] / s_hat['rmse']:.1f}x smaller), SNR {s_raw['snr']:.1f} -> {s_hat['snr']:.1f} dB (+{s_hat['snr'] - s_raw['snr']:.1f} dB), bias {s_raw['bias']:+.3f} -> {s_hat['bias']:+.4f}")
    print(f"  parametric        : RMSE {s_raw['rmse']:.3f} -> {s_par['rmse']:.3f}  ({s_raw['rmse'] / s_par['rmse']:.0f}x smaller), SNR {s_raw['snr']:.1f} -> {s_par['snr']:.1f} dB (+{s_par['snr'] - s_raw['snr']:.1f} dB)")

    # Error budget: the pipeline is linear, so each component's contribution is separable.
    parts = {
        "distortion of the desired signal": through_pipeline(comp["x_true"]) - x_true,
        "15 Hz vibration (leakage)": through_pipeline(comp["vib"]),
        "80 Hz interference (leakage)": through_pipeline(comp["hf"]),
        "bias + drift (leakage)": through_pipeline(comp["bias"]),
        "noise (in-band)": through_pipeline(comp["noise"]),
    }
    check = float(np.max(np.abs(sum(parts.values()) - e_hat)))
    print(f"\nERROR BUDGET of the filtered estimate (each component pushed through the same linear pipeline; parts sum to e[n] to {check:.0e})")
    print(f"  {'source':<36}{'RMS, whole record':>19}{'RMS, interior':>15}{'share of MSE':>14}")
    total_ms = float(np.mean(e_hat**2))
    for name, e in parts.items():
        print(f"  {name:<36}{np.sqrt(np.mean(e ** 2)):>19.4f}{np.sqrt(np.mean(e[inner] ** 2)):>15.4f}{100 * np.mean(e ** 2) / total_ms:>13.1f}%")
    print(f"  {'total (with cross terms)':<36}{np.sqrt(total_ms):>19.4f}{s_hat_i['rmse']:>15.4f}")
    edges = np.r_[e_hat[:EDGE], e_hat[-EDGE:]]
    print(f"  RMS of the first and last 2 s together: {np.sqrt(np.mean(edges ** 2)):.3f}   vs the middle 16 s: {s_hat_i['rmse']:.3f}")
    print(f"  in-band noise alone (middle of the record): {np.sqrt(np.mean(parts['noise (in-band)'][inner] ** 2)):.3f} RMS; the rest of the middle-record error ({s_hat_i['rmse']:.3f} total) is edge error bleeding inward")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    ax.plot(t_n, e_raw, color="lightgray", linewidth=0.8, label=f"raw  (RMSE {s_raw['rmse']:.2f})")
    ax.plot(t_n, e_hat, color="firebrick", linewidth=1.2, label=f"filtered  (RMSE {s_hat['rmse']:.2f})")
    ax.plot(t_n, e_par, color="steelblue", linewidth=1.0, label=f"parametric  (RMSE {s_par['rmse']:.3f})")
    ax.axvspan(0, 2, color="orange", alpha=0.12)
    ax.axvspan(P.DURATION - 2, P.DURATION, color="orange", alpha=0.12, label="record ends (2 s)")
    ax.set_title("Error e[n] = estimate - truth")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 1]
    win = 100
    for e, c, nm in [(e_raw, "gray", "raw"), (e_hat, "firebrick", "filtered"), (e_par, "steelblue", "parametric")]:
        rr = np.sqrt(np.convolve(e**2, np.ones(win) / win, mode="same"))
        ax.semilogy(t_n[win // 2:-win // 2], rr[win // 2:-win // 2], color=c, label=nm)
    ax.set_title("Running RMS of the error (1 s window): the errors sit at the record ends")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("running RMS")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bins = np.linspace(-1.5, 1.5, 80)
    ax.hist(e_hat[inner], bins=bins, density=True, color="firebrick", alpha=0.6, label="filtered, middle 16 s")
    ax.hist(e_par, bins=bins, density=True, color="steelblue", alpha=0.6, label="parametric")
    gx = np.linspace(-1.5, 1.5, 300)
    s_ = s_hat_i["std"]
    ax.plot(gx, np.exp(-0.5 * ((gx - s_hat_i["bias"]) / s_) ** 2) / (s_ * np.sqrt(2 * np.pi)), "k--", label="Gaussian, same mean and std")
    ax.set_title("Distribution of the error")
    ax.set_xlabel("e")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    names = list(parts.keys())
    vals = [np.sqrt(np.mean(e**2)) for e in parts.values()]
    short = ["desired\ndistortion", "15 Hz\nvibration", "80 Hz\ninterf.", "bias +\ndrift", "noise"]
    bars = ax.bar(short, vals, color=["darkorange", "steelblue", "gray", "seagreen", "firebrick"])
    ax.set_yscale("log")
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v * 1.15, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_title("Error budget: RMS contributed by each source (whole record)")
    ax.set_ylabel("RMS")
    for ax in axes.flatten():
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stage9_error.png"), dpi=150)
    print("\nSaved stage9_error.png")

    big = max(parts, key=lambda k: np.mean(parts[k] ** 2))
    print(
        f"""
What this means
---------------
The processing helped, and the numbers say by how much: the RMSE of the filtered estimate
is {s_hat['rmse']:.2f} against {s_raw['rmse']:.2f} for the raw measurement ({s_raw['rmse'] / s_hat['rmse']:.0f}x smaller) and its SNR rises from {s_raw['snr']:.1f} to
{s_hat['snr']:.1f} dB. The raw error is mostly the bias (+{s_raw['bias']:.2f}), the drift and the vibration, and the
filter removes all three, so the filtered estimate's bias is {s_hat['bias']:+.3f}.
The error budget shows what is left and why (shares add to more than 100% because the
sources are correlated):
  - the two biggest terms, "{big}" and the 15 Hz vibration, are EDGE effects. In
    steady state the filter passes the vibration at -81 dB, as designed, yet the whole-record
    vibration term is not small: the first and last 2 s carry an RMS of {np.sqrt(np.mean(edges ** 2)):.2f} against
    {s_hat_i['rmse']:.2f} in the middle. A short record has to be extended past its ends before it can be filtered,
    and the extension of an oscillating component is only approximately right; the 0.15 Hz
    high-pass (memory of several seconds) then spreads that mismatch inward. This is the
    price of removing drift from a 20 s record,
  - leakage of the 80 Hz interference and of the bias/drift is tiny ({np.sqrt(np.mean(parts['80 Hz interference (leakage)'] ** 2)):.3f} and {np.sqrt(np.mean(parts['bias + drift (leakage)'] ** 2)):.3f} RMS), as the
    spectrum-based design predicted,
  - the in-band noise contributes only {np.sqrt(np.mean(parts['noise (in-band)'] ** 2)):.2f} RMS: the floor no linear filter can lower without also removing signal.
Design note: staging the filters (low-pass with mirror padding first, then the high-pass)
was tried during development. It cut the vibration edge term from 0.40 to 0.11 but raised the
desired-signal edge term from 0.52 to 0.66, so the total error got worse (0.40 to 0.63):
the edge error is a trade-off, not something one padding choice removes. A longer record, or
a model-based estimate (below), is the real fix.
The parametric estimate reaches {s_par['rmse']:.3f} RMSE ({s_par['snr']:.0f} dB) because it uses more prior knowledge: it assumes the
signal is two sinusoids, which the spectrum supports, and estimates only four amplitudes,
frequencies and phases instead of 2000 samples. It is the better estimator here, and it
would be the worse one if the motion contained anything the model does not."""
    )
