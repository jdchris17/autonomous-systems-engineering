"""
Module 16: ONE command that regenerates everything.

    python run_all.py

Writes to outputs/:
    config_used.json             frozen configuration, seeds and targets
    scenario_table.csv / .md     every metric for every scenario (demo seed)
    multiseed.csv                nominal and sensor-failure runs over seeds 0-9, with pass/fail
    results.txt                  the printed summary
    trajectories.png             reference, truth and estimate on equal axes, per scenario
    diagnostics.png              tracking error, estimation error with uncertainty, actuator limits
    multiseed.png                per-seed results against the frozen targets
    logs/<scenario>.csv          per-tick reference, truth, estimate, SDs, commands, errors (demo seed)
    logs/<scenario>_P.npz        full 6x6 covariance at every tick (demo seed)
    logs/<scenario>_events.csv   every aiding update: time, sensor, innovation, NIS (demo seed)
"""

import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config as C
from closed_loop import REF, score, simulate

OUT = Path(__file__).resolve().parent / "outputs"
LOGS = OUT / "logs"
COLORS = {"true-state benchmark": "k", "nominal": "tab:blue", "disturbance": "tab:orange",
          "sensor failure": "tab:red", "model mismatch": "tab:purple", "authority limit": "tab:brown"}
TABLE_COLS = [
    ("track_rmse", "pos-tracking RMSE (m)", ".3f"), ("track_peak_from_18", "peak track err >=18 s (m)", ".3f"),
    ("cross_rmse", "cross-track RMSE (m)", ".3f"), ("cross_max", "max |cross-track| (m)", ".3f"),
    ("speed_err_rmse", "speed err vs ref RMSE (m/s)", ".3f"), ("effort_a", "accel effort", ".2f"),
    ("effort_w", "yaw effort", ".3f"), ("t_at_a_lim", "time at accel limit (s)", ".2f"),
    ("t_at_w_lim", "time at yaw limit (s)", ".2f"),
    ("nav_pos_rmse", "nav pos RMSE (m)", ".3f"), ("nav_head_rmse_deg", "nav heading RMSE (deg)", ".2f"),
    ("nav_speed_rmse", "nav speed RMSE (m/s)", ".3f"), ("pos_sd_max_18_28", "max pos SD 18-28 s (m)", ".3f"),
    ("b_a_err", "b_a error (m/s^2)", "+.4f"), ("b_a_sd", "b_a SD", ".4f"),
    ("b_g_err", "b_g error (rad/s)", "+.5f"), ("b_g_sd", "b_g SD", ".5f"),
    ("finite", "all finite", ""), ("min_eig_P", "min eig P", ".2e"), ("max_asym_P", "max |P-P^T|", ".1e"),
]


def fmt(v, f):
    return str(v) if f == "" else format(v, f)


def write_logs(name, L, s):
    slug = name.replace(" ", "_").replace("-", "_")
    sd = np.sqrt(np.einsum("kii->ki", L["P"]))
    head = (["t", "ref_px", "ref_py", "ref_v", "ref_psi", "true_px", "true_py", "true_v", "true_psi",
             "est_px", "est_py", "est_v", "est_psi", "est_ba", "est_bg"]
            + [f"sd_{n}" for n in ("px", "py", "v", "psi", "ba", "bg")]
            + ["u_a_raw", "u_a", "u_w_raw", "u_w", "xi", "v_target", "psi_target", "e_par_hat", "e_perp_hat",
               "true_along_err", "true_cross_err", "true_track_err"])
    with open(LOGS / f"{slug}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for k in range(len(L["t"])):
            row = ([L["t"][k], *REF["p"][k], REF["v"][k], REF["psi"][k], *L["truth"][k], *L["est"][k], *sd[k]]
                   + [L[key][k] for key in ("u_a_raw", "u_a", "u_w_raw", "u_w", "xi", "v_target", "psi_target",
                                            "e_par_hat", "e_perp_hat")]
                   + [s["_along"][k], s["_cross"][k], s["_track"][k]])
            w.writerow([f"{v:.6g}" for v in row])
    np.savez_compressed(LOGS / f"{slug}_P.npz", t=L["t"], P=L["P"])
    with open(LOGS / f"{slug}_events.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "sensor", "innov_1", "innov_2", "NIS"])
        for ev in L["events"]:
            w.writerow([f"{ev[0]:.2f}", ev[1], f"{ev[2]:.6g}", "" if np.isnan(ev[3]) else f"{ev[3]:.6g}", f"{ev[4]:.4f}"])


def scenario_table(demo):
    names = list(demo)
    with open(OUT / "scenario_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric"] + names)
        for key, label, f in TABLE_COLS:
            w.writerow([label] + [fmt(demo[n][1][key], f) for n in names])
    lines = ["| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for key, label, f in TABLE_COLS:
        lines.append(f"| {label} | " + " | ".join(fmt(demo[n][1][key], f) for n in names) + " |")
    (OUT / "scenario_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Scenario table, seed {C.DEMO_SEED}")
    print(f"  {'metric':<28}" + "".join(f"{n[:14]:>15}" for n in names))
    for key, label, f in TABLE_COLS:
        print(f"  {label:<28}" + "".join(f"{fmt(demo[n][1][key], f):>15}" for n in names))


def multiseed():
    rows = []
    for name, key, target in (("nominal", "track_rmse", C.TARGET_NOMINAL_RMSE),
                              ("sensor failure", "track_peak_from_18", C.TARGET_OUTAGE_PEAK)):
        for seed in C.MULTI_SEEDS:
            s = score(simulate(name, seed))
            rows.append({"scenario": name, "seed": seed, "metric": key, "value": s[key], "target": target,
                         "pass": s[key] < target and s["finite"] and s["min_eig_P"] > 0,
                         "track_rmse": s["track_rmse"], "track_peak_from_18": s["track_peak_from_18"],
                         "nav_pos_rmse": s["nav_pos_rmse"], "pos_sd_max_18_28": s["pos_sd_max_18_28"],
                         "finite": s["finite"], "min_eig_P": s["min_eig_P"]})
    with open(OUT / "multiseed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nMulti-seed check (seeds {C.MULTI_SEEDS[0]}-{C.MULTI_SEEDS[-1]}), targets frozen in config.py")
    for name in ("nominal", "sensor failure"):
        rs = [r for r in rows if r["scenario"] == name]
        vals = np.array([r["value"] for r in rs])
        print(f"  {name:<15} {rs[0]['metric']:<19} mean {vals.mean():.3f}  max {vals.max():.3f}  "
              f"target < {rs[0]['target']}  passed {sum(r['pass'] for r in rs)}/{len(rs)}   "
              f"(nav pos RMSE mean {np.mean([r['nav_pos_rmse'] for r in rs]):.3f} m; "
              f"max pos SD in 18-28 s mean {np.mean([r['pos_sd_max_18_28'] for r in rs]):.3f} m)")
        bad = [r for r in rs if not r["pass"]]
        print("     failures: " + (", ".join(f"seed {r['seed']} ({r['value']:.3f} m)" for r in bad) if bad else "none")
              + "; numerical failures: " + ("none" if all(r["finite"] and r["min_eig_P"] > 0 for r in rs) else "SEE CSV"))
    return rows


def plot_trajectories(demo, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (name, (L, s)) in zip(axes.ravel(), demo.items()):
        ax.plot(REF["p"][:, 0], REF["p"][:, 1], "k--", lw=1.5, label="reference")
        ax.plot(L["truth"][:, 0], L["truth"][:, 1], color=COLORS[name] if name != "true-state benchmark" else "tab:green",
                lw=2, label="truth")
        ax.plot(L["est"][:, 0], L["est"][:, 1], color="0.35", lw=0.9, ls=":", label="estimate")
        ax.plot(*REF["p"][0], "ko", ms=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{name}: track RMSE {s['track_rmse']:.2f} m", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Module 16: reference, truth and estimate (equal axes, seed {C.DEMO_SEED}; dot = start)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_diagnostics(demo, path):
    t = REF_T = demo["nominal"][0]["t"]
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(12, 16))
    for a in ax:
        a.axvspan(18, 28, color="tab:red", alpha=0.07, lw=0)
        a.axvspan(15, 22, ymin=0.95, ymax=1.0, color="tab:orange", alpha=0.5, lw=0)
        a.grid(alpha=0.3, which="both")
    for name, (L, s) in demo.items():
        ax[0].semilogy(t, np.maximum(s["_track"], 1e-3), color=COLORS[name], lw=1.2, label=name)
        if name != "authority limit":
            ax[1].plot(t, s["_cross"], color=COLORS[name], lw=1.1, label=name)
    ax[0].axhline(C.TARGET_NOMINAL_RMSE, color="tab:blue", ls="--", lw=0.8)
    ax[0].axhline(C.TARGET_OUTAGE_PEAK, color="tab:red", ls="--", lw=0.8)
    ax[0].text(0.3, C.TARGET_NOMINAL_RMSE * 1.1, "nominal RMSE target 1 m", fontsize=8, color="tab:blue")
    ax[0].text(0.3, C.TARGET_OUTAGE_PEAK * 1.1, "outage peak target 3 m", fontsize=8, color="tab:red")
    ax[0].set_title("(a) true position-tracking error |p_ref − p| (log scale)")
    ax[0].set_ylabel("m"); ax[0].legend(fontsize=8, ncol=3, loc="upper left")
    ax[1].set_title("(b) true cross-track error in the reference frame (authority-limit case omitted: off scale)")
    ax[1].set_ylabel("m"); ax[1].set_ylim(-1.5, 1.5)
    for name in ("nominal", "sensor failure", "model mismatch"):
        L, s = demo[name]
        err = np.hypot(*(L["truth"][:, :2] - L["est"][:, :2]).T)
        ax[2].plot(t, err, color=COLORS[name], lw=1.1, label=f"{name}: |p − p̂|")
        ax[2].plot(t, 2 * L["pos_sd"], color=COLORS[name], lw=0.8, ls="--")
    ax[2].set_title("(c) navigation position error (solid) and filter 2·sqrt(Pxx + Pyy) (dashed)")
    ax[2].set_ylabel("m"); ax[2].set_ylim(0, 3); ax[2].legend(fontsize=8)
    for name in ("nominal", "authority limit"):
        L, s = demo[name]
        ax[3].plot(t, L["u_w_raw"], color=COLORS[name], lw=0.8, ls=":", label=f"{name}: raw")
        ax[3].plot(t, L["u_w"], color=COLORS[name], lw=1.4, label=f"{name}: applied")
    for lim in (C.U_W_LIM, 0.05):
        ax[3].axhline(lim, color="0.4", lw=0.7, ls="--"); ax[3].axhline(-lim, color="0.4", lw=0.7, ls="--")
    ax[3].plot(t, REF["w"], color="k", lw=0.8, label="reference yaw rate")
    ax[3].set_ylim(-0.8, 0.8)
    ax[3].set_title("(d) yaw-rate command (limits ±0.5 nominal, ±0.05 authority-limit case)")
    ax[3].set_ylabel("rad/s"); ax[3].legend(fontsize=8, ncol=3)
    for name in ("nominal", "disturbance", "model mismatch", "authority limit"):
        L, s = demo[name]
        ax[4].plot(t, L["u_a"], color=COLORS[name], lw=1.1, label=name)
    ax[4].axhline(C.U_A_LIM, color="0.4", lw=0.7, ls="--"); ax[4].axhline(-C.U_A_LIM, color="0.4", lw=0.7, ls="--")
    ax[4].set_title("(e) applied acceleration command (limits ±2)")
    ax[4].set_ylabel("m/s²"); ax[4].set_xlabel("time (s)"); ax[4].legend(fontsize=8, ncol=4)
    fig.suptitle(f"Module 16 diagnostics, seed {C.DEMO_SEED} (red band: position outage in the sensor-failure case;"
                 " orange tick: disturbance window)")
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def plot_multiseed(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, target) in zip(axes, (("nominal", C.TARGET_NOMINAL_RMSE), ("sensor failure", C.TARGET_OUTAGE_PEAK))):
        rs = [r for r in rows if r["scenario"] == name]
        ax.bar([r["seed"] for r in rs], [r["value"] for r in rs], color=COLORS[name])
        ax.axhline(target, color="k", ls="--", lw=1, label=f"target {target} m")
        ax.set_xticks(C.MULTI_SEEDS)
        ax.set_xlabel("seed"); ax.set_ylabel("m")
        ax.set_title(f"{name}: {'tracking RMSE' if name == 'nominal' else 'peak tracking error from 18 s'}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    (OUT / "config_used.json").write_text(json.dumps(C.snapshot(), indent=2), encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        demo = {}
        for name in C.SCENARIOS:
            L = simulate(name, C.DEMO_SEED)
            s = score(L)
            demo[name] = (L, s)
            write_logs(name, L, s)
        scenario_table(demo)
        rows = multiseed()
        plot_trajectories(demo, OUT / "trajectories.png")
        plot_diagnostics(demo, OUT / "diagnostics.png")
        plot_multiseed(rows, OUT / "multiseed.png")
        print(f"\nOutputs written to {OUT}")
    text = buf.getvalue()
    (OUT / "results.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
