"""
Experiment C: robustness and model-based comparison.

Stochastic dynamics: intended move 0.8, each perpendicular slip 0.1 (goal-oriented reward, gamma 0.95).
Agents (5 seeds each):
  deterministic agent  Experiment A's agent (trained in the deterministic world, eps 0.2, 3000 episodes)
  stochastic agent     trained in the stochastic world, 5000 episodes, eps 1.0 -> 0.05 (k 0.002)
Value iteration on the known model (tolerance 1e-12) -> greedy policy = model-based optimal reference.
Environmental change: the cell (0,3) on the learned top route becomes an obstacle (route blocked); the
only remaining route is the 12-move left route. The stochastic agent is evaluated without retraining,
then continues learning in the changed world (epsilon fixed 0.1, alpha 0.1) for 1000 episodes (frozen-policy check every 10 episodes, 100 evaluation episodes).
Every evaluation: frozen greedy policy, 2000 episodes, independent random stream. Failure = 100-step
time limit reached without the goal. V^pi(start) = exact discounted value of the policy (model).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp_a_qlearning import train_default
from gridworld import OBSTACLES, TXT, GridWorld, policy_value, value_iteration
from qlearning import QLearningAgent, evaluate, exp_decay, fixed, train

OUT = Path(__file__).resolve().parent / "outputs"
SEEDS, SLIP, GAMMA, N_EVAL = range(5), 0.2, 0.95, 2000
BLOCK = (0, 3)
ADAPT_EPISODES, ADAPT_CHECK = 1000, 10


def env_original(slip=SLIP):
    return GridWorld(slip=slip)


def env_changed(slip=SLIP):
    return GridWorld(slip=slip, obstacles=OBSTACLES + (BLOCK,))


def optimal_fraction(env, policy, Qstar, states=None):
    states = env.free_states() if states is None else states
    states = [s for s in states if env.rc(s) != env.goal]
    V = Qstar.max(axis=1)
    return np.mean([Qstar[s, policy[s]] >= V[s] - 1e-6 for s in states])


def main():
    OUT.mkdir(exist_ok=True)
    eo, ec, ed = env_original(), env_changed(), env_original(0.0)
    Vo, Qo, it_o = value_iteration(eo, GAMMA)
    Vc, Qc, it_c = value_iteration(ec, GAMMA)
    pol_vo, pol_vc = Qo.argmax(1), Qc.argmax(1)
    print("Experiment C: stochastic transitions (0.8 intended / 0.1 / 0.1), gamma 0.95, 5 seeds, 2000 evaluation episodes")
    print(f"  value iteration converged in {it_o} (original) / {it_c} (changed) sweeps; V*(start) = {Vo[0]:.3f} / {Vc[0]:.3f}")
    for name, e, pol in (("original", eo, pol_vo), ("changed (0,3) blocked", ec, pol_vc)):
        print(f"    VI policy, {name}:")
        for r in range(5):
            print("      " + " ".join("G" if (r, c) == e.goal else "#" if (r, c) in e.obstacles else TXT[pol[e.idx((r, c))]]
                                      for c in range(5)))
    Vd, Qd, _ = value_iteration(ed, GAMMA)
    print(f"  deterministic-world VI policy equals stochastic-world VI policy in "
          f"{np.mean([Qd.argmax(1)[s] == pol_vo[s] for s in eo.free_states() if eo.rc(s) != eo.goal]):.0%} of cells")

    rows = {k: [] for k in ("det_det", "det_sto", "sto_sto", "vi_orig", "sto_changed", "adapt_changed", "vi_changed",
                            "det_changed_det", "det_adapt_det", "vi_changed_det")}
    extra = {"opt_det_agent": [], "opt_sto_agent": [], "v_det": [], "v_sto": [], "opt_onpath": [], "adapt_first": [],
             "opt_adapt": [], "v_adapt": [], "curves": []}
    vi_visit = None
    for sd in SEEDS:
        # Part 1: deterministic agent (Experiment A) and a stochastic agent
        _, det_agent, _ = train_default(seed=sd)
        sto_agent = QLearningAgent(eo.n_states, eo.n_actions, alpha=0.1, gamma=GAMMA, seed=sd)
        train(env_original(), sto_agent, 5000, exp_decay(1.0, 0.05, 0.002), seed=sd)
        pd, ps = det_agent.greedy_policy(), sto_agent.greedy_policy()
        rows["det_det"].append(evaluate(ed, pd, N_EVAL, 10000 + sd))
        rows["det_sto"].append(evaluate(eo, pd, N_EVAL, 10000 + sd))
        rows["sto_sto"].append(evaluate(eo, ps, N_EVAL, 10000 + sd))
        rows["vi_orig"].append(evaluate(eo, pol_vo, N_EVAL, 10000 + sd))
        extra["opt_det_agent"].append(optimal_fraction(eo, pd, Qo))
        extra["opt_sto_agent"].append(optimal_fraction(eo, ps, Qo))
        extra["v_det"].append(policy_value(eo, pd, GAMMA)[0])
        extra["v_sto"].append(policy_value(eo, ps, GAMMA)[0])
        if vi_visit is None:                                    # states the optimal policy actually visits
            vi_visit = set()
            e = env_original()
            e.reset(seed=1)
            for _ in range(500):
                s, _ = e.reset()
                while True:
                    vi_visit.add(s)
                    s, r, term, trunc, _ = e.step(pol_vo[s])
                    if term or trunc:
                        break
        extra["opt_onpath"].append(optimal_fraction(eo, ps, Qo, sorted(vi_visit)))

        # Part 3: block (0,3); evaluate without retraining, then keep learning
        rows["sto_changed"].append(evaluate(ec, ps, N_EVAL, 20000 + sd))
        rows["vi_changed"].append(evaluate(ec, pol_vc, N_EVAL, 20000 + sd))
        curve = []

        def check(ag, _curve=curve):
            ev = evaluate(env_changed(), ag.greedy_policy(), 100, 30000)
            _curve.append(ev)
            return ev["success"]

        h = train(env_changed(), sto_agent, ADAPT_EPISODES, fixed(0.1), seed=100 + sd, check_every=ADAPT_CHECK, check_fn=check)
        succ_ck = np.array([c[1] for c in h["checks"]])
        extra["adapt_first"].append((np.argmax(succ_ck >= 0.95) + 1) * ADAPT_CHECK if (succ_ck >= 0.95).any() else np.nan)
        extra["curves"].append(([c["success"] for c in curve], [c["return"] for c in curve], h["return"]))
        pa = sto_agent.greedy_policy()
        rows["adapt_changed"].append(evaluate(ec, pa, N_EVAL, 20000 + sd))
        extra["opt_adapt"].append(optimal_fraction(ec, pa, Qc))
        extra["v_adapt"].append(policy_value(ec, pa, GAMMA)[0])

        # same change in the deterministic world (Experiment A agent)
        edc = env_changed(0.0)
        rows["det_changed_det"].append(evaluate(edc, pd, 1, 0))
        train(edc, det_agent, ADAPT_EPISODES, fixed(0.1), seed=200 + sd)
        rows["det_adapt_det"].append(evaluate(edc, det_agent.greedy_policy(), 1, 0))
        Vcd, Qcd, _ = value_iteration(edc, GAMMA)
        rows["vi_changed_det"].append(evaluate(edc, Qcd.argmax(1), 1, 0))

    def agg(key, f):
        v = np.array([r[f] for r in rows[key]], float)
        return np.nanmean(v), np.nanstd(v)

    print("\n  Part 1/2: original stochastic world (mean +- sd over 5 seeds)")
    print(f"    {'policy':<44}{'success':>15}{'mean return':>15}{'failure':>10}{'steps to goal':>15}")
    for key, name in (("det_det", "deterministic agent, deterministic world"), ("det_sto", "deterministic agent, STOCHASTIC world"),
                      ("sto_sto", "stochastic-trained agent, stochastic world"), ("vi_orig", "value iteration, stochastic world")):
        s, r, f, st = agg(key, "success"), agg(key, "return"), agg(key, "failure"), agg(key, "steps_to_goal")
        print(f"    {name:<44}{s[0]:>8.3f}+-{s[1]:.3f}{r[0]:>9.2f}+-{r[1]:.2f}{f[0]:>10.3f}{st[0]:>12.2f}")
    print(f"    exact discounted value V(start): optimal {Vo[0]:.3f}; deterministic agent {np.mean(extra['v_det']):.3f}"
          f" (+-{np.std(extra['v_det']):.3f}); stochastic agent {np.mean(extra['v_sto']):.3f} (+-{np.std(extra['v_sto']):.3f})")
    print(f"    cells where the Q-learning action is optimal under the true stochastic model: deterministic agent"
          f" {np.mean(extra['opt_det_agent']):.0%}, stochastic agent {np.mean(extra['opt_sto_agent']):.0%} of all 17;"
          f" stochastic agent on the {len(vi_visit) - 1} cells the optimal policy actually visits: {np.mean(extra['opt_onpath']):.0%}")

    print(f"\n  Part 3: route blocked at {BLOCK} (stochastic world) - REQUIRED COMPARISON (mean +- sd over 5 seeds)")
    print(f"    {'policy':<34}{'environment':<26}{'success':>14}{'mean return':>16}{'failure':>14}{'steps':>8}")
    for key, name, envn in (("sto_sto", "Original Q-learning", "Original"), ("sto_changed", "Original Q-learning", "Changed"),
                            ("adapt_changed", f"Adapted Q-learning (+{ADAPT_EPISODES} ep)", "Changed"),
                            ("vi_changed", "Value iteration", "Changed, known dynamics")):
        s, r, f, st = agg(key, "success"), agg(key, "return"), agg(key, "failure"), agg(key, "steps_to_goal")
        print(f"    {name:<34}{envn:<26}{s[0]:>7.3f}+-{s[1]:.3f}{r[0]:>9.2f}+-{r[1]:.2f}{f[0]:>7.3f}+-{f[1]:.3f}"
              f"{st[0]:>8.2f}")
    af = np.array(extra["adapt_first"], float)
    print(f"    adaptation: greedy success >= 0.95 after {np.nanmedian(af):.0f} episodes (median; range"
          f" {np.nanmin(af):.0f}-{np.nanmax(af):.0f}) of continued learning with epsilon 0.1")
    print(f"    adapted policy: optimal action in {np.mean(extra['opt_adapt']):.0%} of cells; V(start) {np.mean(extra['v_adapt']):.3f}"
          f" vs optimal {Vc[0]:.3f}")
    print("    same change in the DETERMINISTIC world (Experiment A agent):"
          f" before adaptation success {agg('det_changed_det', 'success')[0]:.2f}, after {agg('det_adapt_det', 'success')[0]:.2f}"
          f" (steps {agg('det_adapt_det', 'steps_to_goal')[0]:.0f}); value iteration {agg('vi_changed_det', 'steps_to_goal')[0]:.0f} steps")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(1, len(extra["curves"][0][0]) + 1) * ADAPT_CHECK
    S = np.array([c[0] for c in extra["curves"]])
    R = np.array([c[1] for c in extra["curves"]])
    for k in range(len(SEEDS)):
        ax[0].plot(x, S[k], color="tab:blue", alpha=0.3)
        ax[1].plot(x, R[k], color="tab:blue", alpha=0.3)
    ax[0].plot(x, S.mean(0), color="tab:blue", lw=2, label="adapting Q-learning (mean of 5 seeds)")
    ax[1].plot(x, R.mean(0), color="tab:blue", lw=2, label="adapting Q-learning (mean of 5 seeds)")
    ax[0].axhline(agg("vi_changed", "success")[0], color="k", ls=":", label="value iteration (known changed model)")
    ax[1].axhline(agg("vi_changed", "return")[0], color="k", ls=":", label="value iteration (known changed model)")
    ax[0].set_ylabel("greedy success rate")
    ax[1].set_ylabel("greedy mean return")
    for a in ax:
        a.set_xlabel("episodes of continued learning in the changed world")
        a.set_xscale("log")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"Experiment C: recovery after blocking {BLOCK} (stochastic world, frozen-policy evaluations every {ADAPT_CHECK} episodes)")
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_adaptation.png", dpi=120)
    plt.close(fig)
    print("  saved", OUT / "exp_c_adaptation.png")


if __name__ == "__main__":
    main()
