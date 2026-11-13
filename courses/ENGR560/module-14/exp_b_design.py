"""
Experiment B: reward design, discounting and exploration (small controlled comparison).

Same deterministic grid, alpha 0.1, 3000 training episodes. Baseline = goal-oriented reward, gamma 0.95,
fixed epsilon 0.2. One factor changed per configuration. Seed 0 first, then seeds 0-4.
Random tie-breaking rows: success / steps are averaged over 200 independent tie-break draws of the greedy policy.
Target performance: the frozen greedy policy follows a shortest (8-move) path to the goal; "episodes to
target" = first checkpoint (every 10 episodes) after which every later checkpoint also meets it.
Extra row (not required): sparse reward with gamma = 1 - a reward that is maximised by ANY path.
"""

from pathlib import Path

import numpy as np

from gridworld import TXT, GridWorld, rollout, value_iteration
from qlearning import QLearningAgent, exp_decay, fixed, train

OUT = Path(__file__).resolve().parent / "outputs"
EPISODES, SEEDS = 3000, range(5)
CONFIGS = {
    "Goal-oriented reward (baseline)": dict(reward="goal", gamma=0.95, sched=fixed(0.2)),
    "Sparse reward": dict(reward="sparse", gamma=0.95, sched=fixed(0.2)),
    "Low discount factor (gamma 0.5)": dict(reward="goal", gamma=0.5, sched=fixed(0.2)),
    "High discount factor (gamma 0.99)": dict(reward="goal", gamma=0.99, sched=fixed(0.2)),
    "Decaying exploration (1.0 -> 0.01)": dict(reward="goal", gamma=0.95, sched=exp_decay(1.0, 0.01, 0.005)),
    "Sparse + random tie-breaking (extra)": dict(reward="sparse", gamma=0.95, sched=fixed(0.2), tie="random"),
    "Sparse + decaying exploration (extra)": dict(reward="sparse", gamma=0.95, sched=exp_decay(1.0, 0.01, 0.005)),
    "Sparse, gamma 1.0, random ties (extra)": dict(reward="sparse", gamma=1.0, sched=fixed(0.2), tie="random"),
}


def run(cfg, seed):
    env = GridWorld(reward=cfg["reward"])
    ref = GridWorld(reward=cfg["reward"])
    opt_len = int(env.bfs_distance()[0])
    agent = QLearningAgent(env.n_states, env.n_actions, alpha=0.1, gamma=cfg["gamma"], seed=seed,
                           tie_break=cfg.get("tie", "first"))

    def check(ag):
        path, _, ok = rollout(ref, ag.greedy_policy())
        return len(path) - 1 if ok else np.inf

    h = train(env, agent, EPISODES, cfg["sched"], seed=seed, check_every=10, check_fn=check)
    ck = np.array([c[1] for c in h["checks"]])
    eps = np.array([c[0] for c in h["checks"]])
    good = ck == opt_len
    stable = [i for i in range(len(ck)) if good[i:].all()]
    path, ret, ok = rollout(ref, agent.greedy_policy())
    steps = len(path) - 1 if ok else np.nan
    if cfg.get("tie") == "random":
        # tied greedy actions: evaluate 200 independent tie-break draws instead of one arbitrary choice
        outs = []
        for k in range(200):
            rr = np.random.default_rng(1000 + k)
            pol_k = np.array([int(rr.choice(np.flatnonzero(q == q.max()))) for q in agent.Q])
            p_k, _, ok_k = rollout(ref, pol_k)
            outs.append((ok_k, len(p_k) - 1))
        ok = np.mean([o[0] for o in outs])
        steps = np.mean([o[1] for o in outs if o[0]]) if ok > 0 else np.nan
    sa = env.shortest_actions()
    pol = agent.greedy_policy()
    return {"success": float(ok), "steps": steps, "to_target": eps[stable[0]] if stable else np.nan,
            "cells_opt": sum(pol[s] in sa[s] for s in sa), "n_cells": len(sa),
            "train_ret_last": h["return"][-500:].mean(), "train_succ_last": h["success"][-500:].mean(),
            "collisions": h["collisions"].sum(), "agent": agent, "path": path,
            "train_goals": h["success"].sum()}


def fmt_target(v):
    v = np.asarray(v, float)
    if np.all(np.isnan(v)):
        return "never"
    ok = v[~np.isnan(v)]
    s = f"{np.median(ok):.0f} [{ok.min():.0f}-{ok.max():.0f}]"
    return s + (f" ({np.sum(np.isnan(v))} never)" if np.isnan(v).any() else "")


def main():
    print("Experiment B: reward design, discount factor, exploration (alpha 0.1, 3000 episodes)")
    results = {name: [run(cfg, sd) for sd in SEEDS] for name, cfg in CONFIGS.items()}

    print("\n  seed 0 only (initial comparison):")
    print(f"    {'configuration':<38}{'success':>8}{'steps':>7}{'episodes to target':>20}{'cells optimal':>15}")
    for name, rs in results.items():
        r = rs[0]
        print(f"    {name:<38}{r['success']:>8.2f}{r['steps']:>7.0f}{fmt_target([r['to_target']]):>20}"
              f"{r['cells_opt']:>9}/{r['n_cells']}")

    print(f"\n  all {len(SEEDS)} seeds (frozen greedy policy; deterministic environment, so one evaluation rollout is exact):")
    print(f"    {'configuration':<38}{'success rate':>13}{'mean steps':>11}{'episodes to target, median [range]':>36}"
          f"{'cells optimal':>14}{'train return*':>14}{'collisions*':>12}")
    for name, rs in results.items():
        succ = np.mean([r["success"] for r in rs])
        steps = np.nanmean([r["steps"] for r in rs]) if succ > 0 else np.nan
        print(f"    {name:<38}{succ:>13.2f}{steps:>11.1f}{fmt_target([r['to_target'] for r in rs]):>36}"
              f"{np.mean([r['cells_opt'] for r in rs]):>10.1f}/{rs[0]['n_cells']}"
              f"{np.mean([r['train_ret_last'] for r in rs]):>14.2f}{np.mean([r['collisions'] for r in rs]):>12.0f}")
    print("    * train return = mean return over the last 500 TRAINING episodes (exploration on; goal-oriented scale"
          " except sparse rows); collisions = obstacle bumps during all 3000 training episodes")

    print("\n  discounting: how much does the distant goal still separate the actions? (exact value iteration)")
    env = GridWorld()
    d = env.bfs_distance()
    print(f"    {'gamma':>6}{'V*(start)':>11}{'gap at start':>14}{'gap 1 step from goal':>22}{'goal term at start':>20}"
          f"{'learned gap at start (seed 0)':>31}")
    key = {0.5: "Low discount factor (gamma 0.5)", 0.95: "Goal-oriented reward (baseline)", 0.99: "High discount factor (gamma 0.99)"}
    for g in (0.5, 0.95, 0.99):
        V, Q, _ = value_iteration(env, g)
        s0, s1 = 0, env.idx((3, 4))
        gap = lambda q: np.sort(q)[-1] - np.sort(q)[-2]
        Ql = results[key[g]][0]["agent"].Q[0]
        print(f"    {g:>6}{V[s0]:>11.4f}{gap(Q[s0]):>14.4f}{gap(Q[s1]):>22.4f}{10 * g ** (d[0] - 1):>20.4f}{gap(Ql):>31.4f}")

    a = results["Sparse reward"][0]["agent"]
    print(f"\n  sparse reward, problem-set agent (np.argmax ties -> 'up'), seed 0: Q still exactly 0 in"
          f" {np.sum(np.all(a.Q == 0, axis=1))} of 25 states; training episodes that reached the goal:"
          f" {int(results['Sparse reward'][0]['train_goals'])} of {EPISODES}")
    for name in ("Sparse + random tie-breaking (extra)", "Sparse, gamma 1.0, random ties (extra)"):
        Q = results[name][0]["agent"].Q
        print(f"  {name}, seed 0: Q(start) = " + " ".join(f"{q:.4f}" for q in Q[0])
              + f"; Q one step from goal (3,4) = " + " ".join(f"{q:.4f}" for q in Q[19]))
    Q = results["Sparse, gamma 1.0, random ties (extra)"][0]["agent"].Q
    print(f"  gamma 1.0: 10 - Q(start) = " + " ".join(f"{10 - q:.1e}" for q in Q[0])
          + " -> the greedy choice rests on <1e-6 leftovers of incomplete convergence (actions on the path already"
          " used converge first), not on any preference expressed by the reward")
    V1, Q1, it1 = value_iteration(GridWorld(reward="sparse"), 1.0)
    print(f"  exact sparse, gamma 1.0: Q*(start) = " + " ".join(f"{q:.4f}" for q in Q1[0])
          + f" -> every action is optimal (all paths that eventually reach the goal are worth 10)")


if __name__ == "__main__":
    main()
