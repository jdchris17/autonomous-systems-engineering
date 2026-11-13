"""
Experiment A: 5 x 5 deterministic grid world, tabular Q-learning (alpha 0.1, gamma 0.95, epsilon 0.2 fixed -
the problem set's defaults), 3000 training episodes, seed 0. Moving averages over 100 episodes.
Every 10 episodes the frozen greedy policy is rolled out once (deterministic environment -> one rollout is
exact) to see when the greedy path becomes the shortest one.
After training: frozen greedy policy evaluated on separate episodes; compared with the BFS shortest path.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld import ARROWS, TXT, GridWorld, rollout
from qlearning import QLearningAgent, evaluate, fixed, moving_average, train

OUT = Path(__file__).resolve().parent / "outputs"
EPISODES, SEED = 3000, 0


def greedy_len(env):
    def f(agent):
        path, ret, ok = rollout(env, agent.greedy_policy())
        return len(path) - 1 if ok else np.inf
    return f


def train_default(seed=SEED, episodes=EPISODES, **env_kw):
    env = GridWorld(**env_kw)
    agent = QLearningAgent(env.n_states, env.n_actions, alpha=0.1, gamma=0.95, epsilon=0.2, seed=seed)
    h = train(env, agent, episodes, fixed(0.2), seed=seed, check_every=10, check_fn=greedy_len(GridWorld(**env_kw)))
    return env, agent, h


def policy_figure(env, agent, path, path_opt, fname, title):
    pol = agent.greedy_policy()
    visited = np.any(agent.Q != 0, axis=1)
    opt = env.shortest_actions()
    fig, ax = plt.subplots(figsize=(6, 6))
    for r in range(env.size):
        for c in range(env.size):
            s = env.idx((r, c))
            if (r, c) in env.obstacles:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="0.25"))
                continue
            if (r, c) == env.goal:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#b7e4b7"))
                ax.text(c, r, "G", ha="center", va="center", fontsize=20, weight="bold")
                continue
            if (r, c) == env.start:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#cfe2f3"))
                ax.text(c - 0.35, r - 0.32, "S", fontsize=10, weight="bold")
            good = pol[s] in opt.get(s, set())
            col = "k" if good else "tab:red"
            ax.text(c, r, ARROWS[pol[s]], ha="center", va="center", fontsize=26, color=col if visited[s] else "0.7")
    pr = [env.rc(s) for s in path_opt]
    ax.plot([p[1] for p in pr], [p[0] for p in pr], "-", color="tab:green", lw=8, alpha=0.25, label="BFS shortest path")
    pl = [env.rc(s) for s in path]
    ax.plot([p[1] + 0.06 for p in pl], [p[0] + 0.06 for p in pl], "--", color="tab:blue", lw=2,
            label=f"learned greedy path ({len(path) - 1} moves)")
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.grid(False)
    for k in range(env.size + 1):
        ax.axhline(k - 0.5, color="0.6", lw=0.8)
        ax.axvline(k - 0.5, color="0.6", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=120)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    env, agent, h = train_default()
    dist = env.bfs_distance()
    print("Experiment A: deterministic 5 x 5 grid world, Q-learning (alpha 0.1, gamma 0.95, epsilon 0.2 fixed)")
    print("  layout (S start, G goal, # obstacle):")
    for r in range(5):
        print("    " + " ".join("S" if (r, c) == env.start else "G" if (r, c) == env.goal else "#" if (r, c) in env.obstacles
                                else "." for c in range(5)))
    ma_s, ma_r, ma_l = moving_average(h["success"]), moving_average(h["return"]), moving_average(h["length"])
    print(f"\n  training ({EPISODES} episodes, exploration on), 100-episode moving averages:")
    for ep in (100, 250, 500, 1000, 2000, 3000):
        print(f"    episode {ep:>5}: success {ma_s[ep - 1]:.2f}  return {ma_r[ep - 1]:+6.2f}  length {ma_l[ep - 1]:5.1f}")
    ck = np.array([c[1] for c in h["checks"]])
    eps_ck = np.array([c[0] for c in h["checks"]])
    first_ok = eps_ck[np.argmax(np.isfinite(ck))]
    stable = eps_ck[[i for i in range(len(ck)) if np.all(ck[i:] == dist[0])][0]]
    print(f"  greedy policy first reaches the goal after {first_ok} episodes; greedy path is the 8-move shortest path"
          f" from episode {stable} onwards (checked every 10 episodes)")

    ev = evaluate(env, agent.greedy_policy(), n=100, seed=999)
    path, ret, ok = rollout(env, agent.greedy_policy())
    _, Qvi_pol_path = None, None
    from gridworld import value_iteration
    V, Qvi, _ = value_iteration(env, 0.95)
    path_opt, _, _ = rollout(env, Qvi.argmax(1))
    print(f"\n  frozen greedy policy, 100 evaluation episodes: success {ev['success']:.2f}, mean return {ev['return']:.2f},"
          f" steps to goal {ev['steps_to_goal']:.1f}, collisions/episode {ev['collisions']:.2f}")
    print(f"  learned path: {len(path) - 1} moves  "
          + " ".join(f"({env.rc(s)[0]},{env.rc(s)[1]})" for s in path))
    print(f"  BFS shortest: {int(dist[0])} moves  " + " ".join(f"({env.rc(s)[0]},{env.rc(s)[1]})" for s in path_opt))
    print(f"  optimal return along the shortest path: {10 - 0.1 * (dist[0] - 1):.2f}; learned {ret:.2f}")

    opt = env.shortest_actions()
    pol = agent.greedy_policy()
    visited = np.any(agent.Q != 0, axis=1)
    on_opt = [s for s in opt if pol[s] in opt[s]]
    bad = [s for s in opt if pol[s] not in opt[s]]
    print(f"\n  policy quality over all {len(opt)} reachable non-goal cells: {len(on_opt)} choose a shortest-path action,"
          f" {len(bad)} do not: " + ", ".join(f"({env.rc(s)[0]},{env.rc(s)[1]}) {TXT[pol[s]]}" for s in bad))
    for s in bad:
        s2, kind = env._move(s, pol[s])
        print(f"    cell ({env.rc(s)[0]},{env.rc(s)[1]}): BFS distance {int(dist[s])}, greedy goes {TXT[pol[s]]}"
              f" ({'collision' if kind == 'obstacle' else 'to distance ' + str(int(dist[s2]))}); Q row "
              + " ".join(f"{q:+.2f}" for q in agent.Q[s]) + f"; optimal Q* row " + " ".join(f"{q:+.2f}" for q in Qvi[s]))
    print(f"  max |Q - Q*| on the shortest path's (state, action) pairs: "
          f"{max(abs(agent.Q[s, pol[s]] - Qvi[s, pol[s]]) for s in path_opt[:-1]):.4f}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(1, EPISODES + 1)
    ma_r, ma_s, ma_l = (moving_average(h[k], 25) for k in ("return", "success", "length"))
    axs[0].plot(x, ma_r, color="tab:blue")
    axs[0].axhline(10 - 0.1 * 7, color="k", ls=":", label="optimal (8 moves) = 9.30")
    axs[0].set_title("episode return (25-ep moving avg)")
    axs[1].plot(x, ma_s, color="tab:green")
    axs[1].set_title("success rate (25-ep moving avg)")
    axs[1].set_ylim(0, 1.05)
    axs[2].plot(x, ma_l, color="tab:orange", label="training episode length (epsilon 0.2)")
    axs[2].step(eps_ck, np.where(np.isfinite(ck), ck, 100), where="post", color="tab:purple", lw=1,
                label="greedy path length (100 = fails)")
    axs[2].axhline(8, color="k", ls=":", label="shortest path (8)")
    axs[2].set_yscale("log")
    axs[2].set_title("episode length")
    for a in axs:
        a.set_xlabel("training episode (log scale)")
        a.set_xscale("log")
        a.grid(alpha=0.3)
        a.legend(fontsize=8) if a.get_legend_handles_labels()[0] else None
    fig.suptitle("Experiment A: Q-learning learning curves (exploration on during training)")
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_learning_curve.png", dpi=120)
    plt.close(fig)
    policy_figure(env, agent, path, path_opt, "exp_a_policy.png",
                  "Experiment A: learned greedy action per reachable cell\n(black = on a shortest path, red = not,"
                  " grey = never updated)")
    print("  saved", OUT / "exp_a_learning_curve.png", "and", OUT / "exp_a_policy.png")


if __name__ == "__main__":
    main()
