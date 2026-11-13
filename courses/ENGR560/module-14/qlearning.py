"""
Tabular Q-learning (the agent from the problem set) plus training / evaluation helpers.

TD target: r + gamma * max_a' Q(s', a') unless the episode TERMINATED (goal). A TRUNCATED episode (time
limit) still bootstraps - stopping the clock is not a property of the state, so cutting the target there
would teach the agent that wandering states are worth nothing.
"""

import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.2, seed=0, tie_break="first"):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.tie_break = tie_break          # "first" = np.argmax as in the problem set; "random" = among maxima

    def _argmax(self, q):
        if self.tie_break == "random":
            best = np.flatnonzero(q == q.max())
            return int(best[self.rng.integers(len(best))]) if len(best) > 1 else int(best[0])
        return int(np.argmax(q))

    def select_action(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.Q.shape[1]))
        return self._argmax(self.Q[state])

    def update(self, state, action, reward, next_state, terminated):
        target = reward
        if not terminated:
            target += self.gamma * np.max(self.Q[next_state])
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return td_error

    def greedy_policy(self):
        """Deterministic greedy policy (with random tie-breaking, one fixed choice per state)."""
        if self.tie_break == "random":
            r = np.random.default_rng(0)
            return np.array([int(r.choice(np.flatnonzero(q == q.max()))) for q in self.Q])
        return np.argmax(self.Q, axis=1)


def fixed(eps):
    return lambda ep: eps


def exp_decay(eps0=1.0, eps_min=0.05, k=0.005):
    """epsilon_t = max(eps_min, eps0 * exp(-k t)), t = episode index."""
    return lambda ep: max(eps_min, eps0 * np.exp(-k * ep))


def train(env, agent, episodes, schedule, seed=0, check_every=None, check_fn=None):
    """Returns per-episode arrays (return, length, success, collisions) and optional greedy checkpoints."""
    ret, length, success, coll = (np.zeros(episodes) for _ in range(4))
    checks = []
    env.reset(seed=seed)
    for ep in range(episodes):
        agent.epsilon = schedule(ep)
        s, _ = env.reset()
        G = 0.0
        while True:
            a = agent.select_action(s)
            s2, r, term, trunc, info = env.step(a)
            agent.update(s, a, r, s2, term)
            G += r
            coll[ep] += info["collision"]
            s = s2
            if term or trunc:
                break
        ret[ep], length[ep], success[ep] = G, info["t"], term
        if check_every and (ep + 1) % check_every == 0:
            checks.append((ep + 1, check_fn(agent)))
    return {"return": ret, "length": length, "success": success, "collisions": coll, "checks": checks}


def evaluate(env, policy, n=1000, seed=12345):
    """Frozen policy (no exploration, no learning) over n episodes with their own random stream."""
    env.reset(seed=seed)
    rets, steps, succ, coll = [], [], [], []
    for _ in range(n):
        s, _ = env.reset()
        G, c = 0.0, 0
        while True:
            s, r, term, trunc, info = env.step(policy[s])
            G += r
            c += info["collision"]
            if term or trunc:
                break
        rets.append(G)
        steps.append(info["t"])
        succ.append(term)
        coll.append(c)
    rets, steps, succ = np.array(rets), np.array(steps), np.array(succ, bool)
    return {"success": succ.mean(), "failure": 1 - succ.mean(), "return": rets.mean(),
            "steps_to_goal": steps[succ].mean() if succ.any() else np.nan, "collisions": float(np.mean(coll))}


def moving_average(x, w=100):
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out
