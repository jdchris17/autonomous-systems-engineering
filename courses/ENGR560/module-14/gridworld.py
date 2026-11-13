"""
Module 14 environment: 5 x 5 grid world (Gymnasium-style API) plus model-based tools.

Layout (row 0 at the top; S start, G goal, # obstacle):
    r0:  S  .  .  .  .
    r1:  .  #  #  #  .
    r2:  .  #  .  .  .
    r3:  .  #  .  #  .
    r4:  .  .  .  #  G
Two routes: the TOP route (right along row 0, then down column 4) is the unique shortest path, 8 moves;
the LEFT route (down column 0, along row 4, up through (3,2), (2,2), across row 2, down column 4) is 12 moves.

Actions 0 up, 1 down, 2 left, 3 right.  State index = 5 * row + col.
Rewards "goal" (goal-oriented): reach goal +10, ordinary move -0.1, attempt to enter an obstacle -1
(agent stays put).  "sparse": goal +10, everything else 0.
Grid edge: not specified in the problem set; here a move off the grid leaves the agent in place and is
charged as an ordinary move (-0.1 / 0).
Episode ends: terminated = goal reached (true MDP termination); truncated = 100 steps without the goal
(time limit, NOT an MDP termination - the TD target should still bootstrap).
Stochastic transitions (slip > 0): intended move with prob 1 - slip, each perpendicular move with slip / 2.
"""

from collections import deque

import numpy as np

ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
NAMES = ("up", "down", "left", "right")
ARROWS = ("↑", "↓", "←", "→")       # for figures
TXT = "^v<>"                                           # for console output
PERP = {0: (2, 3), 1: (3, 2), 2: (1, 0), 3: (0, 1)}
OBSTACLES = ((1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (3, 3), (4, 3))
REWARDS = {"goal": {"goal": 10.0, "move": -0.1, "obstacle": -1.0},
           "sparse": {"goal": 10.0, "move": 0.0, "obstacle": 0.0}}


class GridWorld:
    def __init__(self, size=5, start=(0, 0), goal=(4, 4), obstacles=OBSTACLES, slip=0.0, max_steps=100,
                 reward="goal"):
        self.size, self.start, self.goal = size, tuple(start), tuple(goal)
        self.obstacles = frozenset(map(tuple, obstacles))
        self.slip, self.max_steps, self.reward_name = slip, max_steps, reward
        self.R = REWARDS[reward]
        self.n_states, self.n_actions = size * size, 4
        self.rng = np.random.default_rng()
        self.state, self.t = self.idx(self.start), 0

    # --- indexing -------------------------------------------------------------------------------
    def idx(self, rc):
        return rc[0] * self.size + rc[1]

    def rc(self, s):
        return divmod(int(s), self.size)

    def free_states(self):
        return [self.idx((r, c)) for r in range(self.size) for c in range(self.size) if (r, c) not in self.obstacles]

    # --- dynamics -------------------------------------------------------------------------------
    def _move(self, s, a):
        r, c = self.rc(s)
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return s, "move"                                   # edge: stay, ordinary cost
        if (nr, nc) in self.obstacles:
            return s, "obstacle"                               # blocked: stay, collision cost
        return self.idx((nr, nc)), "move"

    def outcomes(self, s, a):
        """[(prob, next_state, reward, terminated)] - the known model used by value iteration."""
        if self.rc(s) == self.goal:
            return [(1.0, s, 0.0, True)]
        acts = [(1 - self.slip, a)] + ([(self.slip / 2, PERP[a][0]), (self.slip / 2, PERP[a][1])] if self.slip else [])
        out = []
        for p, act in acts:
            s2, kind = self._move(s, act)
            term = self.rc(s2) == self.goal
            out.append((p, s2, self.R["goal"] if term else self.R[kind], term))
        return out

    # --- Gymnasium-style API --------------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state, self.t = self.idx(self.start), 0
        return self.state, {"t": 0}

    def step(self, action):
        a = int(action)
        if self.slip:
            u = self.rng.random()
            if u >= 1 - self.slip:
                a = PERP[a][0] if u < 1 - self.slip / 2 else PERP[a][1]
        s2, kind = self._move(self.state, a)
        self.t += 1
        terminated = self.rc(s2) == self.goal
        reward = self.R["goal"] if terminated else self.R[kind]
        truncated = (not terminated) and self.t >= self.max_steps
        self.state = s2
        return s2, reward, terminated, truncated, {"t": self.t, "actual_action": a, "collision": kind == "obstacle"}

    # --- reference solutions --------------------------------------------------------------------
    def bfs_distance(self):
        """Shortest number of moves to the goal from every free cell (deterministic moves)."""
        dist = np.full(self.n_states, np.inf)
        g = self.idx(self.goal)
        dist[g] = 0
        q = deque([g])
        while q:
            s = q.popleft()
            for a in range(4):                                  # moves are reversible on a grid
                s2, kind = self._move(s, a)
                if kind == "move" and s2 != s and dist[s2] == np.inf:
                    dist[s2] = dist[s] + 1
                    q.append(s2)
        return dist

    def shortest_actions(self):
        d = self.bfs_distance()
        return {s: {a for a in range(4) if d[self._move(s, a)[0]] == d[s] - 1} for s in self.free_states()
                if np.isfinite(d[s]) and d[s] > 0}


def value_iteration(env, gamma, tol=1e-12, max_iter=100000):
    nS, nA = env.n_states, env.n_actions
    model = {(s, a): env.outcomes(s, a) for s in env.free_states() for a in range(nA)}
    V = np.zeros(nS)
    for it in range(1, max_iter + 1):
        Q = np.zeros((nS, nA))
        for (s, a), outs in model.items():
            Q[s, a] = sum(p * (r + (0.0 if term else gamma * V[s2])) for p, s2, r, term in outs)
        Q[env.idx(env.goal)] = 0.0
        Vn = Q.max(axis=1)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    return V, Q, it


def policy_value(env, policy, gamma):
    """Exact discounted value of a deterministic policy (linear solve; ignores the 100-step limit)."""
    nS = env.n_states
    P, r = np.zeros((nS, nS)), np.zeros(nS)
    for s in env.free_states():
        for p, s2, rew, term in env.outcomes(s, int(policy[s])):
            r[s] += p * rew
            if not term:
                P[s, s2] += p
    g = env.idx(env.goal)
    P[g], r[g] = 0, 0
    return np.linalg.solve(np.eye(nS) - gamma * P, r)


def rollout(env, policy, seed=None, max_steps=None):
    s, _ = env.reset(seed=seed)
    path, ret = [s], 0.0
    while True:
        s, r, term, trunc, info = env.step(policy[s])
        path.append(s)
        ret += r
        if term or trunc:
            return path, ret, term
