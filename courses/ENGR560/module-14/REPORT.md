# Module 14 — Reinforcement learning in a grid world

All numbers come from `python run_all.py`, logged in `outputs/run_all_log.txt` (about 40 s on CPU; NumPy only).

| File | Contents |
|---|---|
| `gridworld.py` | `GridWorld` (`reset(seed)`, `step(action)` → next state, reward, terminated, truncated, info), the known model `outcomes(s, a)`, BFS shortest paths, `value_iteration`, exact `policy_value` |
| `qlearning.py` | The problem set's `QLearningAgent`, plus a tie-breaking option, training/evaluation loops and epsilon schedules |
| `exp_a_qlearning.py` | Experiment A: the two required figures |
| `exp_b_design.py` | Experiment B: reward, discount and exploration table (5 seeds) |
| `exp_c_robustness.py` | Experiment C: stochastic transitions, value iteration, blocked route, adaptation |

**Screenshots.** A few lines were partly hidden by the page's "…" button (for example the description of `step()`'s return values, and the γ set in Experiment B). Each was readable in the next screenshot, so nothing appears to be missing.

## Environment

```
S . . . .      S start (0,0), G goal (4,4), # obstacle
. # # # .      top route:  right x4, down x4          =  8 moves (unique shortest path)
. # . . .      left route: down x4, right x2, up x2,
. # . # .                  right x2, down x2          = 12 moves
. . . # G
```

**Rewards.**
- Goal +10, ordinary move −0.1, attempting to enter an obstacle −1 (the agent stays put).
- Moving off the edge isn't specified in the problem set. Here it leaves the agent in place and is charged as an ordinary move.

**Episode end, following the Gymnasium convention.**
- `terminated` means the goal was reached. The TD target uses no bootstrap there.
- `truncated` means 100 steps passed without reaching the goal. The target **still bootstraps**: a time limit is not a property of the state, and cutting the target would teach the agent that wandering states are worthless.

**Stochastic dynamics (Experiment C).** The intended move succeeds with probability 0.8; each perpendicular move happens with probability 0.1.

---

## Experiment A — Q-learning in the deterministic world

**Setup.** Exactly the problem set's agent: α 0.1, γ 0.95, ε 0.2 fixed, ties broken by `np.argmax`. 3,000 episodes, seed 0.

**Figures.**
- `outputs/exp_a_learning_curve.png`: return, success rate and episode length as 25-episode moving averages on a log episode axis, because learning is essentially done within the first 100 episodes. It also shows the greedy path length, checked every 10 episodes.
- `outputs/exp_a_policy.png`: the greedy action in every reachable cell (black = on a shortest path, red = not), with the learned path drawn over the BFS shortest path.

**Training** (100-episode moving averages, exploration on):

| Episode | Success | Return | Length |
|---|---|---|---|
| 100 | 0.98 | +7.34 | 17.7 |
| 500 | 1.00 | +8.76 | 10.3 |
| 3000 | 1.00 | +8.76 | 10.2 |

Even after learning, training episodes average about 10 moves, not 8. That is the price of 20% random actions, not a learning failure.

**Frozen greedy policy** (100 separate evaluation episodes): success 1.00, 8.0 steps, return **9.30**, 0 collisions.
- Learned path: (0,0)→(0,1)→(0,2)→(0,3)→(0,4)→(1,4)→(2,4)→(3,4)→(4,4). This is identical to the BFS shortest path and achieves the optimal return 10 − 0.7 = 9.30.
- The greedy path became the shortest path at **episode 30** and never changed afterwards.
- On the path's (state, action) pairs, Q equals the exact optimum Q* (max |Q − Q*| below 1e-4).

**Engineering question: an efficient policy, or one that merely reaches the goal?**
- **Along the route the agent actually uses, it is efficient.** It takes the unique shortest path, and its values there have converged to the true optimal values.
- **Away from that route, it is not.**
  - 16 of the 17 reachable cells choose a shortest-path action. Cell (3,0) chooses "up", which moves away from the goal.
  - Its Q row (+0.10, −0.25, −0.25, −0.45) bears no resemblance to the optimal Q* (+5.18, +5.96, +5.56, +4.66). That cell was almost never visited.
- **Q-learning is only as good as the states it has visited.** The policy is optimal from the start state, not everywhere.
- That matters as soon as the agent can start elsewhere or be pushed off its route. Experiment C tests the push.

---

## Experiment B — reward design, discounting, exploration

**Setup.**
- Same world, α 0.1, 3,000 episodes. One factor is changed per configuration.
- Seed 0 first, then 5 seeds (0–4).
- **Target performance:** the frozen greedy policy follows an 8-move shortest path. "Episodes to target" is the first checkpoint (every 10 episodes) after which every later checkpoint also meets it.
- The world is deterministic, so a single greedy rollout is an exact evaluation.

**Required table (5 seeds):**

| Configuration | Success rate | Mean steps to goal | Episodes to target, median [range] | Cells with an optimal action | Training return, last 500 episodes |
|---|---|---|---|---|---|
| **Goal-oriented reward** (γ 0.95, ε 0.2) | 1.00 | 8.0 | 40 [30–40] | 14.8 / 17 | 8.81 |
| **Sparse reward** (problem-set agent) | **0.00** | — | **never** | 4.0 / 17 | 0.00 |
| **Low discount factor** (γ 0.5) | 1.00 | 8.0 | 40 [30–50] | 13.8 / 17 | 8.81 |
| High discount factor (γ 0.99) | 1.00 | 8.0 | 40 [30–40] | 14.8 / 17 | 8.81 |
| **Decaying exploration** (ε 1.0 → 0.01, k = 0.005) | 1.00 | 8.0 | **30 [20–40]** | **16.8 / 17** | **9.28** |
| *Extra:* sparse + random tie-breaking | 1.00 | 8.0 | 20 [20–20] | 14.4 / 17 | 10.00 |
| *Extra:* sparse + decaying exploration | 1.00 | 8.0 | 20 [10–20] | 14.8 / 17 | 10.00 |
| *Extra:* sparse, γ = 1, random tie-breaking | 1.00 | 8.0 | 20 [20–20] | 14.4 / 17 | 10.00 |

Seed 0 alone gave the same picture. The goal-oriented baseline reached the target at 30 episodes, and the sparse reward never did.

**Reward design: sparse rewards failed completely, and the cause was not what I expected.**
- With all-zero rewards, every Q value stays exactly 0, so `np.argmax` always returns action 0, "up".
- The agent therefore pushes against the top wall 80% of the time. It **reached the goal in 0 of 3,000 training episodes**, so it never received a single non-zero learning signal.
- **The goal-oriented reward hides this problem.**
  - Its −0.1 step cost makes every action the agent has tried score below the untried ones (still 0).
  - That turns zero initialisation into *optimistic* exploration: the agent keeps trying actions it hasn't taken yet.
- **The fix is not the reward.** With random tie-breaking, or with ε starting at 1.0, the sparse reward learns the shortest path in about 20 episodes, as fast as or faster than the goal-oriented reward.
- **Lesson:** "sparse rewards are hard" partly comes from how exploration interacts with tie-breaking and initial values. A reward change can silently change exploration behaviour.

**Discount factor.** The distant reward does lose its grip on early actions (exact value iteration):

| γ | V*(start) | Gap between the best two actions at the start | Same gap one step from the goal |
|---|---|---|---|
| 0.5 | **−0.120** | **0.040** | 5.10 |
| 0.95 | 6.380 | 0.419 | 0.60 |
| 0.99 | 8.641 | 0.186 | 0.20 |

- With γ = 0.5, the goal is worth only 10·0.5⁷ = 0.078 at the start, less than the step costs. The start state's value is negative, and the right action wins by only 0.04.
- In this deterministic world Q-learning still learns that tiny gap exactly, so γ = 0.5 reached the target (median 40 episodes). It left more off-route cells wrong (13.8 / 17).
- With noise in the value estimates (Experiment C dynamics, or function approximation), a 0.04 margin is fragile: a small error flips the start action.
- γ = 0.99 shrinks the start gap for the opposite reason: a 12-move detour costs little when discounting is weak.
- **γ is part of the task specification, not a tuning knob.**

**Exploration.**
- Decaying ε (from 1.0) reached the target slightly faster and fixed nearly every cell (16.8 / 17), because the early random phase visits every cell.
- Its late training return (9.28) is almost optimal, against 8.81 with ε fixed at 0.2. A fixed ε keeps paying for exploration forever.
- The same pattern held across all 5 seeds, so it is not a lucky run. The differences in episodes to target, however, are within one or two checkpoints.

**Engineering question: why might an agent maximise its assigned reward and still behave undesirably?**
- The reward is a proxy. Whatever it does not price, the agent is free to do.
- **Sparse reward, γ = 1.** Every path that eventually reaches the goal is worth exactly 10 (exact Q*(start) = 10, 10, 10, 10).
  - "Optimal" therefore includes 12-move detours, wandering, and bumping into obstacles, which cost nothing.
  - Our agent took the 8-move path only by accident. Its Q values differ by less than 1e-6, the leftovers of incomplete convergence, because actions on the path already in use converge first.
  - The reward expresses no preference at all.
- **Sparse rewards in general.** Obstacles are free, and sparse-reward agents bumped into them about 1,000–1,100 times during training.
- **Problem-set agent with a sparse reward.** It maximised its expected return (0) by a policy that never reaches the goal.
- **The general point:** reward misspecification, discounting choices and exploration side-effects all produce "correct" optimisation of the wrong objective. Inspecting the *behaviour* (paths, collisions, off-route cells) is the only check.

---

## Experiment C — robustness and comparison with a model (the most important experiment)

**Setup.**
- Stochastic dynamics (0.8 / 0.1 / 0.1), γ 0.95, goal-oriented reward, 5 seeds.
- Every evaluation uses the frozen greedy policy for 2,000 episodes on an independent random stream.
- Failure means the 100-step limit was reached without the goal.

### Parts 1–2 — stochastic transitions and value iteration

- **Value iteration** used the known model and converged in 51 sweeps to V*(start) = 4.944.
- Its optimal stochastic policy is **identical to the deterministic optimum in every cell**: the top route.

| Policy | World | Success | Mean return | Failure | Steps to goal |
|---|---|---|---|---|---|
| Deterministic-trained agent (Experiment A) | deterministic | 1.000 | 9.30 | 0 | 8.00 |
| Deterministic-trained agent, **no retraining** | **stochastic** | 1.000 | 8.44 ± 0.03 | 0 | 10.46 |
| Stochastic-trained agent (5,000 episodes, ε 1.0 → 0.05) | stochastic | 1.000 | 8.44 ± 0.03 | 0 | 10.46 |
| Value iteration (known model) | stochastic | 1.000 | 8.44 ± 0.03 | 0 | 10.46 |

**Exact discounted value from the start:** optimal 4.944; deterministic agent 4.944; stochastic agent 4.944. All three policies take the same action in every state the optimal policy visits.

**The policies differ only where it does not matter:**
- The Q-learning action is optimal under the true model in 87% of the 17 cells (deterministic agent) and 93% (stochastic agent).
- On the 9 cells the optimal policy actually visits, the stochastic agent agrees with value iteration on 100%.

**Why the deterministic policy didn't degrade.** In *this* grid, slipping only costs time: the expected path rises from 8 to 10.5 moves. There is no cliff or hazard that would make a different route safer, so the deterministic optimum stays optimal. That is a property of this environment, and I would not generalise it. With a hazard next to the short route, a model-aware policy would detour and the deterministic policy would not.

### Part 3 — environmental change: block the learned route at (0,3)

The only remaining route is the 12-move left route. Value iteration on the changed model gives V*(start) = 2.787 and turns the top row around ("v < <").

**Required comparison** (stochastic world, mean ± sd over 5 seeds, 2,000 evaluation episodes each):

| Policy | Environment | Success rate | Mean episode return | Failure rate |
|---|---|---|---|---|
| Original Q-learning | Original | 1.000 | 8.44 ± 0.03 | 0.000 |
| Original Q-learning | **Changed** | **0.000** | **−88.98 ± 0.03** | **1.000** |
| Retrained / adapted Q-learning (+1,000 episodes) | Changed | 1.000 | 7.19 ± 0.02 | 0.000 |
| Value iteration | Changed, known dynamics | 1.000 | 7.19 ± 0.02 | 0.000 |

**Without retraining, the policy fails every time.**
- The greedy policy walks to (0,2) and keeps choosing "right", into the new obstacle.
- That costs −1 per step for the remaining roughly 97 steps, until truncation.
- A frozen learned policy has no mechanism to notice that the world changed.

**Continued learning recovers fully, and fast.**
- With ε = 0.1, greedy success reached at least 0.95 after a median of **50 episodes** (range 40–80). See `outputs/exp_c_adaptation.png`.
- After 1,000 episodes, the adapted policy's value from the start (2.787) **equals value iteration's**, with the optimal action in 99% of cells.
- Steps to goal: 15.8, against 10.5 before the change.

**Deterministic world.** The same change gives the same story: the Experiment A agent goes from success 0 to 1 after adaptation, with 12 moves, equal to value iteration.

**Engineering question: what does RL offer when the model is known accurately?**
- **Here, essentially nothing.**
  - Value iteration needs 51–59 sweeps over 17 states (milliseconds) and returns the provably optimal policy for *every* state, including off-route ones that Q-learning left wrong.
  - Q-learning needed thousands of episodes to match it on the visited states.
  - When the route was blocked, value iteration needed only the corrected model; Q-learning needed new experience, and was useless (100% failure) until it got it.
- **RL earns its keep when the model is unknown, wrong or too expensive.**
  - Unknown or inaccurate transition probabilities.
  - State spaces too large to sweep.
  - A world that changes in ways nobody writes into the model.
- In those cases, the ability to learn from interaction, demonstrated by the 50-episode recovery, is the advantage.
- For a small, fully known grid world, the answer "very little" is the valid result. The model-based solution is the reference that tells us Q-learning got the visited states right.

## Summary

1. **Q-learning found the shortest path in 30 episodes,** and its values on that path match the exact optimum. Off-route cells can stay wrong, so efficiency holds only where the agent has experience.
2. **Reward design interacts with exploration.** The sparse reward failed only because constant Q values plus first-index tie-breaking stopped exploration. The step penalty was silently doing the exploring. A low γ shrinks the start-state action gap 10×. A fixed ε keeps costing return; a decaying ε avoids it.
3. **This grid's slips cost time but not safety,** so the deterministic policy remained optimal under stochastic dynamics.
4. **Blocking the learned route broke the frozen policy completely** (100% failure). Fifty episodes of continued learning restored the value-iteration optimum.
5. **With a known, small model, value iteration is strictly better.**
