# TD3 Notes: Twin Delayed DDPG for CartPole Swing-up

This folder implements a continuous-action TD3 agent for CartPole swing-up + balance.
The environment and reward shaping are **identical** to the DDPG example so the two can
be compared directly. The only differences are in the learning algorithm.

> **Reference:** Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods", ICML 2018.

---

## 1. Why TD3 Exists: Problems with DDPG

DDPG suffers from three well-known failure modes:

### Problem 1 — Q-value overestimation

DDPG uses a single critic:

$$
y = r + \gamma Q_{\phi^-}(s', \mu_{\theta^-}(s')).
$$

Because $Q$ is trained with bootstrapping, errors accumulate.
With a single network, the estimate is systematically optimistic:

$$
\mathbb{E}[\max_a Q(s, a)] \ge \max_a \mathbb{E}[Q(s, a)].
$$

An overestimated Q pushes the actor toward fictitiously good actions — the policy diverges.

### Problem 2 — High variance targets

The target $y$ depends on the actor (via $\mu_{\theta^-}(s')$).
Any sharp peak in the actor's output around $s'$ becomes an unstable target for the critic.

### Problem 3 — Actor and critic updating simultaneously

In DDPG, both networks update at every step.
Because the critic is inaccurate early on, frequent actor updates based on a noisy critic
propagate errors and destabilize learning.

---

## 2. TD3 Solution: Three Targeted Fixes

TD3 adds exactly three modifications to DDPG. Everything else is identical.

---

### Fix 1 — Twin Critics (addresses overestimation)

TD3 maintains **two independent Q-networks**, $Q_1$ and $Q_2$.

The Bellman target uses the **minimum** of the two:

$$
y = r + \gamma \min(Q_1^-(s', \tilde{a}'),\; Q_2^-(s', \tilde{a}')).
$$

**Why min works:**

Both critics overestimate by different random amounts.
Taking the minimum gives a lower, more conservative bound.
This prevents the actor from exploiting an inflated region.

$$
\min(Q_1, Q_2) \le Q_1, \quad \min(Q_1, Q_2) \le Q_2.
$$

Each critic is still trained on the same target $y$:

$$
\mathcal{L}_{Q_i} = \mathbb{E}\left[(Q_i(s,a) - y)^2\right], \quad i \in \{1, 2\}.
$$

The actor is updated using only $Q_1$ (either critic works; $Q_1$ is conventional):

$$
\mathcal{L}_\pi = -\mathbb{E}\left[Q_1(s, \mu_\theta(s))\right].
$$

---

### Fix 2 — Target Policy Smoothing (addresses high variance targets)

Before computing the Bellman target, TD3 adds **clipped Gaussian noise** to the target actor's output:

$$
\tilde{a}' = \mu_{\theta^-}(s') + \epsilon, \quad \epsilon \sim \mathrm{clip}(\mathcal{N}(0, \tilde\sigma),\; -c,\; c).
$$

The noisy action is then clipped to the valid action range:

$$
\tilde{a}' = \mathrm{clip}(\tilde{a}',\; -a_{\max},\; a_{\max}).
$$

**Why this helps:**

If the actor has a sharp, narrow peak in action space, the target $Q(s', \mu(s'))$ becomes very sensitive to small changes.
By smoothing the target action, TD3 averages Q over a small neighborhood:

$$
y \approx r + \gamma \mathbb{E}_\epsilon\left[\min(Q_1^-, Q_2^-)(s', \mu^-(s') + \epsilon)\right].
$$

This makes the target more robust and reduces the variance of the Bellman backup.

---

### Fix 3 — Delayed Actor and Target Updates (addresses simultaneous updating)

TD3 updates the **critic at every step**, but the **actor only every $d$ steps** (default $d = 2$):

```
for each update step t:
    update Q1, Q2

    if t % policy_delay == 0:
        update actor
        soft-update all target networks
```

**Why this helps:**

The critic needs several gradient steps before it becomes accurate enough to give useful gradient signal to the actor.
Delaying the actor update gives the critics time to stabilize before the policy changes.

This also means target networks (and their slow Polyak tracking) only update when the actor does.

---

## 3. Full TD3 Update (Summary)

**Every step:**

$$
\tilde a' = \mathrm{clip}\!\left(\mu_{\theta^-}(s') + \mathrm{clip}(\epsilon, -c, c),\; -a_{\max}, a_{\max}\right)
$$

$$
y = r + \gamma \min(Q_1^-(s', \tilde a'),\; Q_2^-(s', \tilde a'))
$$

$$
\phi_i \leftarrow \phi_i - \alpha_Q \nabla_{\phi_i} \mathbb{E}[(Q_i(s,a) - y)^2], \quad i \in \{1, 2\}
$$

**Every $d$ steps (delayed):**

$$
\theta \leftarrow \theta - \alpha_\pi \nabla_\theta \mathbb{E}[-Q_1(s, \mu_\theta(s))]
$$

$$
\phi_i^- \leftarrow \tau \phi_i + (1-\tau)\phi_i^-, \quad \theta^- \leftarrow \tau\theta + (1-\tau)\theta^-
$$

---

## 4. Reward Shaping

Same shaping as DDPG, SAC, TRPO, and PPO:

$$
r(s) = 1 + \cos(\theta) - 0.1x^2 - 0.01(\dot x^2 + \dot\theta^2).
$$

Near-upright bonus:

$$
\text{bonus} = 0.5 \quad \text{if } |\theta| < 0.25 \text{ and } |x| < 0.5.
$$

---

## 5. Study Case: CartPole Swing-up + Balance

### State

Raw state: $s = [x, \dot x, \theta, \dot\theta]$.

Neural network feature vector: $\phi(s) = [x, \dot x, \sin\theta, \cos\theta, \dot\theta]$.

### Action

Continuous scalar force applied to the cart, $a \in [-F_{\max}, F_{\max}]$.

### Initial-state curriculum

Same as DDPG: mostly swing-up starts, with every 5th episode near-upright.

---

## 6. What the Script Produces

Outputs saved into `outputs/`:

- `td3_training_curves.png` — reward, actor loss, Q1 loss, Q2 loss
- `td3_trajectory.png` — pole angle and cart position over time
- `td3_cartpole.gif` — animated final rollout

---

## 7. How to Run

```bash
cd /home/quangvd7/self_learning/Robot_learning/RL/TD3
python td3_cartpole_swingup_balance.py
```

Useful options:

```bash
python td3_cartpole_swingup_balance.py --episodes 400
python td3_cartpole_swingup_balance.py --policy-delay 2 --policy-noise 0.2
python td3_cartpole_swingup_balance.py --skip-gif
```

---

## 8. Important Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--policy-noise` | 0.2 | std of smoothing noise on target actor (fraction of $F_{\max}$) |
| `--noise-clip` | 0.5 | clip bound for smoothing noise (fraction of $F_{\max}$) |
| `--policy-delay` | 2 | actor update frequency relative to critic |
| `--tau` | 0.005 | Polyak averaging rate for target networks |
| `--noise` | 0.25 | exploration noise std (fraction of $F_{\max}$) |
| `--noise-decay` | 0.995 | per-episode decay of exploration noise |
| `--warmup-steps` | 1000 | random exploration steps before training |

---

## 9. Comparison: DDPG vs TD3

| Aspect | DDPG | TD3 |
|--------|------|-----|
| Number of critics | 1 | 2 (twin) |
| Target value | $Q^-(s', \mu^-(s'))$ | $\min(Q_1^-, Q_2^-)(s', \tilde a')$ |
| Target action | $\mu^-(s')$ (clean) | $\mu^-(s') + \epsilon$ (smoothed) |
| Actor update freq | Every step | Every $d$ steps (delayed) |
| Target update freq | Every step | Every $d$ steps (with actor) |
| Overestimation | Common | Reduced |
| Stability | Can be brittle | More robust |

TD3 is a drop-in improvement over DDPG with almost no extra code.

---

## 10. Comparison With the Repo's Other Algorithms

| Algorithm | Policy type | Off-policy | Trust region / constraint | Key idea |
|-----------|-------------|------------|---------------------------|----------|
| DDPG | Deterministic | Yes | None | Continuous argmax via actor |
| **TD3** | **Deterministic** | **Yes** | **None** | **Twin Q + smoothing + delay** |
| SAC | Stochastic | Yes | None (entropy regularizer) | Max-entropy + double Q |
| TRPO | Stochastic | No | KL constraint (hard) | Natural gradient trust region |
| PPO | Stochastic | No | Ratio clipping (soft) | Clipped surrogate objective |

---

## 11. Intuition & Theory

### Why min(Q1, Q2) does not break the policy

The minimum is still a monotonic function of the true Q.
If action $a$ is genuinely better than action $b$, $\min(Q_1, Q_2)(s, a) > \min(Q_1, Q_2)(s, b)$
in expectation.
The minimum only removes the spurious spike caused by noise, not the true ordering of actions.

### Why smoothing noise is added to the target, not the current policy

Exploration noise is added to the **current** actor during data collection so the agent
tries diverse actions in the environment.

Smoothing noise is added to the **target** actor inside the Bellman update to regularize
the critic's loss surface. These are two different purposes and use different magnitudes.

### One-line Summary

> **TD3 = DDPG + min(Q1, Q2) + target smoothing + delayed actor update**

---

## 12. Takeaway

TD3 fixes the three most common DDPG failure modes with three minimal, targeted changes.
It maintains all the advantages of DDPG (off-policy, replay buffer, deterministic policy,
continuous control) while being significantly more stable in practice.

In the repo's progression:

```
DDPG  →  TD3  (fix overestimation + smoothing + delay)
           ↓
          SAC  (replace deterministic with stochastic + entropy objective)
```
