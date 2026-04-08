# DDPG Notes: Continuous Control for CartPole Swing-up

This folder implements a continuous-action DDPG agent for CartPole swing-up + balance.
The environment is custom (no gym) and matches the dynamics used in the other RL examples.

## 1. Problem Setup (What We Control)

We want a deterministic policy $\mu_\theta(s)$ that outputs a continuous force $a \in [-F_{\max}, F_{\max}]$.
Trajectories follow the dynamics

$$
 s_{t+1} = f(s_t, a_t),\quad a_t = \mu_\theta(s_t) + \epsilon_t
$$

where $\epsilon_t$ is exploration noise (Gaussian or Ornstein-Uhlenbeck).

**Intuition:** DDPG learns a deterministic controller for continuous force.
We still learn a critic $Q(s,a)$, but we cannot do $\arg\max_a$ directly, so
we add an actor network that outputs $a$.

## 2. Critic (Q-function)

The critic approximates the action-value function $Q_\phi(s,a)$.
DDPG trains it with a bootstrapped target:

$$
 y = r + \gamma Q_{\phi^-}(s', \mu_{\theta^-}(s'))
$$

and a regression loss

$$
 \mathcal{L}_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[(Q_\phi(s,a) - y)^2\right].
$$

Here $\phi^-$ and $\theta^-$ are target networks updated slowly.

**Bellman target idea:** use the *target actor* to generate the next action so
the target is stable during learning.

## 3. Actor (Policy)

The actor is trained to maximize the critic value:

$$
 \max_\theta\; \mathbb{E}_{s \sim \mathcal{D}}\left[Q_\phi(s, \mu_\theta(s))\right].
$$

By chain rule, the deterministic policy gradient is

$$
 \nabla_\theta J(\theta) \approx
 \mathbb{E}_{s \sim \mathcal{D}}\left[\nabla_a Q_\phi(s,a)\rvert_{a=\mu_\theta(s)}\nabla_\theta \mu_\theta(s)\right].
$$

**Intuition:** the critic tells the actor which direction in action space
increases $Q$, and the actor updates its parameters to move that way.

## 4. Replay Buffer + Target Networks

Transitions are stored in a replay buffer $\mathcal{D}$ and sampled in mini-batches to
reduce temporal correlation. Target networks are updated with a soft update
(Polyak averaging):

$$
 \phi^- \leftarrow \tau \phi + (1-\tau)\phi^-,\quad
 \theta^- \leftarrow \tau \theta + (1-\tau)\theta^-.
$$

## 5. Reward Shaping for Swing-up + Balance

This implementation uses a smooth reward that encourages upright balance and keeps the cart near center:

$$
 r(s) = 1 + \cos(\theta) - 0.1 x^2 - 0.01(\dot x^2 + \dot\theta^2)
$$

with a small bonus near upright to stabilize balance.

## 6. Algorithm (Step-by-Step)

1) Collect transition using behavior policy:
    $$
    a_t = \mu_\theta(s_t) + \epsilon_t,\quad (s_t,a_t,r_t,s_{t+1})\to\mathcal{D}
    $$
2) Critic update (sample mini-batch from $\mathcal{D}$):
    $$
    y = r + \gamma Q_{\phi^-}(s', \mu_{\theta^-}(s'))
    $$
    $$
    \phi \leftarrow \phi - \alpha_q\nabla_\phi\,\mathbb{E}\left[(Q_\phi(s,a)-y)^2\right]
    $$
3) Actor update (policy gradient through critic):
    $$
    	heta \leftarrow \theta + \alpha_\pi\,\mathbb{E}\left[\nabla_a Q_\phi(s,a)\rvert_{a=\mu_\theta(s)}\nabla_\theta \mu_\theta(s)\right]
    $$
4) Soft-update targets:
    $$
    \phi^- \leftarrow \tau\phi + (1-\tau)\phi^-,\quad
    	heta^- \leftarrow \tau\theta + (1-\tau)\theta^-
    $$

## 7. Relation to DQN (Intuition)

DDPG can be viewed as a DQN-style critic plus a separate actor to handle
continuous actions. The relationship is easiest to see by comparing the update
targets and action selection:

- **DQN (discrete actions):**
    $$
    y = r + \gamma \max_{a'} Q_{\phi^-}(s', a')
    $$
    Action comes from $\arg\max_a Q_\phi(s,a)$ (or epsilon-greedy).

- **DDPG (continuous actions):**
    $$
    y = r + \gamma Q_{\phi^-}(s', \mu_{\theta^-}(s'))
    $$
    Action comes from the actor $a = \mu_\theta(s) + \epsilon$.

So the key change is replacing the discrete argmax with a learned actor that
outputs a continuous action. DDPG still keeps the DQN-style critic, replay
buffer, and target networks, but it needs **two** target networks: one for the
critic and one for the actor.

You can also line up the learning objectives side by side:

- **DQN critic loss:**
    $$
    \mathcal{L}_{\text{DQN}}(\phi) = \mathbb{E}\left[(Q_\phi(s,a) - y)^2\right],\quad
    y = r + \gamma \max_{a'} Q_{\phi^-}(s', a')
    $$

- **DDPG critic loss:**
    $$
    \mathcal{L}_{\text{DDPG}}(\phi) = \mathbb{E}\left[(Q_\phi(s,a) - y)^2\right],\quad
    y = r + \gamma Q_{\phi^-}(s', \mu_{\theta^-}(s'))
    $$

- **DDPG actor objective (deterministic policy gradient):**
    $$
    J(\theta) = \mathbb{E}_{s \sim \mathcal{D}}\left[Q_\phi(s, \mu_\theta(s))\right]
    $$

## 8. Study Case: CartPole Swing-up + Balance

What to look for when running the script:

- **Early training:** high exploration noise, many failed swing-ups.
- **Mid training:** pole reaches upright more often, reward curve rises.
- **Late training:** smooth balance around upright with smaller actions.

Typical outputs (saved under `outputs/`):

- Training curves (reward, critic loss, actor loss)
- Trajectory plot of $x, \dot x, \theta, \dot\theta$
- GIF showing swing-up then balance

## 9. How to Run

```bash
cd /home/quangvd7/self_learning/Robot_learning/RL/DDPG
python ddpg_cartpole_swingup_balance.py
```

Outputs (plots + GIF) are saved under the `outputs/` folder in the repository root.
