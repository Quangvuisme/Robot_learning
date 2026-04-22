# Reinforcement Learning Notes

This folder collects lecture-style notes and runnable examples for both discrete and continuous control.

## Structure

- `theorem/` - core RL concepts and discrete-action examples (Q-learning, DQN, DDQN)
- `DDPG/` - continuous-action control with deterministic policy gradients
- `SAC/` - continuous-action control with maximum-entropy actor-critic
- `TRPO/` - continuous-action trust-region policy optimization

## 1. Problem Setup

We consider trajectories $(s_t, a_t, r_t)$ with dynamics

$$
 s_{t+1} = f(s_t, a_t),\quad r_t = r(s_t, a_t)
$$

The goal is to maximize the discounted return

$$
 J(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^{T} \gamma^t r_t\right].
$$

## 2. Value Functions

State-value and action-value functions are

$$
 V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{T} \gamma^t r_t \mid s_0=s\right],\quad
 Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^{T} \gamma^t r_t \mid s_0=s,a_0=a\right].
$$

The Bellman optimality equation is

$$
 Q^*(s,a) = r(s,a) + \gamma \mathbb{E}_{s'\mid s,a}[\max_{a'} Q^*(s',a')].
$$

## 3. Discrete Actions: Q-Learning and DQN

**Q-learning update:**

$$
 Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big].
$$

**DQN target (with target network):**

$$
 y_{\text{DQN}} = r + \gamma \max_{a'} Q_{\phi^-}(s', a').
$$

**Double DQN target:**

$$
 y_{\text{DDQN}} = r + \gamma Q_{\phi^-}(s', \arg\max_{a'} Q_{\phi}(s',a')).
$$

## 4. Continuous Actions: DDPG, SAC, and TRPO

For continuous actions, DDPG learns a deterministic policy $\mu_\theta(s)$ and critic $Q_\phi(s,a)$.

**Critic loss:**

$$
 \mathcal{L}_Q(\phi) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\left[(Q_\phi(s,a) - y)^2\right],\quad
 y = r + \gamma Q_{\phi^-}(s', \mu_{\theta^-}(s')).
$$

**Actor update (deterministic policy gradient):**

$$
 \nabla_\theta J(\theta) \approx
 \mathbb{E}_{s\sim\mathcal{D}}\left[\nabla_a Q_\phi(s,a)\rvert_{a=\mu_\theta(s)}\nabla_\theta \mu_\theta(s)\right].
$$

SAC instead optimizes a stochastic policy with entropy regularization:

$$
J_{\text{SAC}}(\pi) = \sum_t \mathbb{E}\left[r(s_t,a_t) + \alpha\,\mathcal{H}(\pi(\cdot\mid s_t))\right].
$$

TRPO takes a different route and maximizes a local surrogate under a KL trust-region constraint:

$$
\max_{\theta'}\; \mathbb{E}\left[\frac{\pi_{\theta'}(a\mid s)}{\pi_\theta(a\mid s)} A^{\pi_\theta}(s,a)\right]
\quad \text{such that} \quad
\mathbb{E}\left[D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid s)\|\pi_{\theta'}(\cdot\mid s)\big)\right] \le \delta.
$$

In short:

- **DDPG** - off-policy, deterministic actor, replay buffer, target networks
- **SAC** - off-policy, stochastic actor, entropy maximization
- **TRPO** - on-policy, stochastic actor, natural-gradient trust region

## 5. Quick Start

**Discrete examples (theorem):**

```bash
cd /home/quangvd7/self_learning/Robot_learning/RL/theorem
python 01_basic_concepts.py
python 07_deep_q_network_cartpole.py
python 10_double_dqn_cartpole.py
```

**Continuous examples:**

```bash
cd /home/quangvd7/self_learning/Robot_learning/RL/DDPG
python ddpg_cartpole_swingup_balance.py

cd /home/quangvd7/self_learning/Robot_learning/RL/SAC
python sac_cartpole_swingup_balance.py

cd /home/quangvd7/self_learning/Robot_learning/RL/TRPO
python trpo_cartpole_swingup_balance.py
```
