# SAC Notes: Soft Actor-Critic for CartPole Swing-up

This folder implements a continuous-action SAC agent for CartPole swing-up + balance.
The environment is custom (no gym) and matches the dynamics used in the DDPG example.

## 1. Objective (Maximum Entropy)

SAC maximizes reward and entropy:

$$
J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\pi}\left[r(s_t,a_t) + \alpha\,\mathcal{H}(\pi(\cdot\mid s_t))\right]
$$

The entropy term encourages exploration and robust policies. $\alpha$ controls the
reward-entropy tradeoff.

**Intuition:** SAC does not only chase high reward. It also keeps the policy
stochastic so the agent can keep exploring and avoid brittle solutions.

## 2. Critic (Double Q)

Two critics reduce overestimation:

$$
Q_1, Q_2 \approx Q^\pi
$$

Target uses the minimum:

$$
\hat y = r + \gamma\left(\min(Q_1^-, Q_2^-) - \alpha\log\pi(a'\mid s')\right)
$$

Each critic is trained with MSE:

$$
\mathcal{L}_{Q_i} = \mathbb{E}\left[(Q_i(s,a) - \hat y)^2\right],\quad i \in \{1,2\}
$$

## 3. Actor (Stochastic Policy)

We use a Gaussian policy with tanh squashing. The actor loss is:

$$
\mathcal{L}_\pi = \mathbb{E}\left[\alpha\log\pi(a\mid s) - \min(Q_1,Q_2)\right]
$$

**Reparameterization (sampling):**

$$
u = \mu_\theta(s) + \sigma_\theta(s)\odot\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$
$$
a = \tanh(u)
$$

This makes sampling differentiable so the actor can be optimized by gradient descent.

This pushes the policy to choose actions with high Q while keeping entropy high.

## 4. Temperature (Optional Auto-Tuning)

If auto-tuning is enabled, update $\alpha$ to match a target entropy:

$$
\mathcal{L}_\alpha = -\log\alpha\,(\log\pi(a\mid s) + \mathcal{H}_{\text{target}})
$$

**Intuition:** if entropy is lower than target, $\alpha$ increases and pushes
the policy to explore more (and vice versa).

## 5. Reward Shaping

Same shaping as DDPG swing-up + balance:

$$
 r(s) = 1 + \cos(\theta) - 0.1x^2 - 0.01(\dot x^2 + \dot\theta^2)
$$

## 6. How to Run

```bash
cd /home/quangvd7/self_learning/Robot_learning/RL/SAC
python sac_cartpole_swingup_balance.py
```

Outputs (plots + GIF) are saved under the local `outputs/` folder.

## 7. Algorithm (Step-by-Step)

1) Collect data with stochastic policy:
	$$
	a_t \sim \pi_\theta(\cdot\mid s_t),\quad (s_t,a_t,r_t,s_{t+1})\to\mathcal{D}
	$$
2) Critic update (two Q networks):
	$$
	\hat y = r + \gamma\left(\min(Q_1^-,Q_2^-) - \alpha\log\pi(a'\mid s')\right)
	$$
	$$
	\phi_i \leftarrow \phi_i - \alpha_q\nabla_{\phi_i}\,\mathbb{E}[(Q_i-\hat y)^2]
	$$
3) Actor update (maximize Q and entropy):
	$$
		heta \leftarrow \theta - \alpha_\pi\nabla_\theta\,\mathbb{E}[\alpha\log\pi(a\mid s) - \min(Q_1,Q_2)]
	$$
4) Target update (soft):
	$$
	\phi^- \leftarrow \tau\phi + (1-\tau)\phi^-
	$$

---

## Intuition & Theory: From DDPG → SAC

### 9. Policy Gradient & θ

A policy is a function:

$$
\pi_\theta(a \mid s)
$$

**θ (theta)** represents all neural network parameters (weights & biases). Training = updating θ to maximize expected reward:

$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$

> **Insight:** θ defines the behavior of the agent. Changing θ = changing the policy.

---

### 10. From Q-learning → DDPG

**Q-learning:**

$$
a = \arg\max_a Q(s, a)
$$

- Works for discrete actions, not feasible for continuous control.

**DDPG:**

$$
a = \mu_\theta(s) \approx \arg\max_a Q(s, a)
$$

Actor update:

$$
\nabla_\theta J = \mathbb{E}\left[\nabla_a Q(s,a) \cdot \nabla_\theta \mu(s)\right]
$$

> **Insight:** DDPG replaces **argmax** with a **learned differentiable function (actor)**.

---

### 11. Problems with DDPG

- Deterministic policy → poor exploration
- Q-value overestimation
- Unstable learning

---

### 12. Entropy

$$
\mathcal{H}(\pi) = -\mathbb{E}[\log \pi(a \mid s)]
$$

- High entropy → more randomness (exploration)
- Low entropy → deterministic behavior

> **Insight:** Entropy is NOT maximized alone — it is a **regularizer**.

---

### 13. SAC Objective

$$
J = \mathbb{E}[r + \alpha \mathcal{H}]
$$

- Maximize reward
- Maintain sufficient randomness

---

### 14. Softmax Policy (Soft Argmax)

Instead of hard argmax, SAC uses:

$$
\pi(a \mid s) \propto \exp\!\left(\frac{Q(s,a)}{\alpha}\right)
$$

This is a **soft version of argmax**:
- Higher Q → higher probability
- Lower Q → still possible (exploration)

---

### 15. Role of Temperature α

- $\alpha \downarrow$ → policy becomes deterministic (like DDPG)
- $\alpha \uparrow$ → policy becomes more random

> Controls exploration vs exploitation trade-off.

---

### 16. Why Double Q (Two Critics)?

Problem:

$$
\max(Q + \text{noise}) \Rightarrow \text{overestimation}
$$

Solution:

$$
Q_{\text{safe}} = \min(Q_1, Q_2)
$$

Using the minimum:
- Removes over-optimistic estimates
- Improves stability

---

### 17. SAC Core Components

SAC combines three key ideas:

| Component | Formula | Origin |
|-----------|---------|--------|
| Soft policy (entropy) | $\pi(a\|s) \propto \exp(Q/\alpha)$ | Maximum entropy RL |
| Double Q | $Q = \min(Q_1, Q_2)$ | Double Q-learning |
| Actor-Critic | reparameterized Gaussian actor | Policy gradient |

---

### 18. Key Insights

- SAC does **not** find a single best action → it learns a **distribution over good actions**
- The $\min(Q_1, Q_2)$ does NOT break optimality → it preserves ranking and avoids risky overestimation
- Entropy ensures exploration is **learned, not injected**

---

### 19. One-line Summary

> **SAC = softmax(Q) (via entropy) + double Q (for stability) + actor-critic (for continuous control)**

---

### 20. Intuition (Robot Perspective)

| Agent | Behavior |
|-------|---------|
| DDPG | "Do the best action I know" |
| SAC | "Prefer the best actions, but stay flexible and explore safely" |

---

### 21. Final Takeaway

SAC improves over DDPG by:

1. Replacing hard argmax with **soft probabilistic decisions**
2. Embedding exploration into the objective **(entropy)**
3. Using conservative Q estimates **(double Q)**

> **Result:** more stable, robust, and efficient learning in continuous control

---

## 22. Study Case: CartPole Swing-up + Balance

What to look for in the plots and GIF:

- **Early training:** large action variance, lots of swings with failures.
- **Mid training:** repeated swing-up successes, smoother control.
- **Late training:** stable upright balance with smaller oscillations.

Typical outputs:

- Training curves (reward, Q losses, policy entropy)
- Trajectory plot of $x, \dot x, \theta, \dot\theta$
- GIF showing swing-up then balance
