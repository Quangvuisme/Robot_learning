# PPO Notes: Proximal Policy Optimization for CartPole Swing-up

This folder implements a continuous-action PPO agent for CartPole swing-up + balance.
The environment is custom (no gym) and intentionally matches the dynamics and reward shaping
used in the DDPG, SAC, and TRPO study cases, so the main difference is the learning algorithm.

## 1. Why PPO Exists

TRPO achieves stable policy updates by solving a constrained optimization:

$$
\max_{\theta'} L_\theta(\theta') \quad \text{such that} \quad \bar D_{\mathrm{KL}}(\pi_\theta \| \pi_{\theta'}) \le \delta.
$$

This requires conjugate gradient and Fisher-vector products — expensive and complicated.

PPO keeps the same spirit (don't move too far from the old policy) but enforces it with a
simple **clipping trick** instead of a hard KL constraint:

> "Clip the probability ratio so large updates are automatically penalized."

This makes PPO much simpler to implement and faster per iteration, while retaining most of
the stability benefit of TRPO.

## 2. Policy Improvement View (same as TRPO)

Both TRPO and PPO start from the same identity:

$$
J(\theta') - J(\theta)
= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[\sum_t \gamma^t A^{\pi_\theta}(s_t, a_t)\right].
$$

And use importance sampling to evaluate it under the old policy:

$$
L_\theta(\theta') = \mathbb{E}_{s \sim \rho_{\pi_\theta},\, a \sim \pi_\theta}
\left[\frac{\pi_{\theta'}(a \mid s)}{\pi_\theta(a \mid s)} A^{\pi_\theta}(s,a)\right].
$$

The probability ratio is:

$$
r_t(\theta') = \frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}.
$$

## 3. The Clipped Surrogate Objective (Core PPO Idea)

PPO replaces the KL constraint with a clipped objective:

$$
L^{\mathrm{CLIP}}(\theta')
= \mathbb{E}\left[
\min\!\left(
r_t(\theta')\,A_t,\;
\mathrm{clip}\!\left(r_t(\theta'),\,1-\varepsilon,\,1+\varepsilon\right) A_t
\right)
\right].
$$

**Why this works:**

- If $r_t$ stays inside $[1-\varepsilon, 1+\varepsilon]$, both terms are equal → normal gradient.
- If $r_t$ goes outside the clip range and the advantage is positive, the clipped term is smaller → gradient is reduced.
- The $\min$ takes the **pessimistic** (lower) bound, so the update cannot be too aggressive in either direction.

**Intuition:** PPO says "take as big a step as possible, but automatically stop when the ratio leaves the trust band."

## 4. Full PPO Objective

In practice, PPO trains actor and critic jointly with three terms:

$$
L(\theta') = L^{\mathrm{CLIP}}(\theta') - c_1 L^{VF}(\theta') + c_2 \mathcal{H}(\pi_{\theta'})
$$

where:

- $L^{VF} = \mathbb{E}[(V_\phi(s_t) - \hat R_t)^2]$ is the value function MSE,
- $\mathcal{H}(\pi_{\theta'}) = -\mathbb{E}[\log \pi_{\theta'}(a \mid s)]$ is the entropy bonus (encourages exploration),
- $c_1, c_2$ are weighting coefficients.

## 5. Critic and GAE

Same value baseline and advantage estimation as TRPO.

TD residual:

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t).
$$

Generalized Advantage Estimation (GAE):

$$
\hat A_t = \delta_t + \gamma \lambda \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots
$$

Recursively:

$$
\hat A_t = \delta_t + \gamma \lambda (1 - d_t) \hat A_{t+1}.
$$

Return target for the critic:

$$
\hat R_t = \hat A_t + V_\phi(s_t).
$$

## 6. Multiple Epochs — Key Difference from TRPO

TRPO performs **one update** per collected batch (the KL constraint limits reuse).

PPO performs **multiple minibatch epochs** over the same batch.
The clipping prevents the ratio from drifting too far even after many gradient steps,
so the data can be safely reused.

This makes PPO significantly **more sample-efficient per rollout** than TRPO.

## 7. Algorithm (Step-by-Step)

For each training iteration:

1. Collect a fresh on-policy batch with the current Gaussian policy.
2. Compute GAE advantages and return targets with the current value network.
3. Store old log-probabilities $\log \pi_\theta(a_t \mid s_t)$ before any update.
4. For $K$ epochs:
   - Shuffle data.
   - For each minibatch:
     $$
     \text{compute } r_t(\theta') = \exp(\log\pi_{\theta'}(a_t|s_t) - \log\pi_\theta(a_t|s_t))
     $$
     $$
     L = -L^{\mathrm{CLIP}} + c_1 L^{VF} - c_2 \mathcal{H}
     $$
     $$
     \theta' \leftarrow \theta' - \alpha \nabla_{\theta'} L
     $$
5. Repeat with a new batch from the updated policy.

Notice what is missing compared with TRPO:

- No conjugate gradient.
- No Fisher-vector products.
- No line search.
- One shared optimizer for actor and critic.

## 8. Policy Used in This Example

Same diagonal Gaussian policy with tanh-squashed mean as TRPO:

$$
a = F_{\max} \cdot u, \quad u \sim \mathcal{N}(\mu_\theta(s),\, \sigma_\theta(s)^2).
$$

Log-probability (used for ratio computation):

$$
\log \pi_\theta(a \mid s) = -\frac{1}{2}\left(\frac{(a - \mu)^2}{\sigma^2} + 2\log\sigma + \log 2\pi\right).
$$

## 9. Reward Shaping

Same shaping as DDPG, SAC, and TRPO:

$$
r(s) = 1 + \cos(\theta) - 0.1x^2 - 0.01(\dot x^2 + \dot\theta^2).
$$

Near-upright bonus:

$$
\text{bonus} = 0.5 \quad \text{if } |\theta| < 0.25 \text{ and } |x| < 0.5.
$$

## 10. Study Case: CartPole Swing-up + Balance

### State

Raw state: $s = [x, \dot x, \theta, \dot\theta]$.

Neural network feature vector: $\phi(s) = [x, \dot x, \sin\theta, \cos\theta, \dot\theta]$.

### Action

Continuous scalar control force applied to the cart.

### Initial-state curriculum

Same as TRPO: mostly swing-up starts, with every 5th episode near-upright.

## 11. What the Script Produces

Outputs are saved into the local `outputs/` folder:

- `ppo_training_curves.png`
- `ppo_trajectory.png`
- `ppo_cartpole.gif`

The training-curves figure contains:

- average episode reward per iteration,
- clipped surrogate policy loss,
- value loss,
- policy entropy.

## 12. How to Read the PPO Outputs

### Average Episode Reward

Should trend upward over iterations.

### Policy Loss (clipped surrogate)

Should remain small and relatively stable.
Large spikes may indicate $\varepsilon$ is too large or epochs are too many.

### Value Loss

Tracks how well the critic fits return targets.
Persistently high values → advantage estimates are noisy → policy updates are less reliable.

### Policy Entropy

Should start high (exploration) and decrease gradually as the policy becomes more confident.
If entropy collapses early, reduce `--entropy-coef` or increase `--clip-eps`.

## 13. How to Run

```bash
cd /home/quangvd7/self_learning/Robot_learning/RL/PPO
python ppo_cartpole_swingup_balance.py
```

Useful options:

```bash
python ppo_cartpole_swingup_balance.py --iterations 150 --steps-per-iter 4096
python ppo_cartpole_swingup_balance.py --clip-eps 0.1 --ppo-epochs 5
python ppo_cartpole_swingup_balance.py --skip-gif
```

## 14. Important Hyperparameters

- `--clip-eps`: clipping range $\varepsilon$ (default 0.2). Controls trust region width.
- `--ppo-epochs`: number of gradient epochs per batch (default 10).
- `--minibatch-size`: minibatch size inside each epoch (default 256).
- `--gae-lambda`: bias-variance tradeoff for GAE (default 0.97).
- `--entropy-coef`: weight for entropy bonus (default 0.01).
- `--value-coef`: weight for value loss (default 0.5).
- `--max-grad-norm`: gradient clipping threshold (default 0.5).
- `--steps-per-iter`: on-policy batch size per iteration.

## 15. Comparison With TRPO, SAC, and DDPG

### PPO vs TRPO

| Aspect | TRPO | PPO |
|--------|------|-----|
| Trust region mechanism | KL constraint (hard) | Ratio clipping (soft) |
| Optimization | Conjugate gradient + line search | Standard gradient descent |
| Data reuse per batch | One update | Multiple epochs |
| Implementation complexity | High | Low |
| Sample efficiency | Lower | Higher |

Both are on-policy and use GAE. PPO is the practical default.

### PPO vs SAC

| Aspect | SAC | PPO |
|--------|-----|-----|
| On/off-policy | Off-policy (replay buffer) | On-policy |
| Exploration | Entropy in objective | Entropy bonus + clipping |
| Trust region | None | Ratio clipping |
| Sample efficiency | Very high | Moderate |

SAC reuses past data, making it more sample-efficient. PPO is simpler and more stable on-policy.

### PPO vs DDPG

| Aspect | DDPG | PPO |
|--------|------|-----|
| Policy type | Deterministic | Stochastic Gaussian |
| Off-policy | Yes (replay buffer) | No |
| Exploration | Noise injection | Natural from Gaussian |
| Stability | Can be brittle | More stable |

### Practical summary

| Algorithm | Stability | Sample Efficiency | Complexity |
|-----------|-----------|-------------------|------------|
| DDPG | Low–Medium | High | Medium |
| SAC | High | High | Medium–High |
| TRPO | High | Low | High |
| **PPO** | **High** | **Medium** | **Low** |

PPO is the go-to algorithm when you want a good stability-simplicity tradeoff without a replay buffer.

## 16. Intuition & Theory: From TRPO to PPO

### Why PPO Can Reuse Data

TRPO's KL constraint limits how far $\pi_{\theta'}$ can move in one step.
After one update, the constraint is saturated → you must collect new data.

PPO's clipping also limits the step, but more loosely.
After each minibatch step, the ratio $r_t(\theta')$ may still be inside the clip band.
So you can keep taking gradient steps on the same batch — up to `ppo_epochs` times —
before the policy drifts enough that the old log-probs are stale.

### The clip as a Soft KL Proxy

High KL divergence ≈ many ratios $r_t$ far from 1.
Clipping at $[1-\varepsilon, 1+\varepsilon]$ limits exactly this.
So the clipping acts as an implicit, cheaper version of the KL trust region.

### One-line Summary

> **PPO = TRPO idea (trust region) + clip trick (replace KL constraint) + multi-epoch reuse**

## 17. Takeaway

PPO is best understood as:

- policy gradient,
- plus GAE value baseline,
- plus a probability-ratio clip to prevent destructive updates,
- plus multiple gradient epochs for better data utilization.

That combination gives PPO its reputation as the practical workhorse of deep RL:
simple enough to implement in ~100 lines, stable enough to solve challenging continuous-control tasks.
