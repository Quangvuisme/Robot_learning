# TRPO Notes: Trust Region Policy Optimization for CartPole Swing-up

This folder implements a continuous-action TRPO agent for CartPole swing-up + balance.
The environment is custom (no gym) and intentionally matches the dynamics and reward shaping
used in the DDPG and SAC study cases, so the main difference is the learning algorithm.

## 1. Why TRPO Exists

Vanilla policy gradient updates the policy by following

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) R_t\right].
$$

This is simple, but large steps can change the policy too aggressively.
In practice that creates a common failure mode:

- the new policy is very different from the data-collecting policy,
- the policy-ratio term explodes,
- the update improves the local surrogate but hurts the real return.

TRPO addresses exactly this instability by saying:

"Improve the policy, but only inside a small trust region measured by KL divergence."

## 2. Policy Improvement View

TRPO starts from the policy improvement identity:

$$
J(\theta') - J(\theta)
= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[\sum_t \gamma^t A^{\pi_\theta}(s_t, a_t)\right].
$$

This expression is exact, but not directly usable because the expectation is under the
new policy $\pi_{\theta'}$, while the data usually comes from the old policy $\pi_\theta$.

Using importance sampling,

$$
J(\theta')
= \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\frac{p_{\theta'}(\tau)}{p_\theta(\tau)} R(\tau)\right].
$$

At the trajectory level this ratio becomes a product over time and quickly becomes high variance.
TRPO avoids that full-trajectory ratio by using a local surrogate objective based on per-step ratios.

## 3. Surrogate Objective

The practical surrogate used by TRPO is

$$
L_\theta(\theta')
= \mathbb{E}_{s \sim \rho_{\pi_\theta},\, a \sim \pi_\theta}
\left[
\frac{\pi_{\theta'}(a \mid s)}{\pi_\theta(a \mid s)}
A^{\pi_\theta}(s,a)
\right].
$$

Interpretation:

- $\rho_{\pi_\theta}$ is the state visitation distribution of the old policy.
- The ratio
  $$
  r_t(\theta') = \frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}
  $$
  tells us how much more or less likely the new policy makes each sampled action.
- The advantage $A^{\pi_\theta}(s_t,a_t)$ tells us whether that sampled action was better or worse than baseline.

So the update wants to increase the probability of actions with positive advantage and decrease the probability of actions with negative advantage.

## 4. Trust Region Constraint

Optimizing the surrogate alone is not enough. TRPO imposes a KL constraint:

$$
\max_{\theta'} \; L_\theta(\theta')
\quad \text{such that} \quad
\mathbb{E}_{s \sim \rho_{\pi_\theta}}
\left[D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid s) \;\|\; \pi_{\theta'}(\cdot \mid s)\big)\right]
\le \delta.
$$

This is the core idea of TRPO.

The KL term measures how far the new policy moves from the old one.
If the KL is small, the surrogate is a good approximation to the true improvement in return.

This is why TRPO is often described as a "monotonic improvement motivated" policy optimization method.

## 5. Natural Gradient Interpretation

TRPO solves the constrained problem by approximating both objective and constraint locally.

First-order approximation of the surrogate around current parameters:

$$
L_\theta(\theta + \Delta\theta)
\approx L_\theta(\theta) + g^\top \Delta\theta,
$$

where

$$
g = \nabla_\theta L_\theta(\theta).
$$

Second-order approximation of the KL constraint:

$$
\bar D_{\mathrm{KL}}(\theta, \theta + \Delta\theta)
\approx \frac{1}{2} \Delta\theta^\top F \Delta\theta,
$$

where $F$ is the Fisher information matrix (or equivalently the Hessian of the mean KL at the current policy).

So the local optimization becomes

$$
\max_{\Delta\theta} \; g^\top \Delta\theta
\quad \text{such that} \quad
\frac{1}{2}\Delta\theta^\top F \Delta\theta \le \delta.
$$

Its solution is the natural-gradient direction:

$$
\Delta\theta^* \propto F^{-1} g.
$$

More precisely,

$$
\Delta\theta^* = \sqrt{\frac{2\delta}{g^\top F^{-1} g}}\; F^{-1}g.
$$

This is the main difference from vanilla policy gradient.
Instead of following Euclidean geometry in parameter space, TRPO uses the geometry induced by the policy distribution.

## 6. Why Conjugate Gradient Appears

For neural networks, the Fisher matrix is too large to form explicitly.
TRPO therefore solves the linear system

$$
Fx = g
$$

approximately with conjugate gradient (CG).

The implementation never builds $F$ directly.
It only needs Fisher-vector products:

$$
Fv.
$$

These are computed efficiently by differentiating the mean KL twice.

In the code, this shows up as:

1. compute the mean KL between old and current policy,
2. differentiate KL once to get $\nabla_\theta D_{\mathrm{KL}}$,
3. take the inner product with a vector $v$,
4. differentiate again to obtain $Fv$.

This is the standard practical TRPO implementation pattern.

## 7. Line Search

Even after the local quadratic approximation, TRPO still performs backtracking line search.

Reason:

- the local model is only approximate,
- the KL estimate is local,
- the neural network objective is nonlinear.

So after computing the candidate step $\Delta\theta$, TRPO tests smaller fractions of that step until both conditions hold:

1. surrogate objective improves,
2. mean KL stays below `max_kl`.

This extra step is a big part of why TRPO is more conservative and usually more stable than naive policy gradient.

## 8. Critic and GAE

TRPO uses a value baseline $V_\phi(s)$ to reduce variance.
This implementation trains it by regression to empirical returns.

The one-step TD residual is

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t).
$$

Advantages are estimated with Generalized Advantage Estimation (GAE):

$$
\hat A_t = \delta_t + \gamma \lambda \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots
$$

or recursively,

$$
\hat A_t = \delta_t + \gamma \lambda (1-d_t) \hat A_{t+1}.
$$

Then the return target for the critic is

$$
\hat R_t = \hat A_t + V_\phi(s_t).
$$

This gives the standard TRPO recipe:

- policy update from trust-region optimization,
- value function update from ordinary supervised learning.

## 9. Policy Used in This Example

The actor is a diagonal Gaussian policy over a normalized action $u \in [-1, 1]$.

The environment force is

$$
a = F_{\max} \cdot u.
$$

So the policy learns the direction and relative magnitude of force in normalized space,
while the environment uses the actual physical force.

This keeps optimization numerically stable while preserving the same continuous-control problem as DDPG and SAC.

## 10. Study Case: CartPole Swing-up + Balance

This is the same study case used in the repo's DDPG and SAC folders.

### State

Raw state:

$$
s = [x, \dot x, \theta, \dot\theta].
$$

Neural network feature vector:

$$
\phi(s) = [x, \dot x, \sin\theta, \cos\theta, \dot\theta].
$$

Using $\sin\theta$ and $\cos\theta$ avoids the angle discontinuity at $\pm \pi$.

### Action

Continuous scalar control force applied to the cart.

### Reward shaping

Same shaping as DDPG and SAC:

$$
r(s) = 1 + \cos(\theta) - 0.1x^2 - 0.01(\dot x^2 + \dot\theta^2).
$$

Additional bonus near the upright region:

$$
\text{bonus} = 0.5 \quad \text{if } |\theta| < 0.25 \text{ and } |x| < 0.5.
$$

Interpretation:

- $\cos(\theta)$ rewards upright alignment,
- $x^2$ penalizes drifting away from center,
- velocity penalties discourage violent oscillation,
- the near-upright bonus helps stabilize the final balancing phase.

### Initial-state curriculum

Like the other continuous-control scripts in this repo, the code mixes two initial-state distributions:

- mostly swing-up starts near the downward configuration,
- every few episodes a near-upright start is sampled.

This helps the value baseline and policy learn both phases of the task:

1. getting the pole up,
2. keeping it balanced.

## 11. What the Script Produces

Outputs are saved into the local `outputs/` folder:

- `trpo_training_curves.png`
- `trpo_trajectory.png`
- `trpo_cartpole.gif`

The training-curves figure contains:

- average episode reward per iteration,
- surrogate objective,
- value loss,
- mean KL after each accepted update.

The trajectory plot shows:

- pole angle over time,
- cart position over time.

The GIF shows the final deterministic rollout using the policy mean.

## 12. How to Read the TRPO Outputs

### Average Episode Reward

This should trend upward over iterations if the trust-region step is helping.

### Surrogate Objective

This is the local quantity TRPO actually maximizes under the KL constraint.
It does not have to correlate perfectly with final return every iteration, but large persistent collapse is a warning sign.

### Value Loss

This tracks how well the critic fits return targets.
If it stays extremely large, advantage estimates will be noisy and the policy step becomes less reliable.

### Mean KL

This is the most TRPO-specific diagnostic.
It should usually remain near, but not much above, `max_kl`.

- If it is always tiny, updates may be too conservative.
- If it frequently overshoots, the step or damping settings are too aggressive.

## 13. Algorithm (Step-by-Step)

For each training iteration:

1. Collect a fresh on-policy batch with the current Gaussian policy.
2. Fit the value baseline $V_\phi$ and compute GAE advantages.
3. Build the surrogate
   $$
   L(\theta') = \mathbb{E}[r_t(\theta') \hat A_t].
   $$
4. Build the mean KL constraint.
5. Compute the natural-gradient step with conjugate gradient.
6. Backtrack along that direction until the surrogate improves and KL is below threshold.
7. Repeat with a new batch from the updated policy.

Notice what is missing compared with DDPG and SAC:

- no replay buffer,
- no target networks,
- no off-policy bootstrapping of the actor.

TRPO is explicitly on-policy.

## 14. Comparison With DDPG and SAC

This folder is deliberately comparable to the repo's other continuous-control examples.

### TRPO vs DDPG

- **TRPO**: stochastic on-policy policy optimization with KL trust region.
- **DDPG**: deterministic off-policy actor-critic with replay buffer and target networks.

TRPO is usually more conservative and theoretically motivated.
DDPG is usually more sample-efficient but can be more brittle.

### TRPO vs SAC

- **TRPO**: constrains policy movement using KL.
- **SAC**: encourages exploration by maximizing entropy.

TRPO uses a hard geometric notion of safe update.
SAC uses an entropy-regularized objective and learns from replay.

### Practical tradeoff summary

- **TRPO**: stable updates, expensive on-policy sampling, more math-heavy.
- **DDPG**: efficient data reuse, simpler deployment policy, can be unstable.
- **SAC**: strong performance and robustness, but more moving parts.

## 15. How to Run

```bash
cd /home/quangvd7/self_learning/Robot_learning/RL/TRPO
python trpo_cartpole_swingup_balance.py
```

Useful options:

```bash
python trpo_cartpole_swingup_balance.py --iterations 150 --steps-per-iter 4096
python trpo_cartpole_swingup_balance.py --max-kl 0.005
python trpo_cartpole_swingup_balance.py --skip-gif
```

## 16. Important Hyperparameters

- `--max-kl`: trust-region radius.
- `--cg-iters`: conjugate-gradient iterations.
- `--cg-damping`: numerical damping added to Fisher-vector products.
- `--gae-lambda`: bias-variance tradeoff for the advantage estimator.
- `--value-epochs`: how long to fit the critic after each policy batch.
- `--steps-per-iter`: on-policy batch size per iteration.

If training is too slow or too noisy, these are the first knobs to tune.

## 17. Takeaway

TRPO is best understood as:

- policy gradient,
- plus a value baseline and GAE,
- plus natural-gradient geometry,
- plus an explicit KL trust region.

That combination is what makes TRPO historically important: it turned policy optimization
from "take a gradient step and hope" into "take the largest local step that still looks safe under the policy geometry."