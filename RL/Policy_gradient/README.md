# Policy Gradient Notes (Lecture-Style)

This folder contains four policy gradient examples:

1) A 2-armed bandit (no state transitions)
2) A tiny MDP for Monte Carlo REINFORCE
3) Variance reduction with a baseline
4) Off-policy REINFORCE with importance sampling

The goal is to explain the math step-by-step like a lecture and connect each
formula to the code.

## 1. Big Picture

We want a policy $\pi_\theta(a\mid s)$ that directly maximizes expected return:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1} r(s_t, a_t)\right].
$$

We are not learning $Q$ first. We update the policy parameters $\theta$ so that
good actions become more likely.

## 2. Trajectory Distribution

The probability of a trajectory $\tau=(s_0,a_0,\dots,s_T)$ is

$$
p_\theta(\tau) = p(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t).
$$

Only the policy term depends on $\theta$.

## 3. Log-Derivative Trick (Core Step)

$$
\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\nabla_\theta \log p_\theta(\tau).
$$

Then:

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau\sim p_\theta(\tau)}\left[R(\tau)\nabla_\theta \log p_\theta(\tau)\right].
$$

And because only the policy depends on $\theta$:

$$
\log p_\theta(\tau) = \sum_{t=0}^{T-1} \log \pi_\theta(a_t\mid s_t) + \text{const}.
$$

So:

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau}\left[\sum_{t=0}^{T-1} R(\tau)\nabla_\theta \log \pi_\theta(a_t\mid s_t)\right].
$$

This is REINFORCE.

## 4. Monte Carlo Return $G_t$

Using total return $R(\tau)$ at every time step is noisy. Use the return from time $t$:

$$
G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r(s_k,a_k).
$$

Then the estimator becomes:

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau}\left[\sum_{t=0}^{T-1} G_t\nabla_\theta \log \pi_\theta(a_t\mid s_t)\right].
$$

**Intuition:** if an action leads to higher future return, increase its probability.

## 5. Baseline (Variance Reduction)

We can subtract any baseline $b$ without changing the expected gradient:

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau}\left[\sum_{t=0}^{T-1} (G_t-b)\nabla_\theta \log \pi_\theta(a_t\mid s_t)\right].
$$

In code we use an EMA baseline:

$$
b \leftarrow (1-\beta)b + \beta r.
$$

## 6. Bandit Case (No State Transitions)

Two actions $a \in \{0,1\}$, reward is Bernoulli:

$$
P(r=1\mid a=0)=p_0,\quad P(r=1\mid a=1)=p_1.
$$

$$
\pi_\theta(a) = \frac{\exp(\theta_a)}{\sum_k \exp(\theta_k)}.
$$

For softmax:

$$
\nabla_\theta \log \pi_\theta(a) = \mathbf{1}_a - \pi_\theta.
$$

Update for one step:

$$
\theta \leftarrow \theta + \alpha (r-b)(\mathbf{1}_a - \pi_\theta).
$$

### Why $\nabla_\theta \log \pi_\theta(a) = \mathbf{1}_a - \pi_\theta$ (Chain Rule)

Pipeline:

$$
\theta \rightarrow \text{softmax} \rightarrow \pi \rightarrow \log \rightarrow \log \pi(a)
$$

Write the log-softmax explicitly:

$$
\log \pi_\theta(a) = \theta_a - \log \sum_k \exp(\theta_k).
$$

Differentiate term-by-term.

**Term 1:**

$$
\nabla_\theta \theta_a = \mathbf{1}_a.
$$

**Term 2:**

$$
\nabla_\theta \log \sum_k \exp(\theta_k)
= \frac{1}{\sum_k \exp(\theta_k)}\nabla_\theta \sum_k \exp(\theta_k)
= \frac{\exp(\theta_i)}{\sum_k \exp(\theta_k)} = \pi_\theta(i).
$$

So for the full vector gradient:

$$
\nabla_\theta \log \pi_\theta(a) = \mathbf{1}_a - \pi_\theta.
$$

Interpretation: **one-hot (what happened) minus expectation (what the policy predicted)**.

## 7. Worked Example (Numbers, Step-by-Step)

**Setup:** two actions with reward probabilities

$$
P(r=1\mid a=0)=p_0=0.3,\quad P(r=1\mid a=1)=p_1=0.8.
$$

Policy parameters and hyperparameters:

$$
\theta = [0,0],\quad \alpha=0.1,\quad b=0.5.
$$

Softmax policy at the start:

$$
\pi = \text{softmax}(\theta) = [0.5,0.5].
$$

### Iteration 1

Sample action (assume we picked action 1):

$$
a=1.
$$

Sample reward (assume success):

$$
r=1.
$$

Compute gradient terms:

$$
\mathbf{1}_a=[0,1],\quad \mathbf{1}_a-\pi=[-0.5,+0.5].
$$

Advantage:

$$
r-b = 1-0.5 = 0.5.
$$

Update:

$$
\Delta\theta = \alpha (r-b)(\mathbf{1}_a-\pi)
= 0.1\times0.5\times[-0.5,+0.5]
= [-0.025,+0.025].
$$

New parameters and policy:

$$
\theta=[-0.025,+0.025],\quad \pi\approx[0.487,0.513].
$$

So the probability of action 1 increases.

### Iteration 2

Sample action (assume we picked action 0):

$$
a=0.
$$

Sample reward (assume failure):

$$
r=0.
$$

Compute gradient terms:

$$
\mathbf{1}_a=[1,0],\quad \mathbf{1}_a-\pi=[0.513,-0.513].
$$

Advantage:

$$
r-b = 0-0.5 = -0.5.
$$

Update:

$$
\Delta\theta = 0.1\times(-0.5)\times[0.513,-0.513]
=[-0.02565,+0.02565].
$$

Even though action 0 was chosen, it gave low reward, so the update still
pushes probability toward action 1.

## 8. Example Scripts

- `01_policy_gradient_bandit.py`
- `02_policy_gradient_monte_carlo.py`
- `03_reducing_variance_baseline.py`
- `04_off_policy_policy_gradient.py`

Run the bandit:

```bash
cd /home/quangvd7/self_learning/Robot_learning/RL/Policy_gradient
python 01_policy_gradient_bandit.py
```

Run Monte Carlo REINFORCE:

```bash
cd /home/quangvd7/self_learning/Robot_learning/RL/Policy_gradient
python 02_policy_gradient_monte_carlo.py
```

Run baseline variance demo:

```bash
cd /home/quangvd7/self_learning/Robot_learning/RL/Policy_gradient
python 03_reducing_variance_baseline.py
```

Run off-policy policy gradient:

```bash
cd /home/quangvd7/self_learning/Robot_learning/RL/Policy_gradient
python 04_off_policy_policy_gradient.py
```

## 9. Monte Carlo Example (Tiny MDP)

We use a 1D random-walk MDP with terminal rewards. States are $\{0,1,2,3,4\}$.
Episode starts at state 2. Actions: left (0) or right (1).
Terminal states: 0 gives reward -1, 4 gives reward +1.

### Monte Carlo Intuition (Why It Works)

Monte Carlo REINFORCE waits until the episode ends, then uses the full return
$G_t$ to estimate how good each action was. This estimator is **unbiased**
because it uses the actual sampled return from the environment, but the
variance can be high because a single episode can be noisy.

Compared with TD methods:

- **Monte Carlo:** uses $G_t$ from the episode, no bootstrapping.
- **TD:** uses a one-step target like $r + \gamma V(s')$, lower variance but
	introduces bias.

In policy gradient, the gradient estimate scales by $G_t$ so large positive or
negative returns cause big updates. The baseline $b$ reduces this variance but
does not change the expected gradient.

### Step-by-Step: One Trajectory

Assume we sample a trajectory:

$$
(s_0,a_0,r_0)=(2,1,0),\quad (s_1,a_1,r_1)=(3,1,+1)\quad \text{and terminate at } s_2=4.
$$

So rewards are $[0, +1]$ and the episode ends.

**Compute returns:**

$$
G_1 = r_1 = +1,
$$

$$
G_0 = r_0 + \gamma r_1 = 0 + \gamma\cdot 1.
$$

If $\gamma=0.99$, then $G_0=0.99$.

### Update Rule (Monte Carlo REINFORCE)

For each time step in the episode:

$$
\theta \leftarrow \theta + \alpha (G_t-b)\nabla_\theta \log \pi_\theta(a_t\mid s_t).
$$

This means:

1) At $t=1$, action $a_1=1$ gets reinforced by $G_1=1$.
2) At $t=0$, action $a_0=1$ gets reinforced by $G_0=0.99$.

So both actions in the trajectory increase their probability because they led
to a terminal reward of +1.

### Why Monte Carlo?

We wait until the episode ends, then use the full return $G_t$ to update.
This gives an unbiased gradient estimate, but it can be noisy because the
whole update depends on the sampled trajectory.

### Worked Case Study (2-Step Episode)

**Setup:** start at state 2, $\gamma=0.99$, baseline $b=0$ for simplicity.

Trajectory:

$$
(s_0,a_0,r_0)=(2,1,0),\quad (s_1,a_1,r_1)=(3,1,+1),\quad \text{terminate at } s_2=4.
$$

Returns:

$$
G_1 = r_1 = 1,\quad G_0 = r_0 + \gamma r_1 = 0 + 0.99\cdot 1 = 0.99.
$$

Policy gradient updates (softmax policy):

$$
	heta_{s_t} \leftarrow \theta_{s_t} + \alpha G_t\,\nabla_{\theta_{s_t}}\log \pi_\theta(a_t\mid s_t).
$$

So action $a_1=1$ at state 3 gets a stronger push (scale 1.0), while action
$a_0=1$ at state 2 gets a slightly smaller push (scale 0.99). Both moves are
reinforced because the episode ended with a positive terminal reward.

### What the Script Plots

- Return moving average
- Policy probability of moving right from the center state

## 10. Reducing Variance with a Baseline

We can subtract any baseline $b$ that does not depend on the action $a$.
This keeps the gradient **unbiased** but reduces variance:

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau}\left[\sum_{t=0}^{T-1} (G_t-b)\nabla_\theta \log \pi_\theta(a_t\mid s_t)\right].
$$

In the bandit example, the baseline can be the expected reward:

$$
b = \mathbb{E}[r] = \sum_a \pi_\theta(a)\,p(r=1\mid a).
$$

The script `03_reducing_variance_baseline.py` measures the sample variance of
the gradient with and without the baseline and shows that variance drops.

Plot note: the histogram shows a few discrete spikes because both action and
reward are binary in the bandit. The baseline keeps the mean similar but
shrinks the magnitude, so the distribution is tighter.

## 11. Off-Policy Policy Gradient (Importance Sampling)

We want to optimize a target policy $\pi_\theta$ but data comes from a behavior
policy $\mu$. Use importance sampling:

$$
\rho(a) = \frac{\pi_\theta(a)}{\mu(a)}.
$$

For the bandit (one-step trajectory), the REINFORCE gradient becomes:

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{a\sim\mu}\left[\rho(a)\,r(a)\nabla_\theta \log \pi_\theta(a)\right].
$$

Large $\rho$ can cause high variance, so we often clip $\rho$ in practice.
The script `04_off_policy_policy_gradient.py` shows how the target policy
still moves toward the better action even when samples come from $\mu$.

### Deeper Math (Trajectory Importance Sampling)

We want $J(\theta')$ but sample trajectories from $p_\theta(\tau)$:

$$
J(\theta') = \mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\frac{p_{\theta'}(\tau)}{p_\theta(\tau)} R(\tau)\right].
$$

Take the gradient and use the log-derivative trick:

$$
\nabla_{\theta'} J(\theta')
= \mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\frac{p_{\theta'}(\tau)}{p_\theta(\tau)}\nabla_{\theta'} \log p_{\theta'}(\tau)\,R(\tau)\right].
$$

Because dynamics cancel, the likelihood ratio is a product of policy ratios:

$$
\frac{p_{\theta'}(\tau)}{p_\theta(\tau)}
= \prod_{t=0}^{T-1}\frac{\pi_{\theta'}(a_t\mid s_t)}{\pi_{\theta}(a_t\mid s_t)}.
$$

So the off-policy policy gradient becomes:

$$
\nabla_{\theta'} J(\theta')
= \mathbb{E}_{\tau\sim p_\theta(\tau)}\left[
\left(\prod_{t=0}^{T-1}\frac{\pi_{\theta'}(a_t\mid s_t)}{\pi_{\theta}(a_t\mid s_t)}\right)
\left(\sum_{t=0}^{T-1}\nabla_{\theta'}\log \pi_{\theta'}(a_t\mid s_t)\right)R(\tau)
\right].
$$

### Causality (Reward-to-Go)

Replace $R(\tau)$ with reward-to-go $R_t$:

$$
R_t = \sum_{k=t}^{T-1} r(s_k,a_k).
$$

Then:

$$
\nabla_{\theta'} J(\theta')
= \mathbb{E}_{\tau\sim p_\theta(\tau)}\left[
\sum_{t=0}^{T-1}\nabla_{\theta'}\log \pi_{\theta'}(a_t\mid s_t)
\left(\prod_{i=0}^{t}\frac{\pi_{\theta'}(a_i\mid s_i)}{\pi_{\theta}(a_i\mid s_i)}\right)R_t
\right].
$$

The product of ratios can grow exponentially with horizon, causing high variance.
That is why practical methods clip or normalize importance weights.

### Worked Study Case (2-step episode)

**Task:** two actions LEFT (0) and RIGHT (1). Episode length $T=2$.
Reward rule: only RIGHT-RIGHT yields +1, otherwise 0.

Behavior policy (data source):

$$
\pi_\theta = [0.8, 0.2]\quad \text{(LEFT more likely)}
$$

Target policy (we want to optimize):

$$
\pi_{\theta'} = [0.2, 0.8]\quad \text{(RIGHT more likely)}
$$

**Trajectory A:** LEFT, RIGHT (reward 0)

$$
p_{\theta}(\tau)=0.8\cdot0.2=0.16,\quad
p_{\theta'}(\tau)=0.2\cdot0.8=0.16,
$$

$$
\rho(\tau)=\frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)}=1.
$$

No amplification.

**Trajectory B (rare under behavior):** RIGHT, RIGHT (reward +1)

$$
p_{\theta}(\tau)=0.2\cdot0.2=0.04,\quad
p_{\theta'}(\tau)=0.8\cdot0.8=0.64,
$$

$$
\rho(\tau)=\frac{0.64}{0.04}=16.
$$

So a single rare trajectory can multiply the gradient by 16.

**Why this explodes with horizon:**

$$
\rho(\tau)=\prod_{t=0}^{T-1}\frac{\pi_{\theta'}(a_t\mid s_t)}{\pi_\theta(a_t\mid s_t)}.
$$

If each step ratio is about 4 and $T=10$:

$$
\rho \approx 4^{10} = 1{,}048{,}576.
$$

**Causality (reward-to-go)** reduces noise in the reward term, but does not
fix the exploding importance ratio. That is why off-policy PG is high-variance
unless we clip or normalize $\rho$.

## 12. Algorithm Summary (Lecture-Style)

REINFORCE (episodic Monte Carlo):

1) Run $\pi_\theta$ to collect $(s_t,a_t,r_t)$.
2) Compute $G_t$ for each time step.
3) Update:

$$
\theta \leftarrow \theta + \alpha \sum_t (G_t-b)\nabla_\theta \log \pi_\theta(a_t\mid s_t).
$$

## 13. Next Step

To reduce variance further, replace $G_t$ with an estimated advantage
$A_t = Q(s_t,a_t) - V(s_t)$ (actor-critic). This is the bridge from
REINFORCE to modern policy gradient algorithms.
