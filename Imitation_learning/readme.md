# Imitation Learning Notes: From Basics to Generative Policies

This document is a lecture-style math note for the imitation learning methods implemented in this project:

- Behavioral Cloning (BC) and DAgger (basic_examples)
- Conditional Diffusion (diffusion)
- Conditional VAE (vae)
- Conditional GAN (gan)

The goal is to imitate expert demonstrations for control tasks (pendulum and cart-pole variants).

## 1. Problem Setup

We are given expert trajectories:

$$
\mathcal{D} = \{(s_t, a_t)\}_{t=1}^{N}
$$

where $s_t$ is the state and $a_t$ is the expert action. The target is a policy distribution

$$
\pi_\theta(a \mid s) \approx p_E(a \mid s),
$$

and rollouts follow known dynamics

$$
s_{t+1} = f(s_t, a_t).
$$

## 2. Baseline: Behavioral Cloning (BC)

BC fits a direct regressor $a = g_\theta(s)$ via supervised learning:

$$
\mathcal{L}_{\text{BC}}(\theta) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\left[\|g_\theta(s) - a\|_2^2\right].
$$

It is fast but suffers from covariate shift because it only sees expert states.

## 3. DAgger (Dataset Aggregation)

DAgger fixes covariate shift by iteratively rolling out the current policy and querying the expert:

$$
\mathcal{D}_{k+1} = \mathcal{D}_k \cup \{(s, a_E) \mid s \sim \pi_{\theta_k}\}.
$$

Training still uses a BC-style loss, but on the aggregated dataset. This makes the learner robust
to its own state distribution.

## 4. Why Probabilistic Policies?

Deterministic regression can be brittle for nonlinear control. A conditional distribution can represent
multi-modal actions, uncertainty, and improve robustness through sampling and aggregation.

## 5. Method A: Conditional Diffusion Policy

### 5.1 Forward noising process

Diffusion defines a latent chain over actions:

$$
q(a_t \mid a_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}a_0, (1-\bar\alpha_t)I\big),
$$

with schedule $\beta_t \in (0,1)$, $\alpha_t = 1-\beta_t$, and

$$
\bar\alpha_t = \prod_{i=1}^{t} \alpha_i.
$$

Equivalent reparameterization:

$$
a_t = \sqrt{\bar\alpha_t}a_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0,I).
$$

### 5.2 Reverse denoising model

Train a neural network $\epsilon_\theta(a_t, s, t)$ to predict the noise:

$$
\mathcal{L}_{\text{diff}}(\theta)
= \mathbb{E}_{(s,a_0), t, \epsilon}
\left[\left\|\epsilon - \epsilon_\theta(a_t, s, t)\right\|_2^2\right].
$$

### 5.3 Inference for action selection

Given state $s$, start from Gaussian noise and iteratively denoise to get $\hat a_0$.
In practice, robustness is improved by lower sampling noise, multiple samples, median aggregation,
and hybrid switching (swing-up uses analytic balance near upright).

## 6. Method B: Conditional VAE Policy

### 6.1 Latent-variable model

$$
p_\theta(a \mid s) = \int p_\theta(a \mid s, z)p(z)\,dz,\quad p(z)=\mathcal{N}(0,I).
$$

Approximate posterior and decoder:

$$
q_\phi(z \mid s, a) = \mathcal{N}(\mu_\phi(s,a), \operatorname{diag}(\sigma_\phi^2(s,a))),
$$

$$
p_\theta(a \mid s, z).
$$

### 6.2 ELBO objective

$$
\mathcal{L}_{\text{vae}}(\theta,\phi)
= \mathbb{E}_{q_\phi(z\mid s,a)}[-\log p_\theta(a\mid s,z)]
+ \beta\,D_{\mathrm{KL}}\big(q_\phi(z\mid s,a)\,\|\,p(z)\big).
$$

In code this appears as reconstruction MSE + KL regularization, with a scalar weight $\beta$.

### 6.3 Inference note

Inference-time latent handling is crucial. Deterministic latents (using $z=\mu$) often stabilize
swing-up more than stochastic sampling.

### 6.4 VAE with action chunking (ACT-style)

For long-horizon control, predicting a *sequence* of actions at once can reduce compounding error.
Let an action chunk of length $K$ be:

$$
a_{t:t+K-1} = (a_t, a_{t+1}, \dots, a_{t+K-1}).
$$

An ACT-style conditional VAE models a distribution over chunks:

$$
p_\theta(a_{t:t+K-1} \mid s_t)
= \int p_\theta(a_{t:t+K-1} \mid s_t, z)\,p(z)\,dz,\quad p(z)=\mathcal{N}(0,I).
$$

One common choice of approximate posterior is:

$$
q_\phi(z \mid s_t, a_{t:t+K-1}).
$$

In the transformer version, the encoder consumes $(s_t, a_{t:t+K-1})$ as a token sequence (often
with a special pooled token), and the decoder predicts the whole chunk in parallel.

The (chunk) ELBO is:

$$
\mathcal{L}_{\text{chunk-vae}}(\theta,\phi)
= \mathbb{E}_{q_\phi(z\mid s_t,a_{t:t+K-1})}\left[-\log p_\theta(a_{t:t+K-1}\mid s_t,z)\right]
+ \beta\,D_{\mathrm{KL}}\big(q_\phi(z\mid s_t,a_{t:t+K-1})\,\|\,p(z)\big).
$$

### 6.5 Temporal ensembling for chunks

At inference, chunks overlap in time. If we predict chunks starting at times $t, t+\Delta, t+2\Delta, \dots$,
each step $\tau$ may be covered by multiple chunk predictions. A simple temporal ensemble is a
decayed weighted average:

$$
\hat a_\tau
= \frac{\sum_{i: \tau \in [t_i, t_i+K-1]} w(\tau-t_i)\,\hat a_\tau^{(i)}}{\sum_{i: \tau \in [t_i, t_i+K-1]} w(\tau-t_i)},
\quad w(d)=\gamma^d,\ \gamma\in(0,1].
$$

This smooths action selection and often improves stability.

## 7. Method C: Conditional GAN Policy (GAIL-style)

GAN-style imitation in this project follows the GAIL loop:

1. Roll out trajectories from current student policy.
2. Train discriminator to classify expert vs student state-action pairs.
3. Update student policy to fool the discriminator.

Model form:

$$
a = G_\theta(s, z),\quad D_\phi(s, a) \in (0,1).
$$

Discriminator objective:

$$
\mathcal{L}_D = -\mathbb{E}_{(s,a)\sim\mathcal{D}_E}[\log D_\phi(s,a)]
-\mathbb{E}_{(s,a)\sim\mathcal{D}_\pi}[\log(1 - D_\phi(s,a))].
$$

Policy objective (fool discriminator), with optional small BC regularizer:

$$
\mathcal{L}_\pi = -\mathbb{E}_{(s,a)\sim\mathcal{D}_\pi}[\log D_\phi(s,a)]
+ \lambda_{bc}\,\|G_\theta(s,0)-a_E\|_2^2.
$$

## 8. State Features and Representation

For angular systems, raw angles are discontinuous at $\pm\pi$. A smoother representation is:

$$
[\sin\theta, \cos\theta].
$$

Swing-up policies use this feature map to reduce wrap-around issues.

## 9. Evaluation Metrics

All tasks use tail-window stability metrics:

- mean tail absolute angle error,
- mean tail cart position error (for cart-pole),
- success rate thresholding tail errors and validity constraints.

Conceptually:

$$
	ext{success} = \mathbf{1}[\bar e_\theta < \tau_\theta \land \bar e_x < \tau_x \land \text{valid rollout}].
$$

## 10. Learning Path: Basic to Advanced

### 10.1 Basic teaching examples

- [Imitation_learning/basic_examples/readme.md](basic_examples/readme.md)
- [Imitation_learning/basic_examples/supervised_learning_basic.py](basic_examples/supervised_learning_basic.py)
- [Imitation_learning/basic_examples/dagger_simple.py](basic_examples/dagger_simple.py)
- [Imitation_learning/basic_examples/dagger_double_integrator.py](basic_examples/dagger_double_integrator.py)
- [Imitation_learning/basic_examples/privileged_teacher_swingup_pendulum.py](basic_examples/privileged_teacher_swingup_pendulum.py)

Quick runs from Imitation_learning:

```bash
python3 basic_examples/supervised_learning_basic.py
python3 basic_examples/dagger_simple.py
python3 basic_examples/dagger_double_integrator.py
python3 basic_examples/privileged_teacher_swingup_pendulum.py
```

### 10.2 Balance control studies

- Diffusion: [Imitation_learning/diffusion/diffusion_inverted_pendulum_study.py](diffusion/diffusion_inverted_pendulum_study.py)
- Diffusion: [Imitation_learning/diffusion/diffusion_cartpole_study.py](diffusion/diffusion_cartpole_study.py)
- VAE: [Imitation_learning/vae/vae_inverted_pendulum_study.py](vae/vae_inverted_pendulum_study.py)
- VAE: [Imitation_learning/vae/vae_cartpole_study.py](vae/vae_cartpole_study.py)
- GAN: [Imitation_learning/gan/gan_inverted_pendulum_study.py](gan/gan_inverted_pendulum_study.py)
- GAN: [Imitation_learning/gan/gan_cartpole_study.py](gan/gan_cartpole_study.py)

### 10.3 Swing-up + balance studies

- Diffusion: [Imitation_learning/diffusion/diffusion_cartpole_swingup_balance_study.py](diffusion/diffusion_cartpole_swingup_balance_study.py)
- VAE: [Imitation_learning/vae/vae_cartpole_swingup_balance_study.py](vae/vae_cartpole_swingup_balance_study.py)
- VAE (ACT-style transformer CVAE): [Imitation_learning/vae/act_cartpole_swingup_balance_study.py](vae/act_cartpole_swingup_balance_study.py)
- GAN: [Imitation_learning/gan/gan_cartpole_swingup_balance_study.py](gan/gan_cartpole_swingup_balance_study.py)

## 11. Folder Index

- Diffusion implementations: [Imitation_learning/diffusion/readme.md](diffusion/readme.md)
- VAE implementations: [Imitation_learning/vae/readme.md](vae/readme.md)
- GAN implementations: [Imitation_learning/gan/readme.md](gan/readme.md)
