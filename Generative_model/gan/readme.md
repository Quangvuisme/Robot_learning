# GAN Examples (Imitation Learning)

This folder mirrors the diffusion/VAE studies but uses a conditional GAN to learn $p(a \mid s)$.

Current swing-up implementation follows a GAIL-style loop:
- Step 1: sample student trajectories from current policy
- Step 2: train discriminator to classify expert vs student state-action pairs
- Step 3: update student policy to fool discriminator

For stability/speed, we still keep a short BC warm start before GAIL updates.

## Files
- gan_inverted_pendulum_study.py: inverted pendulum balance (torque control)
- gan_cartpole_study.py: cart-pole balance (force control)
- gan_cartpole_swingup_balance_study.py: cart-pole swing-up + balance (force control)

## Outputs
- outputs/gan_training_curves_pendulum.png
- outputs/gan_theta_trajectory.png
- outputs/gan_pendulum_compare.gif
- outputs/gan_training_curves_cartpole.png
- outputs/gan_cartpole_trajectory.png
- outputs/gan_cartpole_compare.gif
- outputs/gan_training_curves_cartpole_swingup.csv
- outputs/gan_training_curves_cartpole_swingup.png
- outputs/gan_cartpole_swingup_trajectory.png
- outputs/gan_cartpole_swingup_compare.gif

## Run
```bash
python3 gan_inverted_pendulum_study.py
python3 gan_cartpole_study.py
python3 gan_cartpole_swingup_balance_study.py
```

Recommended swing-up command (GAIL-style):
```bash
python3 gan_cartpole_swingup_balance_study.py \
	--skip-expert-check --max-samples 40000 --epochs 600 \
	--rollout-episodes 12 --rollout-horizon 180 \
	--disc-updates 3 --gen-updates 2 \
	--bc-pretrain-steps 500 --bc-weight 0.2 --skip-bonus
```

## Notes
- Generator/Student: $a = G(s, z)$, Discriminator: $D(s, a)$
- Swing-up script uses feature map from diffusion (`state_features`) and the same expert/environment.
- The discriminator loss is BCE expert-vs-student; policy is optimized to fool the discriminator.

## How the GAN becomes a controller

The generator network *is* the controller. Everything else (discriminator, BCE loss) exists only
to shape the generator's weights during training; at deployment time only `G(s, z)` runs, once per
control step.

- **State** $s \in \mathbb{R}^4$ = $[x, \dot{x}, \theta, \dot{\theta}]$ for cart-pole (position,
  velocity, pole angle, angular velocity).
- **Noise** $z \in \mathbb{R}^8$, resampled from $\mathcal{N}(0, I)$ every call. This is what makes
  $G$ a *conditional distribution* $p(a \mid s)$ instead of a single deterministic regression
  $a = f(s)$ — the same state can map to a spread of plausible actions, not just their mean.
- **Action** $a \in \mathbb{R}$ = force applied to the cart.

```
generator:     G(s, z) = a           (2-layer MLP, gan_cartpole_study.py:35-48)
discriminator: D(s, a) = logit        (2-layer MLP, gan_cartpole_study.py:51-63)
```

### Objective (standard conditional GAN / GAIL-style)

$$
\min_G \max_D \; \mathbb{E}_{(s,a)\sim \text{expert}}\big[\log D(s,a)\big] +
\mathbb{E}_{s\sim \text{data},\, z\sim \mathcal{N}(0,I)}\big[\log(1 - D(s, G(s,z)))\big]
$$

In code this is implemented as two separate BCE-with-logits losses, updated in alternation
([gan_cartpole_study.py:264-292](gan_cartpole_study.py#L264-L292)):

1. **Discriminator step** — freeze $G$, update $D$ to separate real vs. fake:
   $$\mathcal{L}_D = \text{BCE}(D(s, a_{\text{real}}), 1) + \text{BCE}(D(s, G(s,z)), 0)$$
   (code uses soft label $0.9$ instead of $1$ for the real target — standard GAN trick to keep $D$
   from becoming overconfident and stalling the generator's gradient.)

2. **Generator step** — freeze $D$, update $G$ to flip the discriminator's verdict on its own
   samples:
   $$\mathcal{L}_G = \text{BCE}(D(s, G(s,z)), 1)$$
   $G$ never sees the expert action directly in this step — its only signal is "did $D$ believe
   this was real." Gradients flow back through $D$ (whose weights are *not* updated here) into
   $G$'s weights via the chain rule.

Both networks only ever see **normalized** states/actions ($z$-scored by dataset mean/std,
[gan_cartpole_study.py:243-249](gan_cartpole_study.py#L243-L249)); actions are un-normalized back
to physical force units afterward.

## Worked toy example (one training step by hand)

The real nets have hundreds of parameters; here's the same math with 1D state, 1D noise, 1D
action, and linear "networks" (no hidden layer, no activation) so every number can be traced.

**Toy nets**
$$G(s,z) = w^g_1 s + w^g_2 z, \qquad D(s,a) = w^d_1 s + w^d_2 a$$

**Init**: $w^g_1=0.10,\ w^g_2=0.05,\quad w^d_1=0.20,\ w^d_2=0.30$

**One expert sample** (from the dataset built by `base.generate_dataset`): pole tilted slightly
right, expert pushes right to correct it: $s = 0.2,\ a_{\text{real}} = 0.5$.

**1. Generator produces a fake action** — sample noise $z=1.0$:
$$a_{\text{fake}} = G(s,z) = 0.10(0.2) + 0.05(1.0) = 0.07$$

**2. Discriminator scores both** (logits, then sigmoid $\sigma$):
$$
\text{logit}_{\text{real}} = 0.20(0.2)+0.30(0.5)=0.19 \Rightarrow \sigma=0.547
$$
$$
\text{logit}_{\text{fake}} = 0.20(0.2)+0.30(0.07)=0.061 \Rightarrow \sigma=0.515
$$
$D$ is barely distinguishing the two yet ($0.547$ vs. $0.515$, should be $\to 1$ vs. $\to 0$) — the
loss $\mathcal{L}_D = -\log(0.547) - \log(1-0.515) \approx 1.33$ pushes $w^d_1, w^d_2$ to widen that
gap.

**3. Generator update — the interesting part.** With $D$ now frozen, backprop
$\mathcal{L}_G=\text{BCE}(\text{logit}_{\text{fake}}, 1)$ through $D$ into $G$:
$$
\frac{\partial \mathcal{L}_G}{\partial \text{logit}_{\text{fake}}} = \sigma(\text{logit}_{\text{fake}}) - 1 = 0.515 - 1 = -0.485
$$
$$
\frac{\partial \text{logit}_{\text{fake}}}{\partial a_{\text{fake}}} = w^d_2 = 0.30
\;\Rightarrow\;
\frac{\partial \mathcal{L}_G}{\partial a_{\text{fake}}} = -0.485 \times 0.30 = -0.146
$$
Chain further into $G$'s own weights ($a_{\text{fake}} = w^g_1 s + w^g_2 z$):
$$
\frac{\partial \mathcal{L}_G}{\partial w^g_1} = -0.146 \times s = -0.146\times0.2 = -0.029,
\qquad
\frac{\partial \mathcal{L}_G}{\partial w^g_2} = -0.146 \times z = -0.146\times1.0 = -0.146
$$
Gradient descent ($w \leftarrow w - \text{lr}\cdot\nabla$) with $\text{lr}=1.0$ for illustration:
$$
w^g_1 \leftarrow 0.10 - (-0.029) = 0.129, \qquad w^g_2 \leftarrow 0.05 - (-0.146) = 0.196
$$
Both weights *increased* — the generator nudges toward producing a **larger** action for this
$(s,z)$, i.e. closer to the real $a=0.5$ it's trying to fool $D$ into accepting. That is the entire
mechanism, repeated for millions of $(s,z)$ draws: $D$ keeps sharpening the real/fake boundary,
$G$ keeps climbing the gradient that pushes its outputs across that boundary, and the only stable
resting point is $G(s,z) \approx$ the expert's conditional action distribution.

## Inference: from a stochastic generator to a single force

Because $G$ takes random $z$, a single call is noisy. `gan_policy_action`
([gan_cartpole_study.py:209-227](gan_cartpole_study.py#L209-L227)) calls $G$ several times per
control step with independent $z$ draws and takes the **median**, then clamps to the actuator
limit:
$$
a_t = \text{clip}\Big(\text{median}\big(G(s_t, z^{(1)}), \dots, G(s_t, z^{(8..10)})\big),\; -F_{\max}, F_{\max}\Big)
$$
Median-of-samples is a cheap variance-reduction trick (robust to any single outlier sample) that
turns the *distribution* $p(a\mid s)$ the GAN learned back into the single deterministic force
the physics simulator needs each timestep.
