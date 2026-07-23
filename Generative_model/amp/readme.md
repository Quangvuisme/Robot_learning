# AMP (Adversarial Motion Priors)

This folder is the RL-based sibling of [gan/gan_cartpole_swingup_balance_study.py](../gan/gan_cartpole_swingup_balance_study.py).
Both use an adversarial discriminator to define "looks like the expert." They differ in
**how the policy is optimized against that discriminator**:

| | GAN/GAIL (`gan/`) | AMP (`amp/`) |
|---|---|---|
| Policy | conditional generator $a=G(s,z)$ | stochastic Gaussian actor $\pi(a\mid s)$ |
| Discriminator sees | $(s, a)$ | $(s, s')$ — a **transition**, no action needed |
| Policy optimized by | backprop **through** $D$ into $G$ (assumes $G$ is a differentiable "environment" in the adversarial game) | **PPO** (policy gradient) using a reward derived from $D$ — never backprops through the physics step |
| Why the difference matters | works because nothing non-differentiable sits between $G$'s output and $D$'s input | required whenever the real environment step is non-differentiable (true for almost every physics simulator / real robot) |
| Discriminator loss | vanilla BCE | least-squares GAN (LSGAN) + gradient penalty |

AMP = Adversarial Motion Priors (Peng, Ma, Abbeel, Levine, Kanazawa — SIGGRAPH 2021). Originally
used to imitate the *style* of mocap clips for physics-based character animation without needing
per-frame action labels (mocap only gives you a sequence of poses, i.e. states). The same trick
works for the cart-pole: the discriminator learns what an expert (state $\to$ next state) step looks
like, and the policy is rewarded for producing transitions the discriminator can't tell apart from
the expert's.

## Files
- `amp_cartpole_swingup_balance_study.py`: cart-pole swing-up + balance, PPO policy + AMP-style
  transition discriminator.

## Outputs
- `outputs/amp_training_curves_cartpole_swingup.csv` / `.png`
- `outputs/amp_cartpole_swingup_trajectory.png`
- `outputs/amp_cartpole_swingup_compare.gif`

## Run
```bash
python3 amp_cartpole_swingup_balance_study.py
```

Quick smoke test (few seconds, not meant to converge):
```bash
python3 amp_cartpole_swingup_balance_study.py \
  --iterations 5 --steps-per-iter 1024 --horizon 100 \
  --expert-episodes 20 --expert-horizon 100 --skip-gif --skip-bonus
```

## How it works

**The controller is the `GaussianPolicy`**, trained entirely by PPO
([amp_cartpole_swingup_balance_study.py:341-386](amp_cartpole_swingup_balance_study.py#L341-L386),
copied mechanics from [RL/PPO/ppo_cartpole_swingup_balance.py](../../RL/PPO/ppo_cartpole_swingup_balance.py)).
The discriminator's only job each iteration is to hand out a reward number; it is frozen while the
policy is updated, and the policy is frozen while the discriminator is updated. Three things happen
per training iteration, in order:

**1. Rollout** ([amp_cartpole_swingup_balance_study.py:244-323](amp_cartpole_swingup_balance_study.py#L244-L323)) —
run the current policy in the environment. At every step, score the transition with the *current*
discriminator and turn that score into a style reward:
$$
r_{\text{style}}(s,s') = \operatorname{clip}\!\Big(1 - \tfrac14\big(D_\phi(s,s') - 1\big)^2,\; 0,\; 1\Big)
$$
This is the AMP least-squares reward: $D=1$ (discriminator fully convinced it's expert-like) gives
reward $1$; any deviation, in *either* direction, reduces it. The step reward fed to PPO is
$$
r_t = w_{\text{task}}\, r_{\text{task}}(s_{t+1}) + w_{\text{style}}\, r_{\text{style}}(s_t, s_{t+1}),
$$
combining a small hand-written task shaping term ($w_{\text{task}}=0.3$ by default — keep upright,
stay centered) with the learned style term ($w_{\text{style}}=0.7$ — most of the "how to move"
signal comes from the discriminator, not from hand tuning). GAE/advantages are computed from $r_t$
exactly as in vanilla PPO.

**2. Discriminator update**
([amp_cartpole_swingup_balance_study.py:326-361](amp_cartpole_swingup_balance_study.py#L326-L361)) —
using this iteration's just-collected transitions as "fake" and sampled expert transitions as
"real," minimize the LSGAN loss with target $+1$ for expert / $-1$ for policy:
$$
\mathcal{L}_D = \mathbb{E}_{(s,s')\sim\text{expert}}\big[(D_\phi(s,s')-1)^2\big]
+ \mathbb{E}_{(s,s')\sim\pi}\big[(D_\phi(s,s')+1)^2\big]
+ w_{gp}\,\mathbb{E}_{(s,s')\sim\text{expert}}\big[\|\nabla_\phi D_\phi(s,s')\|_2^2\big]
$$
The gradient-penalty term (computed only on expert samples) is a stabilizer specific to AMP: it
keeps $D$ from developing sharp, brittle decision boundaries that would turn into a noisy, spiky
reward signal for the policy.

**3. Policy update** — standard clipped-PPO on the batch collected in step 1, using the reward
that already baked in the (now slightly stale) discriminator's opinion. This staleness is
intentional and matches published AMP implementations: scoring happens with the pre-update $D$,
so step 2 and step 3 can run independently without one invalidating the other's data.

## Why not just backprop through $D$ like the GAN script does?

The GAN script's generator update needs $\dfrac{\partial \mathcal{L}}{\partial a} \cdot
\dfrac{\partial a}{\partial \theta_G}$ — that's fine because $a=G_\theta(s,z)$ is one differentiable
function. AMP's actions instead pass through `env.step` (a real physics simulator) before the
*next* state reaches $D$. `env.step` here happens to be differentiable (it's hand-written PyTorch
autograd code), but real robots and most physics engines are not, so AMP treats it as a black box:
sample actions, observe a scalar reward, use policy gradient (PPO) to increase the probability of
actions that led to high reward. This is strictly more general than the GAN script's approach at
the cost of needing many more environment interactions to converge (PPO's variance is much higher
than a clean discriminator gradient).

## Worked toy example: turning a discriminator score into a policy gradient

Continuing the toy numbers from [gan/readme.md](../gan/readme.md) (1D state, linear "networks", no
hidden layer) — but now show the **AMP** half: how a discriminator score becomes a PPO update
instead of a backprop target for the generator.

**Toy discriminator** (LSGAN form): $D(s,s') = w^d_1 s + w^d_2 s'$, with $w^d_1=0.2,\ w^d_2=0.3$.

**One transition the current policy just produced**: $s=0.2,\ s'=0.26$ (policy pushed the cart
right, pole moved a bit further from upright than the expert would allow).
$$
D(s,s') = 0.2(0.2) + 0.3(0.26) = 0.118
$$
**Style reward** (target is $D=1$):
$$
r_{\text{style}} = \operatorname{clip}\!\big(1 - 0.25(0.118-1)^2,\,0,\,1\big) = \operatorname{clip}(1 - 0.25 \times 0.778,\,0,\,1) = 0.806
$$
Suppose task reward for this step is $r_{\text{task}}=0.10$. With $w_{\text{task}}=0.3,\
w_{\text{style}}=0.7$:
$$
r_t = 0.3(0.10) + 0.7(0.806) = 0.594
$$
**This number — not a gradient — is all that reaches the policy.** It gets combined with the value
baseline into an advantage $A_t$ (via GAE, using $V(s)$ from the critic), and PPO's clipped
surrogate objective
$$
\mathcal{L}_{\text{PPO}} = -\min\!\Big(\rho_t A_t,\ \operatorname{clip}(\rho_t, 1{-}\epsilon, 1{+}\epsilon)\,A_t\Big),
\qquad \rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$
pushes up $\log \pi_\theta(a_t\mid s_t)$ if $A_t>0$ (this action was better than the critic expected)
and down if $A_t<0$ — using only the scalar reward, the policy's own action-probability, and a
value estimate. No gradient ever flows from $D$ into $\pi_\theta$; contrast this with the GAN
readme's worked example, where $\dfrac{\partial \mathcal{L}}{\partial a_{\text{fake}}}$ flowed
directly out of $D$ and into $G$'s weights.

**Meanwhile, the discriminator is trained separately** on this same transition (as a "fake"
target $-1$) plus a batch of real expert transitions (target $+1$):
$$
\mathcal{L}_D \ni (D(s,s') - (-1))^2 = (0.118 + 1)^2 = 1.249
$$
Gradient descent on $w^d_1, w^d_2$ pushes $D(s,s')$ down toward $-1$ for this specific transition —
so if the policy keeps producing similar transitions, next iteration's $D$ score (and hence
$r_{\text{style}}$) will be lower, closing the loop: the discriminator keeps raising the bar for
what counts as "expert-like," and PPO keeps hill-climbing on whatever bar is currently set.
