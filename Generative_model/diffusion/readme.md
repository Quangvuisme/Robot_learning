
# Diffusion policy studies (pendulum + cart-pole)

This folder contains 3 main scripts that learn a policy with conditional diffusion from expert demonstrations.

## 1) Inverted pendulum (upright balance)

File: `diffusion_inverted_pendulum_study.py`

Goal: keep the pendulum near upright (theta ~ 0).

Quick run (sanity):

```bash
python3 diffusion_inverted_pendulum_study.py --epochs 800 --episodes 300 --seed 123
```

More thorough run:

```bash
python3 diffusion_inverted_pendulum_study.py --epochs 2000 --episodes 600 --seed 123
```

## 2) Cart-pole (balance-only quanh upright)

File: `diffusion_cartpole_study.py`

Goal: start near upright and keep balancing. This is easier than swing-up because it does not require injecting energy to lift the pole.

Quick run:

```bash
python3 diffusion_cartpole_study.py --epochs 1200 --episodes 500 --seed 123
```

More thorough run:

```bash
python3 diffusion_cartpole_study.py --epochs 2500 --episodes 900 --seed 123
```

## 3) Cart-pole (swing-up + balance) — two parameterizations

Both scripts below solve the exact same problem (same env, same expert, same
`ConditionalDenoiser` architecture, same noise schedule) and differ only in **what the network is
trained to output**:

- `diffusion_cartpole_swingup_balance_study.py` — **epsilon-prediction**: network outputs the
  noise $\epsilon$ that was added to the expert action.
- `diffusion_cartpole_swingup_balance_study_x0.py` — **x0-prediction (predict-action)**: network
  outputs the clean expert action $a_0$ directly.

They're mathematically equivalent parameterizations of the same diffusion model — given $a_t$,
$\epsilon$ and $a_0$ determine each other exactly:
$$
a_0 = \frac{a_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}
\qquad\Longleftrightarrow\qquad
\epsilon = \frac{a_t - \sqrt{\bar\alpha_t}\,a_0}{\sqrt{1-\bar\alpha_t}}
$$
so the x0-prediction script's `sample_action` derives an equivalent $\epsilon$ from its predicted
$a_0$ whenever it needs the epsilon-based DDIM update
([diffusion_cartpole_swingup_balance_study_x0.py:116](diffusion_cartpole_swingup_balance_study_x0.py#L116)),
and its ancestral (non-strided) sampler instead uses the x0-based posterior mean of
$q(a_{t-1}\mid a_t, a_0)$ directly
([:93-95](diffusion_cartpole_swingup_balance_study_x0.py#L93-L95)):
$$
\mu_{t-1} = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,a_0
+ \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,a_t,
\qquad
\sigma_{t-1}^2 = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t.
$$

**Why bother with two?** In practice the two parameterizations can have different training
dynamics for the same architecture/schedule:
- **epsilon-prediction** supervises on a zero-mean, unit-variance target ($\epsilon\sim\mathcal
  N(0,I)$) at *every* noise level — a very well-conditioned regression target, but at low noise
  ($t\to0$, $a_t\approx a_0$) the network has to divide out a near-zero
  $\sqrt{1-\bar\alpha_t}\,\epsilon$ contribution from an $a_t$ that's already almost the answer,
  which can make fine, low-noise corrections harder to fit precisely.
- **x0-prediction** supervises directly on the action's own (well-scaled, bounded) distribution at
  every noise level, which tends to be easier to fit when $a_t$ is close to $a_0$ (low noise / late
  sampling steps), at the cost of a target whose scale and difficulty vary more across noise levels
  (near pure noise, predicting $a_0$ from an almost-uninformative $a_t$ is a much harder
  regression problem than predicting the next $\epsilon$).

Because `diffusion_cartpole_swingup_balance_study_x0.py` reuses the base script's
`ConditionalDenoiser`, environment, expert controller, dataset generation, evaluation, and
plotting via `load_base_module()` (same pattern `gan_cartpole_swingup_balance_study.py` and
`amp_cartpole_swingup_balance_study.py` use to reuse this folder's environment code), the only
code that actually differs between the two scripts is the training loss target
(`F.mse_loss(pred, noise)` vs `F.mse_loss(pred, a0)`) and the reverse-sampling algebra in
`ActionDiffusion`/`ActionDiffusionX0`. Run and compare them head-to-head with identical
hyperparameters:

```bash
python3 diffusion_cartpole_swingup_balance_study.py    --epochs 3000 --seed 123   # epsilon-prediction
python3 diffusion_cartpole_swingup_balance_study_x0.py --epochs 3000 --seed 123   # x0/action-prediction
```

The x0-prediction script writes its outputs to `outputs/x0/` (same filenames as the base script)
so a single run of both doesn't overwrite either result — compare
`outputs/training_curves_cartpole_swingup.png` against
`outputs/x0/training_curves_cartpole_swingup.png` directly.

### File details (epsilon-prediction script)

File: `diffusion_cartpole_swingup_balance_study.py`

Goal: start near the stable downward equilibrium (theta ~ pi), swing up to upright, then balance.

Fast & effective run (recommended):

```bash
python3 diffusion_cartpole_swingup_balance_study.py \
	--episodes 220 --horizon 520 --epochs 5000 \
	--batch-size 4096 --lr 3e-4 --log-interval 30 \
	--eval-episodes 6 --sampling-steps 8 \
	--diffusion-gain 1.4 \
	--skip-bonus --seed 123
```

Important flags:

- `--sampling-steps`: number of reverse diffusion steps when sampling actions. Smaller is faster (e.g., 6–8) but may reduce quality.
- `--diffusion-gain`: scales the diffusion output force; useful if the diffusion policy is underpowered and cannot swing up.
- `--skip-bonus`: skips trajectory plotting + GIF (this part is very slow).
- `--skip-plots`: if enabled, do not save PNG curves; by default plots are saved (if matplotlib is installed).

## Outputs / Plots

All scripts write outputs into the `outputs/` directory (relative to the working directory where you run the command).

For swing-up + balance, after training you will get:

- `outputs/training_curves_cartpole_swingup.csv` (always)
- `outputs/training_curves_cartpole_swingup.png` (only if `matplotlib` is installed)

If you do not see the `.png` file:

```bash
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

If import fails, install it:

```bash
pip install matplotlib
```

## How the diffusion policy works

Instead of directly regressing `action = f(state)` (which collapses multi-modal expert behavior
to a single mean action), the network learns to **denoise**: starting from pure Gaussian noise, it
iteratively removes noise, conditioned on the state, until what remains is a plausible action. This
is the same mechanism as image diffusion models (DDPM), applied here to a 1-D scalar (force)
instead of pixels. All line references below are for
[diffusion_cartpole_swingup_balance_study.py](diffusion_cartpole_swingup_balance_study.py); the two
other scripts in this folder use the identical mechanism on smaller state spaces.

### 1. The network predicts noise, not the action

`ConditionalDenoiser` ([diffusion_cartpole_swingup_balance_study.py:288-304](diffusion_cartpole_swingup_balance_study.py#L288-L304)):

```python
def forward(self, noisy_action, state, t_norm):
    x = torch.cat([noisy_action, state, self.t_embed(t_norm)], dim=-1)
    return self.net(x)   # predicts epsilon (the noise), not the action
```

Input is `(noisy_action, state, timestep)`; output is the network's guess of *what noise was added*
to corrupt the true action into `noisy_action` at that timestep. The timestep is embedded as
$[\sin(2\pi t), \cos(2\pi t)]$ ([:299-300](diffusion_cartpole_swingup_balance_study.py#L299-L300))
rather than fed in raw, so the network can smoothly interpolate across noise levels.

### 2. Forward process — corrupt expert actions with noise

`q_sample` ([:322-332](diffusion_cartpole_swingup_balance_study.py#L322-L332)) implements the closed-form noising equation:

$$
a_t = \sqrt{\bar\alpha_t}\,a_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
$$

`self.alpha_bars` ([:318-320](diffusion_cartpole_swingup_balance_study.py#L318-L320)) is a
precomputed cumulative-product noise schedule over `steps=28` timesteps, built from a linear
$\beta_t$ schedule (`beta_start=1e-4 → beta_end=2e-2`):

$$
\beta_t \in [10^{-4}, 2\times10^{-2}],\qquad \alpha_t = 1-\beta_t,\qquad \bar\alpha_t = \prod_{i=1}^{t}\alpha_i.
$$

At $t=0$, $\bar\alpha_t\approx 1$ so $a_t\approx a_0$ (clean expert action). At $t=27$,
$\bar\alpha_t\approx 0$ so $a_t$ is almost pure noise.

### 3. Training loss — predict the noise that was added

([:583-598](diffusion_cartpole_swingup_balance_study.py#L583-L598)) For each expert `(state,
action)` pair, pick a random timestep, noise the action, and ask the network to recover the exact
noise vector that produced that corruption:

```python
t_idx = torch.randint(0, diffusion.steps, (batch_size,), ...)
at, noise = diffusion.q_sample(a0, t_idx)
pred_noise = model(at, s, t_norm)
loss = F.mse_loss(pred_noise, noise)
```

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a_0)\sim\mathcal D_E,\ t,\ \epsilon\sim\mathcal N(0,I)}
\Big[\big\|\epsilon - \epsilon_\theta(a_t, s, t)\big\|_2^2\Big]
$$

This is plain supervised regression on a *known target* (the noise you just sampled) — no
adversarial game, no policy rollouts needed during training (unlike the [GAN](../gan/readme.md) or
[AMP](../amp/readme.md) scripts in this project). That makes training very stable, at the cost of
needing an iterative sampling loop at inference time.

### 4. Reverse process — generate an action from noise

`sample_action` ([:334-390](diffusion_cartpole_swingup_balance_study.py#L334-L390)). Start from
`a = randn(...)` and run the schedule backwards; at each step, ask the network what noise is
present, then subtract a scaled portion of it:

$$
a_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}\Big(a_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(a_t,s,t)\Big)}_{\text{mean}}
\;+\; \eta\sqrt{\beta_t}\,z,\qquad z\sim\mathcal N(0,I),\ t>0
$$

with no noise re-injected at the final step ($t=0$). Code:

```python
eps = model(a, state, t_norm)
mean = (1/sqrt(alpha_t)) * (a - ((1-alpha_t)/sqrt(1-alpha_bar_t)) * eps)
a = mean + noise_scale * sqrt(beta_t) * randn_like(a)   # skipped at t == 0
```

28 of these steps turn pure noise into a coherent, state-conditioned action. There's also a faster
strided DDIM-style path ([:359-388](diffusion_cartpole_swingup_balance_study.py#L359-L388)) that
skips timesteps for cheaper inference — this is what `--sampling-steps` controls, and what
`diffusion_hybrid_policy_action` uses (`sampling_steps=6-8`) during training-time evaluation and
fast rollout.

### 5. Turning samples into a controller

`diffusion_policy_action` ([:412-441](diffusion_cartpole_swingup_balance_study.py#L412-L441)) runs
the reverse process `n_action_samples` times (independent noise draws for the same state), takes
the **median** across samples (same denoising-by-aggregation trick used by the GAN/AMP scripts'
`gan_policy_action`/`amp_policy_action`), then scales/clamps to the actuator's force limits.

`diffusion_hybrid_policy_action` ([:444-483](diffusion_cartpole_swingup_balance_study.py#L444-L483))
additionally switches to the analytic `balance_controller` once the pole is near-upright
(`|theta|<0.55` and `|theta_dot|<3.5`): diffusion sampling is comparatively expensive and slightly
noisy, so it's reserved for the hard swing-up phase and handed off to a cheap deterministic
stabilizer for the fragile balance phase — this is also why the default run command above uses
`--diffusion-gain 1.4` (diffusion output alone is often underpowered for a strong swing-up) and
`--sampling-steps 8` (trades sample quality for wall-clock speed).

## Worked toy example: same inputs, two targets

Same spirit as the toy examples in [gan/readme.md](../gan/readme.md): tiny numbers, linear "network"
(no hidden layer), so every step can be traced without a debugger. To make the comparison direct,
**both** parameterizations below start from the exact same toy network weights and the exact same
noised sample — only the training target (and the reverse-step formula) differs.

### Part 1 — epsilon-prediction (base script)

**Toy denoiser** (drop the sin/cos time embedding, fold the timestep's effect into fixed
coefficients for this single illustrative step):
$$
\epsilon_\theta(a_t, s) = w_1 a_t + w_2 s + b,\qquad w_1=0.30,\ w_2=0.10,\ b=0.05
$$

**Setup**: expert sample $s=0.2,\ a_0=0.5$. Pick a noise level with $\bar\alpha_t = 0.81$ (so
$\sqrt{\bar\alpha_t}=0.9$, $\sqrt{1-\bar\alpha_t}=\sqrt{0.19}=0.4359$), and — for a single-step toy
schedule — $\alpha_t=\bar\alpha_t=0.81$, so $\beta_t = 1-\alpha_t = 0.19$. Sample noise
$\epsilon = 1.0$.

**1. Forward corruption** ($q$-sample):
$$
a_t = 0.9(0.5) + 0.4359(1.0) = 0.8859
$$

**2. Network's noise guess** (untrained weights above):
$$
\epsilon_\theta = 0.30(0.8859) + 0.10(0.2) + 0.05 = 0.2658 + 0.02 + 0.05 = 0.3358
$$
Far from the true $\epsilon=1.0$ — the network hasn't learned anything yet.

**3. Loss and gradient** (MSE on this one sample):
$$
\mathcal{L} = (\epsilon_\theta - \epsilon)^2 = (0.3358 - 1.0)^2 = 0.4412
$$
$$
\frac{\partial \mathcal{L}}{\partial \epsilon_\theta} = 2(\epsilon_\theta - \epsilon) = -1.3284
$$
Chain into the weights ($\epsilon_\theta = w_1 a_t + w_2 s + b$):
$$
\frac{\partial \mathcal{L}}{\partial w_1} = -1.3284 \times a_t = -1.1768,\quad
\frac{\partial \mathcal{L}}{\partial w_2} = -1.3284 \times s = -0.2657,\quad
\frac{\partial \mathcal{L}}{\partial b} = -1.3284
$$
Gradient descent ($w \leftarrow w - \text{lr}\cdot\nabla$, lr$=1$ for illustration) pushes every
weight **up** (since all three partials are negative) — exactly what's needed to raise
$\epsilon_\theta$ from $0.3358$ toward the true $1.0$. Repeated over millions of
$(s, a_0, t, \epsilon)$ draws, this is the entire training loop: the network becomes a lookup table
from "how corrupted is this action, and by roughly what" to "what noise vector explains it."

**4. Using the (still under-trained) network for one reverse/denoise step** — this is what
inference does, just with a properly trained $\epsilon_\theta$:
$$
a_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(a_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta\Big)
= \frac{1}{0.9}\big(0.8859 - 0.4359 \times 0.3358\big) = \frac{0.8859 - 0.1464}{0.9} = 0.8217
$$
(Here $t{=}0$ is the last step so no extra noise is re-injected, and $a_{t-1}=a_0^{\text{pred}}$
directly.) Compare to the true expert action $a_0=0.5$: with a barely-trained network the estimate
is still far off ($0.82$ vs $0.5$), which is expected — the whole point of the loss in step 3 is to
shrink that gap over many gradient steps until $\epsilon_\theta$ reliably predicts the injected
noise for *any* $(s, a_t, t)$, at which point this same formula reconstructs $a_0$ accurately from
pure noise across all 28 reverse steps.

### Part 2 — x0-prediction (`_x0` script), identical setup

**Toy denoiser**, same architecture, and — deliberately — the exact same starting weights as Part 1
(so we're isolating the effect of the *target*, not of different initialization):
$$
a_0^\theta(a_t, s) = u_1 a_t + u_2 s + c,\qquad u_1=0.30,\ u_2=0.10,\ c=0.05
$$

**Setup**: identical to Part 1 — $s=0.2,\ a_0=0.5,\ \bar\alpha_t=\alpha_t=0.81,\ \beta_t=0.19,\
\epsilon=1.0$, so the forward-corrupted sample is the same $a_t=0.8859$.

**1. Network's action guess** — same weights, same input as Part 1's step 2, so the raw output is
numerically identical:
$$
a_0^\theta = 0.30(0.8859) + 0.10(0.2) + 0.05 = 0.3358
$$
This is the *same number* the epsilon-network produced. What differs is what it's compared against.

**2. Loss and gradient** — now measured against $a_0=0.5$, not $\epsilon=1.0$:
$$
\mathcal{L} = (a_0^\theta - a_0)^2 = (0.3358 - 0.5)^2 = 0.0270
$$
$$
\frac{\partial \mathcal{L}}{\partial a_0^\theta} = 2(a_0^\theta - a_0) = -0.3284
$$
$$
\frac{\partial \mathcal{L}}{\partial u_1} = -0.3284 \times 0.8859 = -0.2910,\quad
\frac{\partial \mathcal{L}}{\partial u_2} = -0.3284 \times 0.2 = -0.0657,\quad
\frac{\partial \mathcal{L}}{\partial c} = -0.3284
$$
Same sign (weights still get pushed up), but **roughly 4x smaller magnitude** than Part 1's
gradients ($-1.1768$ vs $-0.2910$ for the analogous weight). That ratio isn't a coincidence — it's
exactly $\dfrac{|0.3358-1.0|}{|0.3358-0.5|} = \dfrac{0.6642}{0.1642} \approx 4.05$: the same raw
output is *close* to the $a_0$ target but *far* from the $\epsilon$ target at this noise level, so
the two losses see very different error magnitudes from the identical network.

**3. Reverse/denoise step** — at the final step ($t{=}0$) the x0-prediction code doesn't run any
formula at all, it just reads the network's output directly
([diffusion_cartpole_swingup_balance_study_x0.py:83-85](diffusion_cartpole_swingup_balance_study_x0.py#L83-L85)):
```python
if t == 0:
    a = a0_pred
    break
```
$$
a_{t-1} = a_0^\theta = 0.3358
$$

### Side-by-side: same raw number, two very different reconstructions

| | Part 1 (epsilon-prediction) | Part 2 (x0-prediction) |
|---|---|---|
| Raw network output (identical weights/input) | $0.3358$ | $0.3358$ |
| Interpreted as | "the noise $\epsilon$" | "the clean action $a_0$" |
| Loss target | $\epsilon = 1.0$ | $a_0 = 0.5$ |
| Loss value | $0.4412$ | $0.0270$ |
| Gradient magnitude (analogous weight) | $1.1768$ | $0.2910$ ($\approx 4\times$ smaller) |
| Reverse-step formula | $\frac{1}{\sqrt{\alpha_t}}\big(a_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\big)$ | $a_0^\theta$ directly (at $t{=}0$) |
| Reconstructed action | $0.8217$ | $0.3358$ |
| True action $a_0$ | $0.5$ | $0.5$ |
| Error | $0.32$ | $0.16$ |

Two things this makes concrete, echoing the "why bother with two" claim earlier in this readme:

1. **Same network, different learning signal.** Both networks produced the identical raw scalar —
   the entire difference between the two training runs comes from what that scalar is *scored
   against*. This is exactly the $\epsilon\leftrightarrow a_0$ algebraic equivalence in action: one
   number, two valid readings.
2. **The reverse formula can amplify or damp raw error differently.** The epsilon-prediction
   formula divides by $\sqrt{\alpha_t}$ and subtracts a scaled $\epsilon_\theta$ from $a_t$ — at
   this particular (low-noise, near-final) step that arithmetic happens to *stretch* the
   undertrained network's error outward (final error $0.32$ from a raw discrepancy that, read as
   $a_0$ directly, was only $0.16$). This is a toy-scale illustration of the general folklore:
   epsilon-prediction's reverse formula is well-conditioned when noise dominates ($t$ large) but can
   amplify small mis-estimates when noise is small ($t\to 0$), which is precisely where
   x0-prediction's direct readout has the advantage — it degrades gracefully because at $t\to 0$ it
   is *already* predicting the thing you want. (The reverse is true for large $t$: with mostly noise
   and little signal in $a_t$, guessing $a_0$ directly is the harder regression problem, and
   epsilon-prediction's well-scaled unit-variance target tends to fit more easily — see the "why
   bother with two" note above.)

## Diffusion vs. GAN vs. AMP (same imitation-learning problem, three trainers)

| | Diffusion (this folder) | [GAN](../gan/readme.md) | [AMP](../amp/readme.md) |
|---|---|---|---|
| Training signal | supervised noise-prediction MSE | adversarial (discriminator) | RL reward derived from a discriminator |
| Needs env rollouts during training? | no — static expert dataset only | yes (student rollouts each step) | yes (PPO rollouts each iteration) |
| Inference cost | ~6-28 sequential forward passes × `n_action_samples` | 1 forward pass × `n_action_samples` | 1 forward pass |
| Multi-modality source | the full noise-to-action reverse trajectory | latent $z$ fed to the generator | policy's learned Gaussian std |
| Training stability | high (regression, no adversarial game) | can be unstable (needs BC warm start, label smoothing) | needs LSGAN + gradient penalty to stay stable |

Diffusion trades expensive, sequential inference for very stable, rollout-free training — which is
why this script leans hard on the hybrid switch-to-analytic-controller trick and strided DDIM
sampling to stay fast enough for real-time evaluation.

