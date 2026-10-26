# PhD Research Proposal (Idea 8, fundamental learning, deep dive)

**Working title:**
*"World Models as Amortized Randomized Smoothing: Why Learned Dynamics Give Useful Gradients through Contact,
and How to Control It"*

**Track:** fundamentals of robot learning (model-based RL, differentiable optimisation through contact). It builds
directly on the newest Berkeley result: **SuperMimic** (Liao et al., arXiv 2610.28480, CoRL 2026 oral) trains
humanoid tracking by backpropagation-through-time (BPTT) through **one learned world model**, with more than
1000× fewer GPU-hours than model-free RL.

> Written 2026-10-26. Novelty was checked by web search (arXiv full text was blocked here). Re-check before
> committing.

---

## 1. The puzzle

The true simulator has **discontinuous** dynamics at contact (impact, stick/slip, make/break). First-order
gradients through it are useless or explode: Suh et al. (ICML 2022) "Do differentiable simulators give better
policy gradients?" Yet **learned** world models give gradients good enough to train a humanoid (SuperMimic), and
PWM (ICLR 2025) reports that *"well-regularised world models generate smoother optimisation landscapes than the
actual dynamics."*

**Why?** Nobody has a theory, and there is **no knob** to control *how much* smoothing a world model provides.

## 2. What is close (and what is still open)

| Work | What it does | Missing |
|---|---|---|
| PWM (arXiv 2407.02466, ICLR 2025) | Observes that learned WMs are smoother and better for first-order policy learning; toy where a ball sticks to a wall | No theory, no control over the amount of smoothing |
| Bundled gradients / randomized smoothing (Suh, Pang & Tedrake, arXiv 2109.05143, RA-L 2022); Lidec et al. (arXiv 2206.11884) | Smooth contact dynamics by *sampling* noise; the smoothing noise must shrink for convergence | Costly sampling with variance O(1/(Nσ²)); no learned model |
| Bundled Contact Gradients (arXiv 2609.30951) | Contact-local randomized smoothing for first-order policy learning, with real G1 transfer | Smoothing by sampling inside a differentiable simulator, not by a learned model |
| Smooth Neural Surrogates (SNS, arXiv 2601.12169) | Learned legged dynamics with a **Lipschitz** smoothness budget for MPC | Global slope limiting, not tied to randomized smoothing; no noise-level conditioning or annealing; MPC, not policy learning |
| MB-RPG theory (arXiv 2310.19927, NeurIPS 2023) | Model smoothness controls gradient variance; spectral normalisation | No contact-discontinuity analysis, no annealing |
| NEAR (arXiv 2501.14856); NC-LQL (arXiv 2602.07873) | Noise-conditioned **rewards** and **Q-functions** with annealing | Not dynamics models |
| DIAL-MPC; MDPO (arXiv 2605.07520) | Annealed sampling noise in MPC, or exploration noise in differentiable planning | Not learned models |
| Diffusion world models (DIAMOND, Dreamer-4 style) | Noise on the *output* (generative denoising of the next state) | Different object: they do not smooth the dynamics in the *input* |

**Open gap:** (i) a theory that **noise-trained world models perform randomized smoothing**, (ii) the bias
caused by policy-generated data, and (iii) a **noise-conditioned world model** that gives the whole smoothing
family for **annealed** BPTT policy learning.

## 3. Theory

**Setup.** Dynamics s' = f(s, a), piecewise smooth with jumps at contact. Bundled (per-step smoothed) dynamics:
f_σ(s, a) = E_ε[f(s + σε, a)]. Objective J_σ(θ): the policy return under f_σ.

**Proposition A (noise-trained world model = randomized smoothing, amortized).**
Train a model g on inputs s̃ = s + σε (s ~ p, the data density) with targets f(s) and squared loss. The optimum is

  g_σ(s̃) = E[f(s) | s̃] = (f·p ∗ φ_σ)(s̃) / (p ∗ φ_σ)(s̃),

a Nadaraya–Watson smoother of f under p.
- **(i)** If p is locally flat on the σ scale, then **g_σ = f ∗ φ_σ = f_σ**: exactly the bundled dynamics.
- **(ii)** Backpropagating through g_σ gives ∇J_σ *deterministically*: **zero sampling variance**, unlike
  zeroth-order bundled gradients, whose variance is O(1/(Nσ²)). The cost of smoothing is paid once, at training
  time (amortisation).
- **(iii)** Conditioning on σ as an input gives the **whole family {g_σ}** from one model.

**Proposition B (implicit smoothing explains PWM).** A plain world model trained without noise, with finite
capacity, SGD and spectral bias, behaves like g_σ for some *uncontrolled* effective σ_eff that depends on data and
capacity. A diagnostic follows: estimate σ_eff of any world model by matching its predictions to the family
{g_σ}.

**Proposition C (data-density bias: a world model "believes" the behaviour it has seen).**
Take one dimension across a contact boundary, a jump of size Δ (f = Δ·1[s > 0]), and data density p(s) ∝ e^(−βs)
near the boundary (β > 0: most data on the s < 0 side, e.g. "the ball hits the wall"). The posterior s | s̃ is
Gaussian with mean s̃ − βσ², so

  **g_σ(s̃) = Δ · Φ((s̃ − βσ²)/σ).**

The smoothed contact boundary is **shifted by βσ² towards the rarely-visited side**. The gradient that would
pull the policy across the boundary is therefore **attenuated**:

  φ((s̃ − βσ²)/σ)/σ < φ(s̃/σ)/σ   for s̃ < 0.

Consequence: learned world models trained on policy data *slow the discovery of new contact modes*. This is a
model-based exploration problem with a closed form. **Fixes:** importance-reweight the training data by 1/p̂(s)
(density flattening), or choose σ adaptively.

**Proposition D (annealing).** With {g_σ} available for free, run BPTT on J_σ with σ decreasing
(graduated optimisation or continuation). Hazan, Levy & Shalev-Shwartz (ICML 2016) prove global convergence for
"σ-nice" objectives. Large σ escapes flat and cliff regions; small σ removes the smoothing bias at the end.

## 4. Toy experiment: `toy/smoothing_world_model.py` (results in §4a, once the run finishes)

A ball is thrown at a wall and sticks if it hits; both the wall and the ground are sticky. J(α) is the final x.
J is **flat (true gradient = 0)** for launch angles 5–20°, then jumps at the cliff (~22°), with the optimum near
42°.

Tests:
1. **Proposition A:** compare the NC-WM landscape with the Monte-Carlo smoothed true dynamics at each σ.
2. **Proposition C:** compare an NC-WM trained on policy-concentrated data (90% wall hits) with one trained on
   broad data.
3. **Optimisation from the flat region:**
   - true first-order gradients;
   - zeroth-order randomized smoothing;
   - plain WM;
   - NC-WM at fixed σ;
   - **NC-WM annealed**;
   - annealed NC-WM with concentrated data.

## 4a. Results: Proposition A had to be corrected (input noise ≠ smoothing; argument noise = smoothing)

**v2 (input noise, `outputs/idea8_smoothing_v2_input_noise.json`): negative.**
- A world model trained on noisy *inputs* was **further** from the Monte-Carlo smoothed dynamics near contact
  than the unsmoothed simulator was (error 0.57–1.75 versus 0.35–0.64).
- In optimisation it cleared the wall only **3/9** times.
- **Why:** robot state data lies on a low-dimensional manifold of trajectories. By Tweedie,
  E[f(s) | s̃] ≈ f(s̃ + σ²∇log p_σ(s̃)), so off the manifold an input-noise model acts as a **denoiser** (it
  projects back onto the data) and does not smooth isotropically. The "flat density" condition of Proposition A
  fails badly in robotics.

**Corrected Proposition A′ (argument noise).** Train g(s, σ) with *clean* input s and target
**f(s + σε) produced by the simulator** from the perturbed state. The squared-loss optimum is exactly
E_ε[f(s + σε)] = f_σ(s), **independent of the data density**; only coverage matters. This costs nothing in
simulation, and SuperMimic already fits its world model on simulator transitions.

**v3 (`outputs/idea8_smoothing.json`): confirms A′.**

| | unsmoothed sim | input-noise WM | **argument-noise WM** |
|---|---|---|---|
| one-step error to f_σ near contact (σ = 0.03–0.12) | 0.37–0.81 | 0.34–0.75 | **0.11–0.13** |
| rollout landscape MAE to MC-smoothed J | – | 0.30–0.44 | **0.16–0.19** |
| gradient in the flat region (true gradient = 0) | 0 | erratic sign | **consistently towards the cliff** |
| optimisation: wall cleared / mean J (optimum 1.99) | true first-order: 0/9, 1.00 | 3/9, 1.28 | **annealed: 9/9, 1.85**; fixed σ = 0.12: 9/9, 1.88 |
| …with policy-concentrated data | | | annealed: 4/9, 1.43 |

Zeroth-order randomized smoothing on α reached 9/9 with J = 1.98, but it uses 16 simulator rollouts per
iteration; the world-model methods use none during policy optimisation. With concentrated data, even the
argument-noise model degrades. The mechanism here is **coverage** (model error in rarely-visited regions), not
the closed-form βσ² shift of Proposition C, which is specific to input noise.

**So Proposition C** (closed form Δ·Φ((s̃ − βσ²)/σ), verified numerically to 4 decimals) describes
**input-noise / denoising** models, which is what one gets when the world model's input is a noisy *observation*
(perception). That is exactly the perceptive-locomotion case in §4b.

## 4b. Extension to perceptive stairs: two different reasons to smooth

| purpose | how | anneal σ to |
|---|---|---|
| **optimisation** (escape flat or cliff regions) | argument noise in the simulator call | 0 |
| **uncertainty** (the heightmap is noisy) | train the world model on the *perceived* terrain, so it learns E[f \| Ĥ], the Bayesian posterior | the true sensor noise, never below it |

**Proposition E (pathwise gradients miss the boundary term).** E[J] = ∫ J(θ, H) p(H | Ĥ) dH has, by the
Leibniz rule, a boundary term at the trip surface {a(θ, Ĥ) = H}. Backpropagation through a model fed the *true*
H only sees the interior: on the clear side it sees only the effort cost, which pushes the swing *down* towards
the riser. **No safety margin is learned, even though the policy observes noisy Ĥ.** A model fed Ĥ learns the
posterior, which turns the boundary term into a smooth interior term that backpropagation can see.

The test suite is `toy/stair_tests.py` (results in §4c).

## 5. Full research plan

1. **Theory note** (L4DC- or NeurIPS-style): Propositions A–D with proofs, the σ_eff diagnostic, and the
   bias/variance comparison against sampling-based bundled gradients.
2. **Toys:** ball and wall (done in §4), cartpole with walls, and a planar pusher (contact-rich, feedback policy).
3. **Humanoid:**
   - Start from the **open reproduction** `CuboiLeo/SuperMimic-Reproduced` (mjlab / MuJoCo-Warp, G1).
   - Make the world model noise-conditioned and anneal σ during BPTT.
   - Measure:
     - return against GPU-hours;
     - success on contact-heavy motions (get-up, rolls, cartwheels, crawling);
     - gradient variance;
     - σ_eff of the original SuperMimic world model.
4. **Beyond tracking:** tasks with **sparse or discontinuous objectives** (jump onto a box, get up from a fall,
   push an object to a goal), where smoothing should matter most. Dense tracking rewards may hide the effect.

## 6. Risks

| Risk | Mitigation |
|---|---|
| On dense tracking rewards the gain is small (the reference keeps the policy near the right contact modes) | Emphasise sparse or contact-switching tasks; the theory and diagnostic stand regardless |
| Overlap with SNS, PWM or Bundled Contact Gradients | The contribution is the *equivalence theorem*, the *density-bias result* and *free annealing from one model*; compare against all three directly |
| Real data density is high-dimensional, so 1/p̂ is hard to estimate | Use a k-NN or flow-based density in latent space, or adaptive σ from model uncertainty |

## 7. Venues

NeurIPS 2027 or ICLR 2028 for the theory and toys (fundamental ML). CoRL 2027 with the humanoid results.
