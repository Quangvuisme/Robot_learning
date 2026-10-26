# Method: Event-Branched, Perception-Calibrated World Models (EB-WM)

*Gradient-based policy learning through contact with uncertain perception: humanoid stairs, terrain and motion
tracking.*

> **Status (2026-10-26).** Every claim below was tested on toy problems in `toy/`. The final combined results are
> stable across 5 world-model seeds (§5). **Not yet tested on a humanoid.** Novelty was checked by web search
> only (arXiv full text was blocked from this environment), so re-check before submission.

---

## 1. One-sentence summary

To learn perceptive locomotion by **backpropagating through a learned world model** (as SuperMimic does for
tracking), the model must make contact **events** explicit and calibrated, and the rollout must **branch over
those events**. The world model should see the **perceived** terrain, not the true terrain, and any extra
optimisation smoothing should be added **only up to a budget** that the perception noise does not already
provide.

## 2. The problem (why naive world-model gradients fail on stairs)

1. **Discontinuities.** A toe either clears the riser or trips. True first-order gradients are zero, or point the
   wrong way, in the trip region.
2. **Proposition E (missing boundary term).** The expected return E[J] = ∫ J(θ, H) p(H | Ĥ) dH has, by the Leibniz
   rule, a term on the trip boundary. Backpropagation through dynamics fed the **true** terrain never sees it, so
   policies learn **no safety margin**. They even push the swing *down*, because only the effort cost is
   visible.
   *Toy:* backprop through the true simulator trips **100%** at every perception-noise level.
3. **Input noise is not smoothing.** Training a world model with noise on its *inputs* gives
   E[f(s) | s̃] ≈ f(s̃ + σ²∇log p): a **denoiser** on low-dimensional robot data (Tweedie), not a smoother.
   *Toy:* 3/9 against 9/9 for argument noise.
4. **Mean-field rollouts break after an event.** Propagating the *mean* of "tripped" and "not tripped" creates
   half-stopped feet that never occur in the data, and gradients become meaningless.
   *Toy:* 73–95% trips.

## 3. The method: four components plus one rule

### C1. Event-factored world model

Use a shared trunk h = φ(s, a, terrain_input, σ) with three heads:
- **event head** p_k = P(event k | ·), trained with **binary cross-entropy on contact-event labels** that the
  simulator gives for free (trip / toe-stub, touchdown, slip, …);
- **branch heads** f_free(·) and f_k(·): next-state regressors, each trained with MSE **only on its own samples**.

Cross-entropy is a proper scoring rule, so the event head converges to the calibrated conditional probability.
*Toy:* with true terrain the boundary width is **0.6–0.9 cm**, where a plain regression MLP blurs it to
3–16 cm and shifts it in a seed-dependent way.

### C2. Perception-calibrated inputs

Feed the model the **perceived** terrain (heightmap with the real sensor-noise model), never the privileged
ground truth. The event head then learns the **Bayesian posterior** P(trip | y, Ĥ) = P(H > y | Ĥ), and margins
calibrate themselves to the sensor noise.
- *Toy:* width 4.2–5.7 cm against a posterior width of 5.1 cm at σ_p = 2 cm, and 9.0–10.2 cm against 10.25 cm at
  4 cm. The shift is about 0. This holds for all 5 seeds.
- *Rule:* the behaviour policy that collects world-model data should depend only on the model's inputs (Ĥ),
  never on privileged H. Otherwise the conditional probabilities are confounded. The measured effect in the toy
  was small, but the rule costs nothing.

### C3. Mode-branched rollouts (for backprop-through-time)

Keep one particle per **contact-mode history** (for example "no trip yet" and "tripped"), each with a weight w,
and merge particles that share a mode (IMM-style, so the count stays bounded):

```
w_free' = w_free · (1 − p_trip(s_free))
w_trip' = w_trip + w_free · p_trip(s_free)
s_trip' = [ w_trip · f_free(s_trip) + w_free · p_trip · f_trip(s_free) ] / w_trip'      (merge within the mode)
s_free' = f_free(s_free)
J = Σ_modes w_mode · J(s_mode)
```

Every particle stays **on the data distribution**. The gradient of p_trip flows through the weights, so the
**boundary term that pathwise gradients miss is recovered**, with no sampling variance.
- *Toy:* moving from mean-field to branched cuts trips from 73% to 17% at σ_p = 2 cm and from 53% to 4.8% at 4 cm.

### C4. Argument-noise optimisation smoothing (amortized randomized smoothing)

Train g(s, a, ·, σ) with a **clean input s** and simulator targets **f(s + σε)**, including the event labels at
the perturbed state. The squared-loss optimum is exactly the bundled dynamics E_ε[f(s + σε)], **independent of
the data density** (Proposition A′). During policy learning, anneal σ: hold it at σ_max for 30% of iterations,
then use a cosine decay to 0 at 90%.
- *Toy (ball and wall):* one-step error to the smoothed dynamics is 0.11–0.13 against 0.34–0.75 for input noise;
  the wall is cleared **9/9** times against 3/9.

### Rule: the smoothing budget

The optimiser already sees the perception posterior width σ_p. Add only what is missing:

  **σ_ARG,max = √(max(0, σ_escape² − σ_p²))**,  where σ_escape ≈ the scale of the initial error (4 cm here).

If σ_ARG,max = 0, **do not train with argument noise at all**, because its σ = 0 slice is blurrier.
- *Toy:* without the budget (σ_max = 5 cm everywhere), the method was worse than no smoothing at σ_p = 4 cm
  (17% ± 32% trips). With the budget it is stable at every noise level (§5).

### Algorithm 1 (EB-WM)

```
inputs: simulator f with contact-event labels, sensor-noise model, policy π_θ(obs incl. perceived terrain)
σ_max ← sqrt(max(0, σ_escape² − σ_p²))
repeat
  1. collect transitions with π_θ + exploration; the behaviour depends only on the perceived terrain
  2. train the event-factored WM on (s, a, Ĥ) → (event labels, next state)
       if σ_max > 0: sample σ ∈ [0, σ_max]; targets and labels come from the simulator at s + σε; σ is an input
  3. policy update: BPTT through H-step mode-branched rollouts of the WM
       (σ annealed by the schedule; critic bootstrap at the horizon, as in SuperMimic)
until converged
```

### Algorithm 1, detailed (single training loop)

```python
# ===== EB-WM: Event-Branched, Perception-Calibrated World Model =====
# Inputs: simulator f (with contact-event labels), sensor-noise model, policy π_θ, critic V_ψ
σ_max = sqrt(max(0, σ_escape**2 - σ_p**2))          # smoothing budget (σ_p = sensor noise)
use_arg_noise = σ_max > 0                            # budget 0 -> do not train with argument noise

for iteration in range(N):
    # ---- 1. collect data (behaviour depends only on the PERCEIVED terrain) ----
    for episode in batch:
        terrain = sample_terrain();  ĥ = perceive(terrain, σ_p)          # noisy heightmap
        for t in steps:
            a = π_θ(o_t, ĥ) + exploration_noise
            s_next, event_label = f(s, a, terrain)                        # simulator gives the event label
            buffer.add(s, a, ĥ, terrain)

    # ---- 2. train the event-factored world model g_φ(s, a, ĥ, σ) ----
    for minibatch (s, a, ĥ, terrain) in buffer:
        σ = sample_uniform(0, σ_max) if use_arg_noise else 0
        s_pert = s + σ * ε                                  # ARGUMENT noise: perturb the state, not the input
        y, e = f(s_pert, a, terrain)                        # target and label from the SIMULATOR at s_pert
        p_k, f_free, f_k = g_φ(s, a, ĥ, σ)                  # clean input s; perceived terrain ĥ
        loss = BCE(p_k, e)                                  # event head (proper scoring rule -> calibrated)
               + MSE(f_free, y | e = none)                  # each branch trained only on its own samples
               + Σ_k MSE(f_k, y | e = k)
        φ ← φ - lr * ∇_φ loss

    # ---- 3. update the policy by BPTT through MODE-BRANCHED rollouts ----
    σ = schedule(iteration)        # hold σ_max for 30 %, cosine decay, 0 from 90 % of training
    particles = {"no_event": (w = 1, s = s0)}            # one particle per contact-mode history
    for t in range(H):
        new = {}
        for mode, (w, s) in particles:
            a = π_θ(s, ĥ)
            p_k, f_free, f_k = g_φ(s, a, ĥ, σ)
            new[mode]          += (w * (1 - Σ_k p_k), f_free)        # no event
            for k: new[mode + k] += (w * p_k, f_k)                   # event k happens
        particles = merge_same_mode(new)                 # IMM: weighted average of states within a mode
        J += Σ_modes w * r(s, a)                         # reward weighted by mode probability
    J += Σ_modes w * V_ψ(s_H)                            # critic bootstrap at the horizon
    θ ← θ + lr * ∇_θ J                                   # gradient flows through the weights w (boundary term)
    ψ ← fit V_ψ on λ-returns from the branched rollouts
```

## 4. Theory (statements; proofs to write)

- **Proposition A′.** Under squared loss, argument-noise training gives g_σ = f ∗ φ_σ (the bundled dynamics) for
  every s in the data support, regardless of the data density. Input-noise training instead gives the
  Nadaraya–Watson / Tweedie denoiser E[f(s) | s̃].
- **Proposition C (input noise, closed form).** For a step of height Δ and a data density ∝ e^(−βs) at the
  boundary, the model learns Δ·Φ((s̃ − βσ²)/σ): the boundary shifts by βσ² towards the rarely-visited side.
  Verified numerically to 4 decimals.
- **Proposition E (boundary term).** Pathwise gradients of E[J] through event-discontinuous dynamics omit the
  boundary integral. A calibrated event model with branched rollouts recovers it as ∂w/∂θ.
- **Proposition F (branched rollout).** For a single event per step, the branched objective equals the exact
  expectation of J under the model. With mode merging it is exact whenever the post-event dynamics are affine in
  the merged state, and approximate otherwise.
- **Smoothing budget.** Smoothing widths add in quadrature (Gaussian convolution), so the effective smoothing is
  √(σ_p² + σ_ARG²). Graduated optimisation needs this to exceed the escape scale early on, and an excess biases
  the optimum when there are effort costs (observed at σ_p = 4 cm).

## 5. Evidence (toy stair-stepping with noisy heightmaps; 5 world-model seeds; mean ± std)

E[J] = 0.4 + 0.2·P(clear) − a²; the optimal margin grows with sensor noise. "Escape" means starting from a
policy that trips 86% of the time.

| perception noise σ_p | method | J | trip rate | 5-stair success | escape from tripping start |
|---|---|---|---|---|---|
| 0 cm | **EB-WM** | 0.554 ± 0.029 | **0%** | **100%** | **5/5** (0% trips) |
| | oracle | 0.583 | 0% | 100% | – |
| 2 cm | **EB-WM** | **0.566 ± 0.005** | **0.14% ± 0.12%** | **99.2%** | **5/5** (0.566 ± 0.004) |
| | ablation: no branching (mean-field) | 0.441 ± 0.054 | 73% | 4% | ✗ |
| | ablation: regression MLP instead of event heads | 0.558 ± 0.006 | 3.3% ± 5.8% | 88% | ✗ (89% trips) |
| | ablation: privileged (true) terrain input | 0.559 ± 0.009 | 2.0% ± 2.5% | 91% | partial (37% trips) |
| | backprop through the true simulator | 0.398 | 100% | 0% | ✗ |
| | model-free zeroth-order (614k sim rollouts) | 0.569 | 0.2% | – | ✓ |
| | oracle | 0.570 | 0.8% | 96% | – |
| 4 cm | **EB-WM** (budget gives σ_ARG = 0) | **0.557 ± 0.005** | 4.8% ± 4.1% | 81% | **5/5** (0.554 ± 0.005) |
| | ablation: ignore the budget (σ_max = 5 cm) | 0.527 ± 0.051 | 17% ± 32% | 76% | 3/5 |
| | oracle | 0.556 | 4.2% | 84% | – |

**Robustness** (policies trained at 2 cm; v1 suite, comparing the perceived-terrain world model with the
privileged one):

| test | perceived terrain | privileged terrain |
|---|---|---|
| deployment noise 4 cm | 6% trips | 37% |
| unseen tall risers (20–26 cm) | 1% | 27% |
| heightmap biased −1.5 cm | 2% | 54% |

The model-based methods use **no simulator rollouts** during policy optimisation. The model-free zeroth-order
baseline used 614,400.

**Known weakness:** at σ_p = 0, 1 of 5 seeds is over-conservative (a 21 cm margin, J = 0.496). Likely fix: a
higher-capacity model, or a final fine-tuning phase on a dedicated σ = 0 model.

## 6. From toy to humanoid (the PhD paper)

- **Platform:** mjlab + MuJoCo-Warp, Unitree G1, starting from the open reproduction
  `CuboiLeo/SuperMimic-Reproduced` (a BPTT world-model tracker).
- **Events (C1):** labels per foot from MuJoCo contacts: toe-stub on a riser edge, touchdown, slip, knee or shin
  contact. Mode particles are per foot (2 feet × a few modes, merged by mode, so K ≤ 8).
- **Perception (C2):** an egocentric heightmap with a measured noise model (dropout, bias, edge blur). The world
  model takes the perceived heightmap.
- **Tasks:**
  - stairs up and down at random rise and run;
  - stepping stones and gaps;
  - rough terrain;
  - **perceptive motion tracking** (tracking human motion clips on stairs and terrain);
  - contact-rich motions (get-up, kneeling) for the mimic chapter.
- **Baselines:**
  - perceptive PPO with height scans (the standard mjlab/legged-gym recipe);
  - SuperMimic-style plain world model with BPTT;
  - EB-WM ablations (C1–C4 and the budget);
  - zeroth-order / model-free sample cost.
- **Metrics:**
  - success and toe-stub rate on stairs, as a function of rise and heightmap noise;
  - learned swing clearance compared with the sensor noise (calibration);
  - GPU-hours to reach a given success rate;
  - sim-to-sim and sim-to-real transfer on a real staircase.
- **Headline claims to test on hardware:** (1) calibrated foot clearance that scales with real heightmap noise;
  (2) far fewer toe-stubs than plain world-model BPTT at equal compute; (3) PPO-level robustness at a fraction of
  the compute.

## 7. Paper outline

- **Title:** *Event-Branched World Models: Calibrated Gradients through Contact for Perceptive Humanoid
  Locomotion*
- **Contributions:**
  1. Proposition E: why pathwise world-model gradients learn no safety margin, and how branching recovers the
     boundary term.
  2. Proposition A′ / C: argument noise versus input noise, i.e. amortized randomized smoothing versus denoising.
  3. The EB-WM method (event heads, perception-calibrated inputs, branched rollouts, smoothing budget).
  4. Toy analysis plus G1 experiments on stairs, terrain and perceptive motion tracking.
- **Venues:** CoRL 2027 or RSS 2027 (with G1 results); a NeurIPS or ICLR version for the theory and toy
  analysis.

## 8. Reproduce the toy evidence

```bash
cd Research_proposal/toy
python3 smoothing_world_model.py                # ball and wall: Proposition A' (argument vs input noise)
python3 stair_tests.py --seeds 3                # stairs v1: T1–T6 (Proposition E, robustness)
python3 stair_tests_v4.py --seeds 5 --sp 0.02   # ablations A–E (run per noise level)
python3 stair_tests_v5.py --sp 0.02             # smoothing-budget rule
```

Requirements: numpy, autograd, matplotlib. No GPU is needed.
