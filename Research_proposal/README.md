# PhD Research Proposal: Consequence-Aware Guidance for Humanoid Diffusion Policies

**Working title:**
*"Look Before You Leap: World-Model-Grounded, Sampling-Based Test-Time Guidance for Versatile Humanoid Whole-Body Control"*

**One-line pitch:** BeyondMimic showed that one diffusion policy can do many humanoid tasks at test time
through cost guidance. That guidance scores the states the diffusion model *imagines*, not what the robot will
*actually do*. It also needs differentiable costs. We score each candidate plan by rolling its actions out in a
learned world model, and we guide the denoising chain with an MPPI-style path-integral update. This makes guidance
consistent with the dynamics, works with non-differentiable costs (falls, contacts, collisions) and stays
cautious where the world model is uncertain.

> Written 2026-10-25. Papers from 2026 move fast. Before you commit, re-check arXiv for the newest work in
> Section 9. Some 2026 entries below come from search snippets that I could not open in full.

---

## 1. Why this topic, and why now

### 1.1 The research line we build on (Qiayuan Liao / Berkeley Hybrid Robotics)

| Year | Work | What it contributed | What it leaves open |
|---|---|---|---|
| 2022–23 | `legged_control` (NMPC + WBC, OCS2) | Model-based, *verifiable* control | Hand-built models, limited agility |
| 2025–26 | **BeyondMimic** (arXiv 2508.08241, *Science Robotics* 2026) | Scalable motion tracking, plus a **state-action diffusion policy** with **test-time cost guidance** (joystick, waypoints, obstacle avoidance, inpainting), zero-shot sim-to-real | Guidance gradients act on *diffused* states. These can be dynamically inconsistent with the actions, and costs must be differentiable. |
| 2026 | **mjlab** (arXiv 2601.22074) | Isaac Lab API on MuJoCo-Warp: fast, open, GPU-parallel | — (it is the tooling) |
| 2026 | **SuperMimic** (arXiv 2610.28480, CoRL 2026 oral) | One **learned differentiable world model** trains whole-body control with far fewer GPU-hours | Uses the world model to *train* policies. It does not use it to *steer* a generative policy at test time. |

Read top to bottom, the trend is model-based control → generative policy → learned model. The next step is to
**close the loop: use the learned world model to tell the generative policy what its plan will actually do.**
This combines the strengths of the three steps above. It is a natural continuation of this line of work, and it
lines up with the code you already have in this repo (Section 7).

### 1.2 The gap, stated precisely

BeyondMimic-style policies denoise a joint trajectory τ = (s_{t:t+H}, a_{t:t+H}). Guidance adds
∇_τ J(τ̂₀), where J is a task cost evaluated on the **diffused states ŝ**. This raises three problems:

1. **Inconsistency.** Gradient guidance can push ŝ somewhere the actions â cannot actually take the robot. The
   low-level system then executes â, and what happens differs from what the cost was scored on. The harder the
   guidance pushes, the larger this gap gets. This is our main hypothesis to measure.
2. **Differentiability.** The costs that matter most for humanoids are discontinuous: falling (an indicator),
   foot-contact schedules, self-collision, collisions with geometry, and joint-limit violations. Today they are
   replaced by smooth proxies.
3. **No notion of risk.** Nothing in the guidance knows *when it is extrapolating*.

Related work covers only pieces of this:
- RGB (arXiv 2606.25123) uses an RL prior inside MPPI, but has no generative prior or learned model.
- Feedback World Model (arXiv 2605.15705) and DynaGuide/LPB guide diffusion with a world model, but only for
  manipulation.
- Diffusion-ES (CVPR 2024) does gradient-free search inside diffusion, but for driving.
- RISE (arXiv 2610.30061) applies test-time steering to humanoid fall recovery, using a kinematic planner.
- DreamControl-v2 (arXiv 2604.00202) trains guided diffusion priors.

**None of them combines (humanoid whole-body) + (learned world model as the consequence evaluator) +
(sampling-based guidance for non-differentiable costs) + (uncertainty-aware) + (real-time, on hardware).**
That combination is the paper.

---

## 2. Research questions and hypotheses

- **RQ1:** Does scoring guidance on world-model rollouts of the *actions*, instead of the diffused states, improve
  task success and reduce falls under strong guidance?
  **H1:** Yes. The gain grows with guidance strength and with how far the task is from the training distribution.
- **RQ2:** Can path-integral (MPPI-style) weighting inside the denoising chain optimise non-differentiable
  humanoid costs as well as, or better than, smooth-proxy gradient guidance, within a real-time budget?
  **H2:** Yes, with 16–64 candidates and guidance applied only in the last few denoising steps.
- **RQ3:** Does an uncertainty penalty (disagreement across a world-model ensemble) prevent the guidance from
  exploiting world-model errors?
  **H3:** Yes. Without it, success in sim improves but sim-to-real transfer gets worse.

---

## 3. Method: WM-PIG (World-Model Path-Integral Guidance)

### 3.1 Components

1. **Experts → data.** Train BeyondMimic-style motion-tracking experts in **mjlab** on LAFAN1/AMASS retargeted
   motions (Unitree G1, 29 DoF). Roll them out with domain randomisation and logged noise injection to collect
   (s, a, s') data.
2. **Diffusion policy p_θ(τ | o_t).** A state-action trajectory transformer like BeyondMimic's, with
   horizon H ≈ 16 and x₀-prediction. Your repo already compares ε- and x₀-prediction in
   `Generative_model/diffusion/`.
3. **World model f_φ (an ensemble of E = 5).** A latent or proprioceptive next-state predictor trained on the same
   data, in the SuperMimic style. Keep it simple: an MLP or transformer on proprioceptive state that predicts
   Δs and a contact/fall logit.
4. **Guidance (the contribution).** At denoising step k, take the current noisy plan τᵏ:

```
for k = K ... 1:
    draw M candidates  τ^{k-1}_i ~ p_θ(· | τ^k, o_t)            # batched
    τ̂₀,i  = x0-prediction(τ^{k-1}_i)                              # Tweedie / x0 head
    roll out actions â_i through every ensemble member e:  ŝ^e_i = f_φe(s_t, â_i)
    J_i = J_task(mean_e ŝ_i)                                      # any cost, may be non-differentiable
        + β · ‖ŝ_i − ŝ^diff_i‖²                                   # consistency: diffused vs rolled-out
        + κ · Var_e(ŝ^e_i)                                        # pessimism / uncertainty
        + γ · 1[fall or collision in ŝ_i]                         # hard events, no gradient needed
    w_i = softmax(−(J_i − min J) / λ)                             # MPPI / path-integral weights
    τ^{k-1} = Σ_i w_i τ^{k-1}_i    (or resample, SMC-style)
    (optional) τ^{k-1} −= η ∇ J_smooth through f_φ                # hybrid gradient step
execute â₀ (receding horizon), warm-start the next call from a shifted plan
```

5. **Real-time tricks.** Use DDIM with K ≈ 5–10 steps. Guide only at the last K_g ≤ 3 steps. Batch all M×E
   rollouts in one forward pass. Warm-start between control steps, as in MPPI. Optionally, *distil* guided
   behaviour back into the policy (as in ReGuide, arXiv 2606.28939) for tasks you see often.

### 3.2 Theory angle

Theory is what gets the paper accepted rather than labelled "engineering". Frame guidance as sampling from the
tilted distribution

  p*(τ) ∝ p_θ(τ | o) · exp(−J(τ) / λ)

This is exactly the optimal distribution in KL-regularised control and MPPI. The path-integral update is a
self-normalised importance-sampling estimate of the mean under p*, so it needs **no ∇J**. Contributions:

- (a) Show that state-space gradient guidance targets a *different* distribution whenever ŝ ≠ f(s, â). Give a
  bound on the guidance error in terms of the dynamics-consistency residual.
- (b) Show that the ensemble-variance penalty gives a lower confidence bound on the true cost, in the same way as
  pessimistic model-based RL (MOPO).
- (c) Report how effective sample size scales with λ and M, and use this to set λ adaptively, which is a standard
  MPPI trick.

---

## 4. Experiments

**Platform:** mjlab (training) → MuJoCo sim-to-sim (a different simulator stack, as a transfer check) →
**Unitree G1** hardware if available. Booster T1 or Berkeley Humanoid Lite are alternatives.

**Tasks (test-time only; no retraining per task):**
1. Joystick velocity tracking while doing expressive motion (BeyondMimic's task, for a direct comparison).
2. Waypoint navigation with **non-convex obstacles** (collision indicator cost).
3. **Stepping-stone / foothold constraints** (discrete contact-location cost; gradient methods struggle here).
4. **Push recovery and fall avoidance** (fall-indicator cost under external pushes).
5. Motion inpainting with keyframes plus a "don't fall" constraint.
6. (Stretch goal) Kicking or pushing an object to a target, where the cost lies entirely in the object's
   consequences.

**Baselines:**
- BeyondMimic gradient guidance on diffused states. This is the main baseline.
- Unguided diffusion plus a task-specific RL policy (the upper bound for a single task).
- RGB-style MPPI with an RL prior (no diffusion).
- Diffusion-ES-style gradient-free search without a world model (scored on diffused states).
- Ours, with gradient-only, sampling-only and hybrid guidance.

**Metrics:** task success, fall rate, cost achieved, **consistency gap** ‖ŝ_diffused − s_real‖, smoothness and
energy, control frequency and latency, and the sim-to-real success drop.

**Ablations:** β (consistency term), κ (uncertainty term), M (number of samples), K_g (number of guided steps),
ensemble size, world-model horizon errors, and whether the world model is trained on expert data only or on
data with injected noise.

**Key figures to plan for:**
- **Fig. 1:** the same diffusion policy and the same cost, guided by state gradients (falls) versus by our method
  (succeeds), on hardware.
- **Fig. 2:** success as a function of guidance strength, for both methods. We expect the baseline to collapse
  and ours to degrade gracefully.
- **Fig. 3:** a tasks × methods table.
- **Fig. 4:** a teaser where uncertainty-aware guidance refuses an infeasible request and stays safe.

---

## 5. Timeline (about 7 months to a strong submission)

| Month | Milestone |
|---|---|
| 0–0.5 | **Toy proof of concept in this repo**: cartpole swing-up with the existing diffusion policy, world model and MPPI code. Show H1 (consistency gap) and H2 (non-differentiable cost) in 2D. This becomes a supplementary figure and de-risks the idea. |
| 0.5–2 | Set up mjlab with the G1. Reproduce the BeyondMimic tracking experts (the `whole_body_tracking` repo is public). Collect data. |
| 2–3 | Train the diffusion policy and the world-model ensemble. Reproduce gradient guidance (the baseline). |
| 3–4.5 | Implement WM-PIG. Run the simulation benchmark and the ablations. Write the theory section. |
| 4.5–6 | Sim-to-sim checks, then hardware: latency engineering and real runs. |
| 6–7 | Writing, video, supplementary material. Open-source on mjlab, which earns goodwill and citations. |

**Compute:** a single RTX 4090/5090-class GPU is enough for mjlab training plus diffusion and world-model
training. Your `requirements.txt` already targets a Blackwell GPU. Onboard inference needs a Jetson Orin or a
laptop GPU tethered to the robot.

## 6. Where to submit (deadlines are approximate; check each venue's site)

| Venue | Typical deadline | Fit |
|---|---|---|
| **RSS 2027** | late Jan 2027 | Excellent if the hardware results are ready (tight) |
| **IROS 2027** | ~1 Mar 2027 | Good. A realistic first target. |
| **CoRL 2027** | ~late May / Jun 2027 | **Best fit** (learning + hardware). Recommended main target. |
| **Humanoids 2027** | ~Jul 2027 | Safe fallback; strong community fit |
| **IEEE RA-L / T-RO** | rolling | Use for the journal version with full theory and extended experiments (a typical PhD chapter) |

Suggested plan: an RA-L submission with an IROS option around March, or CoRL 2027, followed by an extended T-RO
version as a thesis chapter.

---

## 7. How it connects to this repo (your unfair advantage)

You already have every building block, in toy form:

- `MPPI/`: path-integral weighting, warm start, temperature λ. This becomes the guidance rule.
- `Generative_model/diffusion/` (including the x₀ variant): the diffusion policy and the x₀-prediction head
  needed for τ̂₀.
- `World_model/world_model_training_lab/`: learned dynamics models.
- `Generative_model/amp/`, `RL/PPO`: tracking-expert training.
- `Generative_model/basic_examples/privileged_teacher_*`, DAgger: the teacher → student data pipeline.

**First concrete step (1–2 weeks):** write `Research_proposal/toy_wm_pig_cartpole.py`. It should take the trained
cartpole diffusion policy, train a small world-model ensemble on its rollouts, and compare three things on a
**non-differentiable** cost (e.g. "keep the cart out of the zone x ∈ [0.3, 0.6]", or "θ never exceeds a limit"):
(a) unguided, (b) gradient guidance on diffused states with a smooth proxy, (c) WM-PIG. Plot the consistency gap
against guidance strength. If H1 shows up even on cartpole, the idea is alive.

---

## 8. Risks and fallbacks

| Risk | Mitigation or fallback paper |
|---|---|
| Too slow for real time on the robot | Guide only the last 1–3 denoising steps, use a smaller M, warm-start, or distil guided samples into the policy (as in ReGuide). A paper on *"amortised consequence-aware guidance"* still works. |
| World-model errors over horizon H | Short horizons (H ≈ 16 at 50 Hz is 0.32 s), ensemble pessimism, and fine-tuning the world model on data collected under guidance |
| No hardware | A strong sim-to-sim benchmark plus theory still suits RA-L or Humanoids. Contact Berkeley Humanoid Lite or other open-hardware groups. |
| Someone publishes it first | The pieces are separable: the theory (consistency bound), the non-differentiable humanoid benchmark, or the uncertainty-aware safety result can each stand alone. |

## 9. Backup topics (in case the main one is scooped)

1. **Certifiable guided diffusion for humanoid safety:** combine guided-diffusion humanoid control with
   control-barrier-function / reachability filters in a learned latent space, giving probabilistic safety
   guarantees for fall avoidance. This connects to the Sreenath lab's CBF background and to RISE and the
   fall-safety policies in arXiv 2511.07407.
2. **World-model MPPI on top of a tracking prior:** run MPPI *in the latent space* of a SuperMimic-style world
   model, with a BeyondMimic tracker as the sampling prior. This is a learned-model version of RGB
   (arXiv 2606.25123). It is lower risk, but also less novel.

## 10. Must-read list (read in this order)

1. BeyondMimic: arXiv 2508.08241 (and the `HybridRobotics/whole_body_tracking` code)
2. SuperMimic: arXiv 2610.28480
3. mjlab: arXiv 2601.22074 (and the `mujocolab/mjlab` code)
4. RGB: RL-Guided Whole-Body MPPI: arXiv 2606.25123
5. Feedback World Model guides diffusion policy: arXiv 2605.15705
6. ReGuide: arXiv 2606.28939
7. Diffusion-ES (CVPR 2024); Twisted Diffusion Sampler / SMC guidance (Wu et al., 2023)
8. RISE: test-time steering for humanoid fall recovery: arXiv 2610.30061
9. DreamControl / DreamControl-v2: arXiv 2509.14353, 2604.00202
10. Model-based Diffusion (Pan et al., 2024); MPPI (Williams et al., 2017); MOPO (Yu et al., 2020)
11. Survey hub: https://github.com/YanjieZe/awesome-humanoid-robot-learning
