# PhD Research Proposal (Idea 2, a new problem): Thermal Preview Control for Humanoids

**Working title:**
*"Don't Burn Out: Endurance-Aware Whole-Body Humanoid Control by Planning Heat over Minutes and Acting over
Milliseconds"*

> Written 2026-10-25. Novelty was checked against web search only (arXiv full text was blocked from this
> environment). Re-check before committing; Section 9 lists the closest work.

---

## 1. The problem nobody has solved for humanoids

Humanoid demos last seconds, but real deployments last hours. **Motors overheat.** When they do, the firmware
**derates torque**, cutting maximum joint torque by 40% or more according to industry reports. The policy was
trained assuming full torque, so it stumbles or falls.

- FALCON (arXiv 2505.06776) reports that the wrist motors overheat when carrying more than about 2 kg per arm on
  hardware, even though sim carried more than 3 kg.
- A Unitree G1 getting-up study (arXiv 2502.12152) found the factory controller overheats the arm motors. The
  learned policy instead leans on the stronger leg motors. This is evidence that **a humanoid can choose where its
  heat goes.**
- Industry press calls this the *"thermal wall"* of humanoid mass production (Digitimes, May 2026).

**What exists (only for quadrupeds):**
- arXiv 2603.01631 adds temperature to the observation and adds a thermal penalty reward. On an A1, running time
  went from 7 to 27 minutes. Its stated limitation is that the policy becomes conservative even when the motors
  are cool.
- arXiv 2605.27046 adds a residual policy conditioned on thermal state. Its stated future work is to *"extend
  thermal management to the planning layer."*

Both are **reactive.** They only see the current temperature, and they work with a reward horizon of about
2 seconds (γ = 0.99 at 50 Hz). Heat builds up over **minutes**, so the methods cannot *plan* it.

## 2. The key insight (this is what makes it new)

1. **Timescale separation.** Motor heat follows a slow, low-dimensional, nearly linear lumped RC model, with time
   constants of minutes. Control runs at 50 Hz. So do not ask RL to solve a 30-minute credit-assignment problem.
   **Split it**: a learned fast policy handles the milliseconds, and a cheap model-based planner handles the
   minutes.
2. **Humanoid redundancy is a thermal resource.** A humanoid can carry the same box, do the same dance or stand
   in the same pose with very different per-joint loads. It can bend the knees less, or shift load from the wrists
   to the elbows and shoulders, or from the hips to the ankles. Quadrupeds have far less of this freedom. A
   humanoid can therefore move heat between joints, not just slow down.
3. **In motion tracking, the future is known (preview).** A dance show, a demo program or a factory loop is a
   known sequence of reference clips. That means the robot can **budget heat ahead of time**. It can stay cool
   during easy segments to "save" thermal headroom for the hard jump at minute 12, much as a marathon runner
   paces. No reactive method can do this.

## 3. Method

### 3.1 Thermal model (cheap, identifiable, and the robot reports temperature)

For each joint j, using Joule heating from torque (current ∝ torque):

  C_j · dT_j/dt = α_j · τ_j² + p_j − (T_j − T_amb) / R_j      (optionally coupled to neighbouring joints)

Identify α, p, R and C per motor from logged G1 data. The Unitree low-level state reports motor temperature.
Add this model to **mjlab** so heat is simulated in parallel across thousands of environments, essentially for
free.

### 3.2 Fast layer: allowance-conditioned whole-body tracker (learned)

Train π(a | o, g, **b**), where:
- g is the motion reference (as in BeyondMimic tracking);
- **b ∈ [0, 1]^J is a per-joint "heat allowance"**: 1 means use the joint freely, 0 means spare it.

The reward is tracking quality minus Σ_j w(b_j)·τ_j². Randomise b during training so the policy learns *how to
redistribute* load across redundant joints. The fast policy never needs to reason about minutes; it just obeys b.

**Learned surrogate.** From sim rollouts, fit two small networks:
- P̄_j(b, clip), the mean heat input per joint;
- Ē(b, clip), the expected tracking error.

These summarise the fast layer for the slow layer.

### 3.3 Slow layer: thermal preview MPC (model-based)

Given the upcoming program (clips c₁…c_N) and the *measured* temperatures, choose an allowance schedule
b₁…b_N that solves

  min Σ_k Ē(b_k, c_k)   subject to   T_j(t) ≤ T_max − margin for all j and t,   with linear RC dynamics driven by P̄(b_k, c_k)

The decision variables are few (J × N), and the model is cheap. Solve with **MPPI** (your `MPPI/` code) or SQP,
and re-plan every 1–5 s from measured temperatures. This closed loop absorbs errors in the thermal model and the
surrogate. If the program is unknown (e.g. joystick control), use a forecast of the command distribution; that
gives the no-preview ablation.

### 3.4 Theory contribution

- With timescale ratio ε = (control period) / (thermal time constant) of about 10⁻⁴, use singular-perturbation
  arguments to bound how far the hierarchical solution is from optimal, in terms of ε and the surrogate error.
- Show that **preview strictly helps.** With a convex surrogate, the preview solution is never worse than the best
  reactive (myopic) allowance policy. Give a simple example where the reactive policy is forced to derate and the
  preview policy is not (the "save heat for the jump" example).
- Treat ensemble-surrogate uncertainty with a robust or chance constraint on T_max, so that safety holds with high
  probability.

## 4. Contributions (as they would be written in the paper)

1. The first **endurance-aware whole-body humanoid controller**, and a formulation of humanoid thermal limits as a
   *two-timescale* control problem.
2. **Allowance-conditioned tracking policies** that exploit kinematic and dynamic redundancy to move heat between
   joints.
3. **Thermal preview MPC** that budgets heat over known motion programs, with guarantees on suboptimality and
   constraint satisfaction.
4. A **thermal simulation extension for mjlab** and an endurance benchmark: time to first derating, and completion
   rate for a 30-minute program. Open-source contributions to mjlab tend to draw community adoption and citations.
5. Hardware results: a G1 completes long programs that the baselines cannot finish.

## 5. Experiments

**Tasks:**
- **(T1)** A 30-minute dance or demo show made of LAFAN1 clips with hard segments.
- **(T2)** Repeated box carrying, 2–4 kg (the FALCON wrist problem).
- **(T3)** Repeated sit-to-stand and getting up.
- **(T4)** Walking with a payload, with joystick commands (no preview).

**Baselines:**
- **B0:** A nominal tracker plus factory derating. This is what happens today.
- **B1:** A thermal observation plus a thermal penalty (port of arXiv 2603.01631).
- **B2:** A thermal residual policy (port of arXiv 2605.27046).
- **B3:** Ours without preview (myopic allowance).
- **B4:** Ours without redistribution (a single scalar allowance, i.e. just slowing down).

**Ours:** the full method.

**Metrics:** time to first derating or overheat stop; program completion rate; tracking error, averaged over the
whole program and on the *hard segments*; peak and mean temperature; falls.

**Headline figure:** temperature traces over 30 minutes for each method, with derating events marked. The
baseline curve hits T_max at minute 9 and falls; ours "saves heat" and lands the jump at minute 12.

**Ablations:**
- Surrogate accuracy.
- Re-planning rate.
- Preview length.
- Thermal model mismatch: wrong R and C, or different ambient temperature (summer versus winter tests).

## 6. Toy prototype first (1–2 weeks, in this repo)

Use a **3-link planar arm holding a payload**, which is redundant, so load can move between joints. Add an RC
heat model to each joint:
- Train an allowance-conditioned PPO policy, reusing `RL/PPO`.
- Fit the surrogate.
- Run MPPI over the allowance schedule for a sequence of "hold here, then lift there" targets (reusing `MPPI/`).

The test: does preview plus redistribution extend operation before any joint hits T_max, compared with a
reactive thermal penalty? If yes, you have Figure 1 of the paper and a de-risked PhD chapter.

## 7. Timeline (about 6–7 months)

| Month | Milestone |
|---|---|
| 0–0.5 | Toy prototype (Section 6) |
| 0.5–1.5 | Log G1 motor temperatures under typical motions; identify the RC model; add heat to mjlab |
| 1.5–3 | Allowance-conditioned tracker (start from BeyondMimic `whole_body_tracking`); fit the surrogate |
| 3–4 | Preview MPC; simulated endurance benchmark; baselines B0–B4 |
| 4–5.5 | Hardware endurance tests. These are slow, about 30 minutes each, so batch them and keep a conservative T_max. |
| 5.5–7 | Theory write-up, paper, video, open-source release |

**Hardware needed:** one Unitree G1 (or any humanoid that reports motor temperature), a thermal camera
(optional, good for figures), and one GPU.

## 8. Venues

CoRL 2027 (about late May/June) is the best fit. Alternatives are IROS 2027 (about March) and Humanoids 2027
(about July). An extended version with full theory and more platforms fits **IEEE T-RO** or **RA-L**. Check exact
deadlines on each venue's site.

**Why reviewers will like it:**
- It is a real, unsolved deployment problem that industry is talking about.
- The insight is clean (timescale split, redundancy, preview).
- The results are hard to argue with (minutes of operation versus falls).
- The hardware cost is modest, with no huge compute.

## 9. Closest work (read these to position the paper)

- Thermal-aware quadruped RL: arXiv 2603.01631
- Thermal residual policy for quadrupeds: arXiv 2605.27046 (it names the planning layer as future work)
- Thermal-aware Olaf character robot (Disney): arXiv 2512.16705
- Motor core temperature estimation for musculoskeletal humanoids: arXiv 2407.08055; Kumagai et al. 2014
  (torque-limit adaptation by joint temperature, model-based)
- FALCON: arXiv 2505.06776 (evidence that wrists overheat)
- G1 getting-up policies: arXiv 2502.12152 (evidence that a policy can choose where heat goes)
- BeyondMimic: arXiv 2508.08241; mjlab: arXiv 2601.22074 (the tracker and the simulator you build on)
- Olaf (Disney): arXiv 2512.16705. It says thermal timescales are far slower than balance dynamics, which
  supports our timescale split.
- Human-Level Actuation for Humanoids: arXiv 2511.06796 (thermal limits appear only in integrated tests)
- Model-based two-timescale precedents: two-rate actuator policies (Yagi & Morimoto), singular-perturbation
  hierarchical MPC (NAIST thesis), and thermal-constrained manipulator MPC (Pisa thesis)
- Toy evidence that the idea works: `toy/README.md` (preview plus per-joint allowance completes 5/5 programs
  with no derating, versus at most 4/5 for the baselines)

## 10. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Someone publishes thermal-aware *humanoid* RL first | Your differentiators are **preview** and **redistribution**, which a reactive humanoid port will not have. Position the work as the planning layer. |
| The policy ignores b | Add a curriculum on b, and a hard torque clamp scaled by b as a fallback actuation-level constraint |
| The RC model is inaccurate | Closed-loop re-planning from measured temperature, plus online RC parameter estimation (as in arXiv 2407.08055) |
| Risk of hardware damage | Keep T_max well below the firmware limit and use a watchdog. Thermal tests are much less dangerous than agile stunts. |
| No humanoid available | The toy arm, a simulated G1 with realistic RC parameters, and one borrowed quadruped or arm can still support a strong RA-L paper |
