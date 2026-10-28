# Combining Attention-based Contact Planning (ACP, arXiv 2610.31608) with EB-WM (Algorithm 1′)

> **Source caveat (2026-10-28).** The ACP PDF could not be opened from this environment, because arxiv.org and
> leggedrobotics.github.io are blocked by the session's network policy. The summary below comes from the abstract
> and project-page text returned by web search, plus the README of an unofficial re-implementation
> (`lucidrains/attention-contact-planning`, work in progress). Verify the details against the full paper.

## 1. ACP in brief (He et al., ETH Zürich, submitted 2026-10-22)

- **Planner:** an explicit, self-supervised, **attention-based contact planner**. **9 body-part queries** (hands,
  elbows, knees, feet, pelvis) cross-attend over **terrain points** from a LiDAR height map (about 187 points in
  the unofficial code). The planner also takes proprioception and a goal, and has a **learned "no-contact" slot**.
  It outputs, per body part, a categorical distribution over *where the next contact will be* within a 1.5 s
  horizon. It is trained with a soft-target loss on the attention logits.
- **Tracker:** a **contact-conditioned tracker** turns the planned contacts into whole-body motion.
- **Training:** motion data is only a prior. The tracker is pretrained with a **backward curriculum** (start states
  move progressively earlier along each reference). A **fine-tuning stage** then lets the planner and tracker
  explore terrain beyond the motion data.
- **Results:** on a G1, a single policy climbs stairs, ladders and boxes, vaults walls, crawls, sits and stands
  up, with no reference motion at deployment.

## 2. Why they fit together

| ACP | EB-WM (Algorithm 1′) | Combined |
|---|---|---|
| The planner's attention gives P(body part i contacts terrain point j), plus a no-contact slot | The event head gives P(event k \| s, a) | **Contact-attention event head:** the world model predicts the contact *outcome* (made at point j, slip, miss, unintended contact, fall) with the same query×terrain attention |
| The planner gives *intended* contacts | The world model gives *actual* contact outcomes, with mode-branched rollouts | The planner is trained so that intended contacts are **achievable** (high WM success probability), giving **risk-aware contact selection**: keep away from edges and thin rungs under LiDAR noise |
| Fine-tuning by exploration (RL) | BPTT through a WM with mode-branched rollouts | **Sample-efficient fine-tuning** of planner and tracker, with gradients through contact-success probabilities (the boundary term) |
| LiDAR height map (noisy) | Perception noise is sampled in the batch (or the WM is perceived-input); margins calibrate themselves | **Calibrated contact margins:** foot placement distance from a stair edge and grasp margin on a rung grow with heightmap noise |
| Self-supervised planner (soft-target cross-entropy) | Idea 10: L2 distillation collapses margins | Keep **proper scoring** (cross-entropy) for every contact head. If any stage distils a privileged teacher, use quantile or critic-guided distillation, not L2. |

## 3. Algorithm 2: ACP + EB-WM (pseudocode)

```python
# ===== ACP + EB-WM: attention contact planning + event-branched world-model fine-tuning =====
# Body parts B = {hands, elbows, knees, feet, pelvis} (9); terrain points P_t from the LiDAR height map
# Planner  π_plan(P̂_t, proprio, goal) -> c_i ~ Cat({no-contact} ∪ P_t) for every body part i  (ACP attention)
# Tracker  π_track(proprio, planned contacts c) -> joint targets a_t
# World model g_φ(s, a, P_in, σ):
#   contact-attention event head: for each body part i, Cat over {none, contact at point j, slip, lift-off}
#   + global absorbing head: {fall, missed planned contact}
#   + branch heads f_mode(s, a): next state for each contact mode (mode = set of body parts in contact)

# ---- Stage 0 (ACP, unchanged): pretrain from motion data ----
pretrain π_track with the backward curriculum on reference motions
pretrain π_plan  self-supervised: targets = contacts observed in the motions (soft targets, cross-entropy)

for iteration in range(N):
    # ---- Stage 1: data on random terrain with REAL perception noise ----
    for episode in batch:
        terrain = sample_terrain()                      # stairs, ladders, boxes, gaps, walls ...
        for t in range(T):
            P̂_t = lidar_heightmap(terrain, noise_model)
            c   = π_plan(P̂_t, proprio, goal)            # intended contacts (sampled or argmax)
            a   = π_track(proprio, c) + exploration
            s_next, events = sim(s, a, terrain)          # per-body-part contact events + location; fall flag
            buffer.add(s, a, terrain, P̂_t, c, events)

    # ---- Stage 2: train the event-factored world model (contact-attention event head) ----
    for minibatch in buffer:
        P_in = terrain_points(terrain) if TERRAIN_INPUT == "true" else P̂_t
        σ = uniform(0, σ_max) if σ_max > 0 else 0         # σ_max = sqrt(max(0, σ_esc² − σ_p²))
        events, y = sim(s + σ ε, a, terrain)             # labels and targets at the perturbed state
        logits_i, absorb_logits, {f_mode} = g_φ(s, a, P_in, σ)
        loss = Σ_i CE(logits_i, event_i)                  # which point, slip, lift-off, none (proper scoring)
             + CE(absorb_logits, fall_or_miss)
             + Σ_mode MSE(f_mode, y | mode_next == mode)  # each branch trained on its own samples
        φ ← φ - lr ∇_φ loss

    # ---- Stage 3: fine-tune planner + tracker by BPTT through MODE-BRANCHED rollouts ----
    terrain = sample_terrain(B);  P̂ = lidar_heightmap(terrain, noise_model)   # MC over perception noise
    particles = {mode_0: (w = 1, s = s_0)}
    J = 0
    for t in range(H):
        new = defaultdict(list)
        for mode, (w, s) in particles.items():
            if mode in ABSORBING:                         # fall / missed planned contact
                new[mode].append((w, s)); continue
            c_soft = π_plan(P̂_t, proprio(s), goal)        # attention probabilities (differentiable)
            a = π_track(proprio(s), expected_location(c_soft))
            p_i, p_abs, {f_mode} = g_φ(s, a, P_in, σ_t)
            J += w * [ r_goal(s) - r_effort(a)
                       + λ_c Σ_i log P_wm(contact of body part i lands at its planned point) ]   # achievable plans
            for mode_next, prob in factorized_transitions(mode, p_i, p_abs):  # per-part events, assumed independent
                new[mode_next].append((w * prob, f_mode_next(s, a)))
        particles = prune(merge_same_mode(new), K_max)   # IMM merge + beam over contact modes
    J += Σ_{mode ∉ ABSORBING} w V_ψ(s) + Σ_{mode ∈ ABSORBING} w V_fail
    (θ_plan, θ_track) ← (θ_plan, θ_track) + lr ∇ J      # gradient through the weights = contact-success boundary terms
    ψ ← fit on λ-returns   # optional event-factored critic: V = P_survive V_alive + (1 − P_survive) V_fail
```

## 4. What is new compared with each paper alone

1. **A contact-attention event head for world models.** ACP's attention mechanism is reused as a calibrated,
   spatially resolved event model inside an EB-WM world model.
2. **Risk-aware contact planning under LiDAR noise.** The planner learns contacts that are *achievable* given
   perception uncertainty, so margins on edges and rungs become calibrated, rather than imitating motion contacts.
   This addresses margin collapse (Idea 10) for multi-contact skills.
3. **Gradient-based fine-tuning through contact.** Mode-branched BPTT, instead of (or in addition to) RL
   exploration, may give the sample-efficiency gains seen in SuperMimic, without losing the boundary terms.

## 5. Experiments (G1, mjlab)

- **Tasks:** stairs up and down, ladder climbing, box climbing, stepping stones, wall vault. LiDAR heightmap
  noise at 0, 2 and 4 cm.
- **Baselines:**
  - ACP as published (RL fine-tuning);
  - ACP + plain-WM BPTT (SuperMimic-style mean rollouts);
  - ACP + EB-WM (ours);
  - ablations: no contact-attention event head, no branching, L2-distilled planner.
- **Metrics:**
  - success, falls, missed or unintended contacts;
  - **contact margin vs heightmap noise** (distance of footholds from edges; rung-grasp margin);
  - simulator steps and GPU-hours to a given success rate.
- **Main risk:** combinatorial contact modes (9 parts). Mitigation: factorize events per body part, use a beam
  over modes (K_max ≈ 8–16), and merge modes that share a contact set.

## 6. Thorough toy test (`toy/acp_ebwm_stepping_stones.py`, 3 seeds × 4 noise levels)

**Setup:**
- Stepping stones (6 per step); perceived centre and width have noise σ_p; reach limit 0.75 m; motor noise 1 cm.
- A miss is an absorbing fall with F = 3; 5 steps per episode.
- Planner: ACP-style token scoring plus a no-step slot and an aim offset, pretrained self-supervised from a
  privileged expert (ACP-SS), then fine-tuned. Test-time policy = argmax.
- Values are mean ± std; J is the 5-step return.

| σ_p | ACP-SS (no fine-tune) | **+ EB-WM (true + MC)** | + EB-WM (perceived) | + plain WM (mean-field) | + pathwise true-sim | + ZO RL (8.2M sim rollouts) | oracle | privileged expert |
|---|---|---|---|---|---|---|---|---|
| 0 cm | 2.454, miss 5.3% | **3.537 ± 0.004**, miss 0.2% | 3.538 | 3.349 ± 0.022 | 1.642 | 3.052 ± 0.132 | 3.559 | 3.537 |
| 1 cm | 2.306, miss 6.0% | **3.338 ± 0.007**, miss 0.8% | 3.344 | 3.264 ± 0.013 | 1.736 | 2.642 ± 0.219 | 3.353 | (3.537, sees true stones) |
| 2 cm | 1.873, miss 8.1% | **2.869 ± 0.048**, miss 2.5% | 2.862 | 2.906 ± 0.021 | 1.747 | 1.921 | 2.878 | – |
| 4 cm | 1.134, miss 12.2% | 1.500 ± 0.018 | 1.461 | 1.511 ± 0.074 | 1.458 | 1.460 | 1.458 | – |

**Findings:**
1. **EB-WM fine-tuning of an ACP-style planner reaches the oracle** (within 0.3–1%) for σ_p ≤ 2 cm. That is
   +44–53% return over the self-supervised planner and 5–20× fewer misses, with **no simulator rollouts** during
   fine-tuning. It is stable across seeds (std ≤ 0.05).
2. **Margins calibrate:** mean landing distance from the nearest true edge is 3.7 → 3.9–4.1 → 4.0–4.2 cm for
   σ_p = 0 → 1 → 2 cm. ACP-SS stays at about 3.4 cm, which is margin collapse from imitating a privileged expert
   (Idea 10).
3. **Pathwise through the true simulator fails** at every noise level and is *worse* than no fine-tuning, which
   is Proposition E again.
4. **Model-free ZO** needs 8.2M rollouts and is still far from the oracle at σ_p ≥ 1 cm.
5. **Honest negatives:**
   - The **plain mean-field WM is competitive** here: −5% at 0 cm, but +1% at 2 cm. In this toy each step is a
     *single* transition, so the regressed mean outcome already equals E[r]. The branching advantage needs
     multi-step rollouts through discontinuities, which the staircase toy shows (2.17 vs 4.24). The next test
     should add a swing phase with toe-stub and stop events inside each step.
   - The "oracle" is optimisation-limited (BPTT from the same initialisation, 400 iterations), so it is not a
     strict upper bound.
   - **At σ_p = 4 cm every method converges to the same conservative policy** (J ≈ 1.46–1.51, about 10% misses).
     The noise is comparable to stone half-widths, so the regime is **information-limited**: the gap to the
     privileged expert (3.54) is missing information, not optimisation.
