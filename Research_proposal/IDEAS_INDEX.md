# Research ideas: index and status

Updated every loop iteration. Current focus (user request, 2026-10-26): the **fundamental learning track** below. Earlier focuses were applications, then core areas (walking, WBC, perception, imitation, collaboration). "Open" means no matching paper was found in web search as of the date; always
re-check arXiv before committing.

## Fundamental learning track (current focus, at the user's request on 2026-10-26)

| # | Idea | File | Novelty | Evidence so far | Recommendation |
|---|---|---|---|---|---|
| 7 | **What can an action residual fix?** The matched/unmatched condition for ASAP-style sim-to-real corrections, the diagnostic ρ, and a generalized-force fix | `IDEA_7_matched_unmatched_sim2real.md` | **Open**: no paper states the matching condition for delta-action sim-to-real | Cartpole toy: replay error confirms Propositions 1–3. **Closed-loop transfer** (generic features, 3 seeds): matched → every correction 100%; unmatched → 1-step δ 0% (worse than no correction), ASAP-like δ 34% (unstable across seeds), state residual 97% (= oracle). Proposition 4 (DR) is *not* confirmed: DR reached 73% on the unmatched gap | **Top pick** |
| 8 | **Event-Branched, Perception-Calibrated World Models (EB-WM)**: gradients through contact for perceptive humanoid locomotion (stairs, terrain, mimic) | `METHOD_event_branched_world_models.md`, `IDEA_8_smoothing_world_models.md` | Open as a combination: argument- vs input-noise theory, boundary-term proposition, event-branched rollouts, smoothing budget | **Stable toy results (5 seeds):** σ_p = 2 cm J 0.566 ± 0.005, 0.14% trips, escape 5/5; all ablations worse | **⭐ Main recommendation** (Idea 7 second) |
| 9 | Restart distributions for motion imitation from the performance-difference lemma | `IDEA_8_9_fundamental_sketches.md` | Partly open (RCG, prioritized restarts, SFV) | none yet | Third |

Fundamental ideas checked and rejected: **action-chunking theory for behaviour cloning** (Zhang, Pfrommer, Pan,
Matni & Simchowitz, ICLR 2026, arXiv 2507.09061, plus arXiv 2610.01626); **optimal-transport / DTW rewards for
infeasible or misaligned references** (Flow-Matched Motion Priors arXiv 2609.15631, ORCA arXiv 2502.05397, rough
demonstrations arXiv 2206.11693); **adaptive tracking σ** (KungfuBot); **bias of the privileged (asymmetric)
critic** (Baisero & Amato arXiv 2105.11674; Informed AAC arXiv 2509.26000); **closed-loop identifiability with
active probing** (SPI-Active, CoRL 2025; arXiv 2607.21867; ASID arXiv 2404.12308); **plasticity in humanoid
fine-tuning** (PPF arXiv 2504.09833, partly). Open but compute-heavy: **is the critic the bottleneck in universal
motion tracking?** (SuperMimic notes that critic calibration "has not been systematically evaluated").

## Application-driven ideas (earlier; kept for reference)

| # | Idea | File | Novelty (2026-10-26) | Hardware need | Recommendation |
|---|---|---|---|---|---|
| 1 | World-model-grounded sampling guidance for diffusion policies | `README.md` | Combination of existing pieces; crowded area | G1 | Backup only |
| 2 | **Thermal preview control** (plan motor heat over minutes, move load across redundant joints) | `IDEA_2_thermal_preview_humanoid.md` | **Open** for humanoids; quadruped work is reactive only | G1 + thermal logs | **Main paper.** The toy prototype already works (`toy/`). |
| 3 | **Humanoids in moving vehicles** (gravito-inertial ambiguity breaks "projected gravity") | `IDEA_3_humanoid_in_moving_vehicles.md` | **Open** for learned humanoid control (re-checked 2026-10-26); there is a 2026 survey and estimation-only prior work | G1 + motion platform or rolling cart | **Strong second paper.** The toy works: platform-acceleration estimation survives 2–3× the braking of the IMU-vertical design (`toy/`). Escalators are an extension. |
| 4 | Humanoids wading through flowing water | `IDEA_4_humanoid_wading_sketch.md` | **Open** (no learned humanoid water locomotion found) | Needs waterproofing, which is a risk | Sketch only; possible chapter in "the world moves" |
| 5 | **Cane / trekking pole as a third leg *and* a probe** (walking + WBC + perception + imitation) | `IDEA_5_cane_support_and_probing.md` | **Open**: only a 2015 model-based cane humanoid and quadruped probing arms exist; no learned humanoid tool support or probing | G1 + a $50 wrist-clamped pole | **Top pick in the core areas** (cheap, safe, covers every area the user asked about) |
| 6 | **Humanoid physically assisting people** (stand up, walk, catch a stumble) (collaboration + WBC + imitation) | `IDEA_6_physical_assistance.md` | **Open**: there is co-carrying of objects, dance and exoskeleton simulations, but no learned humanoid assistance of a person | Mannequin first, then volunteers (IRB); G1 arms are weak, so balance support only | High impact; ethics and arm strength are risks |

## A PhD thesis storyline built from these ideas

**"Humanoids in the Real World: Control under Slow Hidden States and Moving Frames"**
- Chapter 1, the robot's own slow states: heat (Idea 2), then battery state of charge and voltage sag, then
  mechanical wear. These follow the same two-timescale recipe.
- Chapter 2, the world's motion: moving vehicles (Idea 3), then ships and lifts.
- Shared message: standard humanoid RL assumes a *fresh robot on static ground*. Real deployment violates both
  assumptions, and each violation has a principled fix.

## Ideas checked and rejected (already taken)

| Idea | Why rejected | Evidence |
|---|---|---|
| Humanoid motor or joint-failure fault tolerance | TOLEBI (arXiv 2602.05596) is the first learned humanoid version; there is also FT-WBC (arXiv 2606.24466) and many quadruped papers | crowded |
| Fatigue or wear-aware torque policy | SATA (arXiv 2502.12674) keeps a fatigue state in the policy (quadruped) | partially taken; could become a thesis chapter on gearbox impact damage |
| Battery voltage-sag modelling | arXiv 2608.29769 models sag and adds a penalty (humanoid jumping) | partially taken; *planning over state of charge* is still open (thesis chapter) |
| Hardware co-design from tracking policies | BRIDGE (arXiv 2609.03497), EA-CoRL (arXiv 2509.26082) | taken |
| Humanoid walking in strong wind | Covered in practice by disturbance observers (ADAPT, arXiv 2606.16542) and push randomisation; one biped wind-DDPG paper exists | weak novelty: wind is just a distributed push |
| Active gaze for perceptive humanoid locomotion | TAGA (arXiv 2606.05880) learns terrain-aware active gaze (attention over the height scan) | taken (a physical neck camera remains a small gap) |
| Human–humanoid interaction mimic from two-person motion capture | CoDance (arXiv 2610.05324), PAIR (CVPR 2026), Harmanoid (arXiv 2510.10206) | taken |
| Multi-humanoid cooperative carrying | decMHT (arXiv 2609.17824), COLA (arXiv 2510.14293), MASkillBlender, Humanoid Hanoi (CoRL 2026) | taken |
| Policy-controlled time warping of reference motions | AdaMimic (arXiv 2510.14454); feasibility-index time warping (IEEE 2015) | taken |
| Carrying a box that blocks the view of the feet | HAIC (arXiv 2602.11758), FootQuery (arXiv 2609.21447), RPL (arXiv 2602.03002) | taken |
| Falsification / stress testing of humanoid policies | Stress tests of SPARK safety filters (arXiv 2605.19009); selective adversarial training (arXiv 2507.08303) | partially open, but the contribution would be methodological; parked |
