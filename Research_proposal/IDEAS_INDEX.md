# Research ideas: index and status

Updated every loop iteration. "Open" means no matching paper was found in web search as of the date; always
re-check arXiv before committing.

| # | Idea | File | Novelty (2026-10-26) | Hardware need | Recommendation |
|---|---|---|---|---|---|
| 1 | World-model-grounded sampling guidance for diffusion policies | `README.md` | Combination of existing pieces; crowded area | G1 | Backup only |
| 2 | **Thermal preview control** (plan motor heat over minutes, move load across redundant joints) | `IDEA_2_thermal_preview_humanoid.md` | **Open** for humanoids; quadruped work is reactive only | G1 + thermal logs | **Main paper.** The toy prototype already works (`toy/`). |
| 3 | **Humanoids in moving vehicles** (gravito-inertial ambiguity breaks "projected gravity") | `IDEA_3_humanoid_in_moving_vehicles.md` | **Open** for learned humanoid control (re-checked 2026-10-26); there is a 2026 survey and estimation-only prior work | G1 + motion platform or rolling cart | **Strong second paper.** The toy works: platform-acceleration estimation survives 2–3× the braking of the IMU-vertical design (`toy/`). Escalators are an extension. |
| 4 | Humanoids wading through flowing water | `IDEA_4_humanoid_wading_sketch.md` | **Open** (no learned humanoid water locomotion found) | Needs waterproofing, which is a risk | Sketch only; possible chapter in "the world moves" |

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
| Falsification / stress testing of humanoid policies | Stress tests of SPARK safety filters (arXiv 2605.19009); selective adversarial training (arXiv 2507.08303) | partially open, but the contribution would be methodological; parked |
