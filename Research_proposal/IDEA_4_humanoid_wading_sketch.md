# Idea 4 (sketch, not yet a full proposal): Humanoids that Wade

*"Knee-Deep: Learning Humanoid Locomotion through Flowing Water"*. Status: candidate, found 2026-10-26.

**Problem.** Flood rescue, beaches, rivers and flooded factories are prime humanoid jobs ("go where people
can't safely go"). Water adds four effects:
- quadratic drag on the shins and thighs, which grows with depth and current;
- buoyancy, which unloads the feet and lowers friction;
- added mass on the swinging legs;
- waves, which push periodically.

A policy trained on dry ground would mis-time its swing and slip.

**Novelty check (web search, 2026-10-26):**
- Prior work: a model-based "underwater ZMP" for a hydraulic legged robot (2019), a Waseda humanoid walking in
  water with a waterproof suit (2016), and SLIP models with added mass (FSU dissertation, 2026).
- **No learned or RL humanoid control in water was found.**

**Core idea.** Model per-link drag, buoyancy and added mass as depth-dependent forces in mjlab. These are cheap
analytic terms, applied as external wrenches per link. Train with randomised depth, current and waves. Add
proprioceptive estimation of depth and current: drag shows up as a residual joint torque, in the spirit of the
ADAPT disturbance observer (arXiv 2606.16542). Then adapt the gait by raising the swing, leaning into the
current and taking shorter steps, as human waders do.

**Risks: why this is only a sketch.**
- **Hardware.** A G1 is not waterproof. You would need waterproof leg covers and to test only at ankle or knee
  depth, or partner with a lab that has a sealed humanoid. Without hardware it is a sim-only paper (weaker venue).
- The fluid model's fidelity would be questioned. Validate it with a single-leg drag test in a water tank.

**Relation to the thesis storyline.** It fits "the world moves" (Chapter 2) as a *moving medium*, next to moving
platforms (Idea 3).
