# PhD Research Proposal (Idea 5): A Stick to Lean On and to Feel With

**Working title:**
*"Third Leg, White Cane: Learning Tool-Mediated Support and Haptic Probing for Humanoid Locomotion"*

**Areas covered:** walking + whole-body control + perception + imitation.

> Written 2026-10-26. Novelty was checked by web search (arXiv full text was blocked here). Re-check before
> committing.

---

## 1. The problem

Humanoids balance on a support polygon of two small feet. Humans facing the same problem **pick up a stick**:
- hikers use trekking poles on steep, icy or loose ground;
- elderly people and patients use a cane when one leg is weak;
- blind people use a white cane to *feel* the ground before stepping;
- people in darkness, smoke or murky floodwater probe ahead with a stick.

A hand-held stick does two jobs at once:
1. **Support.** It adds a third contact anywhere within arm's reach. This enlarges the support polygon and gives
   a lever against slips and pushes.
2. **Sensing.** Tapping and sweeping measure the ground: is there a hole, a step edge, ice or mud? This keeps
   working when cameras fail (darkness, smoke, dust, glare, water covering the ground).

Learned humanoid controllers (BeyondMimic, HOVER, ASAP, AMO, FALCON, and so on) never use a held tool for either
job.

## 2. What exists (the gap)

| Work | What it does | Missing |
|---|---|---|
| Cane-supported humanoid walking (Robotics & Autonomous Systems, about 2015) | Model-based cane walking; Q-learning chooses a usage mode (cyclic, leg-like, preventive) | No learned whole-body policy, no probing, old platform |
| SupraPeds (model-based) | Contact-supported humanoid locomotion with hand contacts | Fixed environment contacts, no tool, no learning |
| ANYmal haptic localisation (arXiv 2108.08015) | Feet as touch sensors in darkness and smoke | Quadruped; uses the feet, which costs a support leg |
| Probing arm on a quadruped (Frontiers 2022); leg-probing of planetary soils | Dedicated probe or leg tests collapsing ground | Quadruped, separate hardware, model-based |
| Learned multi-contact humanoid work (hands on walls or platforms, e.g. APEX, arXiv 2602.11143) | Hands contact fixed environment geometry | No *held* tool, contact location not freely chosen, no probing |
| Sole tactile sensing (TactileStep, arXiv 2609.28959) | Touch sensors in the feet | Senses only where the robot already stepped |

**Gap:** no learned humanoid policy uses a held tool as a third leg, and none uses it as a probe that senses the
ground *before* stepping. Nothing combines the two in a single policy.

## 3. Core ideas and contributions

1. **Tool-in-hand simulation in mjlab.**
   - A stick (or trekking pole) attached to the hand through a compliant grasp, with slip and torque limits.
   - Tip contact with randomised friction (rubber on ice versus rock).
   - Tip force estimated from arm joint torques, as on the real robot (a G1 has no wrist force sensor).
2. **A single policy that learns *when* to lean and *when* to probe.**
   - Reward the task (reach the goal without falling) and charge an energy and time cost for probing.
   - *Hypothesis:* human-like strategies emerge. Examples: the cane on the side opposite a weak leg (as in
     clinical gait practice), 2-point and 3-point gaits on ice, and tap-before-step in darkness.
3. **A haptic terrain memory.**
   - Tip contacts (position, force, slip, compliance) are written into a local terrain belief, such as a small
     egocentric height and friction map with uncertainty.
   - The walking policy reads that belief; this is the perception contribution.
   - With the camera on, the policy *fuses* vision and touch, and should switch to probing when vision is
     unreliable.
4. **Imitation prior (mimic).** Use human motion capture of cane and trekking-pole walking as a style prior
   (AMP or a motion-tracking warm start), so the strategies look natural and transfer to a person's
   expectations. If no dataset exists, record a small one with phone-based motion capture.

## 4. Experiments

**Terrains and conditions:**
- an icy slope (friction 0.1–0.2);
- a steep descent;
- loose rubble;
- a narrow beam;
- stairs in darkness;
- a floor with **hidden holes** (camera disabled or covered by water);
- strong pushes;
- a **weak-leg** condition (one knee torque-limited to 40%).

**Baselines:**
- **B0:** a blind locomotion policy without a stick.
- **B1:** a policy that holds the stick but was trained without it (the stick is just a load).
- **B2:** a vision-based perceptive policy, which fails in the dark.
- **B3:** ours with support only.
- **B4:** ours with probing only.
- **Ours:** support plus probing.

**Metrics:** success and fall rate, slips per metre, speed, stability margin (support-polygon area over time),
false steps into holes, taps per metre, and energy.

**Hardware:** a Unitree G1 with a 3D-printed wrist clamp holding a carbon trekking pole with a rubber tip. This
costs about $50, and a clamp means no grasping is needed. An optional cheap load cell in the pole gives the
ground-truth tip force.

**Headline video:** in the dark, a G1 taps its way across a floor with a hidden hole and steps around it. On ice,
it plants the pole and stays up where the baseline slips.

## 5. Why this is a strong PhD paper

- It is a **new capability with a clear human analogue**, which reviewers understand in one sentence.
- It covers **all of walking, WBC, perception and imitation** in one coherent system.
- It is **cheap and safe hardware**: the G1, a pole, sim in mjlab.
- It has **natural follow-ups**:
  - a crutch for a damaged leg (links to fault tolerance);
  - a pole for wading through floodwater (links to Idea 4);
  - a robot guiding a blind person with a cane-like handle (links to Idea 6).

## 6. Toy prototype (1–2 weeks)

A 2D compass-gait or inverted-pendulum walker with a massless "arm stick" contact:
- Show analytically and in simulation that a stick tip within reach enlarges the region of states the walker can
  recover from (the capturability region) on low friction.
- Add a "hidden hole" world in which the walker can tap before stepping. Compare probing versus no probing in
  falls per metre.

## 7. Risks

| Risk | Mitigation |
|---|---|
| The stick slips in the hand | Use a wrist clamp (no grasp) for the main paper; grasped tools become future work |
| Stick contact in sim is unrealistic | MuJoCo handles a point-tip contact well; randomise tip friction and compliance; validate on hardware |
| Seen as a gimmick | Frame it as *tool-extended support and haptic perception for vision-denied disaster response*, and report quantitative capturability gains |
| The arm is too weak for real support | G1 arm torques allow roughly 30–50 N through a pole, enough for balance correction (not full weight bearing). Thermal limits matter here (see Idea 2). |

## 8. Venues

RSS 2027 (late Jan) or CoRL 2027 (about June) for the full system. Humanoids 2027 or RA-L are fallbacks.
