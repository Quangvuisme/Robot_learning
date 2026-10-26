# PhD Research Proposal (Idea 6): Helping Hands

**Working title:**
*"Helping Hands: Learning Whole-Body Humanoid Policies that Physically Assist People to Stand, Walk and Not
Fall"*

**Areas covered:** human–humanoid collaboration + whole-body control + imitation.

> Written 2026-10-26. Novelty was checked by web search only. Re-check before committing.

---

## 1. The problem

Elder care is the most-cited reason to build humanoids, yet a 2026 scoping review of 59 humanoid studies with
older adults (JMIR Aging) found the robots were used mainly for:
- **social interaction** (39%);
- **cognitive training** (25%);
- **physical-activity facilitation** (19%).

Almost none *physically* assist a person.

Physical assistance is the hard part, for four reasons:
1. The robot must **stay balanced while exchanging large forces with a person**, roughly 50–300 N through the
   hands.
2. The person is **active, unpredictable and fragile.** They push back, hesitate, and sometimes suddenly collapse.
3. **Safety** means limiting contact forces, protecting the person's head, and never letting go at the wrong
   moment.
4. **"Assist as needed."** Too much help weakens the patient; too little lets them fall.

## 2. What exists (the gap)

| Work | What it does | Missing |
|---|---|---|
| Human–humanoid co-carrying: COLA (arXiv 2510.14293), Cognition-to-Control (arXiv 2603.03768) | A humanoid and a human jointly carry an *object* | The partner is not the load; no fragile body in the loop |
| CoDance (arXiv 2610.05324); PAIR (CVPR 2026) | Compliant human–humanoid dance and interaction learned from two-person data | Light touch, not weight-bearing; no impairment or safety model |
| Musculoskeletal DRL for sit-to-stand with an idealised exoskeleton (Stevens, 2024; PubMed 2026) | A simulated human with assistance torques at the joints | An exoskeleton, not a separate humanoid that must balance itself |
| Robotic touch for walking guidance of older users (arXiv 2607.09323); caregiving-robot position papers (arXiv 2606.28592); GenPHRI (arXiv 2604.08664) | Human-factors studies, conceptual safety, and generative pHRI simulation | No learned whole-body assistance controller |
| FALCON (arXiv 2505.06776) | A force-adaptive humanoid under external loads | The load is passive; it is not a person with intent |

**Gap:** no learned humanoid whole-body policy physically assists a person to stand, walk or recover from a
stumble while keeping both of them safe.

## 3. Core ideas and contributions

1. **A simulated "person to be helped."** This is a key enabler and a contribution in its own right.
   - A physics human (an SMPL-shaped character in MuJoCo) driven by a motion-prior policy for sit-to-stand,
     walking and stand-to-sit.
   - **Configurable impairment:** strength scaled to 30–100%, delayed reactions, tremor, fatigue, and **sudden
     collapse events.**
   - Train a *population* of such human policies so the robot sees diverse partners. This is co-training, as in
     multi-agent RL.
   - Validate it against recorded human sit-to-stand kinematics (public datasets).
2. **An assistance policy for the robot.** It controls the whole body (balance plus arm impedance) and observes:
   - the person's pose, from cameras or motion capture in the lab;
   - hand-contact forces, estimated from arm joint torques.

   Its objective is the **assist-as-needed** reward: task success plus person stability, minus robot effort
   transmitted, minus how far the person's motion deviates from their natural coordination (following the
   muscle-synergy literature).
3. **Hard safety constraints:** a contact-force cap, a minimum hold time, and a fall-protection mode (lower the
   person and protect their head) enforced by a safety filter on top of the policy.
4. **Imitation (mimic).** Learn assistance styles from **caregiver–patient two-person motion capture**, which
   physical-therapy recordings could provide. Use contact-aware interaction retargeting, as in PAIR, as a prior,
   so the robot assists the way trained caregivers do.

## 4. Experiments

**Tasks:**
- **(T1)** sit-to-stand assistance, by hand-holding or a forearm under the armpit;
- **(T2)** hand-in-hand walking support;
- **(T3)** stumble catching or fall prevention;
- **(T4)** stand-to-sit lowering.

**Evaluation in sim:**
- success across impairment levels;
- peak contact force and the person's joint loads;
- robot stability margin;
- how much assistance was used.

**Evaluation on hardware, staged for safety:**
- **(i)** a weighted, jointed **mannequin or instrumented dummy**;
- **(ii)** healthy volunteers *simulating weakness* (as many assistive-robotics studies do), with a safety
  harness and force-sensing handles, under IRB approval.

**Baselines:** a scripted impedance controller; FALCON-style force adaptation without a partner model; ours
without impairment diversity; ours without the safety filter (sim only).

## 5. Honest constraints

- **Robot strength.** G1 arms are rated for only a few kg of payload per arm, so a G1 can do **balance support
  and guidance** (tens of newtons) but cannot lift a person. Do full sit-to-stand assistance in sim and on a
  stronger humanoid if one is available (H1-class or larger). Sustained arm loads also overheat the motors, which
  connects to Idea 2.
- **Ethics.** Approval for experiments with people takes months, so start the application early. The mannequin
  results can carry the first paper.

## 6. Why this is a strong PhD topic

It has **very high impact** (elder care), a **clear gap** for learned humanoid control, and a reusable
**simulation asset** (the impaired-human population) that others will cite. It spans collaboration, WBC and
imitation, and it scales naturally into a thesis:
- support while standing, then while walking, then fall catching;
- then dressing and transfers (bed to wheelchair).

## 7. Venues

CoRL 2027, RSS 2027 or HRI 2028 (the human-robot interaction community values the human studies). T-RO or
Science Robotics for the full system with user studies.
