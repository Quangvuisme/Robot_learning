# PhD Research Proposal (Idea 3, a new problem): Humanoids that Ride

**Working title:**
*"Which Way Is Down? Learning Humanoid Balance and Walking inside Accelerating Vehicles under Gravito-Inertial
Ambiguity"*

> Written 2026-10-26. Novelty was checked by web search only (arXiv full text was blocked here). Re-check before
> committing.

---

## 1. The problem

Humanoids are meant to work in our world, and our world moves. Examples: delivery robots riding in vans, buses,
trains and lifts; robots on ships and offshore platforms; robots in aircraft cabins and hospital beds being moved.
A braking bus reaches about 0.15–0.3 g. A ship deck rolls by tens of degrees with 8–10 s periods, according to
the survey arXiv 2604.20990.

Almost every learned humanoid controller (BeyondMimic, HumanPlus, ASAP, and so on) feeds the policy a
**"projected gravity"** vector. That vector comes from an IMU orientation filter, and the filter *uses the
accelerometer to find "down."* Inside an accelerating vehicle, the accelerometer measures gravity **plus** the
vehicle's acceleration. This is Einstein's equivalence principle: a single accelerometer cannot tell a tilt from
an acceleration.

So the robot's sense of "down" tilts. When a bus brakes at 0.2 g, the apparent vertical is off by about 11°.
The policy, trained on a static ground, "corrects" a tilt that is not there and falls forward. Human pilots have
the same failure; it is called the **somatogravic illusion**. Bus passengers fall for the same reason.

**This is a principled failure mode of the standard humanoid observation design, and no paper studies it.**

## 2. What exists (the gap)

- **Survey of legged robots in non-inertial environments** (arXiv 2604.20990): defines the area, mostly for
  quadrupeds and model-based control. It says ship motion needs *predictive* adaptation.
- **Invariant EKF for humanoid locomotion on dynamic rigid surfaces** (arXiv 2201.10636): state estimation only,
  with no learned policy.
- **Multi-IMU fusion** (arXiv 2507.11447): notes that the assumption "gravity dominates the accelerometer" fails,
  and that foot IMUs help during contact.
- **Humanoid scooter riding** (Sensors 2026), **HuB extreme balance** (arXiv 2505.07294) and **load-aware
  locomotion** (arXiv 2603.14308): related balance skills, but not in accelerating frames.
- **The Chalmers project** on human balance in public transport (2027–28): it models *humans*, not robots. It is
  useful for data and for a human comparison.

**Gap:** no learned humanoid controller (1) operates in non-inertial frames, (2) resolves gravito-inertial
ambiguity, and (3) uses whole-body strategies (stepping, leaning, grabbing a handrail) to ride vehicles.

## 3. Key ideas (contributions)

1. **Identify and quantify the failure.** Show, in sim and on hardware, that standard policies with projected
   gravity fail under realistic vehicle accelerations. Give a closed-form prediction of the apparent-vertical
   error, atan(a/g), and show that falls match it. This alone is a solid workshop or RA-L result.
2. **Contact-kinematic disambiguation.** While a foot is planted, it moves *with the vehicle*. Leg kinematics
   plus the IMU then give the robot's motion relative to the platform, which separates *platform acceleration*
   from *tilt*. Build a small estimator: an invariant EKF on a moving surface (extending arXiv 2201.10636), or a
   learned estimator trained in sim with privileged platform state. It outputs both the platform-frame gravity
   and the platform acceleration, and both go to the policy.
3. **Platform-aware policy with anticipation.** Train in **mjlab** with a randomised moving platform: bus and
   train acceleration profiles, ship roll, pitch and heave spectra, and lift jerk. Train a teacher with privileged
   platform state and distil it into a student that uses only the estimator. Optional *anticipation*: vehicles
   are predictable (a bus decelerates before a stop, and waves are periodic), so a short-horizon platform-motion
   predictor lets the robot lean *before* braking peaks.
4. **Whole-body riding strategies.** The humanoid's arms are a resource. Learn when to grab a handrail or pole
   versus when to step or lean. This is the humanoid-specific part, since quadrupeds cannot do it. Include a
   "carrying a box on a moving bus" task (loco-manipulation).

## 4. Experiments

**Sim:**
- A rigid platform driven by real recorded profiles (public bus-braking datasets, train acceleration logs, and
  standard ship motion spectra such as JONSWAP waves to roll and pitch).
- Tasks: stand, walk along the aisle, carry a box, and stand without hands versus with a handrail.

**Hardware (no vehicle needed at first):**
- Put the G1 on a **motion platform or a pushed rolling cart**, with a safety gantry. A Stewart platform or a
  6-DoF motion simulator from a driving-simulator lab is ideal, and a heavy-duty rolling cart pulled with a
  force-sensing rope is cheap.
- Later, a real minibus or van in a parking lot, with a safety harness.

**Baselines:**
- Standard projected-gravity policy.
- Same policy with heavy push randomisation.
- Same policy plus an EKF gravity estimate but no platform awareness.
- Teacher with privileged platform state (the upper bound).

**Metrics:** fall rate as a function of peak acceleration and jerk, apparent-vertical error, recovery steps, and
box-tilt or spill.

**Headline figure:** the same braking profile, with the baseline humanoid falling forward and ours leaning or
stepping and staying up, plus the estimator's "down" arrow drawn on both.

## 5. Why it is a strong PhD paper

- A **clear scientific insight** (the equivalence principle versus projected gravity) that every humanoid policy
  inherits. Reviewers remember papers like this.
- **Very demo-friendly.** Video of a humanoid riding a bus or a rocking deck will be shared.
- **Feasible:** sim is easy (a moving rigid body), the estimator is classical, and the policy pipeline is
  standard mjlab.
- It extends naturally into a thesis chapter: ships and offshore, then lifts and stairs in vehicles, then
  humanoids helping passengers.

## 6. Toy prototype (1 week)

Use a 2D inverted pendulum (or the cartpole in this repo) **standing on an accelerating cart**:
- The IMU is simulated as specific force plus gyro. A complementary filter gives the "projected gravity".
- Run three controllers through bus-braking profiles: (a) the filter's tilt estimate, (b) the filter plus
  contact-kinematic disambiguation (the pendulum base is fixed to the cart, and cart acceleration is estimated
  from the foot), and (c) an oracle.

Expected result: (a) leans into a fall at around 0.15–0.2 g, while (b) is close to (c). That is Figure 1.

## 7. Risks

| Risk | Fallback |
|---|---|
| Hardware testing on vehicles is hard | Motion-platform or rolling-cart experiments are accepted at ICRA, IROS and RA-L |
| Someone publishes humanoid-on-ship RL | Keep the focus on the ambiguity insight and the estimator, plus riding with handrails and objects |
| Contact estimation fails during slips | Detect slips with foot IMUs (arXiv 2507.11447) and use a learned estimator with uncertainty |

## 8. Venues

ICRA 2027 has passed (deadline about Sept 2026). The targets are **RSS 2027** (late Jan; the insight-driven
format suits this), **IROS 2027** (March), **CoRL 2027** and **Humanoids 2027**. The journal version could go to
**T-RO** or **IJRR** (non-inertial legged locomotion).
