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
Human pilots have the same illusion; it is called the **somatogravic illusion**.

**Refined after the toy prototype (iteration 2).** The mechanism is subtler than "the robot corrects a tilt that is
not there," and more interesting:

- **(a)** In *steady* braking, the gravito-inertial vertical (what the IMU reports) is actually the *right* balance
  reference. You should lean into the deceleration, as experienced passengers do.
- **(b)** A torso accelerometer measures *contact force / mass*. For a standing body, its "tilt" reading is about
  −u/(m g l), which is the robot's **own ankle torque**, not its tilt. So feedback on a fast IMU tilt estimate with
  stiffness Kp > m g l is *self-destabilising*: positive feedback through the robot's own torque. In the toy, a
  0.1 s filter falls **even when the bus is stationary**.
- **(c)** The orientation filter must therefore be slow (0.5–2 s). A slow filter **lags at braking onset and
  release**, and that is when robots fall.
- **(d)** Using only joint kinematics (treating the floor as inertial) never leans, so it saturates the ankle
  torque above about 0.07 g.

So "projected gravity from the IMU" is wrong in transients, and "upright relative to the floor" is wrong in steady
state. The fix is to estimate the **platform acceleration itself**.

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

## 6a. Toy result (iteration 2): `toy/braking_bus_pendulum.py`

The largest peak braking deceleration (in g) survived with an ankle strategy only, over 2 noise seeds:

| onset (rise) time | 0.2 s | 0.5 s | 1.0 s | 2.0 s |
|---|---|---|---|---|
| C0 IMU vertical, τ = 0.1 s | 0 (falls standing still) | 0 | 0 | 0 |
| C0 IMU vertical, τ = 0.5 s | 0.08 | 0.08 | 0.12 | 0.22 |
| C0 IMU vertical, τ = 2 s | 0.06 | 0.08 | 0.08 | 0.10 |
| C1 kinematic vertical | 0.06 | 0.06 | 0.06 | 0.06 |
| **C2a ours, torso IMU + kinematics** | **0.12** | **0.16** | **0.24** | **0.44** |
| **C2b ours, foot IMU** | **0.12** | **0.16** | **0.24** | **0.46** |
| oracle a_p | 0.12 | 0.18 | 0.28 | 0.52 |
| oracle + 0.3 s preview | 0.40 | 0.28 | 0.46 | 0.52 |

Typical bus braking is 0.1–0.3 g. The standard IMU-vertical design is **2–3× worse** than estimating the platform
acceleration, and ours comes within about 15% of the oracle. **Anticipation** (preview) is the biggest lever for
fast onsets (0.12 → 0.40 g at 0.2 s), which motivates the "predict the vehicle" contribution.

Also note: with only a torso IMU, the estimator *needs angular acceleration*. The model-based alternative
cancels a_p to first order (its coefficient is sin²θ). This favours adding a cheap foot IMU, which is a hardware
recommendation the paper can make.

Caveats:
- This is a point-mass pendulum with no stepping, under a PD controller rather than a learned policy.
- Real RL policies also see joint encoders and history, and might partly learn to disambiguate. Testing a
  standard mjlab G1 policy on a moving platform is the *first* experiment of the real paper.
- The preview curve is non-monotonic because the fixed 0.3 s preview is not tuned to the onset time; this needs
  investigating.

## 6b. Extensions (thesis chapter material)

- **Escalators**, a moving *and* stepped platform. No humanoid escalator work was found (2026-10-26); the closest
  is stair work such as FastStair (arXiv 2601.10365) and arXiv 2605.09944.
- Lifts (vertical acceleration changes the friction cone and the ankle-torque limit), and ship decks (rotating
  platform, where the floor is no longer level and the kinematic vertical is also wrong).

## 6. Toy prototype (1 week): original plan

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
