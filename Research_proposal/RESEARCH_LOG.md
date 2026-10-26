# Research log: Thermal Preview Control for humanoids

Each loop iteration does three things: a literature sweep, one step on the prototype, and an entry here.

## Iteration 1 (2026-10-26)

### Literature sweep: new or missed work
- **Olaf (Disney), arXiv 2512.16705.** A thermal reward for a character robot. It says explicitly that the
  thermal timescale is much slower than the balance dynamics, which makes thermal limits a hard RL objective.
  This directly supports our timescale-split argument. It is reactive (a reward term) and has no planning.
- **Human-Level Actuation for Humanoids, arXiv 2511.06796.** Argues that actuators which look strong on a test
  bench become bottlenecks in thermally constrained humanoids, and that thermal coupling only shows up in
  task-level tests. Good motivation to cite.
- **ETHRoboticsClub/humanoid-sim, issue #44.** A request to replace a torque-scale stand-in with a winding
  temperature model so that long sessions show derating. Evidence that humanoid simulators lack thermal models,
  which supports contribution 4 (a thermal extension for mjlab).
- **Two-rate actuator control (Yagi & Morimoto):** a high-rate current policy plus a low-rate position policy.
  **A NAIST thesis:** singular-perturbation hierarchical MPC for a hybrid actuator. **A Pisa thesis:** manipulator
  MPC with drive-temperature constraints. These are model-based or single-actuator precedents for our two-timescale
  split. None uses a learned whole-body policy, redistribution or preview.
- **Novelty status: still open.** No humanoid work does thermal planning with a learned policy, per-joint
  redistribution or program preview. The quadruped papers (2603.01631, 2605.27046) remain the closest.

### Prototype
- Added `toy/thermal_preview_arm.py` and `toy/README.md`.
- Bugs found and fixed along the way:
  - The heat coefficients were initially about 100× too small.
  - Wrist torque depends only on tool orientation, so the wrist has *no* redistribution freedom. The bottleneck
    was moved to the elbow. **Lesson for the humanoid version:** pick bottleneck joints that have null-space
    freedom, and report the non-redundant joints separately.
  - Unreachable poses were being picked silently. Failures are now counted properly.
  - MPPI was too weak when it called the fast layer directly. It now uses a precomputed table over an 8-level
    allowance grid, with K = 1024 samples, 25 iterations and an annealed noise level.
- **Result:** ours completes 5/5 programs with no derating; the best baseline completes 4/5. See the table in
  `toy/README.md`.

### Next iteration
1. Replace the exact tables with a **learned surrogate** (a small MLP on (b, target, payload) → τ², ori). Add
   **model mismatch** (±30% on R and C, and ambient 25–40 °C) to test closed-loop robustness.
2. Tune the reactive baseline fairly (grid-search threshold and span), and add a "thermal penalty in the fast
   cost" baseline as a port of 2603.01631.
3. Ablate preview length (1, 3, 5, 10 segments, full program) to get the curve for Figure 2.
4. Literature: search Google Scholar-style terms ("thermal-aware MPC robot joint", "motor temperature constraint
   humanoid RL", "endurance humanoid").

## Iteration 1b (2026-10-26): idea search, at the user's request "continue find new idea"
- Brainstormed deployment problems that humanoid papers ignore, then checked each one's novelty (see
  `IDEAS_INDEX.md`).
- **New: Idea 3, humanoids in moving vehicles.** A single accelerometer cannot tell tilt from acceleration, so
  the standard "projected gravity" input is corrupted inside a braking bus (about 11° error at 0.2 g). We found
  no learned humanoid controller for this. Prior work is the 2026 survey arXiv 2604.20990 and an estimation-only
  invariant EKF (arXiv 2201.10636).
- Rejected as already taken: joint-failure tolerance (TOLEBI), fatigue-aware torque control (SATA), battery-sag
  modelling (2608.29769), and hardware co-design (BRIDGE, EA-CoRL).
- Thesis storyline drafted: "Humanoids in the real world: slow hidden states and moving frames."
- Next: build the Idea 3 toy (a pendulum on a braking cart with a simulated IMU and complementary filter), and
  continue the Idea 2 iteration 2 tasks.

## Iteration 2 (2026-10-26): user asked to continue researching

### New ideas checked
- **Accepted as a sketch: Idea 4, humanoid wading in water.** No learned humanoid water locomotion was found; the
  closest work is model-based underwater ZMP and SLIP models with added mass. Hardware waterproofing is the main
  risk.
- **Escalators:** no humanoid escalator work found. Added as an extension of Idea 3.
- **Rejected: wind** (it is just a distributed push; ADAPT and push randomisation already cover it).
- **Parked: falsification of humanoid policies** (SPARK stress tests and selective adversarial training exist).

### Related-work re-check
- Idea 3: a search for humanoid RL on moving or accelerating platforms in 2026 found nothing. Still open.

### Prototype: Idea 3 toy (`toy/braking_bus_pendulum.py`)
- An inverted pendulum with a foot-limited ankle torque on a braking bus floor, with a simulated torso IMU
  (specific force plus gyro) and a foot IMU.
- **Key analytic insight, now confirmed numerically:** the torso accelerometer "tilt" is about −u/(m g l), the
  robot's own ankle torque. So a fast complementary filter combined with stiff balance feedback is positive
  feedback, and it **falls even when the bus is stationary**. The filter must be slow, and a slow filter lags at
  braking onset and release.
- **Survival map:** the IMU-vertical design survives 0.06–0.22 g; kinematic-vertical survives 0.06 g; ours
  (platform-acceleration estimation) survives 0.12–0.46 g; the oracle survives 0.12–0.52 g. Preview gives
  0.40 g even at a 0.2 s onset.
- **Correction to the Idea 3 proposal:** the failure is *not* "corrects a non-existent tilt". In steady state the
  IMU vertical is the right reference; the failure comes from transients and the self-induced accelerometer
  signal. The proposal text has been updated.
- **Open anomaly:** the preview curve is non-monotonic (the fixed 0.3 s preview is mismatched to some onset
  times). Investigate next time.
- **Most important next experiment for the real paper:** does a standard mjlab G1 policy (push-randomised, no
  platform motion) actually fall on a moving platform? If RL policies implicitly learn to disambiguate from
  history, the story changes to "a learned estimator vs implicit learning".

### Next iteration
1. Idea 2: learned surrogate, model mismatch, fair baselines, preview-length ablation (carried over).
2. Idea 3: fix the preview anomaly (optimal lean-ahead), add a stepping strategy (capture point) to show where
   ankle-only fails, and add a ship-deck roll (non-level floor, where the kinematic vertical is also wrong).
3. One new idea check.

## Iteration 3 (2026-10-26)

### Idea 2 robustness (`toy/thermal_preview_robustness.py`, `outputs/idea2_robustness.json`)
- **Honest correction:** a *tuned* reactive baseline (span 40, offset 8, tuned on separate training programs)
  also completes 5/5 programs without derating. Ours' advantage is **task quality**, not completion: hard-block
  orientation error is 0.5° for ours versus 6.8° for the tuned reactive baseline (mean 0.73° versus 2.66°).
- **Parameters ±30%:** ours completes 5/5 with 1.2° hard-block error.
- **Learned surrogate:** a numpy MLP trained on 100 unrelated programs (torque MAE 0.03–0.33 Nm, orientation
  MAE 1.85°) completes 5/5 with 1.8° error.
- **Weakness, hot day (ambient +8 °C unknown to the planner):** ours still completes 5/5, but one run derates
  and hard-block error rises to 9.9°. The learned version completes only 3/5. An online ambient-bias adaptation
  is implemented (`adapt=True`) but **not yet evaluated** on this case; the run was interrupted.
- **Preview ablation:** horizons of 1, 2, 4, 8, 16 and 30 minutes complete 4, 3, 4, 5, 5 and 5 programs; hard-block
  error is 11.0°, 14.0°, 17.0°, 12.5°, 4.2° and 0.5°. The horizon must exceed the elbow's thermal time constant
  (about 11 minutes). Short horizons with no terminal value are worse than myopic planning.

### Pivot (user request): new ideas in walking, WBC, perception, imitation and collaboration
- **Checked and taken:** active gaze (TAGA); two-person interaction mimic (CoDance, PAIR); multi-humanoid carrying
  (decMHT, COLA); time-warped tracking (AdaMimic); box-carrying occlusion (HAIC, FootQuery, RPL).
- **New and open:**
  - **Idea 5:** a cane or trekking pole used as both a third leg and a haptic probe.
  - **Idea 6:** a humanoid that physically assists people.

  See their proposal files.

## Iteration 4 (2026-10-26): user feedback "not good, too application-focused; do more fundamental learning"

- Switched to a **fundamental learning track**. Checked 7 candidates:
  - **Taken:** chunking/compounding-error theory (ICLR 2026); OT/DTW rewards for infeasible references
    (2609.15631, ORCA).
  - **Partly open:** smoothing theory of world-model gradients (PWM, MB-RPG theory, Bundled Contact Gradients);
    restart-distribution theory for imitation (RCG, prioritized restarts).
  - **Open:** the matching condition for delta-action sim-to-real (ASAP), which became **Idea 7**.
- **Idea 7 toy** (`toy/matched_unmatched_delta_action.py`), an underactuated cartpole:
  - matched gap: ρ = 1.0, and the action residual is exact;
  - unmatched gap: ρ = 0.55, and the 1-step residual is stuck at 0.39 relative error;
  - the multi-step (ASAP-like) residual gives 0.33 on the training policy but **0.64 on a new policy**, so it
    overfits to the data-collection policy, as Proposition 3 predicts;
  - the full state residual is about 0, but it is optimistic because its features span the true residual.
- Caveat: an Idea 2 online-adaptation code path exists (`adapt=True`) and is still unevaluated (parked).
- Next: add closed-loop policy fine-tuning to the Idea 7 toy (measure transfer success, not only replay error);
  run a sim-to-sim humanoid with controlled matched and unmatched gaps (mjlab versus MuJoCo).
