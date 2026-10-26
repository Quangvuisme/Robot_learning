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
