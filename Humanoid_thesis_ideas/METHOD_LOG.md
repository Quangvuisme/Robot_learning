# Method log (one entry per 30-minute iteration)

Each entry: method, official repo, what was smoke-tested, result, and the effect on the thesis proposal in `README.md`.

---

## 2026-11-19 #1: DiffContact (arXiv 2610.35852, KAIST)

**What it is.** A smooth analytic contact gradient. The forward pass stays the exact hard-contact solve, so rollouts
stay feasible. Only the backward pass uses a central-path relaxation (width ρ) of the de Saxcé-corrected
second-order-cone complementarity problem, so gradients point toward neighbouring contact modes
(separate / slide / clamp) instead of holding the mode fixed. They report a quadruped trot learned by BPTT from
**one** environment, 1e5 samples, <2 min on a desktop, transferred to hardware; 100x fewer samples than PPO; contact-implicit MPC
that resists 3.4x larger pulls than exact gradients.

**Official repo.** None yet: the GitHub button on the project page (`diffcontact/diffcontact.github.io`) is commented out.
Quadruped only; no humanoid results.

**Smoke test (own analogue, `smoke_tests/grad_relax.py`).** Same stiff-contact hopper and the same finite-difference
reference as README §3c. "RELAX" keeps the forward pass exact but differentiates a softplus/sigmoid contact in the
backward pass (straight-through). This is a penalty-contact analogue, **not** DiffContact's SOCP solver.

| k | EXACT bias / std | CLIP (∇Mimic-style) bias / std | RELAX ρ=1e-3 | RELAX ρ=3e-3 | RELAX ρ=1e-2 |
|---|---|---|---|---|---|
| 1e3 (soft) | 2.06 / 0.16 | 2.06 / 0.16 | **0.28 / 0.05** | 0.32 / 0.04 | **0.14 / 0.05** |
| 1e4 | 0.12 / 0.26 | 0.50 / 0.22 | 0.12 / 0.26 | 0.12 / 0.26 | 0.31 / 0.37 |
| 1e5 (stiff) | 0.035 / 0.12 | 0.13 / 0.11 | 0.035 / 0.12 | 0.035 / 0.12 | 0.035 / 0.12 |

(relative bias and relative std vs. the reference; cosine to the reference ≥ 0.998 for all estimators)

**Readings.**
- Where exact gradients were badly off in magnitude (soft contact), backward-only relaxation cut the bias **7–15x** and
  the noise 3–4x.
- Where exact gradients were already good, small ρ is neutral, while clipping added 4x bias. Too large a ρ
  (1e-2 at k=1e4) hurts, so ρ must be scheduled or tied to contact stiffness.
- The relaxation was never worse than clipping at a sensible ρ.

**Effect on the thesis.** This strengthens Paper 1 and gives it a concrete method. Instead of (or before) per-sample
clipping, put a **backward-only contact relaxation inside the exact humanoid adjoint** (NablaEngine / MuJoCo-Warp),
with ρ adapted per contact pair, and measure bias against the finite-difference reference at G1 scale.
Open questions it raises:
(1) does relaxation remove the rare unbounded samples that forced ∇Mimic to clip (502 contact pairs, self-collisions)?
(2) can ρ be annealed to zero during training, as the paper's convergence result suggests?
Novelty check: DiffContact is quadruped-only with no code, and ∇Mimic uses exact gradients + clipping.
"Relaxed-adjoint humanoid tracking" is open.

---

## 2026-11-19 #2: prior-art check for Paper 2 (HALO 2603.15084, Contact-UAN 2610.18991) + residual-gradient test

**HALO** (Zhejiang/TeleAI, Mar 2026). Differentiable-simulation **system identification** (MuJoCo XLA) of payload mass/CoM,
in two stages (calibrate the base robot without payload, then identify payload links), using only joint encoders and a fixed-foot rig.
Policies are then **retrained with PPO** in mjlab on the identified model, transferring zero-shot to a G1 with 6 kg torso + 2x2 kg wrists.
Real G1 mass identified within 0.01–0.46 kg. No public repo found (`git ls-remote` on likely names failed).

**Contact-UAN** (MIT Improbable AI, CoRL 2026). A **residual actuator network** inside the simulator, fitted on 6.5 min of real walking.
"Contact-consistent replay" infers the stance-foot pose so the residual learns actuator error, not reset artifacts.
Policies are **retrained with RL** (20% lower running torque, 43% lower force error). No public repo found.

**Novelty verdict for Paper 2.** Neither backpropagates *policy* gradients through exact sim + residual, and both
retrain from scratch with PPO. Paper 2 is still open, but it must beat HALO (parametric SysID + retrain) and Contact-UAN
(residual actuator net + retrain) on **adaptation time and data**, not only on final tracking. Two parts to borrow:
HALO's two-stage calibration and Contact-UAN's contact-consistent replay.

**Smoke test (`smoke_tests/grad_residual.py`, `grad_sysid.py`).** The hopper "real robot" has mass x1.3 and motor gain x0.6.
512 rollouts of real data. Reference = finite differences on the real system.

| k | Exact grad, TRUE model | Exact grad, NOMINAL sim | Full learned WM | Sim + MLP residual | Sim + identified params (HALO-style) |
|---|---|---|---|---|---|
| 1e3 | cos 0.995, bias 5.3 | cos 0.986, bias 11.3 | cos 0.92, bias 81 | **cos -0.97**, bias 143 | cos 0.995, bias 5.3 (m=1.30, g=0.60) |
| 1e4 | cos 1.00, bias 0.04 | **cos -1.00**, bias 1.9 | cos 0.42, bias 510 | **cos -1.00**, bias 685 | **cos 1.00, bias 0.18** (m=1.31, g=0.68) |
| 1e5 | cos 1.00, bias 0.04 | cos 1.00, bias 2.0 | cos -0.05, diverges | cos -0.68, diverges | **cos 1.00, bias 0.12** (m=1.30, g=0.70) |

One-step fit loss: MLP residual **5–11x lower** than the full WM. Yet its gradients were as bad or worse.

**Readings.**
1. Mismatch matters for gradients. Exact gradients of the *nominal* sim point the **wrong way** at k=1e4 (cos -1.00).
2. **A good one-step prediction does not give good gradients.** An unconstrained MLP residual's Jacobian is wrong and compounds over
   the 60-step BPTT horizon. So "exact sim + MLP residual" as originally proposed fails.
3. A **structured** residual (physical parameters identified through the exact sim) recovers almost-true gradients
   (bias 0.12–0.18 vs 2.0 nominal). Motor gain is only partly identified (0.68–0.70 vs 0.6) and the gradients are still good.

**Effect on the thesis (Paper 2 revised).** Make the residual **gradient-faithful**, not just predictive:
- First, physics parameters identified through the adjoint (HALO-style, two-stage).
- Then only a small, regularized neural residual for what parameters cannot explain (actuator effects, as in Contact-UAN),
  trained with a **multi-step and Jacobian-consistency (Sobolev) loss**, and evaluated by **gradient fidelity**
  (cosine/bias vs a finite-difference reference), not only by prediction error.
- New contribution claim: "gradient fidelity, not prediction error, is the right metric for models used in first-order policy adaptation".
  This is testable at G1 scale with the P1 finite-difference reference, and neither HALO nor Contact-UAN measures it.

---

## 2026-11-19 #3: Bundled Contact Gradients (BCG, arXiv 2609.30951, USC / ETH / NVIDIA)

**What it is.** Contact-local randomized smoothing for first-order policy learning. When a contact's normal impulse exceeds a threshold
(400 N on the G1), B=10 copies of the contacting chain are perturbed in Cartesian space (σp = 1 cm, σv = 2 cm/s), stepped H=2 control steps
with shared actions, and **averaged**. Backprop goes through the average. Used with SHAC + ADD rewards on the analytically smoothed contact
model of Schwarke et al. (Warp).
**Results:** first zero-shot transfer of a differentiable-sim-trained policy to a real **humanoid** (G1: run, jump, fight, dance);
>10x fewer samples than PPO; lower MuJoCo tracking error on 3/4 motions. Their stated future work: *set the perturbation scales and bundle size automatically*.

**Official repo.** Not yet: the project page (`bundledcontactgradients.github.io`) says "Code (soon)".

**Smoke test (own analogue, `smoke_tests/grad_bcg.py`).** Same hopper and finite-difference reference as #1. At contact, 10 perturbed branches
for 2 steps are averaged in the forward pass, as in BCG.

| k | EXACT bias / std | CLIP | RELAX (#1, best ρ) | BCG σz=1 mm, σv=2 cm/s | BCG σz=3 mm, σv=5 cm/s |
|---|---|---|---|---|---|
| 1e3 | 2.06 / 0.16 | 2.06 / 0.16 | **0.14 / 0.05** | 2.17 / 0.17 | 2.34 / 0.16 |
| 1e4 | 0.12 / 0.26 | 0.50 / 0.22 | 0.12 / 0.26 | **0.061** / 0.28 | 0.076 / 0.30 |
| 1e5 | 0.035 / 0.12 | 0.13 / 0.11 | 0.035 / 0.12 | 0.57 / 0.69 (cos 0.68) | 1.60 / 1.26 (**cos 0.08**) |

**Readings.**
- At medium stiffness BCG **halved the bias** (0.12 → 0.06), the best of all estimators there.
- At high stiffness it **broke**. Averaging *forward* states of branches that bounce differently changes the rollout itself
  (cos fell to 0.68 / 0.08). The penetration depth at k=1e5 is ~0.1 mm, so a 1 mm perturbation is 10x too large.
  **The smoothing scale must match the contact length scale.** BCG's fixed σ (tuned for κ=300 smoothed contact) would not carry over.
- At soft contact it does not fix the magnitude error. Backward-only relaxation does.
- Backward-only relaxation (#1) was never harmful at a small ρ. Forward-averaging methods (BCG) change the objective and can hurt.
  Caveat: 1-D toy with my own analogues, not their implementations.

**Effect on the thesis (Paper 1 sharpened).** There are now **four** published ways to stabilize humanoid first-order gradients,
each tested in isolation and each with a hand-tuned scale:
∇Mimic (per-sample clipping), DiffContact (backward-only relaxation, ρ), BCG (contact-local bundles, σ), AHAC (adaptive horizon).
Paper 1 should be **"A controlled comparison of first-order gradient stabilizers for humanoid tracking, and an automatic, stiffness-aware
smoothing scale"**:
1. Put all four on one G1 benchmark (LAFAN1 on mjlab / MuJoCo-Warp adjoint) and score them with the same finite-difference bias/variance reference.
2. Contribution: choose ρ or σ per contact from local contact quantities (penetration depth, impulse, stiffness) so that the measured bias
   stays below a target. Both DiffContact and BCG list this as future work, and my toy shows the wrong scale is catastrophic.
Timing risk: BCG and ∇Mimic code is coming out, and the space is moving fast (Sep–Oct 2026).

---

## 2026-11-19 #4: Decoupled forward/backward model optimization (DMO; Amigo, Khorrambakht, Mansard, Righetti; arXiv 2509.00215 + 2602.06219)

**What it is.** First-order MBRL where the **forward** rollout comes from an accurate system (simulator, or a heavy diffusion WM in the 2026 paper)
and the **backward** pass uses Jacobians of a cheap **local** model evaluated *at the accurate states*. The local model only needs one-step
accurate Jacobians, so there is no compounding drift. The 2026 version trains manipulation and quadruped-push policies purely from real play data:
9/10 vs PPO 1/10 on Push-T, with 5x fewer samples.
Related field (ML world models): also flagged were "Dreaming smoothly… gradient-penalized latent dynamics" (2605.23089) and
"Closing the train-test gap in WMs for gradient-based planning" (2512.09929). Both say prediction loss alone gives poor model gradients.

**Official repo.** `machines-in-motion/DMO_code` (project page `machines-in-motion/DMO`). Cloned and inspected: DMO-SHAC on DFlex and
Isaac Gym (Go2); the bipedal env is "coming soon". Needs CUDA / Isaac Gym, so **not run**. The core is in `algorithms/dmo_shac.py`
(real `next_obs` stored, learned `dyn_model` for gradients).

**Smoke test (`smoke_tests/grad_dmo.py`).** Same mismatch setup as #2 (real hopper: mass x1.3, motor gain x0.6, 512 rollouts of data).
Implemented as `s = s_model + (s_true - s_model).detach()`: forward values from the true system, backward through the chosen model.

| k | Forward NOMINAL, backward nominal (#2) | **DMO: forward TRUE, backward nominal sim** | DMO: backward full WM | DMO: backward sim + MLP residual | Identified params (#2) |
|---|---|---|---|---|---|
| 1e3 | cos 0.99, bias 11.3 | cos 0.995, bias 9.4 | cos 0.95, bias 108 | cos -0.97 | cos 0.995, bias 5.3 |
| 1e4 | **cos -1.00**, bias 1.93 | **cos 1.00**, bias 1.22 | cos 0.45, diverges | cos -1.00 | cos 1.00, bias 0.18 |
| 1e5 | cos 1.00, bias 1.98 | **cos 1.00, bias 1.09** | diverges | cos -0.76 | cos 1.00, bias 0.12 |

**Readings.**
1. **Decoupling alone fixes the direction** of nominal-sim gradients under mismatch (cos -1.00 → 1.00 at k=1e4) and roughly halves the bias,
   without learning anything. The forward pass must come from the real system.
2. With the MLP residual, the gradients stay wrong **even when decoupled** (no compounding). So the failure in #2 is the **learned Jacobian itself**,
   not error accumulation. This rules out "just train the residual better on prediction loss".
3. Identified physical parameters are still best (bias 0.12–0.18).

**Effect on the thesis: Paper 2 gets its core algorithm.**
"**Real-world first-order adaptation**": roll out the policy **on the real robot** (forward = real data), and backpropagate through the
**exact simulator adjoint** (NablaEngine / MuJoCo-Warp) evaluated at the measured states, with **physical parameters identified online
through the same adjoint** (HALO-style) and, only if needed, a Jacobian-regularized residual.
- This combines DMO (decoupling, NYU/LAAS) + ∇Mimic (exact G1 adjoint) + HALO (adjoint-based SysID). Nobody has done this for humanoids:
  DMO's biped is "coming soon", and DMO uses a *learned* backward model, not an exact adjoint.
- Measurable claims: gradient fidelity vs a finite-difference reference (from P1); minutes of real data to recover from payload / latency /
  weak motors (README §3a), vs HALO, Contact-UAN and HumanoidNDM.
- Open technical issue: the robot's real state must be estimated well enough (base pose/velocity) to evaluate the adjoint.
  Contact-UAN's contact-consistent state inference is the tool for this.

---

## 2026-11-19 #5: LLA-MPPI (arXiv 2610.10465, CMU / MIT)

**What it is.** Online adaptation without training or gradients. A **bank of K simulators** (MuJoCo-Warp, up to 10,000 worlds) with different
physical or structural hypotheses (payload left/right, friction × slope, a locked leg, box mass) is stepped from each measured transition.
The hypothesis with the lowest **windowed one-step prediction error** (W ≈ 20, switching margin γ) is handed to a whole-body MPPI planner on CPU.
**Results:** 97.5% success over 4 Go2 tasks vs 98.5% for an oracle with the true model, 74% for a particle filter and 42.5% for a disturbance observer.
Hardware: payload added mid-walk, a leg disabled, box mass increased on the fly. Stated limits: a finite, design-time bank, and hypotheses are only
distinguishable when the motion excites them.

**Official repo.** None found: the project page `lla-control/lla-control.github.io` has no code link, and the likely repo names do not exist. Quadruped only.

**Smoke test (`smoke_tests/sysid_bank_vs_adjoint.py`).** Identify (mass, motor gain) of the mismatched hopper (true 1.3, 0.6) from **one** recorded
rollout, by an LLA-style 33×33 bank vs HALO-style gradient descent through the exact step (300 Adam steps). The error is the sum of relative errors.
(Note: one rollout has 60 transitions, so W ≥ 60 uses all of them.)

| k | Window | Contact steps in window | Bank (LLA) mass / gain / error | Adjoint (HALO) mass / gain / error |
|---|---|---|---|---|
| any | W=10 | 0 | 0.80 / 0.30 / 0.89 (wrong) | 1.00 / 1.00 / 0.90 (stuck at init) |
| 1e3 (soft) | W ≥ 30 | 5–27 | 1.30 / 0.609 / 0.016 | **1.30 / 0.600 / 0.000** |
| 1e4 | W ≥ 30 | 1 | **1.30 / 0.609 / 0.016** | 1.306 / 0.706 / 0.181 |
| 1e5 (stiff) | W ≥ 30 | 1 | **1.30 / 0.609 / 0.016** | 1.301 / 0.708 / 0.181 |

**Readings.**
1. **Excitation decides identifiability.** With no contact in the window, neither method can identify anything: in flight, mass and motor gain have no effect.
   This matches LLA's own caveat and matters for humanoids, where e.g. hand payloads only show up in certain motions.
2. **Through stiff contact, adjoint SysID stalls** (gain stuck at 0.71 vs 0.60; the same under-identification as in #2).
   A single stiff contact step gives a narrow, badly conditioned gradient. The gradient-free bank gets it right to grid resolution.
3. With soft or long contact, the adjoint is **exact** (0.000 error) where the bank is limited by its grid (0.016).

**Effect on the thesis (Paper 2: the identification component).** Use **bank-then-adjoint** identification. Coarse, global and robust
LLA-style bank selection on MuJoCo-Warp (no gradients, handles discrete faults such as a locked joint) initializes the estimate. Precise adjoint
refinement (HALO-style) then runs only in windows with enough contact excitation, using P1's stiffness-aware smoothing so the stiff-contact
gradient does not stall. The refined parameters then feed the DMO-style real-world first-order policy update (#4).
New baseline for Paper 2: **LLA-MPPI itself**, i.e. adaptation by replanning with an identified model, vs adaptation by updating the policy.
The README §3b result (16-sample MPPI around GMT could not recover a humanoid even with the oracle model) suggests replanning alone is
too weak at humanoid DoF and a real-time budget. That is a testable claim for the paper.

---

## 2026-11-19 #6: State estimation for Paper 2 (CoCo-InEKF, arXiv 2605.15122, ETH / Disney Research) + estimate-noise tolerance test

**What it is.** A **differentiable invariant EKF** (contact-aided, Hartley-style) where a small MLP predicts a full 3×3 **contact-velocity covariance**
per contact candidate instead of a binary contact flag. It is trained end-to-end by BPTT on a body-velocity loss only, with no contact labels,
and the contact candidates are chosen automatically. Body linear-velocity RMSE: **0.046 m/s** (sim dancing), **0.08 m/s** (real ground motions).
In live control on the Lima biped it gave **higher success than mocap-based estimation** (95% vs 92% on trained dances), at 0.42 ms per step.
Also surveyed: FOCUS (2609.02222, foot-observation confidence), InEKF on non-inertial ground (2606.19512), SixthSense whole-body wrench estimation (2605.01427).

**Official repo.** None found (`DisneyResearch/coco-inekf` and variants do not exist). Disney biped only, not a G1.

**Smoke test (`smoke_tests/grad_dmo_estnoise.py`, `grad_dmo_znoise.py`).** DMO-style gradients from #4 (forward = true system), but the adjoint
Jacobian is evaluated at a **noisy state estimate**. Mismatched hopper (mass x1.3, gain x0.6).
*Correction:* the first run swept velocity noise and gave identical results for every σ_v > 0. In this toy the Jacobian does not depend on
velocity, so only the fixed 1 mm height noise in that run mattered. The second run sweeps **height noise σz** (σv = 0.05 m/s fixed).

| k (contact penetration ≈ mg/k) | σz = 0 | 1e-5 m | 1e-4 m | 3e-4 m | 1e-3 m |
|---|---|---|---|---|---|
| 1e4 (≈1 mm), true-params adjoint | cos 1.00, bias 0.04 | 1.00 / 0.04 | 1.00 / 0.04 | 1.00 / 0.04 | 1.00 / 0.02 |
| 1e5 (≈0.1 mm), true-params adjoint | cos 1.00, bias 0.04 | 1.00 / 0.04 | **0.91** / 0.15 | **0.82** / 0.30 | **0.50** / 0.74 |
| 1e5, nominal adjoint | cos 1.00, bias 1.09 | 1.00 / 1.09 | 0.85 / 0.78 | 0.74 / 0.46 | 0.35 / 1.04 |

**Readings.**
1. Adjoint gradients are **insensitive to velocity-estimate error** (in this toy) but **sensitive to contact-geometry error**.
   They degrade once the height error approaches the contact **penetration scale**, because the Jacobian is evaluated in the wrong contact mode.
2. With stiff contact (~0.1 mm penetration, like the G1 on hard ground) even **0.1 mm** of error lowers the cosine to 0.91, and 1 mm to 0.50.
   No proprioceptive estimator gives foot height to sub-millimetre accuracy, so **raw estimates cannot be plugged into the adjoint**.

**Effect on the thesis (links Paper 1 and Paper 2).**
- In Paper 2 the adjoint must be evaluated at a **contact-consistent** state, not the raw filter output. Re-simulate the measured joint
  trajectory from the estimate so that contact modes are self-consistent (this is exactly Contact-UAN's contact-consistent replay), or take
  contact modes from the filter's contact covariances (CoCo-InEKF).
- In Paper 1 the automatic smoothing scale should be **ρ = max(contact length scale, state-estimate uncertainty)**: backward-only relaxation (#1)
  over the uncertain contact band, instead of differentiating in one possibly wrong mode. The filter's covariance gives this uncertainty for free.
  This turns Paper 1's "automatic smoothing scale" into a principled rule tied to deployment, and it is the bridge between the two papers.

---

## 2026-11-19 #7: MJX differentiable simulation on the real Unitree G1 (google-deepmind/mujoco MJX + Menagerie `unitree_g1/scene_mjx.xml`)

**What it is.** MuJoCo's JAX backend (also behind MuJoCo Playground), used here as the **exact-gradient engine** on the actual G1 model:
29 position actuators, 49 contact pairs, 4 ms timestep, implicitfast. This moves the P1 questions from the toy hopper to the real robot model.

**Repo / setup.** Official MuJoCo package `mujoco-mjx==3.17.0` (in `venv_nm`), with the G1 from the Menagerie revision pinned by NablaMimic.
**Blocker found and patched:** reverse-mode autodiff fails ("Reverse-mode differentiation does not work for lax.while_loop") because
MJX's constraint solver (`mjx/_src/solver.py`, line ~637) uses a plain `jax.lax.while_loop`. This is the issue ∇Mimic cites (mujoco#2259).
I replaced it inside the solver module with MJX's own `_while_loop_scan` at the fixed `opt.iterations` (= 5), giving a fixed-iteration unrolled solver.
After the patch, exact gradients compile in ~80 s on CPU and run.

**Smoke test (`smoke_tests/g1_mjx_grad.py`).** θ = a constant 29-D joint-target offset. Each sample starts from the `knees_bent` key with a random
downward and lateral base velocity, which forces **foot impacts**. J = −Σ[(z_pelvis − z0)² + 0.5·tilt²] over 12 control steps (0.24 s, 5 substeps each).
Reference: central finite differences (h = 2e-3) on 96 common samples (self-error 1–2%). Estimators: mean over 32 per-sample exact gradients, 6 trials.

| contact timeconst (solref) | EXACT cos / bias / std | CLIP (3×median) cos / bias / std | max/median per-sample norm | non-finite samples |
|---|---|---|---|---|
| 0.02 (softer) | 0.968 / 0.129 / 0.263 | **0.991 / 0.073 / 0.134** | mean 4.2, **max 15.4** | 0 / 192 |
| 0.005 (stiffer) | **0.999 / 0.060 / 0.061** | identical (clip inactive) | mean 1.75, max 2.2 | **1 / 192 (NaN)** |

**Readings.**
1. **Exact G1 gradients are good on average** at this short horizon: cos 0.97–0.999 to the reference, bias 6–13%.
2. **Heavy tails are real on the G1** (one sample is 15× the median in the softer setting), and there **clipping helped**:
   bias 0.13 → 0.07, std 0.26 → 0.13. This **reverses the toy result** (#1, #3), where clipping only added bias.
   Toy conclusions about stabilizers do **not** transfer to the full humanoid, so P1 must be run at G1 scale.
3. Surprisingly, the *softer* contact gave the heavier tail here. A guess (untested): longer soft contacts give more contact-mode changes per window.
4. The stiffer setting produced a **NaN** per-sample gradient (1/192): the "rare unbounded samples" ∇Mimic describes. Any estimator needs a finiteness guard.
   Caveats: a single θ, a 0.24 s horizon, standing and impact rather than tracking, CPU float32.

**Effect on the thesis (Paper 1).**
- This confirms the paper is needed: whether clipping or relaxation wins depends on the system and the contact setting, so a **G1-scale controlled comparison** is the contribution.
- The "automatic" rule gains a second input: **the measured tail of per-sample gradient norms** (max/median, or a tail-index estimate) decides *whether*
  to clip, while the contact and estimate scale (#6) decides *how much* to relax.
- The **benchmark harness now exists and runs on CPU**: `smoke_tests/g1_mjx_grad.py` (with the solver patch). On your GPU, extend it to
  longer horizons, tracking rewards, and the four stabilizers (clip, relax, BCG, AHAC).
