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
