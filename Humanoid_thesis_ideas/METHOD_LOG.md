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
