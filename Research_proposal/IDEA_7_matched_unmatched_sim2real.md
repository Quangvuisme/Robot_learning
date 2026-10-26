# PhD Research Proposal (Idea 7, fundamental learning): What Can an Action Residual Fix?

**Working title:**
*"What Can an Action Residual Fix? Matched and Unmatched Sim-to-Real Gaps in Learning Humanoid Control"*

**Track:** fundamentals of robot learning (sim-to-real, model learning). The user asked for less application and
more learning fundamentals.

> Written 2026-10-26. Novelty was checked by web search: no paper states the matching condition for
> action-space sim-to-real corrections. Re-check before committing.

---

## 1. The question

A popular recipe for agile humanoids is **ASAP** (arXiv 2502.01143) and its successors:
1. collect real-robot data;
2. learn a **delta-action model** δ(s, a) so that the simulator, driven by a + δ, reproduces the real
   trajectories;
3. fine-tune the policy in this corrected simulator.

It works well in practice, but **nobody has said what it can and cannot fix.** Related work hints at limits:
- a residual actuator model at the sim rate is more expressive than a policy-rate delta action
  (arXiv 2604.11090);
- reviewers report delta actions destabilising training (Actuator Reality Shaping, arXiv 2607.02205).

**Our question:** which sim-to-real gaps can an action-space correction represent, what happens to the rest, and
what should we do instead?

## 2. Theory: the matching condition, an old control idea applied to a new learning practice

- **Setup.** Let the sim be s' = f(s, a) and the real system be s' = f(s, a) + e(s, a). An action residual
  produces f(s, a + δ) ≈ f(s, a) + B(s, a)·δ, where B = ∂f/∂a (the *input directions*).
- **Proposition 1 (one-step matching condition).** An action residual can reproduce the real one-step dynamics
  everywhere **if and only if** e(s, a) ∈ range B(s, a), up to higher-order terms. This is the classical
  *matched uncertainty* condition from adaptive and robust control (Lyapunov redesign).
- **Proposition 2 (irreducible error).** The best possible one-step error is ‖(I − P_B) e‖.
  - For a humanoid, M q̈ + h = Sᵀτ + Jᵀλ. The 6 floating-base coordinates have **no actuator**, so any gap that
    does not act through Sᵀτ is **unmatched**. This includes mass and inertia errors felt by the base, the contact
    model (friction, compliance, restitution) and external disturbances.
  - Joint-level actuator gaps (motor friction, torque constant, PD-gain error, backlash, delay) are **matched**.
- **Proposition 3 (multi-step compensation is policy-specific).** ASAP-style training minimises *multi-step*
  replay error. For unmatched gaps, δ can then partly "steer" the state back through the dynamics coupling. But
  that correction depends on where the trajectory goes next, which is set by the **data-collection policy**. When
  the policy changes (which is the whole point of fine-tuning), the correction can become worse than a plain
  one-step fit.
- **Diagnostic: the matched fraction ρ = E‖P_B e‖² / E‖e‖².** It can be computed from real logs plus the sim,
  **without training anything**. It tells a practitioner *before* building a correction model whether an action
  residual will be enough.

## 3. Toy evidence (`toy/matched_unmatched_delta_action.py`, about 1.5 min on CPU)

An underactuated cartpole (one force, two velocity degrees of freedom), like a humanoid's unactuated base. Real
data comes from a swing-up policy π0, and the models are tested on held-out π0 data and on a different policy π1.
Each number is the 25-step replay error relative to the plain sim (1 = no improvement, 0 = perfect):

| gap | ρ | δ 1-step (π0 → π1) | δ multi-step, ASAP-like (π0 → **π1**) | state residual (π0 → π1) |
|---|---|---|---|---|
| matched: motor gain 0.7, cart friction | **1.00** | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 |
| unmatched: pole friction, heavier pole | **0.55** | 0.39 → 0.39 (stuck at the bound) | 0.33 → **0.64** (overfits to π0) | 0.00 → 0.004 |
| both | 0.91 | 0.11 → 0.12 | 0.11 → 0.11 | 0.00 → 0.00 |

All three propositions show up:
1. Matched gaps are fixed exactly.
2. Unmatched gaps leave an irreducible error, roughly matching 1 − ρ.
3. The multi-step (ASAP-like) correction looks *better* on the training policy but is **almost 2× worse** on a
   new policy.

**Caveat:** the "state residual" is near-perfect here because its hand-picked features happen to span the true
cartpole residual. On a real robot it must be learned, and identifiability becomes the issue (Section 4).

## 3b. Closed-loop transfer: the test that matters (`toy/matched_unmatched_transfer.py`)

**Setup.**
- The corrections now use **generic features**: all monomials up to degree 2 of (x, ẋ, sinθ, cosθ, θ̇, u). These
  are not hand-picked, so the comparison does not favour the state residual.
- A 7-gain swing-up controller is optimised by CEM *inside each corrected simulator*, then deployed on the "real"
  system.
- Gap sizes were chosen by a scan so that a controller tuned in the plain sim fails on the real system while one
  tuned on the real system succeeds; otherwise the test is uninformative.
- Results are the mean over 3 data seeds and 32 test starts.

| real success (return) | plain sim | δ 1-step | δ multi-step (ASAP-like) | state residual | oracle (tuned on real) |
|---|---|---|---|---|---|
| **matched** (gain 0.5, cart friction 3) | 0% (0.50) | **100%** (0.78) | **100%** (0.78) | **100%** (0.78) | 100% (0.78) |
| **unmatched** (pole friction 0.03, pole 0.12 kg) | 0% (0.39) | **0% (0.28)** | 34% (0.52), seeds 53/50/0% | **97%** (0.68) | 97% (0.68) |
| both | 0% (0.02) | 0% (0.11) | 0% (0.07) | 0% (0.20) | 0% (0.21) |

**Findings:**
- **Proposition 1 holds in closed loop.** For matched gaps, every action-residual variant transfers as well as
  the oracle.
- **Propositions 2 and 3 hold in closed loop.** For unmatched gaps:
  - the 1-step action residual never transfers;
  - the multi-step (ASAP-like) residual is **unreliable across data seeds** (53%, 50%, 0%);
  - only the state-level residual matches the oracle.
- **New finding, the partial-correction trap.** For the unmatched gap, the 1-step action residual makes transfer
  *worse than no correction at all* (return 0.28 versus 0.39). It fixes the matched projection of the gap but
  leaves a sim whose remaining error is systematically biased, and the optimiser exploits it. This is worth a
  proposition of its own in the paper.
- **The "both" gap** is too hard for this controller family: even the oracle reaches 0%. So only returns are
  comparable, and their order (state ≈ oracle > δ1 > δH > plain) is consistent with the theory.

## 3c. Proposition 4 (domain randomization): **not confirmed** in the toy (`toy/matched_unmatched_dr.py`)

**Hypothesis:** DR suffices for matched gaps but not for unmatched ones (feedback can dominate matched
uncertainty only).

| | no DR | DR over the gap's parameters | oracle (this setup) | price of robustness (nominal return lost) |
|---|---|---|---|---|
| matched | 0% | **100%** | 100% | 0.030 |
| unmatched | 0% | **73%** | 80% | 0.027 |

DR handled the unmatched gap almost as well as the oracle in this setup, at the same small price. So the
matched/unmatched distinction is sharp for **model-based corrections** (action residuals) but **not clearly for
DR**. The paper should scope its claims to correction models, and treat DR versus unmatched gaps as an open
empirical question for the humanoid. (The oracle here reaches only 80% because CEM used fewer training starts than
in §3b; that confounder should be fixed before drawing stronger conclusions.)

## 4. Method: what to do instead

1. **Diagnose:** compute ρ from real logs. If ρ ≈ 1, an action residual is enough (use ASAP). Otherwise,
   continue.
2. **Generalized-force residual:** learn r(s, a) as generalized forces applied in the simulator (in MuJoCo,
   `qfrc_applied` / `xfrc_applied`, including a **base wrench**). Decompose it into a matched part (equivalent to
   an action residual) and an unmatched part.
3. **Identifiability and anti-exploitation:**
   - regularise the unmatched part to minimum norm and bound it (no "free" forces the policy could exploit);
   - train on data from *several* policies, or add exploration noise (cf. the exploratory-data result of Zhang
     et al., ICLR 2026);
   - check generalisation on a held-out policy, as Proposition 3 requires.
4. Fine-tune the policy in the corrected sim, mixed with domain randomisation over the residual's uncertainty.

## 5. Experiments

- **Toy:** done (cartpole). Add closed-loop policy fine-tuning, measuring transfer success instead of replay
  error.
- **Sim-to-sim humanoid:**
  - Train in mjlab (MuJoCo-Warp). The "real" system is MuJoCo with controlled gaps of each kind:
    - matched: motor friction, gains, delay;
    - unmatched: base inertia, foot friction and compliance, a payload on the torso.
  - Measure ρ, then compare transfer for: system identification, domain randomisation, ASAP delta action, and
    ours.
- **Real G1:**
  - Compute ρ on logs from BeyondMimic-type motions (open data from `whole_body_tracking`).
  - Answer **"how much of the real G1 sim-to-real gap is matched?"** This is a useful scientific measurement
    on its own.
  - Then fine-tune and test on *held-out motions*, which is the Proposition 3 test.

## 6. Why this is a good fundamental PhD paper

- It connects a 50-year-old control concept (matched versus unmatched uncertainty) to a mainstream learning
  practice, and gives **sharp, testable propositions**.
- It produces a **practical diagnostic** (ρ) that any lab can compute from logs.
- It explains a reported failure mode (delta-action instability and poor generalisation) instead of just
  proposing yet another method.
- It applies beyond humanoids: quadrupeds, drones and manipulators carrying objects all have unmatched gaps.
- The toy already works in this repo, and it builds directly on the user's `MPPI/`, `World_model/` and cartpole
  code.

## 7. Risks

| Risk | Answer |
|---|---|
| The real G1 gap turns out mostly matched (ρ ≈ 1) | Still a publishable measurement ("ASAP works *because* the gap is matched"), and the diagnostic still stands; add a payload or soft-ground experiment where ρ is clearly below 1 |
| The generalized-force residual is exploited by the policy | Bound it, use minimum-norm regularisation and randomise over its uncertainty; report exploitation honestly |
| "This is just residual dynamics learning" | The contribution is the *theory* (when action residuals suffice, and why multi-step fitting overfits to the policy), the diagnostic, and the humanoid measurement, not the residual model itself |

## 8. Venues

- **L4DC 2027** (Learning for Dynamics & Control) for the theory and toy.
- **CoRL 2027 or RSS 2027** with the humanoid experiments.
- **T-RO** for the full version (a thesis chapter).
