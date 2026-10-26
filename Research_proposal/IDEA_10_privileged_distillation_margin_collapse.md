# Idea 10: Privileged Teachers Teach Overconfidence (margin collapse in perceptive distillation)

**Working title:** *"Margin Collapse: Why Privileged Teacher–Student Distillation Makes Perceptive Humanoids Trip,
and Two One-Line Fixes"*

**Status:** found 2026-10-26 in loop iteration 1. **Partly open.** The general "imitation gap" is known, but the
asymmetric-risk analysis and the quantile fix look new (see §4). The toy evidence is very stable.

## 1. Claim

Almost every perceptive humanoid pipeline trains a **privileged teacher**, which sees the true terrain, and then
distils it into a **student** that sees a noisy heightmap or depth, using **L2 action regression** or DAgger
(RPL arXiv 2602.03002, DPL arXiv 2510.07152, and others).
- L2 distillation learns E[a_teacher | observation], the **posterior mean of the teacher's actions**.
- The optimal student action is **argmax_a E[J(a, terrain) | observation]**.

When failure is asymmetric (a toe-stub costs far more than a slightly higher swing), these differ by a **safety
margin** that grows with the perception noise. L2 distillation drops the margin entirely: **margin collapse**.

## 2. Toy evidence (`toy/teacher_student_overconfidence.py`, 3 seeds, std ≈ 0)

Stair toy. The teacher acts optimally given the true height (clearance 3 mm); the student sees the height plus
noise σ_p.

| σ_p | L2-BC / DAgger | quantile-BC τ = 0.97 | critic-guided distillation | RL fine-tuning (1.6M rollouts) | oracle |
|---|---|---|---|---|---|
| 1 cm | **38% trips**, clearance 0.3 cm | 1.4% | J = oracle (0.5764), 0.44% | 0.5726 | 0.5764, 0.42% |
| 2 cm | **43% trips**, clearance 0.3 cm | 2.0% | J = oracle (0.571), 0.87% | 0.5689 | 0.571, 0.92% |
| 4 cm | **46% trips**, clearance 0.3 cm | 2.1% | 0.5654 (≥ grid oracle), 1.0% | 0.5633 | 0.5608, 2.4% |

The L2 student's clearance stays at 0.3 cm **regardless of noise**, as decision theory predicts.

## 3. Fixes

1. **Quantile (newsvendor) distillation.** Replace L2 with a pinball loss at τ, which learns the τ-quantile of
   teacher actions given the observation. τ comes from the cost ratio (the newsvendor critical ratio). This is a
   one-line change.
2. **Critic-guided distillation.** The student maximises E[Q_teacher(s_true, π_student(o))] over the data, using
   the teacher's learned (smooth) critic. It needs no extra simulator rollouts and matches the oracle. This is
   close to CritiQ (arXiv 2505.09546); see below.

## 4. Closest work

| Work | Relation |
|---|---|
| CritiQ / ReTRy, "Distilling Realizable Students from Unrealizable Teachers" (arXiv 2505.09546) | Critic-guided distillation for partially observed students. **Fix 2 is close to this.** Our contribution is the margin-collapse analysis and the quantile fix. |
| Student-Informed Teacher Training (ICLR 2025, arXiv 2412.09149) | Joint training so the teacher avoids actions the student cannot imitate |
| Risk-aware RL for mobile manipulation (arXiv 2603.04579) | A risk-sensitive teacher (QR critic) then distillation; it does not analyse L2-distillation margin collapse |
| Imitation gap / adaptive insubordination (Weihs et al. 2021); TGRL | The general phenomenon, without the asymmetric-risk / perception-noise quantification |
| RPL, DPL and other perceptive humanoid pipelines | Use exactly the L2 distillation that the theory predicts collapses margins |

## 5. Plan

- **Theory:** a closed form for the collapse (Gaussian posterior, step-cost) and the newsvendor quantile.
- **G1 in mjlab:** stairs and stepping stones.
  - Teacher: PPO with privileged height scan.
  - Students: L2-DAgger, quantile-DAgger, critic-guided.
  - Measure toe-stub rate as a function of heightmap noise, then test on a real G1 staircase.
- **Unify with EB-WM** as one thesis chapter: *"Safety margins under perception noise: three standard pipelines
  that collapse them (pathwise BPTT, privileged world models, privileged distillation) and principled fixes."*
- **Venues:** RA-L or ICRA 2027 (short), or as part of the EB-WM CoRL paper.
