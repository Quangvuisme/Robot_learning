# Idea 12 (sketch): Event-Factored Critics for Absorbing Failures (falls, trips)

**Status:** found 2026-10-26 in loop iteration 2. **Partly open.** It is a natural extension of EB-WM, whose
critic bootstrap at the horizon needs this.

**Problem.** In humanoid RL, a fall or trip is an **absorbing failure**. The true value function is therefore
discontinuous at the failure boundary: one centimetre of clearance separates "climbs the rest of the stairs" from
"episode over". A neural critic trained by TD learning **smooths the jump** (as CACTO-BIC, arXiv 2602.19699,
observed for basin boundaries). This biases advantages exactly where the policy must learn safety margins, the
critic-side analogue of Proposition E.

**Idea.** Factor the critic as V(s) = P_survive(s) · V_alive(s) + (1 − P_survive(s)) · V_fail:
- P_survive comes from an **event/hazard head**, trained with cross-entropy on simulator fall labels; it is
  calibrated, as the EB-WM event heads were.
- V_alive is a smooth regression of the value of surviving trajectories.

In EB-WM, the critic bootstrap becomes Σ_modes w_mode · V(s_mode), with V factored in this way. For model-free
PPO, it reduces advantage bias near the failure boundary.

**Toy test (next):** the 5-step staircase with a learned critic replacing the analytic tail, comparing a plain
MLP critic with the factored critic, and measuring advantage error near the trip boundary and the resulting
margins.

**Closest work:**
- CACTO-BIC (arXiv 2602.19699): critic smoothing at value discontinuities, in trajectory optimisation;
- Zero Collapse (arXiv 2605.30896): policy-gradient failure under discontinuous rewards;
- absorbing-state value learning in IL (DAC-style wrappers);
- safety critics / recovery RL, which learn a failure probability, but for constraints rather than for value
  factorisation in locomotion.

**Parked companion idea:** **termination as a hidden reward.** Early-termination thresholds in humanoid RL act as
implicit reward shaping and data filtering; DeepMimic notes their impact "has not been well evaluated". It would
be an analysis paper (sweep thresholds with the alive bonus controlled), not a method, so it is parked.
