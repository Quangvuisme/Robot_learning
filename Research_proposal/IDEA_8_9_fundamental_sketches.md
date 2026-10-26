# Fundamental-learning sketches (Ideas 8 and 9)

These are second-tier candidates for the fundamentals track. Both are partly open; the risks are stated
honestly.

---

## Idea 8: Why do learned world models give useful policy gradients through contact?

*"A Smoothing Theory of World-Model Policy Gradients, with Annealed Noise-Conditioned World Models"*

**Context.** This is the newest direction of the Berkeley line: **SuperMimic** (arXiv 2610.28480) trains
humanoid control by backpropagating through a learned world model. **PWM** (ICLR 2025, arXiv 2407.02466)
observed that well-regularised world models give *smoother* optimisation landscapes than the true simulator.
Their argument: we want world models that are smooth with a low optimality gap, not necessarily accurate.

**Gap.** That observation has no theory, and there is no knob that controls *how much* smoothing the model
provides.

**What is close:**
- explicit randomized smoothing of contact (bundled gradients, Suh et al. 2021/2022);
- Bundled Contact Gradients (arXiv 2609.30951, with real G1 transfer);
- model-smoothness theory for model-based reparameterization gradients (arXiv 2310.19927);
- "Do Transformer World Models Give Better Policy Gradients?" (arXiv 2402.05290).

**Idea.**
1. Show that a world model trained with input noise σ approximates the **Gaussian-smoothed dynamics** f_σ, and
   bound the bias and variance of its policy gradients in terms of σ, model capacity and the contact
   discontinuities.
2. Make σ an explicit input (a noise-conditioned world model, as in diffusion training) and **anneal it during
   policy learning**: coarse, smooth gradients first, precise ones later (homotopy or continuation).
3. Toys: PWM's ball-against-a-wall problem, and a cartpole with walls. Then motion tracking with a SuperMimic-style
   world model.

**Risk:** it sits close to PWM, MB-RPG and Bundled Contact Gradients, so the theory must be the main
contribution.

---

## Idea 9: Where to restart? Restart distributions for motion imitation

*"Reference State Initialization, Explained: Restart Distributions for Motion Imitation from the Performance
Difference Lemma"*

**Context.** Every motion tracker (DeepMimic, PHC, BeyondMimic) relies on **reference state initialization**
(RSI) and on heuristic **adaptive sampling** of hard clips or phases. BeyondMimic, for example, samples by
failure rate. Theory (Kakade & Langford 2002; Agarwal et al. 2021) says policy-gradient performance depends on
the restart distribution μ through the concentrability term ‖d^{π*}/μ‖∞, and RSI is exactly a choice of μ.

**What is close:**
- Reverse Curriculum Generation (arXiv 1707.05300): start where the agent "sometimes succeeds";
- prioritized restarts (Tavakoli et al., arXiv 1811.11298);
- SFV's learned initial-state distribution (arXiv 1810.03599);
- prioritized level replay and other unsupervised environment design in RL.

**Gap.** There is no theory-derived restart rule for *multi-clip* motion imitation, and failure-based sampling
has never been compared against a principled alternative.

**Idea.**
1. Derive a bound for multi-clip tracking in which μ trades off coverage (concentrability) against learnability.
2. Propose an advantage-gap or "learnability" restart rule.
3. Test it against uniform RSI and failure-based sampling, on cartpole swing-up (in this repo) and on a G1
   motion-tracking benchmark in mjlab.

**Risk:** reviewers may see it as incremental over RCG, PLR and BeyondMimic's adaptive sampling unless the theory
gives a clearly different, better rule.
