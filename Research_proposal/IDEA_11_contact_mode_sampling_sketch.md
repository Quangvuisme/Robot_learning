# Idea 11 (sketch): Contact-Mode-Balanced Training for Universal Motion Trackers

**Status:** found 2026-10-26 in loop iteration 1. **Partly open.** No paper reweights training by
**contact-mode frequency**.

**Problem.** Universal humanoid trackers (BeyondMimic, GMT, SuperMimic, EGM, PGMT) sample training clips by
**failure rate** (PGMT, OmniTrack) or by **time bins** (EGM, BCCAS). Large motion-capture datasets are dominated
by feet-only contact (walking, dancing). Rare contact modes (hands on the floor, kneeling, sitting, rolling) are
under-sampled, both for the policy and, in world-model methods such as SuperMimic, for the **world model**.
Proposition C (Idea 8) predicts biased event boundaries there.

**Idea.**
1. Label each frame's contact mode from the reference: which links touch the ground or a scene.
2. Sample start states and clips to **flatten the contact-mode distribution** (with a temperature), and weight the
   world-model loss by inverse mode frequency.
3. Report results per contact mode, not just the dataset average.

**Closest work:**
- EGM (arXiv 2512.19043, time bins);
- PGMT (arXiv 2609.08511, failure-driven);
- SceneBot (arXiv 2606.27581, contact commands);
- ContactMimic (arXiv 2607.08742, contact-label flipping);
- Humanoid Mobility via Attention-based Contacts (arXiv 2610.31608, contact plans).

None of them rebalance by contact-mode frequency.

**Risk.** It may be seen as an engineering trick. It is strongest as a section of the EB-WM motion-tracking
experiments (mode-branched rollouts plus mode-balanced data).
