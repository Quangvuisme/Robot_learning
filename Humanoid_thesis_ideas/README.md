# Humanoid PhD thesis: frontier survey, official-repo smoke tests, and a concrete proposal

Date: 2026-11-18. Survey covers whole-body control (WBC), motion tracking, locomotion, planning/MPC,
optimization (differentiable simulation), and world models (WM), plus ideas carried over from nearby fields.
Every number below comes from a script in [`smoke_tests/`](smoke_tests/) that was run on a 4-core CPU with no GPU.
Anything that needs a GPU is marked as not run.

---

## 1. Where the frontier is (late 2026)

| Thread | Key papers (arXiv) | What they established | What they leave open |
|---|---|---|---|
| Universal motion tracking | GMT 2506.14770, BeyondMimic 2508.08241, SONIC 2511.07820, Humanoid-GPT 2606.03985, HoloMotion-1 2605.15336, YAHMP study 2607.19903 | One policy tracks thousands of retargeted clips; zero-shot to G1 hardware | Free-space only; brittle to dynamics mismatch (payload, latency, weak motors); huge PPO compute |
| **World-model gradients** | **SuperMimic 2610.28480** (Berkeley, code out 2026-11-10) | One differentiable learned WM + BPTT, >1000x less compute than SONIC; beats PPO on LAFAN1 | WM learned from sim only; model bias in gradients; flat ground; few hardware motions |
| **Exact simulator gradients** | **∇Mimic 2610.41654** (UC Davis) | Exact MuJoCo adjoint + per-sample median clipping beats SuperMimic (0.82 in 20 min vs 1.2 h) | *"Per-sample clipping biases the clipped gradients, and we have not quantified this bias"*; G1 + flat ground only; no hardware; cannot use real-robot data |
| Classic diff-sim theory | Suh et al. 2202.00817, follow-up 2604.18161, Bundled Contact Gradients 2609.30951 | First-order gradients can be biased/high-variance through contact | Not tested at humanoid-tracking scale |
| Test-time compute / planning on trackers | TTCO 2610.14896, RGB MPPI 2606.25123, ReactiveBFM 2606.30362, LLA-MPPI 2610.10465 | Sampling against a simulator fixes the command-execution gap (TTCO: -79–84% body error) | TTCO: *"not yet fast enough for real-time"*; all assume the planning model matches reality |
| Real-data adaptation | HumanoidNDM 2610.34126, Compliant GMT 2610.22119, ILA 2611.14985, HEFT 2607.02332 | A neural dynamics model from real data + backprop fine-tunes trackers on G1 | Learns the *full* dynamics from scratch from real data; no exact-sim structure |
| Generative / WM planners | BeyondMimic guided diffusion, DreamControl-v2 2604.00202, ω-0 2608.06375, BeJEPA 2610.23127, JEPA-WAM diffusion-MPC 2610.12407 | Diffusion/flow priors + tracker for task-level control | Mostly kinematic; physical feasibility is left to the tracker |
| From nearby fields | Flow Policy Gradients 2602.02481, PhysMoDPO 2603.13228 (DPO), test-time scaling (TTCO), DiffDreamer 2610.31617 (WM + diff-sim for quadrotors), control variates 2611.12643 | Preference optimization, flow-policy RL, compute-optimal test-time scaling, WM+diff-sim coupling | WM+diff-sim coupling has **not** been done for humanoids |

## 2. Official repos: smoke tests

| Repo | Thread | Test run here | Result |
|---|---|---|---|
| [zixuan417/humanoid-general-motion-tracking](https://github.com/zixuan417/humanoid-general-motion-tracking) (GMT) | tracking | Headless sim2sim with the released checkpoint, 5 motions x 10 s (`gmt_headless.py`) | ✅ 5/5 no fall; joint error 0.066–0.100 rad; 6x real time on CPU |
| [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic) + `mjlab_research` | WM | Full pytest | ✅ **208/208 pass** (training needs CUDA) |
| [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) | WM | pytest | ✅ 6 pass (fp64 differentiable G1 FK = MuJoCo), 18 skipped "needs CUDA" |
| [OamIcs/NablaMimic](https://github.com/OamIcs/NablaMimic) | diff-sim | pytest (CPU JAX, pinned Menagerie G1) | ✅ **37/37 pass**; training needs nvcc + GPU + LAFAN1 |
| [OamIcs/NablaEngine](https://github.com/OamIcs/NablaEngine) | diff-sim | — | ⛔ CUDA megakernels, compiled with nvcc on first use |
| [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | sim / WBC | G1 flat velocity env on Warp-CPU, 4 envs x 50 steps (`mjlab_g1.py`) | ✅ obs 99 / act 29, no terminations |
| [fabio-amadio/yahmp](https://github.com/fabio-amadio/yahmp) | tracking | Load + run the shipped ONNX policy | ✅ obs 1727 → 29 actions, 0.29 ms/inference (no motion clips shipped) |
| [LeCAR-Lab/dial-mpc](https://github.com/LeCAR-Lab/dial-mpc) | planning | H1 locomotion, 256 samples, 60 MPC steps (`dial_h1.yaml`) | ✅ base height stays ≈1.0 m; ~10 s CPU per control step, so GPU-only in practice |
| HybridRobotics/whole_body_tracking (BeyondMimic), NVlabs/GR00T-WholeBodyControl (SONIC), LeCAR-Lab/ASAP, HorizonRobotics/HoloMotion, amazon-far/holosoma, nvidia-isaac/WBC-AGILE | various | Cloned; read requirements | ⛔ Isaac Lab / Isaac Gym, need GPU |
| google-deepmind/mujoco_playground, akanazawa/fpo, nicklashansen/tdmpc2 | loco / RL / WM | Cloned only | Not run (time) |

Setup notes: on a headless box, MuJoCo ≥3.17 needs `MUJOCO_GL=disable`; `pypi.nvidia.com` and the
`download.pytorch.org` index were blocked here, so warp/torch came from PyPI (`uv pip install --no-sources`).

## 3. Experiments that point to the thesis gap

### 3a. Where a released universal tracker breaks (`gmt_robustness.py`, GMT checkpoint)

| Perturbation | walk_stand | dance |
|---|---|---|
| Lateral push 200–600 N for 0.1 s | ✅ all survive | ✅ all survive |
| Torso payload 5 / 10 / **15** / 20 kg | ✅ ✅ **❌ fall 2.8 s** ❌ | ✅ all |
| Action latency 20 / 40 / **60 ms** | ✅ ✅ **❌ fall 2.2 s** | ✅ all |
| Foot friction 0.4 / 0.2 | ✅ ✅ | ✅ ✅ |
| Motor stiffness x0.6 / x0.4 | **❌ 5.5 s** / ❌ 1.4 s | ✅ / ❌ 1.9 s |

**Reading:** the tracker handles *disturbances* well and fails under *dynamics mismatch*
(mass, delay, actuator strength). That is the deployment gap HumanoidNDM and HEFT target.

### 3b. Naive test-time sampling does not fix it (`gmt_mppi.py`)

Policy-prior MPPI around the frozen GMT tracker (RGB/TTCO style): 16 samples, noise on the next 3 actions,
0.2 s horizon, MuJoCo as the rollout model, with the true model ("oracle") or the unperturbed one ("nominal").

| Case | No planner | MPPI, oracle model | MPPI, nominal model |
|---|---|---|---|
| Payload 15 kg | fall 2.84 s | fall 4.22 / 4.16 s | fall 4.04 / 3.68 s |
| Motors x0.6 | fall 5.48 s | fall 3.94 / 3.10 s (worse) | fall 4.16 / 3.00 s (worse) |
| Latency 60 ms | fall 2.20 s | fall 3.54 / 4.92 s | — |

**Reading:** at a real-time CPU budget, sampling only *delays* falls, and knowing the true model barely helps.
The fix has to change the **policy** (gradient-based adaptation), not only search around it at run time.
Caveat: this is a small budget; TTCO uses far more compute offline.

### 3c. How good are exact vs. world-model gradients through stiff contact? (`grad_estimators.py`)

Hopper with a penalty ground contact of stiffness k, stochastic linear policy, 60-step horizon, 64 rollouts per estimate,
20 trials. Reference = central finite differences with common random numbers over 200k rollouts
(reference self-error 0.5–4.5%).

| k | Estimator | cos to reference | relative bias | relative std |
|---|---|---|---|---|
| 1e4 | Exact BPTT | **1.00** | **0.12** | 0.26 |
| 1e4 | Exact + median clipping (∇Mimic-style; clips 17% of samples) | 1.00 | **0.50** | 0.22 |
| 1e4 | Learned-WM BPTT (SuperMimic-style, small MLP) | -0.69 | 653 | 222 |
| 1e4 | Likelihood ratio (PPO-like) | -0.19 | 287 | 1229 |
| 1e5 | Exact BPTT | **1.00** | **0.035** | 0.12 |
| 1e5 | Exact + clipping (clips 9%) | 1.00 | **0.13** | 0.11 |
| 1e5 | Learned-WM BPTT | -0.01 | huge (WM failed to fit stiff contact) | — |

**Readings:**
1. **Clipping bias is real and measurable.** It raised bias about 4x (0.12→0.50, 0.035→0.13) and cut std only about 10–15%.
   This is the quantity ∇Mimic says it did not measure.
2. Exact gradients kept the right direction at every stiffness. The predicted explosion was mild in this toy
   (max/median sample norm only 2–14), so this toy does not show *why* ∇Mimic needs clipping on the G1 (502 contact pairs).
3. A learned full-dynamics WM gives poor gradients exactly where contact is stiff. My proposed "replace outlier samples
   with WM gradients" hybrid therefore **failed**. Learning the *full* dynamics is the weak link, which points to
   learning only a **residual** on top of the exact simulator (idea A below).
   Caveat: my WM is far smaller than SuperMimic's (3x2048 MLP, H=32, critic bootstrap), so this is not a verdict on SuperMimic.

## 4. Thesis proposal (recommended)

### Title idea
**"Exact gradients that adapt: first-order whole-body control for humanoids under dynamics mismatch"**

### Core claim
Exact differentiable-simulator gradients (∇Mimic) are the most accurate training signal for humanoid trackers,
but they cannot learn from real-robot data. A learned *residual* world model (real minus sim, much easier to fit than
full dynamics) plugged into the exact adjoint gives gradients that are accurate *and* grounded in real data. That allows
fast, data-efficient adaptation to payloads, latency and actuator change: the failure modes in 3a that test-time
sampling (3b) cannot fix.

### Three papers

**P1: "How biased are first-order gradients for humanoid tracking?"** (6–9 months, analysis + method)
- Measure bias and variance on the G1 (LAFAN1) of: exact (NablaEngine / MuJoCo-Warp adjoint), exact + clipping,
  SuperMimic WM gradients, PPO. Use the common-random-numbers finite-difference reference from 3c, scaled up on GPU.
- Attribute bias to contact events (foot touchdown, self-collision) and to clipping.
- Method: replace clipping with an estimator whose bias is controlled. Candidates: median-of-means over sample groups,
  clipping plus a likelihood-ratio correction term only on clipped samples, or contact-event-aware smoothing
  (from Bundled Contact Gradients).
- Builds directly on both open repos; both papers list this as open. Venue: CoRL / RSS / ICRA.

**P2: "Residual world models for exact-gradient adaptation"** (main contribution)
- *Revised 2026-11-19 (METHOD_LOG #2):* an unconstrained MLP residual fits 5–11x better but gives wrong-direction gradients. Use physics parameters identified through the adjoint (HALO-style) + a small residual trained for **gradient fidelity** (multi-step + Jacobian loss). Must beat HALO and Contact-UAN (both retrain with PPO) on adaptation time.
- Dynamics = exact sim step f_sim + learned residual r_φ(s, a, history), trained on (perturbed-sim, then real) data.
  Policy gradients backpropagate through ∂f_sim (exact adjoint) + ∂r_φ.
- Benchmark = the stress suite in 3a (payload / latency / weak motors / friction) applied to GMT, BeyondMimic, SONIC and ∇Mimic policies.
- Baselines: domain randomization, HumanoidNDM-style full neural dynamics model, TTCO / RGB test-time sampling,
  SuperMimic WM fine-tuning.
- Metric: minutes of real data needed to recover tracking after a change.

**P3: hardware + beyond flat ground**
- Online adaptation on a real G1 (the residual model updated during deployment).
- Extend to terrain or hand-contact loco-manipulation. NablaEngine lacks terrain and SuperMimic is flat-ground only, so this is open.

### Why this is a good PhD bet
- Both base methods are about a month old with **public code** and tests that pass (section 2). You start at the frontier, not behind it.
- The gaps are stated by the authors themselves (clipping bias; sim-only WM; flat ground; no hardware).
- Compute-light: ∇Mimic and SuperMimic train in minutes to hours on **one GPU**, which fits a single Blackwell card.
- Each paper stands alone if a later one stalls.

### Risks
- Fast-moving area (SuperMimic, ∇Mimic, HumanoidNDM all appeared in Oct 2026); expect competitors on P2. Move fast on P1.
- NablaEngine is G1-specific CUDA, so other robots need the MuJoCo-Warp adjoint (slower).
- Residual models can still be hard to fit around contact; check this early with the toy in 3c using a residual target.

## 5. Alternative ideas (smaller or riskier)

| Idea | Pitch | Evidence / risk |
|---|---|---|
| B. Amortized real-time test-time refinement | Learn the sampling proposal (diffusion/flow) for TTCO-style command optimization so it runs at 50 Hz | 3b shows naive real-time sampling fails; crowded (TTCO, RGB, ReactiveBFM) |
| C. Delay-aware universal tracking | 60 ms latency kills GMT walking; train with a delay-conditioned WM / action-chunk interface ("execution gap" 2610.06582) | Clear failure mode, narrow scope (good first paper) |
| D. Preference / flow-policy fine-tuning of trackers | Carry over DPO (PhysMoDPO) and Flow Policy Gradients to fine-tune trackers for style and human preference (HumanTracker benchmark 2608.13555) | From nearby fields; weaker link to control theory |
| E. Gradient-informed sampling MPC | Use exact adjoint gradients to steer MPPI samples ("SMPC as SMC" 2611.10877, GRASP 2611.05754) | Natural bridge between planning and optimization; GPU-heavy |

## 6. Next steps on your Blackwell GPU

1. `NablaMimic`: `bash scripts/setup_env.sh`, then `python -m nablamimic.train --config configs/ours.args --out runs/ours_s0 --hours 1` (download LAFAN1 with `scripts/download_lafan1.sh`).
2. `supermimic` + `mjlab_research` as siblings: train `SuperMimic-GenCoord-G1` for 1 h.
3. Rerun `smoke_tests/grad_estimators.py` with a *residual* WM target, to check whether a residual fits where the full WM failed.
4. Port `smoke_tests/gmt_robustness.py` to the ∇Mimic and SuperMimic policies, to see whether exact/WM-trained trackers share GMT's failure modes. This is the first figure of P2.

## Reproducing the smoke tests

```bash
# GMT (CPU): needs torch, mujoco, tqdm, scipy
GMT_REPO=/path/to/humanoid-general-motion-tracking MUJOCO_GL=disable PYTHONPATH=smoke_tests \
  python smoke_tests/gmt_robustness.py out.json
python smoke_tests/gmt_headless.py /path/to/humanoid-general-motion-tracking 10 walk_stand.pkl dance.pkl
GMT_REPO=... MUJOCO_GL=disable PYTHONPATH=smoke_tests python smoke_tests/gmt_mppi.py out.json   # ~25 min CPU
python smoke_tests/grad_estimators.py out.json      # ~15 min CPU, torch only
MUJOCO_GL=disable python smoke_tests/mjlab_g1.py    # needs mjlab
dial-mpc --config smoke_tests/dial_h1.yaml          # needs dial-mpc (set output_dir)
```
