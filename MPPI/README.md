# MPPI Cartpole (swing-up + balance)

A literal implementation of **Algorithm 2** from the "Information Theoretic MPC
for Model-Based Reinforcement Learning" slide, applied to the classic
underactuated cartpole swing-up task.

```
Given:  F (transition model); K samples; H timesteps
        (u_0, ..., u_{H-1}): warm-started control sequence
        Sigma, lambda: noise / temperature hyperparameters
while task not done:
    x0 <- GetStateEstimate()
    for k in 0..K-1:
        sample eps^k_t ~ N(0, Sigma)                      for t in 0..H-1
        roll out x_t <- F(x_{t-1}, u_{t-1} + eps^k_{t-1})
        S(eps^k) = sum_t [ q(x_t) + lambda u_{t-1}^T Sigma^-1 eps^k_{t-1} ] + phi(x_T)
    beta <- min_k S(eps^k)
    w_k  <- exp(-(S(eps^k)-beta)/lambda) / sum_k exp(-(S(eps^k)-beta)/lambda)
    u_t += sum_k w_k eps^k_t                               for all t
    SendToActuators(u_0)
    shift: u_{t-1} <- u_t   (receding horizon), reinitialize u_{H-1}
```

## Files

- `toy_example.py` — a tiny, no-torch, by-hand-readable MPPI run (see below) —
  start here if the algorithm above is not yet intuitive.
- `toy_compare_mpc.py` — the same toy plant solved by classical gradient-based
  MPC vs. MPPI, on both a smooth cost and a nondifferentiable one (see below).
- `cartpole.py` — batched (torch) cartpole dynamics: state
  `[pos, vel, theta, thetadot]`, `theta=0` hanging down, `theta=pi` upright.
  Symplectic Euler integration, force-limited actuator, rail limits.
- `mppi.py` — the MPPI controller itself, vectorised over `K` rollouts so the
  whole sampling/rollout/weighting loop runs as batched GPU tensor ops (no
  Python loop over samples, only over the `H` timesteps).
- `run_cartpole.py` — runs the closed loop: at every control tick, replan from
  scratch with the controller, step the "real" plant, log everything.
- `render.py` — renders the cart-pole animation (with a faint fan of live MPPI
  rollouts colored by importance weight) next to angle/position/force traces.

## Run it

```bash
python run_cartpole.py --quick            # small/fast sanity run (K=1024)
python run_cartpole.py                    # full run: K=8192, H=40, 500 steps
python run_cartpole.py --steps 800 --cpu  # force CPU instead of CUDA
```

Output goes to `out/`: `mppi_cartpole.mp4`, `mppi_cartpole.gif`, and
`run_data.npz` (raw state/action history).

## Toy example: MPPI by hand, no GPU needed

`toy_example.py` is the same Algorithm 2 stripped down to the smallest system
where you can print every quantity and see what's happening: a 1D point mass
with no pole, `K=5` samples, `H=4`-step horizon, all plain NumPy loops (no
torch, no batching). Read this before `mppi.py`/`cartpole.py` — it's the exact
same math, just spelled out instead of vectorised.

```bash
python toy_example.py
```

Sample output (one control tick):

```
--- control tick 2 (t=0.4s) ---
  x0 = pos +0.056, vel +0.316   (target = 5.0, nominal u_seq = [ 2.   -0.71 -1.72  0.  ])
   k  eps_0..H-1 (sampled noise)             cost S  weight w_k
   0  [-2.52  3.03  2.69  1.56]               81.16       0.000
   1  [ 0.53 -0.63  2.92  3.92]               81.23       0.000
   2  [ 3.6   2.63  0.71 -2.42]               72.82       1.000
   3  [-0.01  1.31 -2.58  0.79]               89.17       0.000
   4  [ 0.86  1.39 -2.37 -1.32]               87.07       0.000
  beta (best cost) = 72.82, eta (norm. const) = 1.0005
  updated u_seq = u_seq + sum_k w_k*eps_k = [ 5.597  1.919 -1.001 -2.414]
  -> apply u_0 = +5.597 to the real plant, then shift the horizon

  plant after applying u_0: pos=+0.344, vel=+1.436
```

What to notice:

- **`eps`** — each of the `K=5` rows is one *randomly perturbed* copy of the
  nominal force sequence (`u_seq + eps`), rolled out `H` steps into the future
  with the plant model `F` (`step()` in the file).
- **`cost S`** — how bad that rollout was: squared distance from the target,
  summed over the horizon. Sample `k=2` happened to push hardest toward the
  target early and got the lowest cost (72.82).
- **`weight w_k`** — `softmax(-S/lambda)`: essentially all of the weight
  (1.000) landed on sample 2, almost none on the others. Lower cost -> higher
  weight -> more say in the update. `lambda` controls how sharp/soft that is.
- **`updated u_seq`** — the nominal sequence shifts toward whatever the
  high-weight samples did (`u_seq + sum_k w_k * eps_k`), which is why
  `u_seq[0]` jumps from 2.0 to 5.6 in this tick.
- Only `u_0` (the *first* action of the newly updated sequence) is ever sent
  to the real plant. Next tick, the horizon shifts by one and the whole thing
  repeats from the new state — that's the "receding horizon" in MPC.

This is precisely what `mppi.py` does for the cartpole, except `K=8192` and
`H=40` rollouts happen as one batched tensor operation on the GPU instead of a
Python `for k in range(K)` loop.

## MPPI (sampling-based MPC) vs. optimization-based MPC

`toy_compare_mpc.py` runs the *same* 1D toy plant through two controllers back
to back so the trade-off is visible instead of asserted:

```bash
python toy_compare_mpc.py
```

**Part 1 — smooth cost, both work, optimization-based MPC is more sample-efficient.**
MPC comes in two families: *optimization-based* (solve an optimization problem
over the horizon each tick — gradient/shooting methods, iLQR, DDP, QP-based
linear MPC) and *sampling-based* (MPPI's family — draw many candidate
control sequences, weight them by rollout cost, no gradient anywhere). Here a
bare-bones optimization-based MPC (a few steps of gradient descent on the rollout
cost, using the model's analytic Jacobian) is implemented alongside MPPI on the
exact same linear/quadratic plant. With an exact gradient available,
optimization-based MPC reaches the target faster using only 20 gradient steps
per tick, vs. MPPI's 5 random samples per tick:

| tick | optimization-based MPC pos | MPPI pos |
|-----:|----------------------------:|---------:|
| 0 | 0.000 | 0.000 |
| 2 | 0.488 | 0.056 |
| 4 | 1.658 | 0.769 |
| 6 | 3.077 | 1.591 |

This is expected and is *not* a knock against MPPI: when the model is smooth
and differentiable and a good gradient exists, following it is a far more
direct signal than "try K random perturbations and average the good ones."

**Part 2 — a hard, nondifferentiable obstacle cost: optimization-based MPC
drives straight through it, MPPI does not.** The same script then adds one
rectangular no-go zone (`pos in [2, 3]` adds a flat `+40` penalty — a step
function, not a slope) between the start and the target, and reruns both
controllers unmodified except for the cost:

```
tick  opt-based MPC      MPPI   (no-go zone: [2.0, 3.0])
   5          2.365 <- IN ZONE     1.449
   6          3.077                1.613
   7          3.749                1.755
   8          4.351                1.838

optimization-based MPC drove through the no-go zone: True
MPPI drove through the no-go zone:          False
```

The indicator's gradient is zero almost everywhere and undefined at the
boundary, so `dJ/du` computed by backprop through the smooth part of the cost
never "sees" the obstacle at all — optimization-based MPC is not making a
mistake, it genuinely has no usable local signal telling it the obstacle
exists, and drives straight through. MPPI needs no gradient of anything: it forward-simulates
each of its `K` sampled rollouts through the *real* cost (obstacle included),
and any rollout that clips the no-go zone simply scores worse and gets
down-weighted. This is the concrete version of the slide's bullet point
*"easy to handle complex costs/dynamics (nondifferentiable, DNN, etc.)"* — the
same argument is why MPPI tolerates a neural-network cost/dynamics model,
contact discontinuities, or a hard constraint with no smooth relaxation,
none of which hand you a usable gradient either.

**Why the cartpole demo in this repo still uses MPPI even though its cost is
smooth:** underactuated swing-up is *highly non-convex* — countless
local-minimum trajectories (small pumping motions, both swing directions,
almost-there-but-stops-short) trap a purely local gradient method long before
it finds the big pump-then-catch policy that works. MPPI's random exploration
across `K=8192` independent rollouts routinely stumbles onto the right regime,
which is also why `sigma` (how much it explores) mattered more in tuning than
almost anything else — see the hyperparameter notes below.

## Default hyperparameters

`K=8192` samples, `H=40` step horizon (0.8 s lookahead at `dt=0.02`),
`sigma=3.0` N action noise, `lambda=0.5` temperature — tuned by grid search
(see cost trade-off below) to reliably swing up in ~2 s and then balance
within a degree or two of upright. On an RTX-class GPU this replans in
~13 ms/step (~75 Hz), matching the "~80 Hz on a laptop GPU" figure on the
source slide.

Cost function: quadratic in angle-to-upright, angular velocity, cart position
and velocity, running throughout the horizon, plus a heavier terminal cost at
the last timestep and a quartic soft barrier keeping rollouts off the rail
ends. Larger `sigma` explores more (helps swing-up escape local optima) but
adds steady-state jitter once balancing; smaller `lambda` sharpens the
importance weights toward the best rollouts (more decisive, less averaging).
