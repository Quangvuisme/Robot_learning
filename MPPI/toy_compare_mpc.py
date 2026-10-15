"""Same toy plant, two controllers: optimization-based (gradient) shooting MPC
vs. MPPI -- so the difference the slide is arguing for is visible on one
screen instead of taken on faith.

Optimization-based MPC here means: at every tick, take the current nominal
control sequence and run a few steps of gradient descent on the SAME rollout
cost J(u_0..u_{H-1}) = sum_t cost(x_t), using the analytic gradient through the
(linear, differentiable) plant model -- i.e. a bare-bones shooting method /
one un-regularized Gauss-Newton-free iLQR sibling. That gradient is exact
*because* this plant is linear and the cost is smooth; that is precisely the
case MPPI does not need and optimization-based MPC does.

MPPI instead only ever *evaluates* the plant/cost forward (no gradient, no
differentiability requirement at all) and combines K noisy rollouts by their
cost -- see toy_example.py for the fully annotated version.
"""

import numpy as np

from toy_example import step, cost, mppi_step

X_TARGET = 5.0
DT = 0.2


# --------------------------------------------------------- analytic gradient
# For this plant, x_{t+1} = A x_t + B u_t with
#   A = [[1, dt], [0, 1]],  B = [[dt^2], [dt]]     (symplectic Euler, matches step())
# and cost(x) = (pos - target)^2 + 0.1 vel^2, the gradient of the rollout cost
# J(u_0..u_{H-1}) w.r.t. every u_t can be obtained by backpropagating through
# the H linear steps -- exactly what autodiff would do, written out by hand so
# no extra dependency is needed.
A = np.array([[1.0, DT], [0.0, 1.0]])
B = np.array([DT ** 2, DT])
Q = np.diag([2.0, 0.2])              # d/dx of cost(x) = Q (x - [target,0])
x_ref = np.array([X_TARGET, 0.0])


def rollout(x0, u_seq):
    xs = [x0]
    x = x0
    for u in u_seq:
        x = step(x, u)
        xs.append(x)
    return xs


def rollout_cost(x0, u_seq):
    xs = rollout(x0, u_seq)
    return sum(cost(x) for x in xs[1:])


def rollout_grad(x0, u_seq):
    """dJ/du_t for every t, via the adjoint (backprop) method -- the same
    computation torch.autograd would do for this linear/quadratic system."""
    H = len(u_seq)
    xs = rollout(x0, u_seq)
    grads = np.zeros(H)
    # backprop the adjoint lambda_t = dJ/dx_t
    lam = np.zeros(2)
    for t in reversed(range(H)):
        lam = lam + Q @ (xs[t + 1] - x_ref)     # add this step's direct cost grad
        grads[t] = B @ lam                       # dJ/du_t = B^T lambda_t
        lam = A.T @ lam                          # propagate to x_t
    return grads


def gradient_mpc_step(x0, u_seq, n_iters=20, lr=0.05, verbose=True):
    """A handful of gradient-descent steps on the rollout cost -- this is the
    "classical" MPC solve: differentiate the model, descend, repeat."""
    u = u_seq.copy()
    trace = []
    for it in range(n_iters):
        g = rollout_grad(x0, u)
        u = u - lr * g
        if verbose and it in (0, n_iters // 2, n_iters - 1):
            trace.append((it, rollout_cost(x0, u), g.copy()))
    if verbose:
        print(f"  x0 = pos {x0[0]:+.3f}, vel {x0[1]:+.3f}  "
             f"(target = 5.0, nominal u_seq = {np.round(u_seq, 2)})")
        for it, J, g in trace:
            print(f"    grad step {it:3d}: J = {J:8.3f}   grad = {np.round(g, 3)}")
        print(f"  converged u_seq = {np.round(u, 3)}")
        print(f"  -> apply u_0 = {u[0]:+.3f} to the real plant\n")
    return u[0], u


def run(controller_name, step_fn, n_ticks=6, **kwargs):
    rng = np.random.default_rng(0)
    x = np.array([0.0, 0.0])
    u_seq = np.zeros(4)
    positions = [x[0]]
    print(f"\n=== {controller_name} ===")
    for tick in range(n_ticks):
        print(f"--- control tick {tick} (t={tick*DT:.1f}s) ---")
        if controller_name.startswith("Optimization"):
            u0, u_seq = step_fn(x, u_seq, **kwargs)
        else:
            u0, u_seq = step_fn(x, u_seq, rng=rng, **kwargs)
        x = step(x, u0)
        u_seq = np.roll(u_seq, -1)
        u_seq[-1] = 0.0
        positions.append(x[0])
        if kwargs.get("verbose", True):
            print(f"  plant after applying u_0: pos={x[0]:+.3f}, vel={x[1]:+.3f}\n")
        else:
            print(f"  u_0={u0:+.3f} -> pos={x[0]:+.3f}, vel={x[1]:+.3f}")
    print(f"final state: pos={x[0]:.3f} (target {X_TARGET}), vel={x[1]:.3f}")
    return positions


# ------------------------------------------------- the nondifferentiable case
# A hard rectangular no-go zone between start and target: cost(x) jumps by a
# constant PENALTY if pos falls inside [obs_lo, obs_hi], otherwise it is the
# same smooth tracking cost. The gradient of an indicator function is zero
# almost everywhere and undefined at the boundary -- there is no useful
# "dJ/du" to descend, so gradient_mpc_step is handed the WRONG gradient
# (computed as if the obstacle did not exist) once its trajectory is near it,
# which is exactly the failure mode Algorithm 2's slide calls out
# ("easy to handle complex costs -- nondifferentiable, DNN, etc").
OBS_LO, OBS_HI, OBS_PENALTY = 2.0, 3.0, 40.0


def cost_with_obstacle(x, x_target=X_TARGET):
    pos, vel = x
    base = (pos - x_target) ** 2 + 0.1 * vel ** 2
    blocked = OBS_LO <= pos <= OBS_HI
    return base + (OBS_PENALTY if blocked else 0.0)


def mppi_step_obstacle(x0, u_seq, K=32, H=4, sigma=2.0, lam=1.0, rng=None):
    """Identical to mppi_step, but scoring rollouts with cost_with_obstacle --
    MPPI needs no gradient of that indicator, it just forward-simulates and
    downweights whichever samples happened to land inside the no-go zone."""
    rng = rng or np.random.default_rng(0)
    H = len(u_seq)
    eps = rng.normal(0.0, sigma, size=(K, H))
    costs = np.zeros(K)
    for k in range(K):
        x = x0.copy()
        S = 0.0
        for t in range(H):
            u_t = u_seq[t] + eps[k, t]
            x = step(x, u_t)
            S += cost_with_obstacle(x) + lam * u_seq[t] * (1.0 / sigma) * eps[k, t]
        costs[k] = S
    beta = costs.min()
    w = np.exp(-(costs - beta) / lam)
    w /= w.sum()
    u_seq_new = u_seq + (w[:, None] * eps).sum(axis=0)
    return u_seq_new[0], u_seq_new


def gradient_mpc_step_obstacle(x0, u_seq, n_iters=20, lr=0.05):
    """Same gradient descent as gradient_mpc_step, but its analytic gradient
    (rollout_grad) is computed for the SMOOTH cost only -- it has no term for
    the obstacle because there is no usable gradient of a hard indicator to
    write down. This is the realistic failure: not a crash, just a controller
    that is blind to the obstacle and drives straight through it."""
    u = u_seq.copy()
    for _ in range(n_iters):
        g = rollout_grad(x0, u)      # <-- ignores the obstacle entirely
        u = u - lr * g
    return u[0], u


def run_obstacle(controller_name, step_fn, n_ticks=8, **kwargs):
    rng = np.random.default_rng(0)
    x = np.array([0.0, 0.0])
    u_seq = np.zeros(4)
    positions, hit = [x[0]], False
    for tick in range(n_ticks):
        if controller_name.startswith("Optimization"):
            u0, u_seq = step_fn(x, u_seq, **kwargs)
        else:
            u0, u_seq = step_fn(x, u_seq, rng=rng, **kwargs)
        x = step(x, u0)
        u_seq = np.roll(u_seq, -1)
        u_seq[-1] = 0.0
        positions.append(x[0])
        if OBS_LO <= x[0] <= OBS_HI:
            hit = True
    return positions, hit


def compare_obstacle():
    print("\n\n" + "=" * 70)
    print("SAME task, now with a hard no-go zone in the way: "
         f"pos in [{OBS_LO}, {OBS_HI}] costs +{OBS_PENALTY:.0f} (a step, not a "
         "slope -- its gradient is 0 a.e.)")
    print("=" * 70)

    pos_grad, hit_grad = run_obstacle(
        "Optimization-based MPC (blind to the obstacle: no usable gradient of it)",
        gradient_mpc_step_obstacle, n_iters=20, lr=0.05)
    pos_mppi, hit_mppi = run_obstacle(
        "MPPI (forward-simulates the obstacle cost directly, K=32)",
        mppi_step_obstacle, K=32, H=4)

    print(f"\n{'tick':>4}  {'opt-based MPC':>13}  {'MPPI':>8}   (no-go zone: "
         f"[{OBS_LO}, {OBS_HI}])")
    for t, (a, b) in enumerate(zip(pos_grad, pos_mppi)):
        flag_g = " <- IN ZONE" if OBS_LO <= a <= OBS_HI else ""
        flag_m = " <- in zone" if OBS_LO <= b <= OBS_HI else ""
        print(f"{t:>4}  {a:13.3f}{flag_g:<12} {b:8.3f}{flag_m}")

    print(f"\noptimization-based MPC drove through the no-go zone: {hit_grad}")
    print(f"MPPI drove through the no-go zone:          {hit_mppi}")
    print("\nOptimization-based MPC's local descent has no signal telling it the "
         "obstacle exists (its gradient never includes the indicator's jump),")
    print("so it beelines straight for the target through it. MPPI never "
         "needed a gradient of the obstacle term -- it just SAW, by forward")
    print("simulation, that rollouts landing inside cost more, and "
         "down-weighted them.")


def main():
    print("Same plant (1D point mass -> target 5.0), same H=4, same 6 ticks.")
    print("Only the controller differs.\n")

    pos_grad = run("Optimization-based MPC (shooting, exact analytic gradient)",
                   gradient_mpc_step, n_iters=20, lr=0.05)
    pos_mppi = run("MPPI (K=5 samples, no gradient at all)",
                   mppi_step, K=5, H=4, verbose=False)

    print("\n=== side-by-side cart position after each tick ===")
    print(f"{'tick':>4}  {'opt-based MPC':>13}  {'MPPI':>8}")
    for t, (a, b) in enumerate(zip(pos_grad, pos_mppi)):
        print(f"{t:>4}  {a:13.3f}  {b:8.3f}")

    compare_obstacle()


if __name__ == "__main__":
    main()
