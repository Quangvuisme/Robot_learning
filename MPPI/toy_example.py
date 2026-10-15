"""A from-scratch, no-torch, by-hand-readable MPPI toy example.

Plant: a 1D point mass ("cart" with no pole) that only has to reach and hold
position `x_target`.  State is just [pos, vel]; action is a force.  This is
the smallest system where every quantity in Algorithm 2 -- eps, rollout, cost,
beta, eta, weight, weighted update -- can be printed and understood by eye.

No GPU/torch, no batching library: everything is a plain Python/NumPy loop so
you can add a `print` anywhere and see exactly what MPPI is doing. Once this
makes sense, `mppi.py` is the same algorithm rewritten as batched torch tensors
so `K` can be 8192 instead of 5.

Run:
    python toy_example.py
"""

import numpy as np

# ----------------------------------------------------------------- the plant
def step(x, u, dt=0.2):
    """x = [pos, vel]; u = force (mass = 1, so force = acceleration)."""
    pos, vel = x
    vel_next = vel + dt * u
    pos_next = pos + dt * vel_next          # symplectic Euler, same as cartpole.py
    return np.array([pos_next, vel_next])


def cost(x, x_target=5.0):
    """Running cost: just squared distance to the target (+ a bit of velocity
    penalty so it settles instead of overshooting forever)."""
    pos, vel = x
    return (pos - x_target) ** 2 + 0.1 * vel ** 2


# ------------------------------------------------------------------- MPPI
def mppi_step(x0, u_seq, K=5, H=4, sigma=2.0, lam=1.0, rng=None, verbose=True):
    """One planning step of Algorithm 2, K samples / H-step horizon, printed.

    u_seq: current warm-started nominal sequence, shape (H,)
    Returns: (u0_to_apply, new_u_seq, per-sample debug rows)
    """
    rng = rng or np.random.default_rng(0)
    H = len(u_seq)

    eps = rng.normal(0.0, sigma, size=(K, H))     # eps^k_t ~ N(0, Sigma)
    costs = np.zeros(K)
    rows = []

    for k in range(K):
        x = x0.copy()
        S = 0.0
        for t in range(H):
            u_t = u_seq[t] + eps[k, t]            # u_{t-1} + eps^k_{t-1}
            x = step(x, u_t)                      # F(x_{t-1}, u_t)
            # S += q(x_t) + lambda * u_{t-1}^T Sigma^-1 eps^k_{t-1}
            S += cost(x) + lam * u_seq[t] * (1.0 / sigma) * eps[k, t]
        costs[k] = S
        rows.append((k, eps[k].copy(), S))

    beta = costs.min()                             # beta <- min_k S(eps^k)
    w_unnorm = np.exp(-(costs - beta) / lam)
    eta = w_unnorm.sum()                           # eta <- sum_k exp(...)
    w = w_unnorm / eta                             # w_k <- (1/eta) exp(...)

    # u_t += sum_k w_k eps^k_t, for every t in the horizon
    u_seq_new = u_seq + (w[:, None] * eps).sum(axis=0)

    if verbose:
        print(f"  x0 = pos {x0[0]:+.3f}, vel {x0[1]:+.3f}   "
             f"(target = 5.0, nominal u_seq = {np.round(u_seq, 2)})")
        print(f"  {'k':>2}  {'eps_0..H-1 (sampled noise)':<34}  "
             f"{'cost S':>9}  {'weight w_k':>10}")
        for k, e, S in rows:
            print(f"  {k:>2}  {np.round(e, 2)!s:<34}  {S:9.2f}  {w[k]:10.3f}")
        print(f"  beta (best cost) = {beta:.2f}, eta (norm. const) = {eta:.4f}")
        print(f"  updated u_seq = u_seq + sum_k w_k*eps_k = {np.round(u_seq_new, 3)}")
        print(f"  -> apply u_0 = {u_seq_new[0]:+.3f} to the real plant, "
             f"then shift the horizon\n")

    return u_seq_new[0], u_seq_new


def main():
    rng = np.random.default_rng(0)
    x = np.array([0.0, 0.0])       # start at rest at the origin
    u_seq = np.zeros(4)            # H=4, warm start at zero force
    K, H = 5, 4                    # tiny: 5 samples, 4-step horizon

    print("=== MPPI toy example: 1D point mass driving to x_target=5.0 ===")
    print(f"K={K} samples, H={H}-step horizon, dt=0.2s, sigma=2.0, lambda=1.0\n")

    for tick in range(6):
        print(f"--- control tick {tick} (t={tick*0.2:.1f}s) ---")
        u0, u_seq = mppi_step(x, u_seq, K=K, H=H, rng=rng)
        x = step(x, u0)                                  # SendToActuators(u_0)
        u_seq = np.roll(u_seq, -1); u_seq[-1] = 0.0       # shift + reinitialize
        print(f"  plant after applying u_0: pos={x[0]:+.3f}, vel={x[1]:+.3f}\n")

    print(f"final state: pos={x[0]:.3f} (target 5.0), vel={x[1]:.3f}")


if __name__ == "__main__":
    main()
