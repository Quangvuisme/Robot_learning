"""
Toy test for Idea 7: what can an action-space sim-to-real correction (ASAP-style "delta action model") fix?

Claim (matching condition): a correction a -> a + delta(s, a) can only change the next state along the input
directions B(s) = d f / d a. A sim-to-real gap e(s, a) = f_real - f_sim splits into
    matched part    P_B e        (fixable by delta)
    unmatched part  (I - P_B) e  (no action residual can produce it).
The cartpole is underactuated (one force, two velocity DOFs), like a humanoid's unactuated floating base.

Gaps tested (the "real" system differs from the nominal sim):
    matched   : actuator gain 0.7 and viscous cart friction (both enter through the cart-force channel)
    unmatched : pole joint friction and heavier pole (act on the pole directly)
    both      : all of the above

Correction models fitted on real data from policy pi0, evaluated on held-out pi0 data and on pi1 (a different
policy, i.e. a different state distribution):
    none          : plain sim
    delta 1-step  : per-sample least-squares action residual, then ridge regression on features
    delta H-step  : ASAP-like, optimise the residual model for H-step open-loop replay error (finite-diff Adam)
    state resid.  : residual on the full next-velocity (the "fix"; can represent unmatched gaps)

Pure numpy; about a minute on CPU.
"""
import argparse
import json
import os

import numpy as np

DT, G = 0.02, 9.81
NOM = dict(mc=1.0, mp=0.1, l=0.5, k=1.0, bc=0.0, bp=0.0)
GAPS = {
    "matched": dict(NOM, k=0.7, bc=1.5),
    "unmatched": dict(NOM, bp=0.08, mp=0.16),
    "both": dict(NOM, k=0.7, bc=1.5, bp=0.08, mp=0.16),
}


def accel(s, u, p):
    """Cartpole with actuator gain k, viscous cart friction bc, pole joint friction bp (theta=0 upright)."""
    x, xd, th, thd = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    F = p["k"] * u - p["bc"] * xd
    mt = p["mc"] + p["mp"]
    st, ct = np.sin(th), np.cos(th)
    tmp = (F + p["mp"] * p["l"] * thd ** 2 * st) / mt
    thdd = (G * st - ct * tmp - p["bp"] * thd / (p["mp"] * p["l"])) / (p["l"] * (4 / 3 - p["mp"] * ct ** 2 / mt))
    xdd = tmp - p["mp"] * p["l"] * thdd * ct / mt
    return xdd, thdd


def step(s, u, p):
    xdd, thdd = accel(s, u, p)
    xd = s[..., 1] + DT * xdd
    thd = s[..., 3] + DT * thdd
    return np.stack([s[..., 0] + DT * xd, xd, s[..., 2] + DT * thd, thd], -1)


def policy(s, gains, rng, noise):
    """Energy swing-up + PD catch, with exploration noise. gains = (kE, kx, kv, Kth, Kthd)."""
    kE, kx, kv, Kth, Kthd = gains
    x, xd, th, thd = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    E = 0.5 * (4 / 3) * NOM["l"] ** 2 * thd ** 2 + G * NOM["l"] * (np.cos(th) - 1)   # 0 at upright rest
    swing = kE * E * thd * np.cos(th) - kx * x - kv * xd
    thw = (th + np.pi) % (2 * np.pi) - np.pi
    catch = Kth * thw + Kthd * thd - 1.0 * x - 2.0 * xd
    u = np.where(np.abs(thw) < 0.4, catch, swing)
    return np.clip(u + rng.normal(0, noise, u.shape), -15, 15)


PI0 = (6.0, 1.0, 1.5, 40.0, 8.0)
PI1 = (12.0, 0.4, 0.8, 30.0, 5.0)


def collect(p, gains, n_traj, T, seed, noise=2.0):
    rng = np.random.default_rng(seed)
    s = np.stack([rng.uniform(-0.5, 0.5, n_traj), np.zeros(n_traj),
                  rng.uniform(np.pi - 0.6, np.pi + 0.6, n_traj), rng.normal(0, 0.5, n_traj)], -1)
    S, U = [s], []
    for _ in range(T):
        u = policy(s, gains, rng, noise)
        s = step(s, u, p)
        S.append(s)
        U.append(u)
    return np.stack(S, 1), np.stack(U, 1)        # (n, T+1, 4), (n, T)


def feats(s, u):
    x, xd, th, thd = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    st, ct = np.sin(th), np.cos(th)
    return np.stack([np.ones_like(x), x, xd, st, ct, thd, u, thd ** 2 * st, ct * u, ct * thd, st * thd, ct ** 2,
                     xd * ct, thd * ct ** 2], -1)


def B_dir(s, u, p=NOM, h=1e-3):
    """Input direction of the next velocities: d v' / d u (finite difference)."""
    return (step(s, u + h, p)[..., [1, 3]] - step(s, u, p)[..., [1, 3]]) / h


def ridge(X, y, lam=1e-3):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def fit_delta_1step(S, U):
    s, u, s1 = S[:, :-1].reshape(-1, 4), U.reshape(-1), S[:, 1:].reshape(-1, 4)
    e = s1[:, [1, 3]] - step(s, u, NOM)[:, [1, 3]]
    b = B_dir(s, u)
    d = (b * e).sum(-1) / (b * b).sum(-1)                  # least-squares action residual per sample
    return ridge(feats(s, u), d)


def fit_state_resid(S, U):
    s, u, s1 = S[:, :-1].reshape(-1, 4), U.reshape(-1), S[:, 1:].reshape(-1, 4)
    e = s1[:, [1, 3]] - step(s, u, NOM)[:, [1, 3]]
    X = feats(s, u)
    return np.stack([ridge(X, e[:, 0]), ridge(X, e[:, 1])], -1)


def rollout(S, U, H, model=None, kind=None):
    """Open-loop replay of real actions from real start states in windows of H steps. Returns mean error."""
    n, T = U.shape
    starts = np.arange(0, T - H, H)
    s = S[:, starts].reshape(-1, 4)
    err = 0.0
    for h in range(H):
        u = U[:, starts + h].reshape(-1)
        if kind == "delta":
            s_next = step(s, u + feats(s, u) @ model, NOM)
        else:
            s_next = step(s, u, NOM)
            if kind == "state":
                r = feats(s, u) @ model
                s_next = s_next.copy()
                s_next[:, 1] += r[:, 0]
                s_next[:, 3] += r[:, 1]
                s_next[:, 0] = s[:, 0] + DT * s_next[:, 1]
                s_next[:, 2] = s[:, 2] + DT * s_next[:, 3]
        s = s_next
        real = S[:, starts + h + 1].reshape(-1, 4)
        err += np.mean(np.sum((s - real)[:, [0, 2]] ** 2, -1))   # position errors (cart, pole angle)
    return err / H


def fit_delta_hstep(S, U, H, w0, iters=300, lr=0.02, seed=0):
    """ASAP-like: optimise the action-residual model for H-step replay error (finite-difference Adam)."""
    rng = np.random.default_rng(seed)
    w = w0.copy()
    m = np.zeros_like(w); v = np.zeros_like(w)
    scale = np.abs(w0) + 1e-2
    for it in range(1, iters + 1):
        idx = rng.choice(S.shape[0], 48, replace=False)
        Ss, Us = S[idx], U[idx]
        g = np.zeros_like(w)
        for j in range(w.size):
            e = np.zeros_like(w); e[j] = 1e-3 * scale[j]
            g[j] = (rollout(Ss, Us, H, w + e, "delta") - rollout(Ss, Us, H, w - e, "delta")) / (2 * e[j])
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w -= lr * scale * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=25)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = {}
    for gname, p in GAPS.items():
        S0, U0 = collect(p, PI0, 64, 400, seed=1)          # training data (pi0)
        S0t, U0t = collect(p, PI0, 32, 400, seed=2)        # held-out pi0
        S1, U1 = collect(p, PI1, 32, 400, seed=3)          # new policy pi1
        # matched fraction of the one-step gap
        s, u, s1 = S0[:, :-1].reshape(-1, 4), U0.reshape(-1), S0[:, 1:].reshape(-1, 4)
        e = s1[:, [1, 3]] - step(s, u, NOM)[:, [1, 3]]
        b = B_dir(s, u)
        Pe = b * ((b * e).sum(-1) / (b * b).sum(-1))[:, None]
        rho = float((Pe ** 2).sum() / (e ** 2).sum())
        w1 = fit_delta_1step(S0, U0)
        wH = fit_delta_hstep(S0, U0, args.H, w1)
        wS = fit_state_resid(S0, U0)
        row = dict(matched_fraction_rho=round(rho, 3))
        for name, (Ss, Us) in dict(pi0_heldout=(S0t, U0t), pi1_new_policy=(S1, U1)).items():
            base = rollout(Ss, Us, args.H)
            row[name] = dict(
                none=1.0,
                delta_1step=round(rollout(Ss, Us, args.H, w1, "delta") / base, 3),
                delta_Hstep=round(rollout(Ss, Us, args.H, wH, "delta") / base, 3),
                state_resid=round(rollout(Ss, Us, args.H, wS, "state") / base, 3))
        res[gname] = row
        print(gname, json.dumps(row))
    res["note"] = ("numbers = H-step open-loop replay error (cart position + pole angle) relative to the plain "
                   f"sim (1.0 = no improvement, 0 = perfect); H = {args.H}")
    with open(os.path.join(args.out, "idea7_matched_unmatched.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
