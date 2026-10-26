"""
Idea 7, closed-loop test: does the choice of sim-to-real correction matter for POLICY TRANSFER?

Pipeline per gap type (matched / unmatched / both) and per data seed:
  1. collect "real" data with policy pi0 (as in matched_unmatched_delta_action.py)
  2. fit corrections with GENERIC features (all degree-<=2 monomials of x, xd, sin, cos, thd, u; no hand-picked
     terms):  delta 1-step,  delta H-step (ASAP-like),  state residual
  3. optimise a swing-up controller (7 gains, CEM) inside each corrected simulator
  4. deploy the optimised controller on the real system: return and success rate
Baselines: plain nominal sim, and "oracle" (optimising directly on the real system = upper bound).
"""
import argparse
import json
import os

import numpy as np

import matched_unmatched_delta_action as T0

NOM, step, B_dir, ridge = T0.NOM, T0.step, T0.B_dir, T0.ridge
# Gap sizes chosen (by a scan) so that a controller tuned in the plain sim FAILS on the real system while one
# tuned on the real system succeeds -- otherwise the transfer test is uninformative.
GAPS = {
    "matched": dict(NOM, k=0.5, bc=3.0),
    "unmatched": dict(NOM, bp=0.03, mp=0.12),
    "both": dict(NOM, k=0.5, bc=3.0, bp=0.03, mp=0.12),
}
U_MAX = 15.0


def feats(s, u):
    z = [s[..., 0], s[..., 1], np.sin(s[..., 2]), np.cos(s[..., 2]), s[..., 3], u]
    out = [np.ones_like(z[0])] + z + [z[i] * z[j] for i in range(6) for j in range(i, 6)]
    return np.stack(out, -1)                                    # 28 generic features


def flat(S, U):
    return S[:, :-1].reshape(-1, 4), U.reshape(-1), S[:, 1:].reshape(-1, 4)


def fit_delta_1step(S, U):
    s, u, s1 = flat(S, U)
    e = s1[:, [1, 3]] - step(s, u, NOM)[:, [1, 3]]
    b = B_dir(s, u)
    return ridge(feats(s, u), (b * e).sum(-1) / (b * b).sum(-1))


def fit_state(S, U):
    s, u, s1 = flat(S, U)
    e = s1[:, [1, 3]] - step(s, u, NOM)[:, [1, 3]]
    X = feats(s, u)
    return np.stack([ridge(X, e[:, 0]), ridge(X, e[:, 1])], -1)


def sim_step(kind, model, p_real):
    def f(s, u):
        if kind == "nominal":
            return step(s, u, NOM)
        if kind == "oracle":
            return step(s, u, p_real)
        if kind in ("delta1", "deltaH"):
            return step(s, np.clip(u + feats(s, u) @ model, -U_MAX, U_MAX), NOM)
        s1 = step(s, u, NOM)
        r = feats(s, u) @ model
        v = s1[..., 1] + r[..., 0], s1[..., 3] + r[..., 1]
        return np.stack([s[..., 0] + T0.DT * v[0], v[0], s[..., 2] + T0.DT * v[1], v[1]], -1)
    return f


def replay_err(S, U, H, f):
    n, T = U.shape
    starts = np.arange(0, T - H, H)
    s = S[:, starts].reshape(-1, 4)
    err = 0.0
    for h in range(H):
        s = f(s, U[:, starts + h].reshape(-1))
        err += np.mean(np.sum((s - S[:, starts + h + 1].reshape(-1, 4))[:, [0, 2]] ** 2, -1))
    return err / H


def fit_delta_H(S, U, H, w0, iters, seed):
    rng = np.random.default_rng(seed)
    w = w0.copy(); m = np.zeros_like(w); v = np.zeros_like(w)
    scale = np.abs(w0) + 1e-2
    for it in range(1, iters + 1):
        idx = rng.choice(S.shape[0], 24, replace=False)
        Ss, Us = S[idx], U[idx]
        g = np.zeros_like(w)
        for j in range(w.size):
            d = np.zeros_like(w); d[j] = 1e-3 * scale[j]
            g[j] = (replay_err(Ss, Us, H, sim_step("delta1", w + d, None)) -
                    replay_err(Ss, Us, H, sim_step("delta1", w - d, None))) / (2 * d[j])
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w -= 0.02 * scale * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return w


# ---------------- closed-loop controller optimisation ----------------
def lqr_gains(p=NOM, h=1e-5):
    """Discrete LQR at the upright equilibrium of the given model (numerical linearisation)."""
    z = np.zeros(4)
    A = np.stack([(step(z + h * np.eye(4)[i], 0.0, p) - step(z - h * np.eye(4)[i], 0.0, p)) / (2 * h)
                  for i in range(4)], 1)
    B = ((step(z, h, p) - step(z, -h, p)) / (2 * h))[:, None]
    Q, R, P = np.diag([1.0, 1.0, 10.0, 1.0]), np.array([[0.1]]), np.eye(4)
    for _ in range(2000):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
    return -K[0]                                               # u = (-K) s  ->  [Kx, Kv, Kth, Kthd]


G_INIT = np.concatenate([[6.0, 1.0, 1.5], lqr_gains()])       # kE, kx, kv | Kx, Kv, Kth, Kthd


def controller(s, g):
    """Energy swing-up + linear catch. g: (P, 7); s: (P, N, 4)."""
    kE, kx, kv, Kx, Kv, Kth, Kthd = [g[..., i][..., None] for i in range(7)]
    x, xd, th, thd = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    E = 0.5 * (4 / 3) * NOM["l"] ** 2 * thd ** 2 + T0.G * NOM["l"] * (np.cos(th) - 1)
    swing = kE * E * thd * np.cos(th) - kx * x - kv * xd
    thw = (th + np.pi) % (2 * np.pi) - np.pi
    catch = Kx * x + Kv * xd + Kth * thw + Kthd * thd
    return np.clip(np.where(np.abs(thw) < 0.4, catch, swing), -U_MAX, U_MAX)


def evaluate(f, gains, s0, T=500):
    """gains (P,5), s0 (N,4) -> mean return (P,), success (P,)"""
    P, N = gains.shape[0], s0.shape[0]
    s = np.broadcast_to(s0, (P, N, 4)).copy()
    ret = np.zeros((P, N)); alive = np.ones((P, N), bool); ok = np.ones((P, N), bool)
    for t in range(T):
        u = controller(s, gains)
        s = f(s.reshape(-1, 4), u.reshape(-1)).reshape(P, N, 4)
        s = np.nan_to_num(s, nan=10.0, posinf=10.0, neginf=-10.0)
        thw = (s[..., 2] + np.pi) % (2 * np.pi) - np.pi
        r = 0.5 * (1 + np.cos(thw)) - 0.1 * s[..., 0] ** 2 - 1e-3 * u ** 2
        alive &= np.abs(s[..., 0]) < 3.0
        ret += np.where(alive, r, -1.0)
        if t >= T - 100:
            ok &= (np.abs(thw) < 0.2) & (np.abs(s[..., 0]) < 1.5) & alive
    return ret.mean(1) / T, ok.mean(1)


def cem(f, s0, seed, iters=25, pop=64, elite=8):
    rng = np.random.default_rng(seed)
    mu = G_INIT.copy(); sd = 0.3 * np.abs(mu) + 0.5
    best = (-np.inf, mu)
    for _ in range(iters):
        G = mu + sd * rng.normal(size=(pop, mu.size))
        G[0] = mu
        J, _ = evaluate(f, G, s0)
        e = G[np.argsort(-J)[:elite]]
        if J.max() > best[0]:
            best = (J.max(), G[J.argmax()])
        mu, sd = e.mean(0), e.std(0) + 0.05
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--H", type=int, default=25)
    ap.add_argument("--h-iters", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    rng = np.random.default_rng(123)
    s0_train = np.stack([rng.uniform(-0.3, 0.3, 16), np.zeros(16), rng.uniform(np.pi - 0.3, np.pi + 0.3, 16),
                         rng.normal(0, 0.3, 16)], -1)
    s0_test = np.stack([rng.uniform(-0.3, 0.3, 32), np.zeros(32), rng.uniform(np.pi - 0.3, np.pi + 0.3, 32),
                        rng.normal(0, 0.3, 32)], -1)
    kinds = ["nominal", "delta1", "deltaH", "state", "oracle"]
    res = {}
    for gname, p in GAPS.items():
        rows = {k: dict(real_return=[], real_success=[], replay_pi1=[]) for k in kinds}
        for sd in range(args.seeds):
            S0, U0 = T0.collect(p, T0.PI0, 64, 400, seed=10 * sd + 1)
            S1, U1 = T0.collect(p, T0.PI1, 32, 400, seed=10 * sd + 3)
            w1 = fit_delta_1step(S0, U0)
            models = dict(nominal=None, delta1=w1, deltaH=fit_delta_H(S0, U0, args.H, w1, args.h_iters, sd),
                          state=fit_state(S0, U0), oracle=None)
            f_real = sim_step("oracle", None, p)
            base = replay_err(S1, U1, args.H, sim_step("nominal", None, p))
            for k in kinds:
                f = sim_step(k, models[k], p)
                g = cem(f, s0_train, seed=sd)
                J, ok = evaluate(f_real, g[None], s0_test)
                rows[k]["real_return"].append(float(J[0]))
                rows[k]["real_success"].append(float(ok[0]))
                rows[k]["replay_pi1"].append(float(replay_err(S1, U1, args.H, f) / base))
            print(gname, "seed", sd, {k: (round(rows[k]["real_return"][-1], 3), rows[k]["real_success"][-1])
                                      for k in kinds}, flush=True)
        res[gname] = {k: dict(real_return=round(float(np.mean(v["real_return"])), 3),
                              real_success=round(float(np.mean(v["real_success"])), 3),
                              replay_err_new_policy=round(float(np.mean(v["replay_pi1"])), 3))
                      for k, v in rows.items()}
        print(gname, json.dumps(res[gname]), flush=True)
    res["note"] = ("controller (7 gains) optimised by CEM inside each simulator, then deployed on the real system; "
                   f"mean over {args.seeds} data seeds and 32 test initial states. real_return is per-step reward; "
                   "replay_err_new_policy is H-step replay error on pi1 data relative to the plain sim.")
    with open(os.path.join(args.out, "idea7_transfer.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
