"""
Idea 8 extension to perceptive stair locomotion: two different reasons to smooth a world model.

A swing foot (state x, y, vx) moves forward at 1 m/s. A stair riser at x = 0.4 has height H (true), but the robot
only perceives  H^ = H + N(0, sigma_p^2)  (heightmap noise, sigma_p = 2 cm). Perceptive policy: swing apex
a = b + k * H^ (parameters theta = (b, k)); the foot follows y = a sin(pi t / Ts). If the foot reaches the riser
below H it TRIPS (stuck at x = 0.4); otherwise it lands on the tread. Objective J = landing x - lam * a^2
(progress minus swing effort). With exact H the optimum is "just clear" (zero margin); under perception noise it
needs a margin.

Hypotheses
  H1 (optimisation smoothing): from a tripping initialisation, the true first-order gradient points the WRONG
      way (only the effort term is visible). An argument-noise world model, annealed, escapes.
  H2 (uncertainty smoothing): pathwise gradients through a model fed the TRUE terrain miss the boundary term of
      the trip discontinuity, so they learn no safety margin even though the policy sees noisy H^. A world model
      fed the PERCEIVED terrain learns E[f | H^] (the posterior over true terrain), turning the boundary term into
      a smooth interior term: it learns a margin calibrated to sigma_p.

World models compared (all MLPs, per-step):
  WM-true      : terrain input = true H                (privileged sim training)
  WM-perceived : terrain input = H^                    (learns E[f | H^])
  ARG-WM-true  : argument-noise (sigma input), true H  (optimisation smoothing only)
  ARG-WM-perc  : argument-noise, terrain input H^      (both)
"""
import argparse
import json
import os

import autograd.numpy as anp
import numpy as np
from autograd import grad

DT, T, VX, TS, XE, LAM, SIG_P = 0.02, 40, 1.0, 0.6, 0.4, 2.0, 0.02
SIGMAS = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05])


def u_cmd(a, t):
    """Vertical velocity command of the swing primitive at step t (apex a)."""
    tt = t * DT
    return a * (np.pi / TS) * np.cos(np.pi * tt / TS) if tt < TS else -1.0 + 0.0 * a


def true_step(s, u, H):
    x, y, vx = s[..., 0], s[..., 1], s[..., 2]
    xn, yn = x + vx * DT, y + u * DT
    trip = (x < XE) & (xn >= XE) & (yn < H)
    xn = anp.where(trip, XE - 1e-3 + 0.0 * xn, xn)
    vxn = anp.where(trip, 0.0 * vx, vx)
    h = anp.where(xn >= XE, H, 0.0 * H)
    land = (yn < h) & (vxn > 0)
    yn = anp.where(yn < h, h, yn)
    vxn = anp.where(land, 0.0 * vxn, vxn)
    return anp.stack([xn, yn, vxn], -1)


def rollout(step_fn, theta, H_dyn, H_obs):
    """Foot rollout; the policy sees H_obs, the dynamics use H_dyn (or a model's terrain input)."""
    b, k = theta[0], theta[1]
    a = b + k * H_obs
    s = anp.stack([0.0 * a, 0.0 * a, VX + 0.0 * a], -1)
    for t in range(T):
        s = step_fn(s, u_cmd(a, t), H_dyn)
    return s[..., 0] - LAM * a ** 2, s


# ---------------- world models ----------------
MU = np.array([0.4, 0.08, 0.5, 0.0, 0.13]); SD = np.array([0.25, 0.08, 0.5, 0.8, 0.05])
DS = np.array([0.02, 0.03, 1.0])


def init_mlp(sizes, rng):
    return [(rng.normal(0, np.sqrt(1.0 / a), (a, b)), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]


def wm(params, s, u, Hin, sigma):
    z = (anp.concatenate([s, u[..., None] + 0.0 * s[..., :1], Hin[..., None] + 0.0 * s[..., :1]], -1) - MU) / SD
    h = anp.concatenate([z, sigma * 40.0 + 0.0 * z[..., :1]], -1)
    for i, (W, bb) in enumerate(params):
        h = anp.dot(h, W) + bb
        if i < len(params) - 1:
            h = anp.tanh(h)
    return s + h * DS


def collect(n_ep, rng):
    S, U, H, Hp = [], [], [], []
    for _ in range(n_ep):
        Hv = rng.uniform(0.05, 0.22); a = rng.uniform(0.0, 0.35)
        s = np.array([0.0, 0.0, VX])
        for t in range(T):
            u = float(u_cmd(a, t)) + rng.normal(0, 0.3)
            S.append(s); U.append(u); H.append(Hv)
            s = np.asarray(true_step(s, u, Hv))
        Hp += [Hv + rng.normal(0, SIG_P)] * T          # one perception per episode
    return np.array(S), np.array(U), np.array(H), np.array(Hp)


def train(D, terrain, arg_noise, rng, steps=3000, batch=512, lr=3e-3):
    S, U, H, Hp = D
    Hin_all = H if terrain == "true" else Hp
    params = init_mlp([6, 96, 96, 3], rng)
    flat = [p for pair in params for p in pair]
    m = [np.zeros_like(p) for p in flat]; v = [np.zeros_like(p) for p in flat]

    def loss(fp, s, u, hin, sig, y):
        ps = [(fp[2 * i], fp[2 * i + 1]) for i in range(len(fp) // 2)]
        return anp.mean(anp.sum(((wm(ps, s, u, hin, sig) - y) / DS) ** 2, -1))

    gl = grad(loss)
    for it in range(1, steps + 1):
        i = rng.integers(0, len(S), batch)
        s, u, hin = S[i], U[i], Hin_all[i]
        if arg_noise:
            sig = rng.choice(SIGMAS, size=(batch, 1))
            pert = s.copy(); pert[:, :2] += rng.normal(size=(batch, 2)) * sig
            y = np.asarray(true_step(pert, u, H[i]))
        else:
            sig = np.zeros((batch, 1))
            y = np.asarray(true_step(s, u, H[i]))
        g = gl(flat, s, u, hin, sig, y)
        a_lr = lr if it < 0.75 * steps else lr * 0.2
        for j in range(len(flat)):
            m[j] = 0.9 * m[j] + 0.1 * g[j]; v[j] = 0.999 * v[j] + 0.001 * g[j] ** 2
            flat[j] = flat[j] - a_lr * (m[j] / (1 - 0.9 ** it)) / (np.sqrt(v[j] / (1 - 0.999 ** it)) + 1e-8)
    return [(flat[2 * i], flat[2 * i + 1]) for i in range(len(flat) // 2)]


# ---------------- policy optimisation ----------------
def optimise(kind, model, theta0, rng, iters=250, lr=0.01, anneal=False):
    th = np.array(theta0, float); m = np.zeros(2); v = np.zeros(2)
    for it in range(1, iters + 1):
        H = rng.uniform(0.06, 0.20, 64); Hp = H + rng.normal(0, SIG_P, 64)
        sig = 0.05 * (1 - it / iters) ** 2 if anneal else 0.0
        if kind == "true":
            f = lambda th_: anp.mean(rollout(true_step, th_, H, Hp)[0])
        else:
            hin = H if model[1] == "true" else Hp
            f = (lambda p, hin, sig: lambda th_: anp.mean(
                rollout(lambda s, u, _: wm(p, s, u, hin, sig), th_, hin, Hp)[0]))(model[0], hin, sig)
        g = np.clip(grad(f)(th), -5, 5)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        th = th + lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return th


def evaluate(th, rng, sig_p=SIG_P, n=4000):
    H = rng.uniform(0.06, 0.20, n); Hp = H + rng.normal(0, sig_p, n)
    J, s = rollout(true_step, th, H, Hp)
    trip = np.abs(np.asarray(s[..., 0]) - (XE - 1e-3)) < 1e-6
    a = th[0] + th[1] * Hp
    margin = float(np.mean(0.866 * a - H))          # swing height at the riser minus riser height
    return dict(J=round(float(np.mean(J)), 4), trip_rate=round(float(trip.mean()), 3), margin_cm=round(100 * margin, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    D = collect(800, rng)
    models = {
        "WM-true": (train(D, "true", False, np.random.default_rng(1)), "true"),
        "WM-perceived": (train(D, "perceived", False, np.random.default_rng(2)), "perceived"),
        "ARG-WM-true": (train(D, "true", True, np.random.default_rng(3)), "true"),
        "ARG-WM-perceived": (train(D, "perceived", True, np.random.default_rng(4)), "perceived"),
    }
    print("trained", flush=True)
    # brute-force reference: best theta on a grid under the TRUE noisy-perception objective
    grid_b = np.linspace(-0.05, 0.12, 35); grid_k = np.linspace(0.6, 1.8, 41)
    er = np.random.default_rng(99)
    best = max(((b, k) for b in grid_b for k in grid_k),
               key=lambda th: evaluate(np.array(th), np.random.default_rng(99), n=1500)["J"])
    res = {"reference_grid_optimum": dict(theta=[round(float(x), 3) for x in best],
                                          **evaluate(np.array(best), np.random.default_rng(7)))}
    print("reference", res["reference_grid_optimum"], flush=True)
    inits = {"tripping init (b=0, k=0.8)": (0.0, 0.8), "clearing init (b=0.05, k=1.3)": (0.05, 1.3)}
    methods = [("true-sim pathwise (privileged H)", "true", None, False),
               ("WM-true", "wm", "WM-true", False),
               ("WM-perceived", "wm", "WM-perceived", False),
               ("ARG-WM-true annealed", "wm", "ARG-WM-true", True),
               ("ARG-WM-perceived annealed", "wm", "ARG-WM-perceived", True)]
    for iname, th0 in inits.items():
        res[iname] = {}
        for name, kind, mkey, ann in methods:
            evs = []
            for sd in range(args.seeds):
                th = optimise(kind, models.get(mkey), th0, np.random.default_rng(10 + sd), anneal=ann)
                e = evaluate(th, np.random.default_rng(7)); e2 = evaluate(th, np.random.default_rng(7), sig_p=0.04)
                evs.append(dict(theta=[round(float(x), 3) for x in th], **e, trip_rate_at_4cm_noise=e2["trip_rate"]))
            res[iname][name] = dict(
                mean_J=round(float(np.mean([e["J"] for e in evs])), 4),
                trip_rate=round(float(np.mean([e["trip_rate"] for e in evs])), 3),
                trip_rate_at_4cm_noise=round(float(np.mean([e["trip_rate_at_4cm_noise"] for e in evs])), 3),
                margin_cm=round(float(np.mean([e["margin_cm"] for e in evs])), 2),
                thetas=[e["theta"] for e in evs])
            print(iname, name, res[iname][name], flush=True)
    res["note"] = ("J = landing x - 2 a^2; trip = foot stuck at the riser; margin = swing height at the riser minus "
                   "true riser height (mean); perception noise sigma_p = 2 cm unless stated")
    with open(os.path.join(args.out, "idea8_stairs.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
