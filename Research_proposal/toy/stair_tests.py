"""
Idea 8 / perceptive stairs: a test suite for "which world model learns a calibrated toe-clearance margin?"

Clean swing model (no progress confound): the swing foot rises to apex a (quarter sine, 0.25 s), HOLDS that
height while moving forward at 1 m/s, passes the riser at x = 0.4, stops at x = 0.6 and descends onto the tread.
  clearance at the riser = a - H (exactly);  trip  <=>  a < H  (foot stuck at x = 0.4)
  J = x_final - LAM * a^2   ->   E[J] = 0.4 + 0.2 * P(clear) - LAM * E[a^2]   (linear in P(clear))
The robot perceives  H^ = H + N(0, sigma_p^2);  perceptive policy  a = b + k * H^.
First-order optimum: 0.2 * phi(m / sigma_p) / sigma_p = 2 * LAM * a  ->  margin m* ~ 0, 2.4, 4.2, 6.6 cm for
sigma_p = 0, 1, 2, 4 cm (with a ~ 0.2 m, LAM = 1).

Methods
  oracle grid        brute-force best theta under the true noisy objective (reference)
  ZO true-sim        model-free antithetic zeroth-order gradient on theta (unbiased, includes boundary terms)
  pathwise true-sim  backprop through the true simulator with privileged H (misses the trip boundary term)
  WM-true            backprop through a world model fed the TRUE riser height (privileged)
  WM-perceived       backprop through a world model fed the PERCEIVED height (learns E[f | H^])
  ARG-WM-true        argument-noise world model (state smoothing), annealed to 0, true height
Tests
  T1 perception-noise sweep (calibration of the learned margin)
  T2 deployment noise != training noise
  T3 out-of-distribution risers (20-26 cm)
  T4 systematic perception bias (-1.5 cm)
  T5 escape from a tripping initialisation
  T6 mechanism: model-implied P(clear | clearance) vs the true Bayesian posterior
"""
import argparse
import json
import os
import time

import autograd.numpy as anp
import numpy as np
from autograd import grad

DT, T, VX, T_APEX, XE, XT, DESC, LAM = 0.02, 40, 1.0, 0.25, 0.4, 0.6, -2.0, 1.0
H_LO, H_HI = 0.06, 0.20
SIGMAS = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05])


def command(a, s, t):
    if t * DT < T_APEX:
        return a * (np.pi / (2 * T_APEX)) * np.cos(np.pi * t * DT / (2 * T_APEX))
    return anp.where(s[..., 0] >= XT - 1e-9, DESC + 0.0 * a, 0.0 * a)


def true_step(s, u, H):
    x, y, vx = s[..., 0], s[..., 1], s[..., 2]
    xn, yn = x + vx * DT, y + u * DT
    trip = (x < XE) & (xn >= XE) & (yn < H)
    xn = anp.where(trip, XE - 1e-3 + 0.0 * xn, xn)
    vxn = anp.where(trip, 0.0 * vx, vx)
    at_target = xn >= XT - 1e-9
    xn = anp.where(at_target, XT + 0.0 * xn, xn)
    vxn = anp.where(at_target, 0.0 * vxn, vxn)
    h = anp.where(xn >= XE, H, 0.0 * H)
    yn = anp.where(yn < h, h, yn)
    return anp.stack([xn, yn, vxn], -1)


def rollout(step_fn, theta, H_in, H_obs, expl=None):
    a = theta[0] + theta[1] * H_obs
    s = anp.stack([0.0 * a, 0.0 * a, VX + 0.0 * a], -1)
    for t in range(T):
        u = command(a, s, t)
        if expl is not None:
            u = u + expl[:, t]
        s = step_fn(s, u, H_in)
    return s[..., 0] - LAM * a ** 2, s, a


# ---------------- world models ----------------
MU = np.array([0.35, 0.12, 0.6, 0.0, 0.13]); SD = np.array([0.2, 0.08, 0.5, 0.8, 0.05])
DS = np.array([0.02, 0.04, 1.0])


def init_mlp(sizes, rng):
    return [(rng.normal(0, np.sqrt(1.0 / a), (a, b)), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]


def wm(params, s, u, Hin, sigma):
    one = 0.0 * s[..., :1]
    z = (anp.concatenate([s, u[..., None] + one, Hin[..., None] + one], -1) - MU) / SD
    h = anp.concatenate([z, sigma * 40.0 + one], -1)
    for i, (W, bb) in enumerate(params):
        h = anp.dot(h, W) + bb
        if i < len(params) - 1:
            h = anp.tanh(h)
    return s + h * DS


def collect(n_ep, sig_p, rng):
    """Exploratory data: random apex, noisy vertical commands, random risers; one perception per episode."""
    H = rng.uniform(0.04, 0.24, n_ep); Hp = H + rng.normal(0, sig_p, n_ep) if sig_p > 0 else H.copy()
    a = rng.uniform(0.0, 0.32, n_ep)
    expl = rng.normal(0, 0.25, (n_ep, T))
    S, U = [], []
    s = np.stack([np.zeros(n_ep), np.zeros(n_ep), np.full(n_ep, VX)], -1)
    for t in range(T):
        u = np.asarray(command(a, s, t)) + expl[:, t]
        S.append(s); U.append(u)
        s = np.asarray(true_step(s, u, H))
    S = np.stack(S, 1).reshape(-1, 3); U = np.stack(U, 1).reshape(-1)
    return S, U, np.repeat(H, T), np.repeat(Hp, T)


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
            pert = s.copy(); pert[:, 1:2] += rng.normal(size=(batch, 1)) * sig
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


# ---------------- optimisation ----------------
def terrains(rng, n, sig_p, lo=H_LO, hi=H_HI, bias=0.0):
    H = rng.uniform(lo, hi, n)
    return H, H + bias + (rng.normal(0, sig_p, n) if sig_p > 0 else 0.0)


def optimise(method, model, theta0, sig_p, rng, iters=300, lr=0.004, batch=128):
    th = np.array(theta0, float); m = np.zeros(2); v = np.zeros(2); sims = 0
    for it in range(1, iters + 1):
        H, Hp = terrains(rng, batch, sig_p)
        if method == "zo":
            d = rng.normal(0, 1, (8, 2)) * np.array([0.01, 0.05])
            Jp = [float(np.mean(rollout(true_step, th + dd, H, Hp)[0])) for dd in d]
            Jm = [float(np.mean(rollout(true_step, th - dd, H, Hp)[0])) for dd in d]
            g = np.mean([(jp - jm) / 2 * dd / np.array([0.01, 0.05]) ** 2 for jp, jm, dd in zip(Jp, Jm, d)], 0)
            sims += 16 * batch
        elif method == "pathwise":
            g = grad(lambda t_: anp.mean(rollout(true_step, t_, H, Hp)[0]))(th); sims += batch
        else:
            p, terr, ann = model
            sig = 0.03 * (1 - it / iters) ** 2 if ann else 0.0
            hin = H if terr == "true" else Hp
            g = grad(lambda t_: anp.mean(rollout(lambda s, u, _: wm(p, s, u, hin, sig), t_, hin, Hp)[0]))(th)
        g = np.clip(g, -5, 5)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        th = th + lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return th, sims


def evaluate(th, sig_p, n=8000, seed=7, lo=H_LO, hi=H_HI, bias=0.0):
    H, Hp = terrains(np.random.default_rng(seed), n, sig_p, lo, hi, bias)
    J, s, a = rollout(true_step, th, H, Hp)
    trip = np.asarray(s[..., 0]) < XE
    return dict(J=round(float(np.mean(J)), 4), trip=round(float(trip.mean()), 4),
                margin_cm=round(100 * float(np.mean(np.asarray(a) - H)), 2))


def oracle(sig_p):
    best, bt = -1e9, None
    for b in np.arange(-0.03, 0.12, 0.003):
        for k in np.arange(0.85, 1.16, 0.03):
            J = evaluate(np.array([b, k]), sig_p, n=3000, seed=11)["J"]
            if J > best:
                best, bt = J, np.array([b, k])
    return bt


def posterior_clear(c, Hp, sig_p):
    """True P(H < y | H^) for y = H^ + c, H ~ U(lo, hi), Gaussian perception noise."""
    Hg = np.linspace(0.04, 0.24, 4001)
    prior = ((Hg >= H_LO) & (Hg <= H_HI)).astype(float)
    w = prior * np.exp(-0.5 * ((Hp - Hg) / sig_p) ** 2)
    return np.array([(w * (Hg < Hp + cc)).sum() / w.sum() for cc in c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    t0 = time.time()
    res = {}
    rng = np.random.default_rng(0)
    D0 = collect(1000, 0.0, rng)
    wm_true = (train(D0, "true", False, np.random.default_rng(1)), "true", False)
    sig_list = (0.0, 0.01, 0.02, 0.04)
    wm_perc = {}
    for sp in sig_list:
        D = collect(1000, sp, np.random.default_rng(int(100 * sp) + 5))
        wm_perc[sp] = (train(D, "perceived", False, np.random.default_rng(2)), "perceived", False)
    D2 = collect(1000, 0.02, np.random.default_rng(7))
    arg_true = (train(D2, "true", True, np.random.default_rng(3)), "true", True)
    print(f"models trained {time.time() - t0:.0f}s", flush=True)

    # ---- T6 mechanism: model-implied P(clear) at the riser vs Bayesian posterior (sigma_p = 2 cm)
    c = np.linspace(-0.06, 0.06, 25); Hp0 = 0.13
    s_edge = np.stack([np.full(25, XE - 0.005), Hp0 + c, np.full(25, VX)], -1)
    p_true = np.asarray(wm(wm_true[0], s_edge, np.zeros(25), np.full(25, Hp0), 0.0))[:, 2] / VX
    p_perc = np.asarray(wm(wm_perc[0.02][0], s_edge, np.zeros(25), np.full(25, Hp0), 0.0))[:, 2] / VX
    res["T6_mechanism"] = dict(clearance_cm=np.round(100 * c, 1).tolist(),
                               posterior_P_clear=np.round(posterior_clear(c, Hp0, 0.02), 3).tolist(),
                               WM_perceived_P_clear=np.round(p_perc, 3).tolist(),
                               WM_true_P_clear_given_true_H=np.round(p_true, 3).tolist(),
                               step_function=[float(cc > 0) for cc in c])
    print("T6", json.dumps(res["T6_mechanism"]), flush=True)

    def run(method, model, th0, sp, seeds):
        out = []
        for sd in range(seeds):
            th, sims = optimise(method, model, th0, sp, np.random.default_rng(20 + sd))
            out.append((th, sims))
        return out

    def summarise(runs, sp, **ev):
        e = [evaluate(th, sp, **ev) for th, _ in runs]
        return dict(J=round(float(np.mean([x["J"] for x in e])), 4),
                    trip=round(float(np.mean([x["trip"] for x in e])), 4),
                    margin_cm=round(float(np.mean([x["margin_cm"] for x in e])), 2),
                    thetas=[[round(float(v), 3) for v in th] for th, _ in runs])

    good0, bad0 = (0.05, 1.0), (-0.03, 1.0)
    # ---- T1 calibration sweep
    res["T1_noise_sweep"] = {}
    trained_policies = {}
    for sp in sig_list:
        row = {}
        orc = oracle(sp)
        row["oracle"] = dict(theta=[round(float(x), 3) for x in orc], **evaluate(orc, sp))
        for name, method, model in [("ZO true-sim", "zo", None), ("pathwise true-sim", "pathwise", None),
                                    ("WM-true", "wm", wm_true), ("WM-perceived", "wm", wm_perc[sp])]:
            runs = run(method, model, good0, sp, args.seeds)
            row[name] = summarise(runs, sp)
            row[name]["sim_rollouts_used"] = int(runs[0][1])
            trained_policies[(sp, name)] = runs
        if sp == 0.02:
            runs = run("wm", arg_true, good0, sp, args.seeds)
            row["ARG-WM-true annealed"] = summarise(runs, sp)
            trained_policies[(sp, "ARG-WM-true annealed")] = runs
            trained_policies[(sp, "oracle")] = [(orc, 0)]
        res["T1_noise_sweep"][f"sigma_p={sp}"] = row
        print(f"T1 sigma_p={sp}", json.dumps(row), f"{time.time() - t0:.0f}s", flush=True)

    # ---- T2/T3/T4 on policies trained at sigma_p = 2 cm
    names = ["oracle", "ZO true-sim", "pathwise true-sim", "WM-true", "WM-perceived", "ARG-WM-true annealed"]
    res["T2_deploy_noise_mismatch"] = {n: {f"deploy_sigma={d}": summarise(trained_policies[(0.02, n)], d)["trip"]
                                           for d in (0.01, 0.02, 0.04)} for n in names}
    res["T3_ood_risers_20_26cm"] = {n: summarise(trained_policies[(0.02, n)], 0.02, lo=0.20, hi=0.26)
                                    for n in names}
    res["T4_perception_bias_minus_1.5cm"] = {n: summarise(trained_policies[(0.02, n)], 0.02, bias=-0.015)
                                             for n in names}
    for key in ("T2_deploy_noise_mismatch", "T3_ood_risers_20_26cm", "T4_perception_bias_minus_1.5cm"):
        print(key, json.dumps(res[key]), flush=True)

    # ---- T5 escape from a tripping initialisation (sigma_p = 2 cm)
    res["T5_tripping_init"] = {}
    for name, method, model in [("ZO true-sim", "zo", None), ("pathwise true-sim", "pathwise", None),
                                ("WM-true", "wm", wm_true), ("WM-perceived", "wm", wm_perc[0.02]),
                                ("ARG-WM-true annealed", "wm", arg_true)]:
        res["T5_tripping_init"][name] = summarise(run(method, model, bad0, 0.02, args.seeds), 0.02)
    res["T5_tripping_init"]["init_policy"] = evaluate(np.array(bad0), 0.02)
    print("T5", json.dumps(res["T5_tripping_init"]), f"{time.time() - t0:.0f}s", flush=True)

    res["note"] = ("E[J] = 0.4 + 0.2 P(clear) - a^2; first-order optimal margin ~0 / 2.4 / 4.2 / 6.6 cm for "
                   "sigma_p = 0 / 1 / 2 / 4 cm. 'trip' = fraction of steps where the toe hits the riser. "
                   "ZO uses 16x more simulator rollouts per iteration than the pathwise methods.")
    with open(os.path.join(args.out, "idea8_stair_tests.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
