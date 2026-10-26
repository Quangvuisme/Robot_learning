"""
Stair tests v3: EVENT-FACTORED world model (the proposed method's model component).

v2 showed that a plain regression MLP learns the trip boundary blurred (implicit width ~4 cm even with zero
perception noise) and shifted in a seed-dependent way, so model error swamps the perception-posterior effect.

Event-factored world model (EF-WM):
  shared trunk(s, u, terrain_input, sigma)  ->  p = P(trip | .)        (sigmoid head, BCE on simulator event labels)
                                            ->  f_free(.), f_trip(.)   (regression heads, MSE on their own samples)
  prediction  E[s'] = p * f_trip + (1 - p) * f_free                     (used for backprop-through-time)
Why: BCE is a proper scoring rule, so with the PERCEIVED terrain as input the head converges to the calibrated
posterior P(H > y | H^); the event boundary is learned explicitly instead of being blurred by MSE regression.
Optionally trained with ARGUMENT noise (labels and targets from the simulator at perturbed states).

Compared (ensembles of 3 model seeds; sigma_p in {0, 2, 4} cm; targeted data):
  MLP-WM perceived (v2 baseline) | EF-WM true terrain | EF-WM perceived | EF-WM perceived + argument-noise annealing
Also from a TRIPPING initialisation (escape test).
"""
import argparse
import json
import os
import time

import autograd.numpy as anp
import numpy as np
from autograd import grad

import stair_tests as ST
import stair_tests_v2 as V


def trip_label(S, U, H):
    x, y, vx = S[:, 0], S[:, 1], S[:, 2]
    return ((x < ST.XE) & (x + vx * ST.DT >= ST.XE) & (y + U * ST.DT < H)).astype(float)


def init(rng, width=96):
    def lin(a, b):
        return [rng.normal(0, np.sqrt(1.0 / a), (a, b)), np.zeros(b)]
    return lin(6, width) + lin(width, width) + lin(width, 1) + lin(width, 3) + lin(width, 3)


def ef_parts(p, s, u, Hin, sigma):
    one = 0.0 * s[..., :1]
    z = (anp.concatenate([s, u[..., None] + one, Hin[..., None] + one], -1) - ST.MU) / ST.SD
    h = anp.concatenate([z, sigma * 40.0 + one], -1)
    h = anp.tanh(anp.dot(h, p[0]) + p[1])
    h = anp.tanh(anp.dot(h, p[2]) + p[3])
    logit = (anp.dot(h, p[4]) + p[5])[..., 0]
    f_free = s + (anp.dot(h, p[6]) + p[7]) * ST.DS
    f_trip = s + (anp.dot(h, p[8]) + p[9]) * ST.DS
    return logit, f_free, f_trip


def ef_wm(p, s, u, Hin, sigma):
    logit, f_free, f_trip = ef_parts(p, s, u, Hin, sigma)
    pt = 1.0 / (1.0 + anp.exp(-logit))
    return pt[..., None] * f_trip + (1 - pt[..., None]) * f_free


def train_ef(D, terrain, arg_noise, rng, steps=3000, batch=512, lr=3e-3):
    S, U, H, Hp = D
    Hin_all = H if terrain == "true" else Hp
    p = init(rng)
    m = [np.zeros_like(q) for q in p]; v = [np.zeros_like(q) for q in p]

    def loss(p_, s, u, hin, sig, y, lab):
        logit, ff, ft = ef_parts(p_, s, u, hin, sig)
        bce = anp.mean(anp.logaddexp(0.0, logit) - lab * logit)
        mse_f = anp.sum((1 - lab)[:, None] * ((ff - y) / ST.DS) ** 2) / (anp.sum(1 - lab) + 1.0)
        mse_t = anp.sum(lab[:, None] * ((ft - y) / ST.DS) ** 2) / (anp.sum(lab) + 1.0)
        return 5.0 * bce + mse_f + mse_t

    gl = grad(loss)
    for it in range(1, steps + 1):
        i = rng.integers(0, len(S), batch)
        s, u, hin, Hi = S[i], U[i], Hin_all[i], H[i]
        if arg_noise:
            sig = rng.choice(ST.SIGMAS, size=(batch, 1))
            pert = s.copy(); pert[:, 1:2] += rng.normal(size=(batch, 1)) * sig
        else:
            sig = np.zeros((batch, 1)); pert = s
        y = np.asarray(ST.true_step(pert, u, Hi)); lab = trip_label(pert, u, Hi)
        g = gl(p, s, u, hin, sig, y, lab)
        a_lr = lr if it < 0.75 * steps else lr * 0.2
        for j in range(len(p)):
            m[j] = 0.9 * m[j] + 0.1 * g[j]; v[j] = 0.999 * v[j] + 0.001 * g[j] ** 2
            p[j] = p[j] - a_lr * (m[j] / (1 - 0.9 ** it)) / (np.sqrt(v[j] / (1 - 0.999 ** it)) + 1e-8)
    return p


def optimise_ef(p, terrain, theta0, sig_p, rng, anneal=False, iters=300, lr=0.004, batch=128):
    th = np.array(theta0, float); m = np.zeros(2); v = np.zeros(2)
    for it in range(1, iters + 1):
        H, Hp = ST.terrains(rng, batch, sig_p)
        sig = 0.03 * (1 - it / iters) ** 2 if anneal else 0.0
        hin = H if terrain == "true" else Hp
        g = grad(lambda t_: anp.mean(ST.rollout(lambda s, u, _: ef_wm(p, s, u, hin, sig), t_, hin, Hp)[0]))(th)
        g = np.clip(g, -5, 5)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        th = th + lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return th


def mechanism_ef(p, sig_p, Hp0=0.13):
    c = np.linspace(-0.08, 0.08, 161)
    s = np.stack([np.full(c.size, ST.XE - ST.VX * ST.DT), Hp0 + c, np.full(c.size, ST.VX)], -1)
    logit, _, _ = ef_parts(p, s, np.zeros(c.size), np.full(c.size, Hp0), 0.0)
    pc = np.maximum.accumulate(1 - 1 / (1 + np.exp(-np.asarray(logit))))   # P(clear), monotone envelope

    def cross(q):
        return float(c[min(np.searchsorted(pc, q), c.size - 1)])
    return dict(shift_cm=round(100 * cross(0.5), 2), width_cm=round(100 * (cross(0.9) - cross(0.1)), 2))


def agg(evs):
    return {k: [round(float(np.mean([e[k] for e in evs])), 4), round(float(np.std([e[k] for e in evs])), 4)]
            for k in ("J", "trip", "clearance_cm", "staircase5_success")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sp", type=float, default=None, help="run a single perception-noise level")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    t0 = time.time()
    res = dict(C_HOLD=round(V.C_HOLD, 4))
    good0, bad0 = (0.05, 1.0), (-0.03, 1.0)
    for sp in ((0.0, 0.02, 0.04) if args.sp is None else (args.sp,)):
        row = {}
        orc = ST.oracle(sp)
        row["oracle"] = dict(theta=[round(float(x), 3) for x in orc], **V.evaluate(orc, sp))
        ip = [V.optimise_ideal_posterior(sp, good0, np.random.default_rng(30 + k)) for k in range(args.seeds)]
        row["ideal posterior model"] = agg([V.evaluate(t, sp) for t in ip])
        for name, terr in (("MLP-WM true terrain (regression baseline)", "true"),
                           ("MLP-WM perceived (regression baseline)", "perceived")):
            evs, evs_bad, mechs = [], [], []
            for sd in range(args.seeds):
                D = V.collect_targeted(1000, sp, np.random.default_rng(1000 * sd + int(1000 * sp) + 7))
                p = ST.train(D, terr, False, np.random.default_rng(50 + sd))
                evs.append(V.evaluate(ST.optimise("wm", (p, terr, False), good0, sp, np.random.default_rng(20 + sd))[0], sp))
                evs_bad.append(V.evaluate(ST.optimise("wm", (p, terr, False), bad0, sp, np.random.default_rng(20 + sd))[0], sp))
                mechs.append(V.mechanism(p, sp))
            row[name] = dict(from_clearing_init=agg(evs), from_tripping_init=agg(evs_bad),
                             mechanism_shift_cm=[m_["shift_cm"] for m_ in mechs],
                             mechanism_width_cm=[m_["width_cm"] for m_ in mechs])
            print(f"sigma_p={sp}", name, json.dumps(row[name]), f"{time.time() - t0:.0f}s", flush=True)
        configs = [("EF-WM true terrain", "true", False, False),
                   ("EF-WM perceived", "perceived", False, False),
                   ("EF-WM perceived + ARG annealed", "perceived", True, True)]
        for name, terr, argn, ann in configs:
            evs, evs_bad, mechs = [], [], []
            for sd in range(args.seeds):
                D = V.collect_targeted(1000, sp, np.random.default_rng(1000 * sd + int(1000 * sp) + 7))
                p = train_ef(D, terr, argn, np.random.default_rng(50 + sd))
                evs.append(V.evaluate(optimise_ef(p, terr, good0, sp, np.random.default_rng(20 + sd), ann), sp))
                evs_bad.append(V.evaluate(optimise_ef(p, terr, bad0, sp, np.random.default_rng(20 + sd), ann), sp))
                mechs.append(mechanism_ef(p, sp))
            row[name] = dict(from_clearing_init=agg(evs), from_tripping_init=agg(evs_bad),
                             mechanism_shift_cm=[m_["shift_cm"] for m_ in mechs],
                             mechanism_width_cm=[m_["width_cm"] for m_ in mechs],
                             posterior_width_cm=round(100 * 2 * 1.2816 * sp, 2))
            print(f"sigma_p={sp}", name, json.dumps(row[name]), f"{time.time() - t0:.0f}s", flush=True)
        res[f"sigma_p={sp}"] = row
        print(f"sigma_p={sp} oracle", row["oracle"], "ideal", row["ideal posterior model"], flush=True)
    res["note"] = ("[mean, std] over 3 world-model seeds; targeted data; clearance = C_HOLD*a - H (cm); "
                   "mechanism shift/width of P(clear) vs posterior (shift 0, width 2.56 sigma_p)")
    suffix = "" if args.sp is None else f"_sp{args.sp}"
    with open(os.path.join(args.out, f"idea8_stair_tests_v3{suffix}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
