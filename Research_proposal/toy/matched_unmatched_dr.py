"""
Idea 7, Proposition 4 test: is domain randomization (DR) enough for matched gaps but not for unmatched gaps?

Robust-control intuition: matched uncertainty enters through the input channel, so feedback can dominate it;
unmatched uncertainty cannot be cancelled by the input directly. Prediction for RL-with-DR:
  DR over matched parameters    -> a controller that works on the real system, with a small nominal cost
  DR over unmatched parameters  -> lower real success and/or a larger "price of robustness"

Controllers (7 gains, as in matched_unmatched_transfer.py) are optimised by CEM on a batch of plants sampled
from the DR range, then evaluated on the real plant and on the nominal plant.
"""
import argparse
import json
import os

import numpy as np

import matched_unmatched_delta_action as T0
import matched_unmatched_transfer as TR

DR = {
    "matched": dict(k=(0.4, 1.1), bc=(0.0, 4.0)),
    "unmatched": dict(bp=(0.0, 0.05), mp=(0.08, 0.16)),
}


def plants(rng, ranges, n):
    p = {k: np.full(n, v, float) for k, v in T0.NOM.items()}
    for k, (lo, hi) in ranges.items():
        p[k] = rng.uniform(lo, hi, n)
    return p


def make_f(p_batch, P):
    tiled = {k: np.tile(v, P) for k, v in p_batch.items()}
    return lambda s, u: T0.step(s, u, tiled)


def cem_dr(ranges, s0, seed, n_plants=16, iters=25, pop=64, elite=8):
    rng = np.random.default_rng(seed)
    mu = TR.G_INIT.copy(); sd = 0.3 * np.abs(mu) + 0.5
    best = (-np.inf, mu)
    for _ in range(iters):
        pb = plants(rng, ranges, n_plants)
        pb = {k: np.repeat(v, s0.shape[0]) for k, v in pb.items()}          # plant x initial state
        S0 = np.tile(s0, (n_plants, 1))
        G = mu + sd * rng.normal(size=(pop, mu.size)); G[0] = mu
        J, _ = TR.evaluate(make_f(pb, pop), G, S0)
        e = G[np.argsort(-J)[:elite]]
        if J.max() > best[0]:
            best = (J.max(), G[J.argmax()])
        mu, sd = e.mean(0), e.std(0) + 0.05
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    s0 = np.stack([rng.uniform(-0.3, 0.3, 4), np.zeros(4), rng.uniform(np.pi - 0.3, np.pi + 0.3, 4),
                   rng.normal(0, 0.3, 4)], -1)
    s0_test = np.stack([rng.uniform(-0.3, 0.3, 32), np.zeros(32), rng.uniform(np.pi - 0.3, np.pi + 0.3, 32),
                        rng.normal(0, 0.3, 32)], -1)
    f_nom = TR.sim_step("nominal", None, None)
    res = {}
    for gname in ("matched", "unmatched"):
        p_real = TR.GAPS[gname]
        f_real = TR.sim_step("oracle", None, p_real)
        rows = {k: dict(real_return=[], real_success=[], nominal_return=[]) for k in ("no DR", "DR", "oracle")}
        for sd in range(args.seeds):
            gains = {"no DR": TR.cem(f_nom, np.tile(s0, (4, 1)), seed=sd),
                     "DR": cem_dr(DR[gname], s0, seed=sd),
                     "oracle": TR.cem(f_real, np.tile(s0, (4, 1)), seed=sd)}
            for k, g in gains.items():
                Jr, okr = TR.evaluate(f_real, g[None], s0_test)
                Jn, _ = TR.evaluate(f_nom, g[None], s0_test)
                rows[k]["real_return"].append(float(Jr[0])); rows[k]["real_success"].append(float(okr[0]))
                rows[k]["nominal_return"].append(float(Jn[0]))
        res[gname] = {k: {m: round(float(np.mean(v)), 3) for m, v in r.items()} for k, r in rows.items()}
        res[gname]["price_of_robustness(nominal_noDR - nominal_DR)"] = round(
            res[gname]["no DR"]["nominal_return"] - res[gname]["DR"]["nominal_return"], 3)
        print(gname, json.dumps(res[gname]), flush=True)
    res["dr_ranges"] = DR
    with open(os.path.join(args.out, "idea7_dr.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
