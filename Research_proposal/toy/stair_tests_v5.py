"""
Stair tests v5: the SMOOTHING BUDGET rule (fix for the over-smoothing found in v4 at sigma_p = 4 cm).

Total smoothing seen by the optimiser = perception posterior width (sigma_p) (+) optimisation (argument) noise.
Add only the missing part:   sigma_ARG,max = sqrt(max(0, sigma_escape^2 - sigma_p^2)),  sigma_escape = 4 cm,
held for 30 % of iterations, cosine-decayed to 0 at 90 %.  Same data, model seeds and initialisations as v4,
so D' is directly comparable with v4's configurations C and D.
"""
import argparse
import json
import os
import time

import numpy as np

import stair_tests as ST
import stair_tests_v2 as V
import stair_tests_v4 as F

SIG_ESCAPE = 0.04


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sp", type=float, required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    t0, sp = time.time(), args.sp
    smax = float(np.sqrt(max(0.0, SIG_ESCAPE ** 2 - sp ** 2)))
    good0, bad0 = (0.05, 1.0), (-0.03, 1.0)
    acc = dict(good=[], bad=[])
    for sd in range(args.seeds):
        D = F.collect_unconfounded(1000, sp, np.random.default_rng(1000 * sd + int(1000 * sp) + 7))
        ef_arg = F.train_ef(D, "perceived", True, np.random.default_rng(70 + sd))      # identical to v4 model D
        for th0, key in ((good0, "good"), (bad0, "bad")):
            th = F.optimise_ef(ef_arg, "perceived", th0, sp, np.random.default_rng(20 + sd), anneal=smax > 0,
                               branched=True, smax=smax, hold=0.3)
            acc[key].append(V.evaluate(th, sp))
        print(f"sigma_p={sp} seed {sd}", acc["good"][-1], acc["bad"][-1], f"{time.time() - t0:.0f}s", flush=True)
    res = {"D' smoothing budget": dict(sigma_ARG_max_cm=round(100 * smax, 2),
                                       from_clearing_init=F.agg(acc["good"]), from_tripping_init=F.agg(acc["bad"]))}
    print(json.dumps(res), flush=True)
    with open(os.path.join(args.out, f"idea8_stair_tests_v5_sp{sp}.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
