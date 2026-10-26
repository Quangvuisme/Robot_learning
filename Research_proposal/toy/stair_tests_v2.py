"""
Stair tests v2: separate WORLD-MODEL ERROR from the PERCEPTION-POSTERIOR effect (Idea 8, Proposition E).

v1 (stair_tests.py) found that the perceived-terrain world model learns a ~5 cm toe-clearance margin even with
NO perception noise, while the true-terrain model (same information at sigma_p = 0) learned a negative margin:
the margin was dominated by seed-dependent model error near the riser, not by the posterior. v2 adds:

  B  ensembles: 3 world-model seeds per condition (mean +- std of the learned clearance)
  C  targeted data: 70 % of episodes swing close to the riser (better coverage of the contact boundary)
  D  mechanism per model: shift (clearance where P(clear) = 0.5) and width (10-90 %) of the model-implied
     P(clear) curve vs the Bayesian posterior (width 2 * 1.2816 * sigma_p)
  E  staircase: success over 5 consecutive steps with independent risers / perception errors
  +  "ideal posterior model": backprop through the exact expected objective (Gaussian posterior), i.e. what a
     perfect E[f | H^] model would teach
Clearance is measured correctly here: hold height = C_HOLD * a (Euler-integrated rise), clearance = C_HOLD*a - H.
"""
import argparse
import json
import os
import time
import autograd.numpy as anp
import numpy as np
from autograd import grad

import stair_tests as ST


def _hold_factor():
    a = np.array([0.1]); s = np.stack([0 * a, 0 * a, ST.VX + 0 * a], -1)
    for t in range(int(0.4 / ST.DT)):
        s = np.asarray(ST.true_step(s, np.asarray(ST.command(a, s, t)), np.full(1, -1.0)))
    return float(s[0, 1] / a[0])


C_HOLD = _hold_factor()


def norm_cdf(z):
    """Smooth normal CDF (tanh approximation, |error| < 1e-3), autograd-friendly; scipy is not available."""
    return 0.5 * (1.0 + anp.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))


def collect_targeted(n_ep, sig_p, rng, frac=0.7):
    H = rng.uniform(0.04, 0.24, n_ep); Hp = H + (rng.normal(0, sig_p, n_ep) if sig_p > 0 else 0.0)
    near = rng.random(n_ep) < frac
    a = np.where(near, np.clip(H / C_HOLD + rng.normal(0, 0.03, n_ep), 0, 0.35), rng.uniform(0.0, 0.32, n_ep))
    expl = rng.normal(0, 0.1, (n_ep, ST.T))
    S, U = [], []
    s = np.stack([np.zeros(n_ep), np.zeros(n_ep), np.full(n_ep, ST.VX)], -1)
    for t in range(ST.T):
        u = np.asarray(ST.command(a, s, t)) + expl[:, t]
        S.append(s); U.append(u)
        s = np.asarray(ST.true_step(s, u, H))
    return (np.stack(S, 1).reshape(-1, 3), np.stack(U, 1).reshape(-1), np.repeat(H, ST.T), np.repeat(Hp, ST.T))


def evaluate(th, sig_p, n=8000, seed=7):
    H, Hp = ST.terrains(np.random.default_rng(seed), n, sig_p)
    J, s, a = ST.rollout(ST.true_step, th, H, Hp)
    trip = np.asarray(s[..., 0]) < ST.XE
    # 5-step staircase: independent risers and perception errors per step
    r = np.random.default_rng(seed + 1)
    H5 = r.uniform(ST.H_LO, ST.H_HI, (n // 4, 5)); Hp5 = H5 + (r.normal(0, sig_p, H5.shape) if sig_p > 0 else 0.0)
    a5 = th[0] + th[1] * Hp5
    stair5 = float(np.mean(np.all(C_HOLD * a5 >= H5, 1)))
    return dict(J=round(float(np.mean(J)), 4), trip=round(float(trip.mean()), 4),
                clearance_cm=round(100 * float(np.mean(C_HOLD * np.asarray(a) - H)), 2),
                staircase5_success=round(stair5, 4))


def mechanism(model_params, sig_p, Hp0=0.13):
    c = np.linspace(-0.08, 0.08, 161)
    s = np.stack([np.full(c.size, ST.XE - ST.VX * ST.DT), Hp0 + c, np.full(c.size, ST.VX)], -1)
    p = np.clip(np.asarray(ST.wm(model_params, s, np.zeros(c.size), np.full(c.size, Hp0), 0.0))[:, 2] / ST.VX, 0, 1)
    p = np.maximum.accumulate(p)                      # monotone envelope for robust crossing points

    def cross(q):
        i = np.searchsorted(p, q)
        return float(c[min(i, c.size - 1)])
    return dict(shift_cm=round(100 * cross(0.5), 2), width_cm=round(100 * (cross(0.9) - cross(0.1)), 2),
                posterior_width_cm=round(100 * 2 * 1.2816 * sig_p, 2))


def optimise_ideal_posterior(sig_p, theta0, rng, iters=300, lr=0.004, batch=256):
    """Backprop through the exact expected objective E[J | H^] = 0.4 + 0.2 P(clear | H^) - LAM a^2."""
    th = np.array(theta0, float); m = np.zeros(2); v = np.zeros(2)
    sp = max(sig_p, 1e-3)
    for it in range(1, iters + 1):
        H, Hp = ST.terrains(rng, batch, sig_p)

        def EJ(t_):
            a = t_[0] + t_[1] * Hp
            return anp.mean(0.4 + 0.2 * norm_cdf((C_HOLD * a - Hp) / sp) - ST.LAM * a ** 2)
        g = np.clip(grad(EJ)(th), -5, 5)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        th = th + lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    t0 = time.time()
    res = dict(C_HOLD=round(C_HOLD, 4))
    good0 = (0.05, 1.0)
    for sp in (0.0, 0.02, 0.04):
        row = {}
        orc = ST.oracle(sp)
        row["oracle"] = dict(theta=[round(float(x), 3) for x in orc], **evaluate(orc, sp))
        ip = [optimise_ideal_posterior(sp, good0, np.random.default_rng(30 + k)) for k in range(args.seeds)]
        row["ideal posterior model"] = dict(**{k: round(float(np.mean([evaluate(t, sp)[k] for t in ip])), 4)
                                               for k in ("J", "trip", "clearance_cm", "staircase5_success")})
        for regime in ("broad", "targeted"):
            for terr in ("true", "perceived"):
                evs, mechs = [], []
                for sd in range(args.seeds):
                    r = np.random.default_rng(1000 * sd + int(1000 * sp) + (7 if regime == "targeted" else 0))
                    D = ST.collect(1000, sp, r) if regime == "broad" else collect_targeted(1000, sp, r)
                    p = ST.train(D, terr, False, np.random.default_rng(50 + sd))
                    th, _ = ST.optimise("wm", (p, terr, False), good0, sp, np.random.default_rng(20 + sd))
                    evs.append(evaluate(th, sp)); mechs.append(mechanism(p, sp))
                key = f"WM-{terr} ({regime} data)"
                row[key] = {k: [round(float(np.mean([e[k] for e in evs])), 4), round(float(np.std([e[k] for e in evs])), 4)]
                            for k in ("J", "trip", "clearance_cm", "staircase5_success")}
                row[key]["mechanism_shift_cm"] = [m_["shift_cm"] for m_ in mechs]
                row[key]["mechanism_width_cm"] = [m_["width_cm"] for m_ in mechs]
                row[key]["posterior_width_cm"] = mechs[0]["posterior_width_cm"]
                print(f"sigma_p={sp}", key, json.dumps(row[key]), f"{time.time() - t0:.0f}s", flush=True)
        res[f"sigma_p={sp}"] = row
        print(f"sigma_p={sp} oracle", row["oracle"], "ideal", row["ideal posterior model"], flush=True)
    res["note"] = ("values for WM rows are [mean, std] over world-model seeds; clearance = C_HOLD*a - H (cm); "
                   "staircase5_success = P(no trip on 5 consecutive steps); shift = clearance at which the model "
                   "believes P(clear) = 0.5 (posterior: ~0); width = 10-90 % span (posterior: 2.56 sigma_p)")
    with open(os.path.join(args.out, "idea8_stair_tests_v2.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
