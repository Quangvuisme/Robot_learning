"""
EB-WM loop iteration 1 (A): neural perceptive policy on a 5-step staircase (compounding, absorbing failure).

Two caveats of the earlier stair toys were a 2-parameter linear policy and a single event. Here:
  * policy = MLP  H^ -> swing apex a  (nonlinear, 1-32-32-1)
  * a staircase of 5 risers per episode, H_i ~ U(6, 20) cm, perceived H^_i = H_i + N(0, sigma_p^2)
  * tripping is ABSORBING (the episode ends); reward +1 per step climbed, effort LAM5 * a^2 per attempted step
      J = sum_i alive_i - LAM5 * sum_i alive_{i-1} a_i^2,   alive_i = alive_{i-1} * 1[clear_i]
The single-swing world models from stair_tests_v4 give P(clear | a, H^) per riser:
  EB-WM (branched)     P = final weight of the "no trip" particle (mode-branched rollout)
  mean-field           P = (x_final - x_trip) / (x_target - x_trip) of the mean-field rollout
  privileged EB-WM     same as EB-WM but the model is fed the TRUE riser height (policy still sees H^)
  regression MLP-WM    mean-field rollout of the plain MLP world model
Reference: "ideal posterior" = the same NN trained on the exact expected objective (Gaussian posterior).
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
import stair_tests_v4 as F
import ef_multievent as M3

N_STEPS, LAM5 = 5, 5.0                # LAM5 / 1 keeps the effort-to-reward ratio of the single-step toy (1 : 0.2)
C = V.C_HOLD


def init_policy(rng):
    return [rng.normal(0, 1.0, (1, 32)), np.zeros(32), rng.normal(0, 0.2, (32, 32)), np.zeros(32),
            rng.normal(0, 0.2, (32, 1)), np.zeros(1)]


def apex(pp, Hp):
    h = anp.tanh(anp.dot(((Hp - 0.13) / 0.05)[..., None], pp[0]) + pp[1])
    h = anp.tanh(anp.dot(h, pp[2]) + pp[3])
    return 0.35 / (1 + anp.exp(-(anp.dot(h, pp[4]) + pp[5])[..., 0]))


def pretrain(pp, offset, rng, iters=1500):
    """Supervised warm start a ~ H^/C + offset (clearing: +4 cm, tripping: -3 cm)."""
    m = [np.zeros_like(q) for q in pp]; v = [np.zeros_like(q) for q in pp]
    for it in range(1, iters + 1):
        Hp = rng.uniform(0.03, 0.24, 256)
        g = grad(lambda q: anp.mean((apex(q, Hp) - (Hp / C + offset)) ** 2))(pp)
        for j in range(len(pp)):
            m[j] = 0.9 * m[j] + 0.1 * g[j]; v[j] = 0.999 * v[j] + 0.001 * g[j] ** 2
            pp[j] = pp[j] - 3e-3 * (m[j] / (1 - 0.9 ** it)) / (np.sqrt(v[j] / (1 - 0.999 ** it)) + 1e-8)
    return pp


def p_clear_branched(p, a, hin, sig=0.0):
    s = anp.stack([0.0 * a, 0.0 * a, ST.VX + 0.0 * a], -1)
    w_free = 1.0 + 0.0 * a
    for t in range(ST.T):
        lg, ff, _ = F.ef_parts(p, s, ST.command(a, s, t), hin, sig)
        w_free = w_free * (1 - 1 / (1 + anp.exp(-lg)))
        s = ff
    return w_free


def p_clear_meanfield(model_fn, a, hin):
    s = anp.stack([0.0 * a, 0.0 * a, ST.VX + 0.0 * a], -1)
    for t in range(ST.T):
        s = model_fn(s, ST.command(a, s, t), hin)
    return anp.clip((s[..., 0] - (ST.XE - 1e-3)) / (ST.XT - (ST.XE - 1e-3)), 0.0, 1.0)


def staircase_J(P, a):
    """P, a: (batch, N_STEPS)."""
    J, alive = 0.0, 1.0
    for i in range(N_STEPS):
        J = J - LAM5 * alive * a[:, i] ** 2
        alive = alive * P[:, i]
        J = J + alive
    return anp.mean(J)


def make_objective(kind, model, sig_p):
    def obj(pp, H, Hp):
        a = apex(pp, Hp)
        if kind == "ideal":
            sp = max(sig_p, 1e-3)
            P = 0.5 * (1 + anp.tanh(0.7978845608 * ((C * a - Hp) / sp + 0.044715 * ((C * a - Hp) / sp) ** 3)))
        elif kind == "branched":
            P = p_clear_branched(model, a, Hp)
        elif kind == "branched_privileged":
            P = p_clear_branched(model, a, H)
        elif kind == "branched3":
            P = M3.p_clear(model, a, Hp)
        elif kind == "branched3_privileged":
            P = M3.p_clear(model, a, H)
        elif kind == "meanfield_ef":
            P = p_clear_meanfield(lambda s, u, h: F.ef_wm(model, s, u, h, 0.0), a, Hp)
        elif kind == "meanfield_mlp":
            P = p_clear_meanfield(lambda s, u, h: ST.wm(model, s, u, h, 0.0), a, Hp)
        return staircase_J(P, a)
    return obj


def train_policy(obj, pp, sig_p, rng, iters=400, lr=2e-3, batch=48):
    m = [np.zeros_like(q) for q in pp]; v = [np.zeros_like(q) for q in pp]
    for it in range(1, iters + 1):
        H = rng.uniform(ST.H_LO, ST.H_HI, (batch, N_STEPS))
        Hp = H + (rng.normal(0, sig_p, H.shape) if sig_p > 0 else 0.0)
        g = grad(lambda q: obj(q, H.reshape(-1), Hp.reshape(-1)) if False else obj_wrap(obj, q, H, Hp))(pp)
        for j in range(len(pp)):
            gj = np.clip(g[j], -5, 5)
            m[j] = 0.9 * m[j] + 0.1 * gj; v[j] = 0.999 * v[j] + 0.001 * gj ** 2
            pp[j] = pp[j] + lr * (m[j] / (1 - 0.9 ** it)) / (np.sqrt(v[j] / (1 - 0.999 ** it)) + 1e-8)
    return pp


def obj_wrap(obj, q, H, Hp):
    return obj(q, H, Hp)


def evaluate(pp, sig_p, n=4000, seed=7):
    r = np.random.default_rng(seed)
    H = r.uniform(ST.H_LO, ST.H_HI, (n, N_STEPS)); Hp = H + (r.normal(0, sig_p, H.shape) if sig_p > 0 else 0.0)
    a = np.asarray(apex(pp, Hp))
    clear = C * a >= H
    alive = np.cumprod(clear, 1)
    steps = alive.sum(1)
    effort = LAM5 * (np.concatenate([np.ones((n, 1)), alive[:, :-1]], 1) * a ** 2).sum(1)
    grid = np.linspace(0.07, 0.19, 7)
    return dict(J=round(float(np.mean(steps - effort)), 4), steps_climbed=round(float(steps.mean()), 3),
                all5=round(float((steps == N_STEPS).mean()), 4),
                trip_per_step=round(float(1 - clear[:, 0].mean()), 4),
                clearance_cm_vs_Hp={f"{h:.2f}": round(100 * float(C * np.asarray(apex(pp, np.array([h])))[0] - h), 2)
                                    for h in grid})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sp", type=float, default=0.02)
    ap.add_argument("--kinds", default="ideal,branched,meanfield_ef,branched_privileged,meanfield_mlp")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    t0, sp = time.time(), args.sp
    res = {}
    kinds = args.kinds.split(",")
    acc = {k: {"clearing": [], "tripping": []} for k in kinds}
    for sd in range(args.seeds):
        D = F.collect_unconfounded(1000, sp, np.random.default_rng(1000 * sd + int(1000 * sp) + 7))
        models = {}
        if "branched" in kinds or "meanfield_ef" in kinds:
            models["branched"] = F.train_ef(D, "perceived", False, np.random.default_rng(60 + sd))
            models["meanfield_ef"] = models["branched"]
        if "branched_privileged" in kinds:
            models["branched_privileged"] = F.train_ef(D, "true", False, np.random.default_rng(80 + sd))
        if "meanfield_mlp" in kinds:
            models["meanfield_mlp"] = ST.train(D, "perceived", False, np.random.default_rng(50 + sd))
        if "branched3" in kinds:
            models["branched3"] = M3.train(D, "perceived", np.random.default_rng(60 + sd))
        if "branched3_privileged" in kinds:
            models["branched3_privileged"] = M3.train(D, "true", np.random.default_rng(80 + sd))
        for init, off in (("clearing", 0.04), ("tripping", -0.03)):
            base = pretrain(init_policy(np.random.default_rng(5 + sd)), off, np.random.default_rng(6 + sd))
            for k in kinds:
                pp = [q.copy() for q in base]
                pp = train_policy(make_objective(k, models.get(k), sp), pp, sp, np.random.default_rng(20 + sd))
                acc[k][init].append(evaluate(pp, sp))
            print(f"seed {sd} {init}", {k: acc[k][init][-1]["J"] for k in kinds}, f"{time.time() - t0:.0f}s", flush=True)
        res["init_policies"] = {"clearing": evaluate(pretrain(init_policy(np.random.default_rng(5)), 0.04,
                                                              np.random.default_rng(6)), sp),
                                "tripping": evaluate(pretrain(init_policy(np.random.default_rng(5)), -0.03,
                                                              np.random.default_rng(6)), sp)}
    for k in kinds:
        res[k] = {}
        for init in ("clearing", "tripping"):
            e = acc[k][init]
            res[k][init] = {m_: [round(float(np.mean([x[m_] for x in e])), 4), round(float(np.std([x[m_] for x in e])), 4)]
                            for m_ in ("J", "steps_climbed", "all5", "trip_per_step")}
            res[k][init]["clearance_cm_vs_Hp_seed0"] = e[0]["clearance_cm_vs_Hp"]
        print(k, json.dumps(res[k]), flush=True)
    res["note"] = (f"sigma_p={sp}; 5-riser staircase, absorbing trips; J = steps climbed - {LAM5} * effort; "
                   "[mean, std] over world-model seeds; 'ideal' = NN trained on the exact posterior objective")
    with open(os.path.join(args.out, f"ebwm_staircase_nn{args.tag}_sp{sp}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
