"""Policy-prior MPPI around the frozen GMT tracker.
Samples noise on the next M actions, the policy fills the rest of the horizon; the MuJoCo model is the rollout oracle.
plan_model = 'oracle' (knows payload/kp) or 'nominal' (does not)."""
import json, sys, time, numpy as np
from gmt_core import Tracker
def run(case, plan_model, motion="walk_stand.pkl", secs=6.0, K=16, H=10, M=3, sigma=0.4, lam=0.1, seed=0, **kw):
    rng = np.random.default_rng(seed); tr = Tracker(motion, **kw)
    true_mass = tr.m.body_mass.copy(); true_kp = tr.kp.copy()
    nom = Tracker.__new__(Tracker)  # nominal params
    nom_mass = true_mass.copy(); torso = 16
    if "payload_kg" in kw: nom_mass[np.argmax(true_mass - Tracker(motion).m.body_mass)] -= kw["payload_kg"]
    nom_kp = true_kp / kw.get("kp_scale", 1.0)
    n = int(secs/0.02); errs = []; t0 = time.time()
    for k in range(n):
        obs, prop = tr.obs(); a0 = tr.policy(obs[None])[0]
        if K > 0:
            snap = tr.snapshot()
            if plan_model == "nominal": tr.m.body_mass[:] = nom_mass; tr.kp = nom_kp
            eps = rng.normal(0, sigma, (K, M, 23)); eps[0] = 0  # sample 0 = pure policy
            costs = np.zeros(K)
            for i in range(K):
                tr.restore(snap)
                for h in range(H):
                    o, p = tr.obs(); a = tr.policy(o[None])[0]
                    if h < M: a = a + eps[i, h]
                    tr.apply(a, p); costs[i] += tr.cost()
                    if tr.d.qpos[2] < 0.45: costs[i] += 50*(H-h); break
            tr.m.body_mass[:] = true_mass; tr.kp = true_kp; tr.restore(snap)
            w = np.exp(-(costs-costs.min())/(lam*max(costs.std(), 1e-6))); w /= w.sum()
            a0 = a0 + (w[:, None]*eps[:, 0]).sum(0)
        tr.apply(a0, prop)
        if tr.d.qpos[2] < 0.45: return dict(case=case, plan=plan_model, K=K, fell=True, t=round(k*0.02,2), wall=round(time.time()-t0))
        errs.append(np.abs(tr.d.qpos[-23:]-tr.ref(tr.k)[1]).mean())
    return dict(case=case, plan=plan_model, K=K, fell=False, joint_err=round(float(np.mean(errs)),4), wall=round(time.time()-t0))
cases = {"payload15": dict(payload_kg=15.0), "kp0.6": dict(kp_scale=0.6), "latency60": dict(latency_steps=3)}
out = []
for c, kw in cases.items():
    for plan, K in [("none", 0), ("oracle", 16), ("nominal", 16)]:
        if c == "latency60" and plan == "nominal": continue
        for seed in [0, 1]:
            r = run(c, plan, K=K, seed=seed, **kw); out.append(r); print(r, flush=True)
            if K == 0: break
json.dump(out, open(sys.argv[1], "w"), indent=1)
