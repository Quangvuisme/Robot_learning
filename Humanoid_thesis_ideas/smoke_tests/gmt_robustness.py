"""Stress-test GMT: lateral push magnitude, torso payload, action latency, low friction, weak motors."""
import json, os, sys, numpy as np
from gmt_core import Tracker
def run(motion="walk_stand.pkl", secs=6.0, push_N=0.0, push_t=2.0, **kw):
    tr = Tracker(motion, **kw); n = int(secs/0.02); errs=[]
    for k in range(n):
        obs, prop = tr.obs(); a = tr.policy(obs[None])[0]
        push = np.array([0, push_N, 0]) if push_t <= k*0.02 < push_t+0.1 else None
        tr.apply(a, prop, push)
        if tr.d.qpos[2] < 0.45: return dict(fell=True, t=k*0.02)
        errs.append(np.abs(tr.d.qpos[-23:]-tr.ref(tr.k)[1]).mean())
    return dict(fell=False, joint_err=float(np.mean(errs)))
res = {}
for m in ["walk_stand.pkl", "dance.pkl"]:
    for F in [0, 200, 300, 400, 500, 600]: res[f"{m}|push {F}N/0.1s"] = run(m, push_N=F)
    for kg in [5, 10, 15, 20]: res[f"{m}|payload {kg}kg"] = run(m, payload_kg=kg)
    for L in [1, 2, 3]: res[f"{m}|latency {20*L}ms"] = run(m, latency_steps=L)
    for mu in [0.4, 0.2]: res[f"{m}|friction {mu}"] = run(m, friction=mu)
    for s in [0.6, 0.4]: res[f"{m}|kp x{s}"] = run(m, kp_scale=s)
for k, v in res.items(): print(f"{k:38s} {v}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
