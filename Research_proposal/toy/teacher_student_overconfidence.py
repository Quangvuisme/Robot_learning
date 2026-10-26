"""
New topic (loop iteration 1, B1): "Privileged teachers teach overconfidence" in perceptive locomotion.

Setting (stair toy): clear iff C*a >= H;  E[J] = 0.4 + 0.2 P(clear) - a^2.  A privileged TEACHER sees the true
riser H and acts optimally for itself: a_T(H) = (H + 3 mm) / C  (essentially zero margin, which is optimal when H is
known). A STUDENT sees only H^ = H + N(0, sigma_p^2), as in every teacher-student perceptive pipeline (DAgger /
L2 action regression: RPL, DPL, ...).

Decision-theory prediction:  L2 distillation learns  E[a_T(H) | H^]  (posterior MEAN of teacher actions), but the
optimal student action is  argmax_a E[J(a, H) | H^]. Under an asymmetric cost (a trip is far worse than a slightly
higher swing) the posterior mean has ~zero margin, so the student trips about 50 % of the time ("margin collapse").

Students compared (all linear in H^: a = b + k H^, fitted on the same data):
  L2-BC / DAgger            least squares on (H^, a_T(H))
  quantile-BC (tau)         pinball loss: learns the tau-quantile of teacher actions given H^ (newsvendor fix)
  critic-guided distill.    maximise E[Q_T(H, a_S(H^))] with a learned, smooth teacher critic Q_T(H, a)
  RL fine-tuning (ZO)       model-free zeroth-order on the student's own objective (reference; expensive)
  oracle                    best linear student under the true noisy objective
"""
import argparse
import json
import os

import autograd.numpy as anp
import numpy as np
from autograd import grad

C, LAM, H_LO, H_HI = 1.0635, 1.0, 0.06, 0.20


def teacher(H):
    return (H + 0.003) / C


def J_true(a, H):
    return 0.4 + 0.2 * (C * a >= H) - LAM * a ** 2


def evaluate(b, k, sp, n=200000, seed=3):
    r = np.random.default_rng(seed)
    H = r.uniform(H_LO, H_HI, n); Hp = H + r.normal(0, sp, n)
    a = b + k * Hp
    return dict(J=round(float(J_true(a, H).mean()), 4), trip=round(float((C * a < H).mean()), 4),
                clearance_cm=round(100 * float((C * a - H).mean()), 2))


def data(sp, n, rng):
    H = rng.uniform(H_LO, H_HI, n)
    return H, H + rng.normal(0, sp, n)


def l2_bc(H, Hp):
    X = np.stack([np.ones_like(Hp), Hp], 1)
    return np.linalg.lstsq(X, teacher(H), rcond=None)[0]


def quantile_bc(H, Hp, tau, rng, iters=4000):
    w = l2_bc(H, Hp).copy(); m = np.zeros(2); v = np.zeros(2)
    for it in range(1, iters + 1):
        i = rng.integers(0, len(H), 1024)
        r = teacher(H[i]) - (w[0] + w[1] * Hp[i])
        g = -np.array([np.mean(np.where(r > 0, tau, tau - 1)), np.mean(np.where(r > 0, tau, tau - 1) * Hp[i])])
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w = w - 1e-3 * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return w


def critic_guided(H, Hp, rng, iters=3000):
    """Teacher critic Q_T(H, a) learned by regression on teacher-collected (H, a, return) with action noise,
    then the student maximises the mean critic value over its own actions (no privileged input at test time)."""
    # 1) learn a smooth teacher critic from data
    n = 60000
    Hc = rng.uniform(H_LO, H_HI, n); ac = teacher(Hc) + rng.normal(0, 0.03, n)
    y = J_true(ac, Hc)
    feat = lambda h, a: anp.stack([(C * a - h) / 0.01, a / 0.1, h / 0.1], -1)
    W1 = rng.normal(0, 0.5, (3, 64)); b1 = np.zeros(64); W2 = rng.normal(0, 0.1, (64, 1)); b2 = np.zeros(1)
    P = [W1, b1, W2, b2]
    Q = lambda P, h, a: (anp.dot(anp.tanh(anp.dot(feat(h, a), P[0]) + P[1]), P[2]) + P[3])[..., 0]
    mP = [np.zeros_like(p) for p in P]; vP = [np.zeros_like(p) for p in P]
    for it in range(1, 4001):
        i = rng.integers(0, n, 1024)
        g = grad(lambda P_: anp.mean((Q(P_, Hc[i], ac[i]) - y[i]) ** 2))(P)
        for j in range(4):
            mP[j] = 0.9 * mP[j] + 0.1 * g[j]; vP[j] = 0.999 * vP[j] + 0.001 * g[j] ** 2
            P[j] = P[j] - 3e-3 * (mP[j] / (1 - 0.9 ** it)) / (np.sqrt(vP[j] / (1 - 0.999 ** it)) + 1e-8)
    # 2) student maximises E[Q_T(H, b + k H^)] over (H, H^) pairs
    w = l2_bc(H, Hp).copy(); m = np.zeros(2); v = np.zeros(2)
    for it in range(1, iters + 1):
        i = rng.integers(0, len(H), 1024)
        g = grad(lambda w_: anp.mean(Q(P, H[i], w_[0] + w_[1] * Hp[i])))(w)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w = w + 1e-3 * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return w


def zo_rl(sp, rng, iters=400):
    w = np.array([0.0, 1.0 / C]); m = np.zeros(2); v = np.zeros(2); sims = 0
    for it in range(1, iters + 1):
        H, Hp = data(sp, 256, rng)
        d = rng.normal(0, 1, (8, 2)) * np.array([0.01, 0.05])
        g = np.mean([((J_true(w[0] + dd[0] + (w[1] + dd[1]) * Hp, H).mean()
                       - J_true(w[0] - dd[0] + (w[1] - dd[1]) * Hp, H).mean()) / 2) * dd / np.array([0.01, 0.05]) ** 2
                     for dd in d], 0)
        sims += 16 * 256
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w = w + 4e-3 * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return w, sims


def oracle(sp):
    best = (-1, None)
    for b in np.arange(-0.03, 0.15, 0.002):
        for k in np.arange(0.7, 1.2, 0.02):
            J = evaluate(b, k, sp, n=20000, seed=11)["J"]
            if J > best[0]:
                best = (J, (b, k))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    res = {}
    for sp in (0.01, 0.02, 0.04):
        row = {"oracle": evaluate(*oracle(sp), sp)}
        acc = {}
        for sd in range(args.seeds):
            rng = np.random.default_rng(sd)
            H, Hp = data(sp, 50000, rng)
            studs = {"L2-BC / DAgger (privileged teacher)": l2_bc(H, Hp)}
            for tau in (0.9, 0.97):
                studs[f"quantile-BC tau={tau}"] = quantile_bc(H, Hp, tau, rng)
            studs["critic-guided distillation"] = critic_guided(H, Hp, rng)
            w, sims = zo_rl(sp, rng)
            studs["RL fine-tuning (ZO)"] = w
            for name, w_ in studs.items():
                acc.setdefault(name, []).append(evaluate(w_[0], w_[1], sp))
        for name, e in acc.items():
            row[name] = {k: [round(float(np.mean([x[k] for x in e])), 4), round(float(np.std([x[k] for x in e])), 4)]
                         for k in ("J", "trip", "clearance_cm")}
        row["RL fine-tuning (ZO)"]["sim_rollouts"] = sims
        res[f"sigma_p={sp}"] = row
        print(f"sigma_p={sp}", json.dumps(row), flush=True)
    res["note"] = "[mean, std] over 3 seeds; trip = fraction of steps where the toe hits the riser"
    with open(os.path.join(args.out, "teacher_student_overconfidence.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
