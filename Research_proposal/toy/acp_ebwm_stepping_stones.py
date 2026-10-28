"""
Thorough test of Algorithm 2 (ACP contact planning + EB-WM) on a stepping-stone toy.

ACP structure kept: the planner scores each terrain TOKEN (stone: perceived centre, perceived width) with a shared
network (attention-style), plus a learned "no-step" slot, and outputs an aim offset per stone. Contact outcome =
which stone the foot lands on, or MISS (absorbing fall).

Per step: N = 6 stones ahead, centres c_j ~ 0.15 j + U(-4, 4) cm, widths w_j ~ U(4, 16) cm.
Perception: c^_j = c_j + N(0, sp^2), w^_j = w_j + N(0, sp^2). Reach limit R = 0.75 m; motor noise 1 cm.
Landing x = min(target, R) + N(0, 1 cm); success iff it lands inside a TRUE stone.
Reward: + landing distance per successful step; a miss costs F = 3 and ends the episode (5 steps max).
Planner: p = softmax over {no-step, stone_1..6}; target_j = c^_j + offset_j. Test-time policy = argmax.

Methods
  expert (privileged)   sees TRUE stones; greedy per-step choice (reference only, not deployable)
  ACP-SS                self-supervised planner: cross-entropy on the expert's choice + L2 on its aim (perceived input)
  fine-tuned from ACP-SS by:
    EB-WM (true + MC)   contact-attention EVENT head on true stones, MC over perception noise, branched alive/fail
    EB-WM (perceived)   same event head fed the perceived stones (learns the posterior)
    plain WM            regression world model of the step outcome (miss encoded as -F), mean-field, no branching
    pathwise true-sim   backprop through the sampled simulator (one motor-noise sample; misses the boundary term)
    ZO RL               model-free antithetic zeroth-order on all planner parameters (unbiased, expensive)
  oracle                BPTT through the exact expected outcome (analytic Gaussian motor noise, true stones, MC)
"""
import argparse
import json
import os
import time

import autograd.numpy as anp
import numpy as np
from autograd import grad

N, R, SM, F, STEPS = 6, 0.75, 0.01, 3.0, 5        # F = 3: a fall is expensive


def ncdf(z):
    return 0.5 * (1.0 + anp.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))


def terrain(rng, n, sp):
    c = 0.15 * np.arange(1, N + 1)[None] + rng.uniform(-0.04, 0.04, (n, N))
    w = rng.uniform(0.04, 0.16, (n, N))
    ch = c + (rng.normal(0, sp, c.shape) if sp > 0 else 0.0)
    wh = np.maximum(w + (rng.normal(0, sp, w.shape) if sp > 0 else 0.0), 0.01)
    return c, w, ch, wh


# ---------------- planner (ACP-style token scoring) ----------------
def init_planner(rng, h=32):
    return [rng.normal(0, 0.5, (2, h)), np.zeros(h), rng.normal(0, 0.3, (h, h)), np.zeros(h),
            rng.normal(0, 0.3, (h, 2)), np.zeros(2), np.array([0.0])]


def planner(P, ch, wh):
    f = anp.stack([(ch - 0.5) / 0.3, (wh - 0.1) / 0.04], -1)                 # (n, N, 2) tokens
    h = anp.tanh(anp.dot(f, P[0]) + P[1]); h = anp.tanh(anp.dot(h, P[2]) + P[3])
    o = anp.dot(h, P[4]) + P[5]
    logits = anp.concatenate([P[6] + 0.0 * o[..., :1, 0], o[..., 0]], -1)       # (n, N+1): no-step first
    target = ch + 0.5 * wh * anp.tanh(o[..., 1])                              # aim within the perceived stone
    return logits, target


def softmax(z):
    e = anp.exp(z - anp.max(z, -1, keepdims=True))
    return e / anp.sum(e, -1, keepdims=True)


# ---------------- outcome models ----------------
def p_land_exact(x, c, w):
    """P(success) for aim x (n, N) given TRUE stones, analytic Gaussian motor noise (oracle model)."""
    xr = anp.minimum(x, R)
    tot = 0.0
    for k in range(N):
        tot = tot + ncdf((c[:, k:k + 1] + w[:, k:k + 1] / 2 - xr) / SM) - ncdf((c[:, k:k + 1] - w[:, k:k + 1] / 2 - xr) / SM)
    return anp.minimum(tot, 1.0)


def land_sample(x, c, w, rng):
    xl = np.minimum(x, R) + rng.normal(0, SM, np.shape(x))
    inside = np.zeros(np.shape(x), bool)
    for k in range(N):
        inside |= np.abs(xl - c[:, k:k + 1]) <= w[:, k:k + 1] / 2
    return xl, inside


def init_event(rng, h=48):
    return [rng.normal(0, 0.5, (3, h)), np.zeros(h), rng.normal(0, 0.3, (h, h)), np.zeros(h),
            rng.normal(0, 0.3, (h, 1)), np.zeros(1), rng.normal(0, 0.5, (1, h)), np.zeros(h),
            rng.normal(0, 0.3, (h, 1)), np.zeros(1)]


def event_logits(E, x, c, w):
    """Contact-attention event head: query = aim x (n, M); keys = stone tokens (n, N).
    Returns logits (n, M, N+1) over {miss, land on stone k}."""
    f = anp.stack([(x[..., None] - c[:, None, :]) / 0.05, (w[:, None, :] + 0.0 * x[..., None] - 0.1) / 0.04,
                   (x[..., None] + 0.0 * c[:, None, :] - 0.5) / 0.3], -1)     # (n, M, N, 3)
    h = anp.tanh(anp.dot(f, E[0]) + E[1]); h = anp.tanh(anp.dot(h, E[2]) + E[3])
    lk = (anp.dot(h, E[4]) + E[5])[..., 0]                                     # (n, M, N)
    hm = anp.tanh(anp.dot(((x - 0.5) / 0.3)[..., None], E[6]) + E[7])
    lm = (anp.dot(hm, E[8]) + E[9])                                             # (n, M, 1)
    return anp.concatenate([lm, lk], -1)


def train_event(rng, sp, terrain_input, steps=3000):
    E = init_event(rng); m = [np.zeros_like(q) for q in E]; v = [np.zeros_like(q) for q in E]

    def loss(E_, x, c, w, lab):
        lg = event_logits(E_, x, c, w)[:, 0, :]
        lse = anp.log(anp.sum(anp.exp(lg - anp.max(lg, -1, keepdims=True)), -1)) + anp.max(lg, -1)
        return anp.mean(lse - anp.sum(lg * np.eye(N + 1)[lab], -1))
    gl = grad(loss)
    for it in range(1, steps + 1):
        c, w, ch, wh = terrain(rng, 512, sp)
        j = rng.integers(0, N, 512)
        x = (ch[np.arange(512), j] + rng.normal(0, 0.06, 512))[:, None]       # exploration around stones
        x = np.where(rng.random((512, 1)) < 0.2, rng.uniform(0, 0.95, (512, 1)), x)
        xl = np.minimum(x[:, 0], R) + rng.normal(0, SM, 512)
        lab = np.zeros(512, int)
        for k in range(N):
            hit = (np.abs(xl - c[:, k]) <= w[:, k] / 2) & (lab == 0)
            lab[hit] = k + 1
        cin, win = (c, w) if terrain_input == "true" else (ch, wh)
        g = gl(E, x, cin, win, lab)
        lr = 3e-3 if it < 2250 else 6e-4
        for q in range(len(E)):
            m[q] = 0.9 * m[q] + 0.1 * g[q]; v[q] = 0.999 * v[q] + 0.001 * g[q] ** 2
            E[q] = E[q] - lr * (m[q] / (1 - 0.9 ** it)) / (np.sqrt(v[q] / (1 - 0.999 ** it)) + 1e-8)
    return E


def init_reg(rng, h=64):
    return [rng.normal(0, 0.3, (2 * N + 1, h)), np.zeros(h), rng.normal(0, 0.2, (h, h)), np.zeros(h),
            rng.normal(0, 0.2, (h, 1)), np.zeros(1)]


def reg_outcome(Q, x, c, w):
    """Plain regression world model: predicts the step outcome value (landing x on success, -F on miss)."""
    feat = anp.concatenate([((x[..., None] - c[:, None, :]) / 0.05), ((w[:, None, :] + 0.0 * x[..., None]) - 0.1) / 0.04,
                            ((x - 0.5) / 0.3)[..., None]], -1)
    h = anp.tanh(anp.dot(feat, Q[0]) + Q[1]); h = anp.tanh(anp.dot(h, Q[2]) + Q[3])
    return (anp.dot(h, Q[4]) + Q[5])[..., 0]


def train_reg(rng, sp, steps=3000):
    Q = init_reg(rng); m = [np.zeros_like(q) for q in Q]; v = [np.zeros_like(q) for q in Q]
    gl = grad(lambda Q_, x, c, w, y: anp.mean((reg_outcome(Q_, x, c, w)[:, 0] - y) ** 2))
    for it in range(1, steps + 1):
        c, w, ch, wh = terrain(rng, 512, sp)
        j = rng.integers(0, N, 512)
        x = (ch[np.arange(512), j] + rng.normal(0, 0.06, 512))[:, None]
        x = np.where(rng.random((512, 1)) < 0.2, rng.uniform(0, 0.95, (512, 1)), x)
        xl, ok = land_sample(x, c, w, rng)
        y = np.where(ok[:, 0], xl[:, 0], -F)
        g = gl(Q, x, ch, wh, y)                                                 # deployable: perceived input
        lr = 3e-3 if it < 2250 else 6e-4
        for q in range(len(Q)):
            m[q] = 0.9 * m[q] + 0.1 * g[q]; v[q] = 0.999 * v[q] + 0.001 * g[q] ** 2
            Q[q] = Q[q] - lr * (m[q] / (1 - 0.9 ** it)) / (np.sqrt(v[q] / (1 - 0.999 ** it)) + 1e-8)
    return Q


# ---------------- objectives (expected over the planner's categorical choice, STEPS absorbing steps) ----------------
def staircase_return(p, ps, prog):
    """p: (n, N+1) choice probs; ps: (n, N+1) success prob per choice; prog: (n, N+1) progress per choice.
    Steps are i.i.d., so one step's statistics are reused for all STEPS (absorbing failure)."""
    q = anp.sum(p * ps, -1)                                                      # P(success) this step
    r = anp.sum(p * ps * prog, -1) - F * anp.sum(p * (1 - ps), -1)
    J, alive = 0.0, 1.0
    for _ in range(STEPS):
        J = J + alive * r
        alive = alive * q
    return anp.mean(J)


def make_obj(kind, model, rng_holder):
    def obj(P, c, w, ch, wh):
        logits, target = planner(P, ch, wh)
        p = softmax(logits)
        zero = 0.0 * target[:, :1]
        if kind == "oracle":
            ps = p_land_exact(target, c, w)
        elif kind == "ebwm_true":
            ps = 1 - softmax(event_logits(model, target, c, w))[..., 0]
        elif kind == "ebwm_perc":
            ps = 1 - softmax(event_logits(model, target, ch, wh))[..., 0]
        elif kind == "pathwise":
            xl = anp.minimum(target, R) + rng_holder["rng"].normal(0, SM, target.shape)
            inside = 0.0 * xl
            for k in range(N):
                inside = inside + anp.where(anp.abs(xl - c[:, k:k + 1]) <= w[:, k:k + 1] / 2, 1.0, 0.0)
            ps = anp.minimum(inside, 1.0) + 0.0 * xl
        if kind == "plain_wm":
            y = reg_outcome(model, target, ch, wh)                                # mean-field outcome value
            r = anp.sum(p[:, 1:] * y, -1)
            return anp.mean(STEPS * r)
        ps = anp.concatenate([1.0 + zero, ps], -1)
        prog = anp.concatenate([zero, anp.minimum(target, R)], -1)
        return staircase_return(p, ps, prog)
    return obj


def evaluate(P, sp, n=20000, seed=5):
    rng = np.random.default_rng(seed)
    J = np.zeros(n); alive = np.ones(n, bool); misses = 0; steps_taken = 0; margins = []
    for _ in range(STEPS):
        c, w, ch, wh = terrain(rng, n, sp)
        logits, target = planner(P, ch, wh)
        choice = np.argmax(np.asarray(logits), -1)
        x = np.asarray(target)[np.arange(n), np.maximum(choice - 1, 0)]
        xl, ok = land_sample(x[:, None], c, w, rng)
        xl, ok = xl[:, 0], ok[:, 0]
        step = choice > 0
        succ = ~step | ok
        J += alive * np.where(step, np.where(ok, xl, -F), 0.0)
        misses += int((alive & step & ~ok).sum()); steps_taken += int((alive & step).sum())
        d = np.min(np.abs(np.stack([xl[:, None] - (c - w / 2), (c + w / 2) - xl[:, None]], 0)), 0)
        inside = (np.abs(xl[:, None] - c) <= w / 2)
        mg = np.where(inside, d, np.inf).min(1)
        margins += list(mg[alive & step & ok])
        alive = alive & succ
    return dict(J=round(float(J.mean()), 4), all5=round(float(alive.mean()), 4),
                miss_per_step=round(misses / max(steps_taken, 1), 4),
                edge_margin_cm=round(100 * float(np.mean(margins)) if margins else 0.0, 2))


def expert_choice(c, w):
    ps = np.asarray(p_land_exact(c, c, w))                                      # aim at the true centre
    val = ps * np.minimum(c, R) - F * (1 - ps)
    best = np.argmax(val, -1)
    choice = np.where(val[np.arange(len(c)), best] > 0, best + 1, 0)
    return choice


def train_acp_ss(rng, sp, iters=3000):
    P = init_planner(rng); m = [np.zeros_like(q) for q in P]; v = [np.zeros_like(q) for q in P]

    def loss(P_, ch, wh, lab, aim):
        logits, target = planner(P_, ch, wh)
        lse = anp.log(anp.sum(anp.exp(logits - anp.max(logits, -1, keepdims=True)), -1)) + anp.max(logits, -1)
        ce = anp.mean(lse - anp.sum(logits * np.eye(N + 1)[lab], -1))
        idx = np.maximum(lab - 1, 0)
        t_sel = anp.sum(target * np.eye(N)[idx], -1)
        l2 = anp.sum((lab > 0) * (t_sel - aim) ** 2) / (np.sum(lab > 0) + 1) / 0.01 ** 2
        return ce + 0.1 * l2
    gl = grad(loss)
    for it in range(1, iters + 1):
        c, w, ch, wh = terrain(rng, 512, sp)
        lab = expert_choice(c, w)
        aim = c[np.arange(512), np.maximum(lab - 1, 0)]
        g = gl(P, ch, wh, lab, aim)
        for q in range(len(P)):
            m[q] = 0.9 * m[q] + 0.1 * g[q]; v[q] = 0.999 * v[q] + 0.001 * g[q] ** 2
            P[q] = P[q] - 3e-3 * (m[q] / (1 - 0.9 ** it)) / (np.sqrt(v[q] / (1 - 0.999 ** it)) + 1e-8)
    return P


def finetune(P0, kind, model, sp, rng, iters=400, lr=2e-3, batch=256):
    P = [q.copy() for q in P0]; m = [np.zeros_like(q) for q in P]; v = [np.zeros_like(q) for q in P]
    holder = {"rng": rng}; obj = make_obj(kind, model, holder); sims = 0
    for it in range(1, iters + 1):
        c, w, ch, wh = terrain(rng, batch, sp)
        if kind == "zo":
            flat = np.concatenate([q.ravel() for q in P]); g = np.zeros_like(flat)
            for _ in range(8):
                d = rng.normal(0, 0.02, flat.shape)
                def unflat(fv):
                    out, i = [], 0
                    for q in P:
                        out.append(fv[i:i + q.size].reshape(q.shape)); i += q.size
                    return out
                jp = zo_eval(unflat(flat + d), c, w, ch, wh, rng); jm = zo_eval(unflat(flat - d), c, w, ch, wh, rng)
                g += (jp - jm) / 2 * d / 0.02 ** 2 / 8
                sims += 2 * batch * STEPS
            i = 0; gs = []
            for q in P:
                gs.append(g[i:i + q.size].reshape(q.shape)); i += q.size
        else:
            gs = grad(lambda P_: obj(P_, c, w, ch, wh))(P)
        for q in range(len(P)):
            gq = np.clip(gs[q], -5, 5)
            m[q] = 0.9 * m[q] + 0.1 * gq; v[q] = 0.999 * v[q] + 0.001 * gq ** 2
            P[q] = P[q] + lr * (m[q] / (1 - 0.9 ** it)) / (np.sqrt(v[q] / (1 - 0.999 ** it)) + 1e-8)
    return P, sims


def zo_eval(P, c, w, ch, wh, rng):
    """Model-free return estimate from sampled outcomes (stochastic choice for a smooth estimator)."""
    logits, target = planner(P, ch, wh)
    p = np.asarray(softmax(logits)); n = len(c)
    choice = (np.cumsum(p, -1) < rng.random((n, 1))).sum(-1).clip(0, N)        # vectorised categorical sample
    x = np.asarray(target)[np.arange(n), np.maximum(choice - 1, 0)]
    xl, ok = land_sample(x[:, None], c, w, rng)
    step = choice > 0
    r = np.where(step, np.where(ok[:, 0], xl[:, 0], -F), 0.0)
    q = np.where(step, ok[:, 0], True).astype(float)
    return float(np.mean(r * (1 - q ** STEPS) / np.maximum(1 - q, 1e-9) * (q < 1) + r * STEPS * (q >= 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sp", type=float, required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    t0, sp = time.time(), args.sp
    acc = {}
    for sd in range(args.seeds):
        P_ss = train_acp_ss(np.random.default_rng(10 + sd), sp)
        E_true = train_event(np.random.default_rng(20 + sd), sp, "true")
        E_perc = train_event(np.random.default_rng(30 + sd), sp, "perceived")
        Q = train_reg(np.random.default_rng(40 + sd), sp)
        runs = {"ACP-SS (self-supervised, no fine-tune)": (P_ss, 0)}
        for name, kind, model in [("ACP-SS + EB-WM (true + MC)", "ebwm_true", E_true),
                                  ("ACP-SS + EB-WM (perceived)", "ebwm_perc", E_perc),
                                  ("ACP-SS + plain WM (mean-field)", "plain_wm", Q),
                                  ("ACP-SS + pathwise true-sim", "pathwise", None),
                                  ("ACP-SS + ZO RL (model-free)", "zo", None),
                                  ("oracle (exact expected outcome)", "oracle", None)]:
            runs[name] = finetune(P_ss, kind, model, sp, np.random.default_rng(50 + sd))
        for name, (P, sims) in runs.items():
            e = evaluate(P, sp); e["sim_rollouts"] = sims
            acc.setdefault(name, []).append(e)
        print(f"sigma_p={sp} seed {sd}", {k: v[-1]["J"] for k, v in acc.items()}, f"{time.time() - t0:.0f}s", flush=True)
    # privileged expert (not deployable): uses TRUE stones
    rng = np.random.default_rng(5); n = 20000; J = np.zeros(n); alive = np.ones(n, bool)
    for _ in range(STEPS):
        c, w, ch, wh = terrain(rng, n, sp); ch_ = expert_choice(c, w)
        x = c[np.arange(n), np.maximum(ch_ - 1, 0)]
        xl, ok = land_sample(x[:, None], c, w, rng); step = ch_ > 0
        J += alive * np.where(step, np.where(ok[:, 0], xl[:, 0], -F), 0.0); alive &= (~step | ok[:, 0])
    res = {"privileged expert (true stones, not deployable)": dict(J=round(float(J.mean()), 4), all5=round(float(alive.mean()), 4))}
    for name, e in acc.items():
        res[name] = {k: [round(float(np.mean([x[k] for x in e])), 4), round(float(np.std([x[k] for x in e])), 4)]
                     for k in ("J", "all5", "miss_per_step", "edge_margin_cm")}
        res[name]["sim_rollouts"] = e[0]["sim_rollouts"]
    print(json.dumps(res), flush=True)
    with open(os.path.join(args.out, f"acp_ebwm_stones_sp{sp}.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
