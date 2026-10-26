"""
Multi-event EB-WM (loop iteration 2). Lesson from iteration 1: EVERY discontinuity must be an event.

In the stair swing there are two discontinuities: TRIP (the toe hits the riser) and STOP (the foot reaches its
target x = XT and its forward velocity jumps to 0 before descending onto the tread). The 2-event model left STOP
to the regression branch, which smoothed the jump (vx -> -0.7, -1.4, ... -8), drove the rollout off-distribution,
and made the event head fire false trips (p ~ 0.9). That was harmless when the objective only used the final x,
but catastrophic when P(clear) is used directly (multi-step staircase).

Model: shared trunk -> 3-way event head {none, trip, stop} (softmax cross-entropy on simulator labels)
                    -> branches f_none, f_trip, f_stop (each MSE on its own samples).
Branched rollout modes: "free" (keeps moving), "tripped" (absorbing failure), "landed" (absorbing success;
the stop event ends the swing). P(clear) = weight(landed) + weight(free at the horizon).
"""
import autograd.numpy as anp
import numpy as np
from autograd import grad

import stair_tests as ST


def labels(S, U, H):
    x, y, vx = S[:, 0], S[:, 1], S[:, 2]
    xn, yn = x + vx * ST.DT, y + U * ST.DT
    trip = (x < ST.XE) & (xn >= ST.XE) & (yn < H)
    stop = (~trip) & (vx > 0) & (xn >= ST.XT - 1e-9)
    return np.where(trip, 1, np.where(stop, 2, 0))


def init(rng, width=96):
    def lin(a, b):
        return [rng.normal(0, np.sqrt(1.0 / a), (a, b)), np.zeros(b)]
    return lin(6, width) + lin(width, width) + lin(width, 3) + lin(width, 3) + lin(width, 3) + lin(width, 3)


def parts(p, s, u, Hin, sigma):
    one = 0.0 * s[..., :1]
    z = (anp.concatenate([s, u[..., None] + one, Hin[..., None] + one], -1) - ST.MU) / ST.SD
    h = anp.concatenate([z, sigma * 40.0 + one], -1)
    h = anp.tanh(anp.dot(h, p[0]) + p[1])
    h = anp.tanh(anp.dot(h, p[2]) + p[3])
    logits = anp.dot(h, p[4]) + p[5]                                     # (..., 3): none, trip, stop
    f_none = s + (anp.dot(h, p[6]) + p[7]) * ST.DS
    f_trip = s + (anp.dot(h, p[8]) + p[9]) * ST.DS
    f_stop = s + (anp.dot(h, p[10]) + p[11]) * ST.DS
    return logits, f_none, f_trip, f_stop


def softmax(lg):
    e = anp.exp(lg - anp.max(lg, axis=-1, keepdims=True))
    return e / anp.sum(e, axis=-1, keepdims=True)


def train(D, terrain, rng, steps=3000, batch=512, lr=3e-3):
    S, U, H, Hp = D
    Hin_all = H if terrain == "true" else Hp
    p = init(rng)
    m = [np.zeros_like(q) for q in p]; v = [np.zeros_like(q) for q in p]

    def loss(p_, s, u, hin, y, lab):
        lg, fn, ft, fs = parts(p_, s, u, hin, 0.0)
        lse = anp.log(anp.sum(anp.exp(lg - anp.max(lg, -1, keepdims=True)), -1)) + anp.max(lg, -1)
        ce = anp.mean(lse - anp.sum(lg * np.eye(3)[lab], -1))
        out = 0.0
        for k, f in enumerate((fn, ft, fs)):
            w = (lab == k).astype(float)
            out = out + anp.sum(w[:, None] * ((f - y) / ST.DS) ** 2) / (w.sum() + 1.0)
        return 5.0 * ce + out

    gl = grad(loss)
    for it in range(1, steps + 1):
        i = rng.integers(0, len(S), batch)
        s, u, hin, Hi = S[i], U[i], Hin_all[i], H[i]
        y = np.asarray(ST.true_step(s, u, Hi)); lab = labels(s, u, Hi)
        g = gl(p, s, u, hin, y, lab)
        a_lr = lr if it < 0.75 * steps else lr * 0.2
        for j in range(len(p)):
            m[j] = 0.9 * m[j] + 0.1 * g[j]; v[j] = 0.999 * v[j] + 0.001 * g[j] ** 2
            p[j] = p[j] - a_lr * (m[j] / (1 - 0.9 ** it)) / (np.sqrt(v[j] / (1 - 0.999 ** it)) + 1e-8)
    return p


def p_clear(p, a, hin):
    """Mode-branched rollout with absorbing 'tripped' and 'landed' modes; returns P(clear)."""
    s = anp.stack([0.0 * a, 0.0 * a, ST.VX + 0.0 * a], -1)
    w_free, w_land = 1.0 + 0.0 * a, 0.0 * a
    for t in range(ST.T):
        lg, fn, _, _ = parts(p, s, ST.command(a, s, t), hin, 0.0)
        pr = softmax(lg)
        w_land = w_land + w_free * pr[..., 2]
        w_free = w_free * pr[..., 0]
        s = fn
    return w_land + w_free
