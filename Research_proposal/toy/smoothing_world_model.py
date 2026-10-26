"""
Idea 8 toy: are noise-trained world models "amortized randomized smoothing"? Does annealing the noise help?

System: a ball launched from the origin at speed V0 and angle alpha (the policy parameter). A thin wall at x = 1
with height H: if the ball crosses x = 1 below H it sticks to the wall (vx = 0) and slides down. The ground is
sticky (velocity zeroed). Objective J(alpha) = final x after T steps (throw as far as possible).

Landscape (launch height 0.25 m): J is FLAT (= 1, ball stuck on the wall) for alpha in ~5-20 deg, jumps to ~1.66
at ~22 deg (clears the wall), peaks at ~42 deg (J ~ 1.99), then drops. From the flat region the true first-order
gradient is exactly zero.

Claims tested
  Prop A: a model trained on noisy inputs  s~ = s + sigma*eps  to predict f(s) learns E[f(s) | s~], which for
          locally flat data density equals the Gaussian-smoothed dynamics f * phi_sigma (= what randomized
          smoothing / bundled gradients estimate by sampling). One noise-conditioned model g(s, sigma) gives the
          whole family.
  Prop C: with policy-concentrated data the smoothing is density-weighted (biased toward behaviour seen often).
  Annealing: optimise alpha through g(., sigma) with sigma decreasing (continuation / graduated optimisation).

Methods compared for maximising J from starts in the flat region (6-19 deg):
  true FO        autograd through the true (piecewise) simulator
  ZO-RS          zeroth-order randomized smoothing on alpha (8 samples per step)
  WM sigma=0     plain learned world model
  NC-WM sigma=s  noise-conditioned model at fixed sigma
  NC-WM anneal   sigma from 0.12 -> 0 over the optimisation
"""
import argparse
import json
import os

import autograd.numpy as anp
import numpy as np
from autograd import grad

G, V0, WALL_X, WALL_H, DT, T = 9.81, 4.0, 1.0, 0.3, 0.03, 30
SCALE = np.array([1.0, 1.0, 1.0, 1.0])          # noise scale per state dim (positions m, velocities m/s)
Y0 = 0.25                                       # launch height (starting ON the sticky ground breaks smoothing)
SIGMAS = np.array([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12])


def true_step(s):
    x, y, vx, vy = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    xn, yn, vyn = x + vx * DT, y + vy * DT, vy - G * DT
    hit = (x < WALL_X) & (xn >= WALL_X) & (yn < WALL_H)
    xn = anp.where(hit, WALL_X - 1e-3, xn)
    vxn = anp.where(hit, 0.0 * vx, vx)
    ground = yn < 0.0
    yn = anp.where(ground, 0.0 * yn, yn)
    vxn = anp.where(ground, 0.0 * vxn, vxn)
    vyn = anp.where(ground, 0.0 * vyn, vyn)
    return anp.stack([xn, yn, vxn, vyn], -1)


def s0(alpha_deg):
    a = alpha_deg * np.pi / 180.0
    return anp.stack([0.0 * a, 0.0 * a + Y0, V0 * anp.cos(a), V0 * anp.sin(a)], -1)


def J_true(alpha_deg):
    s = s0(alpha_deg)
    for _ in range(T):
        s = true_step(s)
    return s[..., 0]


def J_mc_smoothed(alpha_deg, sigma, n=1024, seed=0):
    """Reference: per-step Gaussian-smoothed true dynamics, E_eps[f(s + sigma*eps)], by Monte Carlo."""
    rng = np.random.default_rng(seed)
    s = np.asarray(s0(np.asarray(alpha_deg, float)))
    for _ in range(T):
        eps = rng.normal(size=(n,) + s.shape) * sigma * SCALE
        s = np.asarray(true_step(s[None] + eps)).mean(0)
    return s[..., 0]


# ---------------- world model: MLP (s, sigma) -> next state ----------------
def init_mlp(sizes, rng):
    return [(rng.normal(0, np.sqrt(1.0 / a), (a, b)), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]


MU_S = np.array([0.8, 0.4, 2.0, 0.0])
SD_S = np.array([0.5, 0.3, 1.5, 3.0])
D_SCALE = np.array([0.12, 0.12, 1.0, 1.0])


def wm(params, s, sigma):
    """Predict next state from (possibly noisy) state s and noise level sigma."""
    z = (s - MU_S) / SD_S
    h = anp.concatenate([z, sigma * anp.ones(z.shape[:-1] + (1,)) * 10.0], -1)
    for i, (W, b) in enumerate(params):
        h = anp.dot(h, W) + b
        if i < len(params) - 1:
            h = anp.tanh(h)
    return s + h * D_SCALE


def make_data(alphas, rng):
    S = []
    for a in alphas:
        s = np.asarray(s0(a))
        traj = [s]
        for _ in range(T):
            s = np.asarray(true_step(s))
            traj.append(s)
        S.append(np.stack(traj))
    S = np.stack(S)                                   # (n, T+1, 4)
    return S[:, :-1].reshape(-1, 4), S[:, 1:].reshape(-1, 4)


def train_wm(X, Y, noise_conditioned, rng, steps=8000, batch=1024, lr=3e-3, weights=None):
    """noise_conditioned: False | "input" (noise on the model input, target f(s)  -> learns E[f(s)|s~], a
    density-dependent DENOISER) | "argument" (clean input s, target f(s + sigma*eps) from the simulator ->
    learns E_eps[f(s + sigma*eps)] = bundled dynamics, independent of the data density)."""
    if noise_conditioned is True:
        noise_conditioned = "input"
    params = init_mlp([5, 128, 128, 4], rng)
    flat = [p for pair in params for p in pair]
    m = [np.zeros_like(p) for p in flat]; v = [np.zeros_like(p) for p in flat]
    p_w = None if weights is None else weights / weights.sum()

    def loss(flat_params, x, y, sig):
        ps = [(flat_params[2 * i], flat_params[2 * i + 1]) for i in range(len(flat_params) // 2)]
        pred = wm(ps, x, sig)
        return anp.mean(anp.sum(((pred - y) / D_SCALE) ** 2, -1))

    gl = grad(loss)
    for it in range(1, steps + 1):
        idx = rng.choice(len(X), batch, p=p_w)
        x, y = X[idx], Y[idx]
        if noise_conditioned == "input":
            sig = rng.choice(SIGMAS, size=(batch, 1))
            x = x + rng.normal(size=x.shape) * sig * SCALE
        elif noise_conditioned == "argument":
            sig = rng.choice(SIGMAS, size=(batch, 1))
            y = np.asarray(true_step(x + rng.normal(size=x.shape) * sig * SCALE))
        else:
            sig = np.zeros((batch, 1))
        g = gl(flat, x, y, sig)
        a = lr if it < 0.75 * steps else lr * 0.2
        for j in range(len(flat)):
            m[j] = 0.9 * m[j] + 0.1 * g[j]; v[j] = 0.999 * v[j] + 0.001 * g[j] ** 2
            flat[j] = flat[j] - a * (m[j] / (1 - 0.9 ** it)) / (np.sqrt(v[j] / (1 - 0.999 ** it)) + 1e-8)
    return [(flat[2 * i], flat[2 * i + 1]) for i in range(len(flat) // 2)]


def J_wm(params, alpha_deg, sigma):
    s = s0(alpha_deg)
    for _ in range(T):
        s = wm(params, s, sigma)
    return s[..., 0]


# ---------------- optimisation of alpha ----------------
def optimise(Jfun, a0, iters=150, lr=1.0, sigma_sched=None, zo=False, rng=None):
    """Adam ascent on alpha (degrees). Jfun(alpha, sigma) must be autograd-differentiable unless zo."""
    a, m, v = float(a0), 0.0, 0.0
    for it in range(1, iters + 1):
        sig = 0.0 if sigma_sched is None else sigma_sched(it, iters)
        if zo:
            d = rng.normal(0, 5.0, 8)
            g = float(np.mean((np.asarray(J_true(a + d)) - float(J_true(a))) * d) / 25.0)
        else:
            g = float(np.clip(grad(lambda al: Jfun(al, sig))(a), -10.0, 10.0))   # clip exploding BPTT grads
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        a = float(np.clip(a + lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8), 1, 89))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {}

    # data: broad coverage (alpha ~ U(3, 85)) vs policy-concentrated (90% in the wall-hit region)
    X, Y = make_data(rng.uniform(3, 85, 400), rng)
    a_conc = np.concatenate([rng.uniform(5, 20, 360), rng.uniform(3, 85, 40)])
    Xc, Yc = make_data(a_conc, rng)
    print("data", X.shape, flush=True)
    wm_plain = train_wm(X, Y, False, np.random.default_rng(1))
    wm_nc = train_wm(X, Y, True, np.random.default_rng(2))
    wm_nc_conc = train_wm(Xc, Yc, True, np.random.default_rng(3))
    wm_arg = train_wm(X, Y, "argument", np.random.default_rng(4))
    wm_arg_conc = train_wm(Xc, Yc, "argument", np.random.default_rng(5))
    print("trained", flush=True)

    # ---- Prop A, one-step test (no compounding): g(s~, sigma) vs MC  f_sigma(s~) = E f(s~ + sigma eps)
    rs = np.random.default_rng(9)
    idx = rs.choice(len(X), 4000, replace=False)
    one = {}
    for sg in (0.03, 0.05, 0.08, 0.12):
        st = X[idx]                                          # on-distribution states
        f_s = np.asarray(true_step(st[None] + rs.normal(size=(2048, 4000, 4)) * sg * SCALE)).mean(0)
        f_raw = np.asarray(true_step(st))
        g_nc = np.asarray(wm(wm_nc, st, sg)); g_pl = np.asarray(wm(wm_plain, st, 0.0))
        g_cc = np.asarray(wm(wm_nc_conc, st, sg))
        g_ar = np.asarray(wm(wm_arg, st, sg)); g_arc = np.asarray(wm(wm_arg_conc, st, sg))
        near = np.abs(f_s - f_raw).sum(-1) > 0.05            # states where smoothing matters (near contact)
        err = lambda g: float(np.abs((g - f_s) / D_SCALE)[near].sum(-1).mean())
        one[sg] = dict(n_near_contact=int(near.sum()),
                       err_to_smoothed__unsmoothed_true=round(err(f_raw), 3),
                       err_to_smoothed__plain_wm=round(err(g_pl), 3),
                       err_to_smoothed__nc_wm=round(err(g_nc), 3),
                       err_to_smoothed__nc_wm_concentrated=round(err(g_cc), 3),
                       err_to_smoothed__ARG_nc_wm=round(err(g_ar), 3),
                       err_to_smoothed__ARG_nc_wm_concentrated=round(err(g_arc), 3))
    res["propA_one_step_near_contact"] = one
    print("propA one-step", json.dumps(one), flush=True)

    # ---- Prop A check: NC-WM(sigma) vs Monte-Carlo smoothed true dynamics
    grid = np.linspace(5, 80, 76)
    land = {"true": np.asarray(J_true(grid)).tolist()}
    propA = {}
    for sg in (0.0, 0.03, 0.05, 0.08, 0.12):
        mc = J_mc_smoothed(grid, sg) if sg > 0 else np.asarray(J_true(grid))
        nc = np.asarray(J_wm(wm_nc, grid, sg))
        ncc = np.asarray(J_wm(wm_nc_conc, grid, sg))
        arg = np.asarray(J_wm(wm_arg, grid, sg)); land[f"arg_sigma{sg}"] = arg.tolist()
        land[f"mc_sigma{sg}"] = mc.tolist(); land[f"nc_sigma{sg}"] = nc.tolist(); land[f"ncconc_sigma{sg}"] = ncc.tolist()
        propA[sg] = dict(mae_nc_vs_mc=round(float(np.abs(nc - mc).mean()), 4),
                         mae_concentrated_vs_mc=round(float(np.abs(ncc - mc).mean()), 4),
                         mae_ARG_vs_mc=round(float(np.abs(arg - mc).mean()), 4))
    land["plain_wm"] = np.asarray(J_wm(wm_plain, grid, 0.0)).tolist()
    res["propA_landscape_error"] = propA
    print("propA", propA, flush=True)

    # ---- gradient at the flat region: does it point toward the cliff (+)?
    pts = np.array([8.0, 12.0, 16.0, 19.0])
    gtab = {}
    for name, fn in [("true FO", lambda a: J_true(a))] + \
            [(f"NC-WM s={sg}", (lambda sg: lambda a: J_wm(wm_nc, a, sg))(sg)) for sg in (0.0, 0.03, 0.05, 0.08, 0.12)] + \
            [(f"NC-WM concentrated s={sg}", (lambda sg: lambda a: J_wm(wm_nc_conc, a, sg))(sg)) for sg in (0.05, 0.12)] + \
            [(f"ARG-WM s={sg}", (lambda sg: lambda a: J_wm(wm_arg, a, sg))(sg)) for sg in (0.03, 0.05, 0.08, 0.12)]:
        gtab[name] = [round(float(grad(fn)(float(a))), 4) for a in pts]
    res["dJ/dalpha_at_8_12_16_19deg"] = gtab
    print("grads", json.dumps(gtab), flush=True)

    # ---- optimisation from the flat region
    starts = np.linspace(6, 19, 9)
    anneal = lambda it, n: 0.12 * (1 - it / n) ** 2
    methods = {
        "true FO": dict(Jfun=lambda a, s: J_true(a)),
        "ZO-RS (8 samples, 5 deg)": dict(Jfun=None, zo=True),
        "plain WM": dict(Jfun=lambda a, s: J_wm(wm_plain, a, 0.0)),
        "input-noise NC-WM annealed 0.12->0": dict(Jfun=lambda a, s: J_wm(wm_nc, a, s), sigma_sched=anneal),
        "ARG-WM sigma=0.05": dict(Jfun=lambda a, s: J_wm(wm_arg, a, 0.05)),
        "ARG-WM sigma=0.12": dict(Jfun=lambda a, s: J_wm(wm_arg, a, 0.12)),
        "ARG-WM annealed 0.12->0": dict(Jfun=lambda a, s: J_wm(wm_arg, a, s), sigma_sched=anneal),
        "ARG-WM annealed, concentrated data": dict(Jfun=lambda a, s: J_wm(wm_arg_conc, a, s), sigma_sched=anneal),
    }
    opt = {}
    for name, kw in methods.items():
        finals = [optimise(a0=a0, rng=np.random.default_rng(int(a0)), **kw) for a0 in starts]
        Jf = np.asarray(J_true(np.array(finals)))
        opt[name] = dict(final_alpha_deg=[round(f, 1) for f in finals], mean_true_J=round(float(Jf.mean()), 3),
                         cleared_wall=f"{int((Jf > 1.05).sum())}/{len(starts)}")
        print(name, opt[name], flush=True)
    res["optimisation_from_flat_region"] = opt
    res["note"] = "optimum: alpha ~ 42 deg, J ~ 1.99; flat region (stuck on wall) J = 1.0; cleared = J > 1.05"
    with open(os.path.join(args.out, "idea8_smoothing.json"), "w") as f:
        json.dump(res, f, indent=2)

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        ax[0].plot(grid, land["true"], "k", lw=2, label="true J")
        for sg, c in zip((0.03, 0.08, 0.12), ("C0", "C1", "C2")):
            ax[0].plot(grid, land[f"mc_sigma{sg}"], color=c, ls="--", label=f"MC-smoothed true dynamics, sigma={sg}")
            ax[0].plot(grid, land[f"arg_sigma{sg}"], color=c, label=f"argument-noise WM, sigma={sg}")
        ax[0].plot(grid, land["plain_wm"], color="C4", ls=":", label="plain WM (sigma=0)")
        ax[0].axvspan(6, 19, color="0.9", zorder=0)
        ax[0].set_xlabel("launch angle alpha [deg]"); ax[0].set_ylabel("J = final x [m]")
        ax[0].set_title("Argument-noise world model vs randomized smoothing (grey: optimisation starts)", fontsize=9)
        ax[0].legend(fontsize=6.5)
        for sg, c in zip((0.05, 0.12), ("C1", "C2")):
            ax[1].plot(grid, land[f"mc_sigma{sg}"], color=c, ls="--", label=f"MC-smoothed, sigma={sg}")
            ax[1].plot(grid, land[f"nc_sigma{sg}"], color=c, label=f"NC-WM broad data, sigma={sg}")
            ax[1].plot(grid, land[f"ncconc_sigma{sg}"], color=c, ls=":", lw=2, label=f"NC-WM concentrated data, sigma={sg}")
        ax[1].plot(grid, land["true"], "k", lw=1)
        ax[1].set_xlabel("launch angle alpha [deg]")
        ax[1].set_title("Input-noise world models (broad / concentrated data) are denoisers, not smoothers", fontsize=9)
        ax[1].legend(fontsize=6.5)
        fig.tight_layout()
        p = os.path.join(args.out, "idea8_smoothing.png")
        fig.savefig(p, dpi=130)
        print("saved", p)


if __name__ == "__main__":
    main()
