"""
Toy prototype for "Thermal Preview Control" (Research_proposal/IDEA_2_thermal_preview_humanoid.md).

A redundant 3-link planar arm holds a payload at a sequence of targets (a known "program").
Each joint motor heats with a lumped RC model:  C dT/dt = alpha * tau^2 - (T - T_amb) / R.
If a joint reaches T_max its torque limit is derated to 50%; asking for more than the limit
counts as a failure (payload drop).

Two timescales:
  fast layer : given a per-joint heat allowance b in (0,1]^3, pick the arm configuration that
               trades tool-orientation error against allowance-weighted squared torque
               (exact argmin over the 1-DoF redundancy; stands in for the learned policy).
  slow layer : choose the allowance schedule b_1..b_N over the program segments.

Compared methods
  B0 nominal      b = 1 always (energy-optimal, thermally blind)
  B1 reactive     b from the current temperature each second (the quadruped-paper idea)
  B3 myopic MPPI  plans only the current segment
  B4 scalar MPPI  preview, but one shared allowance for all joints (can slow down, cannot redistribute)
  OURS            preview MPPI over the whole remaining program, per-joint allowance

Pure numpy; runs in well under a minute on CPU.
"""
import argparse
import json
import os

import numpy as np

# ---------------- arm + thermal parameters ----------------
L = np.array([0.35, 0.30, 0.15])          # link lengths [m]
M = np.array([1.5, 1.0, 0.5])             # link masses [kg] (at link midpoints)
G = 9.81
TAU_MAX = np.array([40.0, 12.0, 3.0])     # peak torque per joint [Nm]
ALPHA = np.array([0.05, 0.60, 2.6])       # Joule-heating coefficient [W / Nm^2]
R_TH = np.array([3.0, 3.0, 3.5])          # thermal resistance [K / W]
C_TH = np.array([400.0, 220.0, 70.0])     # heat capacity [J / K]  (wrist motor is small and fast)
T_AMB, T_MAX, MARGIN = 30.0, 80.0, 3.0
DERATE = 0.5

N_PHI = 180
HARD = (18, 25)                            # hard block (segment indices, inclusive)
PHI_GRID = np.linspace(-np.pi, np.pi, N_PHI, endpoint=False)
W_ORI, W_TAU, EPS_B = 1.0, 2e-3, 0.02


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def config_table(target, payload):
    """All configurations reaching `target` (360 tool angles x 2 elbow signs).
    Returns gravity torques (n, 3), tool angles (n,), valid mask (n,)."""
    p = np.asarray(target)
    phi = np.concatenate([PHI_GRID, PHI_GRID])
    sign = np.concatenate([np.ones(N_PHI), -np.ones(N_PHI)])
    w = p[None, :] - L[2] * np.stack([np.cos(phi), np.sin(phi)], 1)
    d2 = (w ** 2).sum(1)
    c2 = (d2 - L[0] ** 2 - L[1] ** 2) / (2 * L[0] * L[1])
    valid = np.abs(c2) <= 1.0
    q2 = sign * np.arccos(np.clip(c2, -1, 1))
    q1 = np.arctan2(w[:, 1], w[:, 0]) - np.arctan2(L[1] * np.sin(q2), L[0] + L[1] * np.cos(q2))
    a1, a2, a3 = q1, q1 + q2, phi                       # absolute link angles
    x0 = np.zeros_like(a1)
    x1 = x0 + L[0] * np.cos(a1)
    x2 = x1 + L[1] * np.cos(a2)
    x3 = x2 + L[2] * np.cos(a3)
    com = [x0 + L[0] / 2 * np.cos(a1), x1 + L[1] / 2 * np.cos(a2), x2 + L[2] / 2 * np.cos(a3)]
    joints = [x0, x1, x2]
    tau = np.zeros((phi.size, 3))
    for j in range(3):                                   # static gravity torque at joint j
        t = payload * (x3 - joints[j])
        for b in range(j, 3):
            t = t + M[b] * (com[b] - joints[j])
        tau[:, j] = G * t
    return tau, phi, valid


def make_program(seed=0, n_seg=30, seg_len=60):
    """A 30-minute program: mostly easy holds, a hard far/heavy block late in the program."""
    rng = np.random.default_rng(seed)
    segs = []
    for k in range(n_seg):
        hard = HARD[0] <= k <= HARD[1]
        r = rng.uniform(0.62, 0.70) if hard else rng.uniform(0.35, 0.55)
        ang = rng.uniform(-0.25, 0.25) if hard else rng.uniform(-0.6, 0.9)
        payload = 1.2 if hard else rng.uniform(0.3, 0.9)
        segs.append(dict(target=[r * np.cos(ang), r * np.sin(ang)], payload=payload,
                         phi_pref=float(rng.uniform(-0.2, 0.2)), dur=seg_len, hard=hard))
    return segs


def precompute(program):
    taus, oris = [], []
    for s in program:
        tau, phi, valid = config_table(s["target"], s["payload"])
        ori = wrap(phi - s["phi_pref"]) ** 2
        ori = np.where(valid & np.all(np.abs(tau) <= TAU_MAX, 1), ori, 1e6)   # infeasible -> huge
        taus.append(tau)
        oris.append(ori)
    return np.stack(taus), np.stack(oris)               # (N, n, 3), (N, n)


def fast_layer(tau, ori, b):
    """Batched argmin. tau (..., n, 3), ori (..., n), b (..., 3) -> chosen index (...)."""
    cost = W_ORI * ori + W_TAU * (tau ** 2 / (b[..., None, :] + EPS_B)).sum(-1)
    return cost.argmin(-1)


NOMINAL_PLANT = dict(alpha=ALPHA, R=R_TH, C=C_TH, T_amb=T_AMB)


def thermal_segment(T0, tau_sq, dur, plant=NOMINAL_PLANT):
    """Exact RC solution for constant torque over `dur` seconds (end temperature; monotone in time)."""
    T_ss = plant["T_amb"] + plant["alpha"] * plant["R"] * tau_sq
    return T_ss + (T0 - T_ss) * np.exp(-dur / (plant["R"] * plant["C"]))


# ---------------- slow layer: MPPI over the allowance schedule ----------------
B_LEVELS = np.array([0.0, 0.03, 0.08, 0.15, 0.3, 0.5, 0.75, 1.0])
NL = B_LEVELS.size


def quantize(b):
    return np.abs(np.asarray(b)[..., None] - B_LEVELS).argmin(-1)


def allowance_tables(taus, oris):
    """Fast-layer response on the allowance grid: per segment and per (b1,b2,b3) level combo,
    the squared torques (N, NL^3, 3) and orientation cost (N, NL^3). This is the exact version of
    the learned surrogate P_bar(b, clip), E_bar(b, clip) from the proposal."""
    g = np.stack(np.meshgrid(B_LEVELS, B_LEVELS, B_LEVELS, indexing="ij"), -1).reshape(-1, 3)
    idx = fast_layer(taus[:, None], oris[:, None], g[None])                 # (N, NL^3)
    tau_sq = np.take_along_axis(taus, idx[..., None], axis=1) ** 2          # (N, NL^3, 3)
    ori = np.take_along_axis(oris, idx, axis=1)
    return tau_sq, ori


def mppi_plan(T0, tab_tau_sq, tab_ori, durs, u_init, per_joint=True, K=1024, iters=25, rng=None,
              model=NOMINAL_PLANT):
    """MPPI over continuous allowance logits u; each sample is snapped to the grid for evaluation."""
    n_seg = tab_ori.shape[0]
    dim = 3 if per_joint else 1
    u = u_init.copy()
    best_u, best_c = u.copy(), np.inf
    seg = np.arange(n_seg)[None]
    for it in range(iters):
        sigma = 2.0 * (0.85 ** it) + 0.2
        noise = rng.normal(0, sigma, size=(K, n_seg, dim))
        noise[0] = 0.0
        U = u[None] + noise
        B = 1.0 / (1.0 + np.exp(-U))
        if not per_joint:
            B = np.repeat(B, 3, axis=-1)
        q = quantize(B)
        combo = q[..., 0] * NL * NL + q[..., 1] * NL + q[..., 2]           # (K, n_seg)
        tau_sq = tab_tau_sq[seg, combo]                                     # (K, n_seg, 3)
        ori = tab_ori[seg, combo]
        T = np.repeat(T0[None], K, 0)
        cost = np.zeros(K)
        for k in range(n_seg):
            T = thermal_segment(T, tau_sq[:, k], durs[k], model)
            over = np.maximum(T - (T_MAX - MARGIN), 0.0)
            cost += durs[k] * ori[:, k] + 5e3 * (over ** 2).sum(-1) + 1e3 * over.sum(-1)
        i = cost.argmin()
        if cost[i] < best_c:
            best_c, best_u = cost[i], U[i].copy()
        lam_k = max(np.std(cost) * 0.05, 1e-6)
        w = np.exp(-(cost - cost.min()) / lam_k)
        w /= w.sum()
        u = u + (w[:, None, None] * noise).sum(0)
    return best_u


# ---------------- closed-loop simulation (1 s thermal steps) ----------------
def simulate(program, method, seed=0, plant=None, tables=None, horizon=None, reactive=(25.0, MARGIN),
             adapt=False):
    """plant   : true thermal parameters (the planner always assumes NOMINAL_PLANT)
    tables  : planner's model of the fast layer (default: exact tables; pass a learned surrogate's tables)
    horizon : preview length in segments for "ours" (default: the whole remaining program)
    reactive: (span, offset) of the reactive baseline b = clip((T_max - offset - T) / span)
    adapt   : online bias adaptation -- after each segment, compare the planner's own temperature
              prediction with the measurement and fit a shared ambient offset (least squares)"""
    rng = np.random.default_rng(seed)
    plant = plant or NOMINAL_PLANT
    taus, oris = precompute(program)
    tab_tau_sq, tab_ori = tables if tables is not None else allowance_tables(taus, oris)
    durs = np.array([s["dur"] for s in program], float)
    T = np.full(3, plant["T_amb"])
    limit = TAU_MAX.copy()
    log = dict(T=[], ori_deg=[], tau=[], b=[], fail=[])
    first_derate, first_fail = None, None
    u_plan = None
    model = dict(NOMINAL_PLANT)
    amb_bias = 0.0
    t = 0
    for k, s in enumerate(program):
        T_pred = T.copy()                                # planner-model prediction over this segment
        if method in ("ours", "myopic", "scalar"):
            per_joint = method != "scalar"
            dim = 3 if per_joint else 1
            H = 1 if method == "myopic" else (horizon or len(program))
            H = min(H, len(program) - k)
            prev = u_plan
            u_plan = np.full((H, dim), 3.0)
            if prev is not None and prev.shape[1] == dim:       # warm start: shift, extend with last row
                m = min(H, prev.shape[0] - 1)
                if m > 0:
                    u_plan[:m] = prev[1:1 + m]
                u_plan[m:] = prev[-1]
            u_plan = mppi_plan(T, tab_tau_sq[k:k + H], tab_ori[k:k + H], durs[k:k + H],
                               u_plan, per_joint=per_joint, rng=rng, model=model)
            b_seg = B_LEVELS[quantize(1.0 / (1.0 + np.exp(-u_plan[0])))]
            if not per_joint:
                b_seg = np.repeat(b_seg, 3)
        for _ in range(int(s["dur"])):
            if method == "nominal":
                b = np.ones(3)
            elif method == "reactive":
                b = np.clip((T_MAX - reactive[1] - T) / reactive[0], 0.02, 1.0)
            else:
                b = b_seg
            ori_k = np.where(np.all(np.abs(taus[k]) <= limit, 1), oris[k], 1e6)  # respect current limits
            i = fast_layer(taus[k], ori_k, b)
            tau = taus[k][i]
            fail = bool(ori_k[i] >= 1e6)            # no reachable configuration within current torque limits
            if fail and first_fail is None:
                first_fail = t
            T = thermal_segment(T, tau ** 2, 1.0, plant)
            T_pred = thermal_segment(T_pred, tau ** 2, 1.0, model)
            if np.any(T >= T_MAX):
                if first_derate is None:
                    first_derate = t
                limit = np.where(T >= T_MAX, TAU_MAX * DERATE, limit)
            limit = np.where(T < T_MAX - 10.0, TAU_MAX, limit)            # hysteresis recovery
            log["T"].append(T.copy())
            log["ori_deg"].append(np.degrees(np.sqrt(min(oris[k][i], 1e6))))
            log["tau"].append(tau.copy())
            log["b"].append(np.array(b, float))
            log["fail"].append(fail)
            t += 1
        if adapt:                                        # residual = bias * (1 - exp(-dur / RC))
            f = 1.0 - np.exp(-s["dur"] / (R_TH * C_TH))
            r = T - T_pred
            amb_bias = 0.5 * amb_bias + 0.5 * (amb_bias + float((r * f).sum() / (f * f).sum()))
            model = dict(NOMINAL_PLANT, T_amb=T_AMB + amb_bias)
    for key in log:
        log[key] = np.array(log[key])
    hard = np.concatenate([[s["hard"]] * int(s["dur"]) for s in program])
    ori_ok = np.where(log["fail"], np.nan, log["ori_deg"])
    summary = dict(
        method=method,
        first_derate_min=None if first_derate is None else round(first_derate / 60, 2),
        first_fail_min=None if first_fail is None else round(first_fail / 60, 2),
        fail_seconds=int(log["fail"].sum()),
        mean_ori_err_deg=round(float(np.nanmean(ori_ok)), 2),
        hard_ori_err_deg=round(float(np.nanmean(ori_ok[hard])), 2),
        peak_T=[round(float(x), 1) for x in log["T"].max(0)],
    )
    return summary, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    methods = ["nominal", "reactive", "myopic", "scalar", "ours"]
    results = {m: [] for m in methods}
    logs0 = {}
    for seed in range(args.seeds):
        prog = make_program(seed)
        for m in methods:
            summ, log = simulate(prog, m, seed)
            results[m].append(summ)
            if seed == 0:
                logs0[m] = log
            print(seed, summ)
    table = {}
    for m in methods:
        rs = results[m]
        table[m] = dict(
            programs_completed=f"{sum(r['fail_seconds'] == 0 for r in rs)}/{len(rs)}",
            mean_fail_seconds=float(np.mean([r["fail_seconds"] for r in rs])),
            programs_derated=f"{sum(r['first_derate_min'] is not None for r in rs)}/{len(rs)}",
            mean_ori_err_deg=round(float(np.mean([r["mean_ori_err_deg"] for r in rs])), 2),
            hard_ori_err_deg=round(float(np.nanmean([r["hard_ori_err_deg"] for r in rs])), 2),
        )
    print(json.dumps(table, indent=2))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(table=table, per_seed=results), f, indent=2)

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        names = ["shoulder", "elbow", "wrist"]
        for j, ax in enumerate(axes):
            for m in methods:
                ax.plot(np.arange(len(logs0[m]["T"])) / 60, logs0[m]["T"][:, j], label=m, lw=1.4)
            ax.axhline(T_MAX, color="k", ls="--", lw=1)
            ax.axvspan(HARD[0], HARD[1] + 1, color="0.85", zorder=0)
            ax.set_ylabel(f"{names[j]} T [°C]")
        axes[0].legend(ncol=5, fontsize=8, loc="upper left")
        axes[0].set_title("Seed 0: motor temperatures (grey = hard far/heavy block, dashed = derating threshold)")
        axes[-1].set_xlabel("time [min]")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "temperatures_seed0.png"), dpi=130)
        print("saved", os.path.join(args.out, "temperatures_seed0.png"))


if __name__ == "__main__":
    main()
