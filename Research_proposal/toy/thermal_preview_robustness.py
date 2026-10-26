"""
Iteration-3 robustness study for the thermal-preview toy (see thermal_preview_arm.py).

  1. Fair reactive baseline: grid-search (span, offset) on TRAIN programs (seeds 100-104), test on seeds 0-4.
  2. Learned surrogate: a numpy MLP maps (target, payload, preferred angle, allowance b) -> (|tau_j|, ori error),
     trained on 100 unrelated programs; the planner then uses its predictions instead of the exact tables.
  3. Model mismatch: the true plant differs from the planner's model
        params+-30% : alpha, R, C of every joint scaled by U(0.7, 1.3)
        hot day     : ambient 38 C while the planner assumes 30 C
  4. Preview-length ablation: horizon H in {1, 2, 4, 8, 16, 30} segments.
"""
import argparse
import json
import os
import time

import numpy as np

import thermal_preview_arm as A

TEST_SEEDS = range(5)
TRAIN_SEEDS = range(100, 105)


# ---------------- 1. tuned reactive baseline ----------------
def tune_reactive():
    best = None
    for span in (5.0, 10.0, 25.0, 40.0):
        for off in (3.0, 8.0, 15.0, 25.0):
            fails, oris = 0, []
            for s in TRAIN_SEEDS:
                summ, _ = A.simulate(A.make_program(s), "reactive", s, reactive=(span, off))
                fails += summ["fail_seconds"]
                oris.append(summ["mean_ori_err_deg"])
            key = (fails, float(np.mean(oris)))
            if best is None or key < best[0]:
                best = (key, (span, off))
    return best[1], best[0]


# ---------------- 2. learned surrogate (numpy MLP) ----------------
G3 = np.stack(np.meshgrid(A.B_LEVELS, A.B_LEVELS, A.B_LEVELS, indexing="ij"), -1).reshape(-1, 3)
OUT_SCALE = np.array([20.0, 8.0, 2.0, 1.0])


def seg_features(program):
    f = np.array([[s["target"][0], s["target"][1], s["payload"], s["phi_pref"]] for s in program])
    n = f.shape[0]
    X = np.concatenate([np.repeat(f[:, None], G3.shape[0], 1),
                        np.repeat(G3[None], n, 0)], -1)                     # (N, 512, 7)
    return X


def exact_targets(program):
    taus, oris = A.precompute(program)
    tsq, ori = A.allowance_tables(taus, oris)
    return np.concatenate([np.sqrt(tsq), np.sqrt(np.minimum(ori, 4.0))[..., None]], -1) / OUT_SCALE


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, x):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
            hs.append(x)
        return x, hs

    def step(self, x, y, lr=1e-3):
        out, hs = self.forward(x)
        g = 2 * (out - y) / len(x)
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.insert(0, hs[i].T @ g)
            gb.insert(0, g.sum(0))
            if i > 0:
                g = (g @ self.W[i].T) * (hs[i] > 0)
        self.t += 1
        params, grads = self.W + self.b, gW + gb
        for j, (p, gr) in enumerate(zip(params, grads)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * gr
            self.v[j] = 0.999 * self.v[j] + 0.001 * gr * gr
            mh = self.m[j] / (1 - 0.9 ** self.t)
            vh = self.v[j] / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return float(((out - y) ** 2).mean())


def train_surrogate(n_programs=100, steps=6000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([seg_features(A.make_program(1000 + i)).reshape(-1, 7) for i in range(n_programs)])
    Y = np.concatenate([exact_targets(A.make_program(1000 + i)).reshape(-1, 4) for i in range(n_programs)])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    net = MLP([7, 128, 128, 4], rng)
    for it in range(steps):
        idx = rng.integers(0, len(X), 2048)
        net.step((X[idx] - mu) / sd, Y[idx], lr=1e-3 if it < steps * 0.7 else 3e-4)
    return net, mu, sd


def surrogate_tables(net, mu, sd, program):
    X = seg_features(program)
    out, _ = net.forward((X.reshape(-1, 7) - mu) / sd)
    out = np.maximum(out.reshape(X.shape[0], X.shape[1], 4), 0) * OUT_SCALE
    return out[..., :3] ** 2, out[..., 3] ** 2


def surrogate_error(net, mu, sd):
    errs = []
    for s in TEST_SEEDS:
        p = A.make_program(s)
        X = seg_features(p).reshape(-1, 7)
        out, _ = net.forward((X - mu) / sd)
        Y = exact_targets(p).reshape(-1, 4)
        errs.append(np.abs(out - Y) * OUT_SCALE)
    e = np.concatenate(errs)
    return dict(mae_tau_Nm=[round(float(x), 3) for x in e[:, :3].mean(0)],
                mae_ori_deg=round(float(np.degrees(e[:, 3].mean())), 2))


# ---------------- 3. plants ----------------
def plant_for(cond, seed):
    if cond == "exact":
        return None
    if cond == "params+-30%":
        r = np.random.default_rng(500 + seed)
        return dict(alpha=A.ALPHA * r.uniform(0.7, 1.3, 3), R=A.R_TH * r.uniform(0.7, 1.3, 3),
                    C=A.C_TH * r.uniform(0.7, 1.3, 3), T_amb=A.T_AMB)
    if cond == "hot day (38C)":
        return dict(alpha=A.ALPHA, R=A.R_TH, C=A.C_TH, T_amb=38.0)
    raise ValueError(cond)


def summarize(rs):
    return dict(completed=f"{sum(r['fail_seconds'] == 0 for r in rs)}/{len(rs)}",
                derated=f"{sum(r['first_derate_min'] is not None for r in rs)}/{len(rs)}",
                fail_s=round(float(np.mean([r["fail_seconds"] for r in rs])), 1),
                ori_deg=round(float(np.mean([r["mean_ori_err_deg"] for r in rs])), 2),
                hard_ori_deg=round(float(np.nanmean([r["hard_ori_err_deg"] for r in rs])), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = {}
    t0 = time.time()

    reactive_params, reactive_train = tune_reactive()
    res["reactive_tuned_params(span,offset)"] = reactive_params
    print("tuned reactive:", reactive_params, "train (fail_s, ori):", reactive_train, f"{time.time()-t0:.0f}s")

    net, mu, sd = train_surrogate()
    res["surrogate_test_error"] = surrogate_error(net, mu, sd)
    print("surrogate error:", res["surrogate_test_error"], f"{time.time()-t0:.0f}s")

    res["mismatch"] = {}
    for cond in ("exact", "params+-30%", "hot day (38C)"):
        rows = {m: [] for m in ("nominal", "reactive (default)", "reactive (tuned)", "ours (exact model)",
                                "ours (learned surrogate)", "ours (exact model) + adapt", "ours (learned) + adapt")}
        for s in TEST_SEEDS:
            prog, plant = A.make_program(s), plant_for(cond, s)
            rows["nominal"].append(A.simulate(prog, "nominal", s, plant=plant)[0])
            rows["reactive (default)"].append(A.simulate(prog, "reactive", s, plant=plant)[0])
            rows["reactive (tuned)"].append(A.simulate(prog, "reactive", s, plant=plant,
                                                       reactive=reactive_params)[0])
            rows["ours (exact model)"].append(A.simulate(prog, "ours", s, plant=plant)[0])
            tabs = surrogate_tables(net, mu, sd, prog)
            rows["ours (learned surrogate)"].append(A.simulate(prog, "ours", s, plant=plant, tables=tabs)[0])
            rows["ours (exact model) + adapt"].append(A.simulate(prog, "ours", s, plant=plant, adapt=True)[0])
            rows["ours (learned) + adapt"].append(
                A.simulate(prog, "ours", s, plant=plant, tables=tabs, adapt=True)[0])
        res["mismatch"][cond] = {m: summarize(v) for m, v in rows.items()}
        print(cond, json.dumps(res["mismatch"][cond], indent=1), f"{time.time()-t0:.0f}s")

    res["preview_ablation"] = {}
    for H in (1, 2, 4, 8, 16, 30):
        rs = [A.simulate(A.make_program(s), "ours", s, horizon=H)[0] for s in TEST_SEEDS]
        res["preview_ablation"][H] = summarize(rs)
        print("H =", H, res["preview_ablation"][H], f"{time.time()-t0:.0f}s")

    with open(os.path.join(args.out, "idea2_robustness.json"), "w") as f:
        json.dump(res, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Hs = list(res["preview_ablation"].keys())
    comp = [int(res["preview_ablation"][h]["completed"].split("/")[0]) for h in Hs]
    hard = [res["preview_ablation"][h]["hard_ori_deg"] for h in Hs]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(Hs, comp, "o-", label="programs completed (of 5)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("preview horizon [minutes]")
    ax.set_ylabel("completed programs")
    ax2 = ax.twinx()
    ax2.plot(Hs, hard, "s--", color="C1", label="hard-block orientation error [deg]")
    ax2.set_ylabel("hard-block error [deg]")
    fig.legend(loc="upper center", fontsize=8, ncol=2)
    ax.set_title("How far ahead does the robot need to plan its heat?", fontsize=10, pad=24)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "idea2_preview_ablation.png"), dpi=130)
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
