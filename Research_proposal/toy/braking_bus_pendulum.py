"""
Toy prototype for Idea 3 (Research_proposal/IDEA_3_humanoid_in_moving_vehicles.md).

A standing "humanoid" = inverted pendulum (point mass m at height l) with ankle torque limited by the foot
(|u| <= m g d), standing on a bus floor that brakes with a trapezoid deceleration profile
(rise r, hold 2.5 s, release r). Platform frame, theta = forward lean:

    m l^2 theta_dd = m g l sin(theta) - m a_p l cos(theta) + u

The torso IMU measures *specific force* (contact forces / mass) and angular rate. For this body the
accelerometer "tilt" equals -u/(m g l) to first order: it measures the robot's own ankle torque, and only
equals the tilt in quasi-static conditions. That is the gravito-inertial ambiguity in its sharpest form.

Controllers (same PD gains, same torque limit; they differ only in what they think "upright" is):
  C0(tau)   IMU vertical: complementary filter (gyro + accelerometer tilt, time constant tau). This is
            what "projected gravity" gives a learned policy.
  C1        kinematic vertical: ankle encoder, assumes the floor is an inertial frame (never leans).
  C2a ours  torso IMU + kinematics: a_p = R(theta) f - l(theta_dd cos - w^2 sin), reference lean atan(a_p/g)
  C2b ours  foot IMU: the planted foot moves with the floor, so it measures a_p directly
  Oracle    true a_p;   Oracle+preview  true a_p 0.3 s ahead (anticipation, e.g. a bus approaching a stop)

Output: the largest survivable peak deceleration for each rise time (a "survival map"), plus an example trace.
Pure numpy (matplotlib only for plots).
"""
import argparse
import json
import os

import numpy as np

M, L, G, D_FOOT = 60.0, 0.9, 9.81, 0.10
U_MAX = M * G * D_FOOT                         # ankle torque the foot can transmit (~59 Nm)
KP, KD = 1500.0, 350.0
DT = 0.002
HOLD, T_PRE, T_POST = 2.5, 1.0, 2.5
FALL = 0.5                                      # rad
SIG_ACC, SIG_GYRO, GYRO_BIAS, SIG_ENC = 0.10, 0.003, 0.002, 1e-4
PREVIEW = 0.3


def profile(t, A, r):
    """Platform acceleration (forward +). Braking = negative. A, r broadcast over the batch."""
    t1, t2, t3, t4 = T_PRE, T_PRE + r, T_PRE + r + HOLD, T_PRE + 2 * r + HOLD
    up = np.clip((t - t1) / r, 0, 1)
    down = np.clip((t - t3) / r, 0, 1)
    return -A * (up - down) * (t <= t4 + 1e9)


def simulate(A, r, ctrl, tau_f=0.5, seed=0, record=False):
    """Batched simulation over arrays A (m/s^2), r (s). Returns survived mask (and a trace if record)."""
    rng = np.random.default_rng(seed)
    n = A.size
    T_end = T_PRE + 2 * r.max() + HOLD + T_POST
    steps = int(T_end / DT)
    th = np.zeros(n); w = np.zeros(n); u = np.zeros(n)
    th_f = np.zeros(n)                          # complementary-filter tilt
    w_prev = np.zeros(n); alpha_f = np.zeros(n); ap_f = np.zeros(n)
    alive = np.ones(n, bool)
    trace = {k: [] for k in ("t", "a_p", "theta", "theta_ref", "u")}
    for k in range(steps):
        t = k * DT
        a_p = profile(t, A, r)
        # ---- true dynamics for the current torque -> mass acceleration -> IMU readings
        th_dd = (G * np.sin(th) - a_p * np.cos(th)) / L + u / (M * L * L)
        ax = a_p + L * (th_dd * np.cos(th) - w ** 2 * np.sin(th))
        az = L * (-th_dd * np.sin(th) - w ** 2 * np.cos(th))
        fx, fz = ax, az + G                    # specific force, world frame
        f_bx = fx * np.cos(th) - fz * np.sin(th) + rng.normal(0, SIG_ACC, n)
        f_bz = fx * np.sin(th) + fz * np.cos(th) + rng.normal(0, SIG_ACC, n)
        gyro = w + GYRO_BIAS + rng.normal(0, SIG_GYRO, n)
        th_kin = th + rng.normal(0, SIG_ENC, n)     # floor is level: ankle angle = body tilt
        # ---- estimators -> reference / measured tilt
        if ctrl == "C0":
            th_acc = np.arctan2(-f_bx, f_bz)
            th_f = th_f + gyro * DT
            th_f = th_f + (DT / tau_f) * (th_acc - th_f)
            err, th_ref = th_f, np.zeros(n)
        else:
            if ctrl == "C1":
                a_hat = np.zeros(n)
            elif ctrl == "C2a":
                alpha_f += (DT / 0.02) * ((gyro - w_prev) / DT - alpha_f)
                fwx = f_bx * np.cos(th_kin) + f_bz * np.sin(th_kin)
                raw = fwx - L * (alpha_f * np.cos(th_kin) - gyro ** 2 * np.sin(th_kin))
                ap_f += (DT / 0.05) * (raw - ap_f)
                a_hat = ap_f
            elif ctrl == "C2b":
                ap_f += (DT / 0.03) * (a_p + rng.normal(0, SIG_ACC, n) - ap_f)
                a_hat = ap_f
            elif ctrl == "oracle":
                a_hat = a_p
            elif ctrl == "preview":
                a_hat = profile(t + PREVIEW, A, r)
            th_ref = np.arctan2(a_hat, G)       # lean into the gravito-inertial vertical
            err = th_kin - th_ref
        w_prev = gyro
        u = np.clip(-KP * err - KD * gyro, -U_MAX, U_MAX)
        # ---- integrate with the new torque (semi-implicit Euler)
        th_dd = (G * np.sin(th) - a_p * np.cos(th)) / L + u / (M * L * L)
        w = w + th_dd * DT
        th = th + w * DT
        alive &= np.abs(th) < FALL
        th = np.where(alive, th, np.sign(th) * FALL)
        w = np.where(alive, w, 0.0)
        if record and k % 5 == 0:
            for key, val in (("t", t), ("a_p", a_p.copy()), ("theta", th.copy()),
                             ("theta_ref", (th - err).copy()), ("u", u.copy())):
                trace[key].append(val)
    if record:
        return alive, {k: np.array(v) for k, v in trace.items()}
    return alive


def survival_map(ctrl, tau_f, A_grid, r_grid, seeds):
    Ag, rg = np.meshgrid(A_grid, r_grid, indexing="ij")
    ok = np.ones(Ag.shape, bool)
    for s in range(seeds):
        ok &= simulate(Ag.ravel(), rg.ravel(), ctrl, tau_f, seed=s).reshape(Ag.shape)
    # largest A such that it and every smaller A survive
    best = []
    for j in range(len(r_grid)):
        col = ok[:, j]
        m = np.argmin(col) if not col.all() else len(col)
        best.append(float(A_grid[m - 1] / G) if m > 0 else 0.0)
    return best, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    A_grid = np.arange(0.0, 0.62, 0.02) * G
    r_grid = np.array([0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0])
    ctrls = [("C0 IMU-vertical tau=0.1s", "C0", 0.1), ("C0 IMU-vertical tau=0.5s", "C0", 0.5),
             ("C0 IMU-vertical tau=2s", "C0", 2.0), ("C1 kinematic-vertical", "C1", None),
             ("C2a ours: torso IMU+kinematics", "C2a", None), ("C2b ours: foot IMU", "C2b", None),
             ("oracle a_p", "oracle", None), ("oracle + 0.3 s preview", "preview", None)]
    results = {}
    for name, c, tf in ctrls:
        best, _ = survival_map(c, tf or 0.5, A_grid, r_grid, args.seeds)
        results[name] = best
        print(f"{name:34s}", " ".join(f"{b:5.2f}" for b in best))
    print("rise times [s]:                   ", " ".join(f"{r:5.2f}" for r in r_grid))
    with open(os.path.join(args.out, "idea3_survival.json"), "w") as f:
        json.dump(dict(rise_times=r_grid.tolist(), max_survivable_decel_g=results,
                       note="largest peak deceleration (in g) survived, all smaller ones also survived, "
                            f"over {args.seeds} noise seeds"), f, indent=2)

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        styles = ["--", "--", "--", ":", "-", "-", "-", "-."]
        for (name, _, _), st in zip(ctrls, styles):
            ax[0].plot(r_grid, results[name], st, marker="o", ms=3, label=name)
        ax[0].axhspan(0.1, 0.3, color="0.9", zorder=0)
        ax[0].text(1.25, 0.31, "typical bus braking 0.1-0.3 g", fontsize=8)
        ax[0].set_xlabel("braking onset (rise) time [s]")
        ax[0].set_ylabel("max survivable deceleration [g]")
        ax[0].set_title("Ankle-strategy survival map (higher is better)")
        ax[0].legend(fontsize=7)
        A1, r1 = np.array([0.15 * G]), np.array([0.75])
        for name, c, tf in [ctrls[1], ctrls[3], ctrls[4], ctrls[6]]:
            _, tr = simulate(A1, r1, c, tf or 0.5, seed=0, record=True)
            ax[1].plot(tr["t"], np.degrees(tr["theta"][:, 0]), label=name)
        ax[1].plot(tr["t"], np.degrees(np.arctan2(tr["a_p"][:, 0], G)), "k:", label="gravito-inertial vertical")
        ax[1].set_xlabel("time [s]")
        ax[1].set_ylabel("body lean [deg] (+ forward)")
        ax[1].set_title("0.15 g braking, 0.75 s onset")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(args.out, "idea3_survival_map.png")
        fig.savefig(p, dpi=130)
        print("saved", p)


if __name__ == "__main__":
    main()
