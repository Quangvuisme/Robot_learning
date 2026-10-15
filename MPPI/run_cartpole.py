"""Run MPPI on cartpole swing-up + balance, and render an animation.

    python run_cartpole.py [--quick] [--steps N] [--cpu]

The "real" plant is stepped once per control tick with the same dynamics model
(no model mismatch here -- MPPI itself is the point of the demo). At every
tick we replan from scratch with K rollouts of horizon H, exactly following
the receding-horizon MPPI algorithm.
"""

import argparse
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from cartpole import CartpoleParams, dynamics, angle_wrap
from mppi import MPPIController

OUT = os.path.join(os.path.dirname(__file__), "out")


def make_costs(params: CartpoleParams, device):
    """Running and terminal cost. theta=pi is upright (swing-up target)."""
    w_theta, w_thd, w_pos, w_vel = 8.0, 0.15, 1.2, 0.05
    w_theta_T, w_thd_T, w_pos_T = 80.0, 1.0, 10.0
    rail = params.rail_limit

    def q(x):
        pos, vel, th, thd = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        dth = angle_wrap(th - torch.pi)
        cost = (w_theta * dth ** 2 + w_thd * thd ** 2
               + w_pos * pos ** 2 + w_vel * vel ** 2)
        # soft rail barrier: keep samples from wanting to run off the track
        over = (pos.abs() - 0.9 * rail).clamp(min=0.0)
        cost = cost + 400.0 * over ** 2
        return cost

    def phi(x):
        pos, vel, th, thd = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        dth = angle_wrap(th - torch.pi)
        return (w_theta_T * dth ** 2 + w_thd_T * thd ** 2 + w_pos_T * pos ** 2)

    return q, phi


def main(steps=500, horizon=40, samples=8192, dt=0.02, device="auto",
         seed=0, sigma=3.0, lam=0.5):
    dev = ("cuda" if (device == "auto" and torch.cuda.is_available()) else
          ("cpu" if device == "auto" else device))
    torch.manual_seed(seed)

    params = CartpoleParams()
    q_cost, phi_cost = make_costs(params, dev)

    def F(x, u, dt_):
        return dynamics(x, u, dt_, params)

    ctrl = MPPIController(F, q_cost, phi_cost, nx=4, nu=1, horizon=horizon,
                          num_samples=samples, dt=dt, noise_sigma=sigma,
                          lam=lam, u_min=-params.force_limit,
                          u_max=params.force_limit, device=dev, seed=seed)

    # true initial state: pole hanging straight down, cart centred
    x = torch.tensor([0.0, 0.0, 0.0, 0.0], device=dev)

    hist_x = np.zeros((steps + 1, 4), dtype=np.float32)
    hist_u = np.zeros(steps, dtype=np.float32)
    hist_cost_spread = np.zeros(steps, dtype=np.float32)  # for diagnostics
    hist_x[0] = x.cpu().numpy()
    plan_snapshots = []   # (step, rollouts_subset, weights_subset) at a few ticks

    print(f"MPPI cartpole swing-up: device={dev}, K={samples}, H={horizon}, "
         f"dt={dt}, steps={steps}")
    t0 = time.time()
    snap_every = max(1, steps // 6)
    for k in range(steps):
        u = ctrl.command(x)
        x = F(x, u, dt)
        hist_x[k + 1] = x.cpu().numpy()
        hist_u[k] = u.item()
        hist_cost_spread[k] = (ctrl.last_weights.max() * ctrl.K).item()

        if k % snap_every == 0 or k == steps - 1:
            idx = torch.randperm(ctrl.K, device=dev)[:120]
            plan_snapshots.append((
                k,
                ctrl.last_rollouts[idx].cpu().numpy(),
                ctrl.last_weights[idx].cpu().numpy(),
            ))
        if (k + 1) % max(1, steps // 20) == 0:
            done = (k + 1) / steps
            el = time.time() - t0
            print(f"  [{100*done:5.1f}%] t={k*dt:6.2f}s  "
                  f"theta={np.degrees(angle_wrap(torch.tensor(hist_x[k+1,2])).item()):+7.2f}deg  "
                  f"eta={el*(1-done)/max(done,1e-9):5.1f}s")
    total_time = time.time() - t0
    print(f"done in {total_time:.1f}s ({1000*total_time/steps:.2f} ms/step)")

    return dict(params=params, hist_x=hist_x, hist_u=hist_u, dt=dt,
               plan_snapshots=plan_snapshots, samples=samples, horizon=horizon,
               device=dev, total_time=total_time)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    steps = args.steps or (150 if args.quick else 500)
    samples = 1024 if args.quick else 8192
    device = "cpu" if args.cpu else "auto"
    result = main(steps=steps, samples=samples, device=device)

    np.savez(os.path.join(OUT, "run_data.npz"),
             hist_x=result["hist_x"], hist_u=result["hist_u"],
             dt=result["dt"])
    print("saved", os.path.join(OUT, "run_data.npz"))

    from render import animate, use_dark_style
    use_dark_style()
    animate(result, os.path.join(OUT, "mppi_cartpole"),
           speed=1.0 if args.quick else 1.0,
           tail_frames=None)
