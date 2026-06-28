import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import imageio.v2 as imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# state s = [x, v], action u
# x_{t+1} = x_t + dt * v_t
# v_{t+1} = v_t + dt * u_t
class DoubleIntegratorEnv:
    def __init__(self, dt: float = 0.1, x_limit: float = 6.0, v_limit: float = 6.0, u_limit: float = 2.0):
        self.dt = dt
        self.x_limit = x_limit
        self.v_limit = v_limit
        self.u_limit = u_limit

    def step(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = s[:, 0:1]
        v = s[:, 1:2]
        u = torch.clamp(u, -self.u_limit, self.u_limit)

        x_next = x + self.dt * v
        v_next = v + self.dt * u

        x_next = torch.clamp(x_next, -self.x_limit, self.x_limit)
        v_next = torch.clamp(v_next, -self.v_limit, self.v_limit)
        return torch.cat([x_next, v_next], dim=-1)


# PD/LQR-like expert
@torch.no_grad()
def expert_policy(s: torch.Tensor, u_limit: float) -> torch.Tensor:
    x = s[:, 0:1]
    v = s[:, 1:2]
    u = -(2.2 * x + 1.6 * v)
    return torch.clamp(u, -u_limit, u_limit)


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


def train_bc(states: torch.Tensor, actions: torch.Tensor, epochs: int = 500, lr: float = 1e-3) -> nn.Module:
    model = PolicyNet(hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        pred = model(states)
        loss = F.mse_loss(pred, actions)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def make_narrow_bc_dataset(n: int, env: DoubleIntegratorEnv) -> tuple[torch.Tensor, torch.Tensor]:
    x = (torch.rand(n, 1) * 2.0 - 1.0) * 0.8
    v = (torch.rand(n, 1) * 2.0 - 1.0) * 0.6
    s = torch.cat([x, v], dim=-1)
    a = expert_policy(s, env.u_limit)
    return s, a


@torch.no_grad()
def evaluate(env: DoubleIntegratorEnv, model: nn.Module, episodes: int = 250, horizon: int = 40) -> tuple[float, float]:
    success = 0
    tail_errs = []

    for _ in range(episodes):
        x0 = (torch.rand(1, 1) * 2.0 - 1.0) * 5.0
        v0 = (torch.rand(1, 1) * 2.0 - 1.0) * 4.0
        s = torch.cat([x0, v0], dim=-1)

        traj = []
        failed = False
        for _ in range(horizon):
            u = torch.clamp(model(s), -env.u_limit, env.u_limit)
            s = env.step(s, u)
            x = s[:, 0:1]
            v = s[:, 1:2]
            err = torch.sqrt(x**2 + 0.2 * v**2).item()
            traj.append(err)
            if torch.abs(x).item() > 0.98 * env.x_limit:
                failed = True

        tail = sum(traj[-10:]) / 10.0
        tail_errs.append(tail)
        if (tail < 0.18) and (not failed):
            success += 1

    return sum(tail_errs) / len(tail_errs), success / episodes


@torch.no_grad()
def rollout(env: DoubleIntegratorEnv, model: nn.Module, s0: torch.Tensor, horizon: int):
    s = s0.clone()
    xs = [s[:, 0].item()]
    vs = [s[:, 1].item()]
    us = []
    for _ in range(horizon):
        u = torch.clamp(model(s), -env.u_limit, env.u_limit)
        us.append(u.item())
        s = env.step(s, u)
        xs.append(s[:, 0].item())
        vs.append(s[:, 1].item())
    return xs, vs, us


def dagger(
    env: DoubleIntegratorEnv,
    base_states: torch.Tensor,
    base_actions: torch.Tensor,
    iterations: int,
    rollouts_per_iter: int,
    horizon: int,
) -> tuple[nn.Module, list[tuple[int, float, float]]]:
    states = base_states.clone()
    actions = base_actions.clone()

    model = train_bc(states, actions, epochs=500)
    metrics = []

    for it in range(1, iterations + 1):
        beta = max(0.10, 0.70 ** it)
        new_states = []

        for _ in range(rollouts_per_iter):
            x0 = (torch.rand(1, 1) * 2.0 - 1.0) * 5.0
            v0 = (torch.rand(1, 1) * 2.0 - 1.0) * 4.0
            s = torch.cat([x0, v0], dim=-1)

            for _ in range(horizon):
                u_model = torch.clamp(model(s), -env.u_limit, env.u_limit)
                u_expert = expert_policy(s, env.u_limit)
                u = u_expert if (torch.rand(1).item() < beta) else u_model
                new_states.append(s.detach().clone())
                s = env.step(s, u)

        if new_states:
            ns = torch.cat(new_states, dim=0)
            na = expert_policy(ns, env.u_limit)
            states = torch.cat([states, ns], dim=0)
            actions = torch.cat([actions, na], dim=0)

        model = train_bc(states, actions, epochs=500)
        err, sr = evaluate(env, model, episodes=250, horizon=horizon)
        metrics.append((it, err, sr))
        print(f"DAgger iter {it:2d} | beta={beta:.2f} | mean tail err={err:.4f} | success={100.0 * sr:.1f}%")

    return model, metrics


def plot_metrics(output_dir: Path, bc_metric: tuple[float, float], dagger_metrics: list[tuple[int, float, float]]):
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip metric plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    iters = [0] + [m[0] for m in dagger_metrics]
    errs = [bc_metric[0]] + [m[1] for m in dagger_metrics]
    srs = [100.0 * bc_metric[1]] + [100.0 * m[2] for m in dagger_metrics]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(iters, errs, "-o", color="#0072B2")
    ax1.set_xlabel("DAgger iteration (0 = BC)")
    ax1.set_ylabel("Mean tail error (lower better)")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(iters, srs, "-s", color="#D55E00")
    ax2.set_ylabel("Success %")
    ax2.set_ylim(0, 105)

    ax1.set_title("BC vs DAgger on Double Integrator")
    fig.tight_layout()
    out_file = output_dir / "double_integrator_bc_vs_dagger.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def make_rollout_gif(
    output_dir: Path,
    dt: float,
    bc_x: list[float],
    bc_v: list[float],
    dag_x: list[float],
    dag_v: list[float],
):
    if not HAS_MATPLOTLIB or not HAS_IMAGEIO:
        print("matplotlib/imageio missing -> skip GIF.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(bc_x), len(dag_x))
    x_lim = max(4.5, max(max(abs(x) for x in bc_x), max(abs(x) for x in dag_x)) + 0.5)
    v_lim = max(3.5, max(max(abs(v) for v in bc_v), max(abs(v) for v in dag_v)) + 0.5)

    frame_paths = []
    for i in range(n):
        t = i * dt
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

        # position strip (top-down lane)
        for ax, x, title, color in [
            (axes[0], bc_x[i], "BC", "#CC79A7"),
            (axes[1], dag_x[i], "DAgger", "#009E73"),
        ]:
            ax.axhline(0.0, color="black", linewidth=1.2)
            ax.scatter([x], [0.0], s=180, color=color)
            ax.scatter([0.0], [0.0], s=45, color="black")
            ax.set_xlim(-x_lim, x_lim)
            ax.set_ylim(-1.0, 1.0)
            ax.set_yticks([])
            ax.set_xlabel("x")
            ax.set_title(f"{title} | t={t:.2f}s")
            ax.grid(alpha=0.2)

        fig.tight_layout()
        frame_path = output_dir / f"double_integrator_frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = output_dir / "double_integrator_bc_vs_dagger.gif"
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, duration=0.04)
    print(f"Saved: {gif_path}")

    for p in frame_paths:
        p.unlink(missing_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="BC vs DAgger on a double-integrator system with GIF visualization.")
    p.add_argument("--bc-samples", type=int, default=180)
    p.add_argument("--dagger-iters", type=int, default=8)
    p.add_argument("--rollouts-per-iter", type=int, default=24)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    env = DoubleIntegratorEnv()
    output_dir = Path("outputs")

    print("Example: BC vs DAgger on double-integrator (with GIF)")

    bc_states, bc_actions = make_narrow_bc_dataset(args.bc_samples, env)
    bc_model = train_bc(bc_states, bc_actions, epochs=500)
    bc_err, bc_sr = evaluate(env, bc_model, episodes=250, horizon=args.horizon)
    print(f"BC baseline   | mean tail err = {bc_err:.4f} | success = {100.0 * bc_sr:.1f}%")

    dag_model, dag_metrics = dagger(
        env,
        bc_states,
        bc_actions,
        iterations=args.dagger_iters,
        rollouts_per_iter=args.rollouts_per_iter,
        horizon=args.horizon,
    )

    d_err, d_sr = evaluate(env, dag_model, episodes=250, horizon=args.horizon)
    print(f"Final DAgger  | mean tail err = {d_err:.4f} | success = {100.0 * d_sr:.1f}%")

    plot_metrics(output_dir, (bc_err, bc_sr), dag_metrics)

    s0 = torch.tensor([[4.8, -2.0]])
    bc_x, bc_v, _ = rollout(env, bc_model, s0, args.horizon)
    dag_x, dag_v, _ = rollout(env, dag_model, s0, args.horizon)
    make_rollout_gif(output_dir, env.dt, bc_x, bc_v, dag_x, dag_v)


if __name__ == "__main__":
    main()
