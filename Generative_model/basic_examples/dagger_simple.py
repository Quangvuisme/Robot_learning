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


# -----------------------------
# Toy control environment
# -----------------------------
# state: x (1D position)
# dynamics: x_{t+1} = x_t + dt * u_t
# goal: stabilize x -> 0
class Toy1DEnv:
    def __init__(self, dt: float = 0.2, x_limit: float = 6.0, u_limit: float = 3.0):
        self.dt = dt
        self.x_limit = x_limit
        self.u_limit = u_limit

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(u, -self.u_limit, self.u_limit)
        # Nonlinear open-loop drift makes out-of-distribution states harder.
        drift = 0.10 * (x**3)
        x_next = x + self.dt * (u + drift)
        return torch.clamp(x_next, -self.x_limit, self.x_limit)


# Piecewise expert: gentle near zero, saturated farther away.
def expert_policy(x: torch.Tensor) -> torch.Tensor:
    # Cancels nonlinear drift + linear stabilization.
    u = -(0.10 * (x**3) + 1.4 * x)
    return torch.clamp(u, -3.0, 3.0)


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_policy_supervised(states: torch.Tensor, actions: torch.Tensor, epochs: int = 400, lr: float = 1e-3):
    model = PolicyNet(hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        pred = model(states)
        loss = F.mse_loss(pred, actions)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


@torch.no_grad()
def rollout(env: Toy1DEnv, policy_fn, x0: torch.Tensor, horizon: int):
    x = x0.clone()
    xs = [x.item()]
    for _ in range(horizon):
        u = policy_fn(x)
        x = env.step(x, u)
        xs.append(x.item())
    return xs


@torch.no_grad()
def evaluate(env: Toy1DEnv, model: nn.Module, episodes: int = 200, horizon: int = 25):
    success = 0
    tail_abs = []

    for _ in range(episodes):
        x = (torch.rand(1, 1) * 2.0 - 1.0) * 3.2  # broad initial states [-3.2, 3.2]
        hist = []
        failed = False
        for _ in range(horizon):
            u = torch.clamp(model(x), -env.u_limit, env.u_limit)
            x = env.step(x, u)
            hist.append(torch.abs(x).item())
            if torch.abs(x).item() > (0.98 * env.x_limit):
                failed = True

        tail = sum(hist[-8:]) / 8.0
        tail_abs.append(tail)
        if (tail < 0.25) and (not failed):
            success += 1

    return sum(tail_abs) / len(tail_abs), success / episodes


def make_bc_dataset(n: int = 300):
    # Intentionally narrow state coverage near zero to create covariate shift.
    s = (torch.rand(n, 1) * 2.0 - 1.0) * 0.55
    a = expert_policy(s)
    return s, a


def dagger(env: Toy1DEnv, base_states: torch.Tensor, base_actions: torch.Tensor, iterations: int, rollouts_per_iter: int, horizon: int):
    states = base_states.clone()
    actions = base_actions.clone()

    model = train_policy_supervised(states, actions)

    metrics = []
    for it in range(1, iterations + 1):
        beta = max(0.10, 0.65 ** it)
        # Collect on-policy states from current learner.
        new_states = []
        for _ in range(rollouts_per_iter):
            x = (torch.rand(1, 1) * 2.0 - 1.0) * 3.2
            for _ in range(horizon):
                u_model = torch.clamp(model(x), -env.u_limit, env.u_limit)
                u_expert = expert_policy(x)
                use_expert = torch.rand(1).item() < beta
                u = u_expert if use_expert else u_model
                new_states.append(x.detach().clone())
                x = env.step(x, u)

        if new_states:
            new_states = torch.cat(new_states, dim=0)
            new_actions = expert_policy(new_states).detach()
            states = torch.cat([states, new_states], dim=0)
            actions = torch.cat([actions, new_actions], dim=0)

        model = train_policy_supervised(states, actions, epochs=500)
        err, sr = evaluate(env, model, episodes=200, horizon=horizon)
        metrics.append((it, err, sr))
        print(f"DAgger iter {it:2d} | beta={beta:.2f} | mean tail |x| = {err:.4f} | success = {100.0 * sr:.1f}%")

    return model, metrics


def maybe_plot(output_dir: Path, bc_metrics: tuple[float, float], dagger_metrics: list[tuple[int, float, float]], bc_rollout, dagger_rollout):
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # Left: improvement over DAgger iterations.
    iters = [0] + [x[0] for x in dagger_metrics]
    errs = [bc_metrics[0]] + [x[1] for x in dagger_metrics]
    srs = [100.0 * bc_metrics[1]] + [100.0 * x[2] for x in dagger_metrics]

    ax1 = axes[0]
    ax1.plot(iters, errs, "-o", color="#0072B2", label="Mean tail |x|")
    ax1.set_xlabel("DAgger iteration (0 = BC)")
    ax1.set_ylabel("Mean tail |x| (lower better)")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(iters, srs, "-s", color="#D55E00", label="Success %")
    ax2.set_ylabel("Success %")
    ax2.set_ylim(0, 105)

    ax1.set_title("BC vs DAgger on Toy 1D Control")

    # Right: one rollout comparison from same initial state.
    t = list(range(len(bc_rollout)))
    axes[1].plot(t, bc_rollout, color="#CC79A7", linewidth=2.0, label="BC rollout")
    axes[1].plot(t, dagger_rollout, color="#009E73", linewidth=2.0, label="DAgger rollout")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("x")
    axes[1].set_title("Single Rollout from Same Initial State")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    out_file = output_dir / "dagger_vs_bc_toy.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def parse_args():
    p = argparse.ArgumentParser(description="Simple DAgger example (compare with Behavior Cloning).")
    p.add_argument("--bc-samples", type=int, default=140)
    p.add_argument("--dagger-iters", type=int, default=8)
    p.add_argument("--rollouts-per-iter", type=int, default=25)
    p.add_argument("--horizon", type=int, default=35)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    env = Toy1DEnv()
    output_dir = Path("outputs")

    print("Example: Behavior Cloning vs DAgger (toy 1D control)")

    # 1) Behavior Cloning from narrow state distribution.
    bc_states, bc_actions = make_bc_dataset(args.bc_samples)
    bc_model = train_policy_supervised(bc_states, bc_actions)
    bc_err, bc_sr = evaluate(env, bc_model, episodes=200, horizon=args.horizon)
    print(f"BC baseline   | mean tail |x| = {bc_err:.4f} | success = {100.0 * bc_sr:.1f}%")

    # 2) DAgger improves by querying expert on learner-visited states.
    dagger_model, dagger_metrics = dagger(
        env,
        bc_states,
        bc_actions,
        iterations=args.dagger_iters,
        rollouts_per_iter=args.rollouts_per_iter,
        horizon=args.horizon,
    )

    d_err, d_sr = evaluate(env, dagger_model, episodes=200, horizon=args.horizon)
    print(f"Final DAgger  | mean tail |x| = {d_err:.4f} | success = {100.0 * d_sr:.1f}%")

    # one rollout comparison from the same initial condition
    x0 = torch.tensor([[2.6]])
    bc_rollout = rollout(env, lambda x: torch.clamp(bc_model(x), -env.u_limit, env.u_limit), x0, args.horizon)
    d_rollout = rollout(env, lambda x: torch.clamp(dagger_model(x), -env.u_limit, env.u_limit), x0, args.horizon)

    maybe_plot(output_dir, (bc_err, bc_sr), dagger_metrics, bc_rollout, d_rollout)


if __name__ == "__main__":
    main()
