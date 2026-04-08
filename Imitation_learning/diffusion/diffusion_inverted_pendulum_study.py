import math
import random
from dataclasses import dataclass
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


# -----------------------------
# 1) Simple inverted pendulum
# -----------------------------
@dataclass
class PendulumConfig:
    g: float = 9.81
    length: float = 1.0
    mass: float = 1.0
    damping: float = 0.05
    dt: float = 0.02
    u_max: float = 20.0


class InvertedPendulum:
    def __init__(self, cfg: PendulumConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device

    def _wrap_angle(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta + math.pi) % (2.0 * math.pi) - math.pi

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        state: [batch, 2] -> [theta, theta_dot]
        action: [batch, 1] torque in [-u_max, u_max]
        """
        theta = state[:, 0:1]
        theta_dot = state[:, 1:2]

        u = torch.clamp(action, -self.cfg.u_max, self.cfg.u_max)

        theta_ddot = (
            (self.cfg.g / self.cfg.length) * torch.sin(theta)
            + u / (self.cfg.mass * self.cfg.length * self.cfg.length)
            - self.cfg.damping * theta_dot
        )

        new_theta_dot = theta_dot + self.cfg.dt * theta_ddot
        new_theta = theta + self.cfg.dt * new_theta_dot
        new_theta = self._wrap_angle(new_theta)

        return torch.cat([new_theta, new_theta_dot], dim=-1)


# -----------------------------
# 2) Expert policy to create data
# -----------------------------
def expert_policy(state: torch.Tensor, cfg: PendulumConfig) -> torch.Tensor:
    """
    Hybrid controller:
    - Near upright: PD stabilization
    - Far from upright: simple energy pumping term
    """
    theta = (state[:, 0:1] + math.pi) % (2.0 * math.pi) - math.pi
    theta_dot = state[:, 1:2]

    # Feedback linearization around true dynamics gives a much stronger teacher.
    # theta_ddot = g/l*sin(theta) + u/(m*l^2) - d*theta_dot
    # Choose u to cancel nonlinear terms and apply stable PD in theta-space.
    kp = 10.0
    kd = 5.0
    inv_actuator = cfg.mass * (cfg.length * cfg.length)
    u = inv_actuator * (
        -(cfg.g / cfg.length) * torch.sin(theta)
        - kp * theta
        - (kd - cfg.damping) * theta_dot
    )

    return torch.clamp(u, -cfg.u_max, cfg.u_max)


def generate_dataset(
    env: InvertedPendulum,
    episodes: int = 220,
    horizon: int = 160,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    states = []
    actions = []

    for _ in range(episodes):
        # Balance-focused study case: start around upright instead of full swing-up range.
        theta0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.8
        theta_dot0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 2.5
        s = torch.cat([theta0, theta_dot0], dim=-1)

        for _ in range(horizon):
            u = expert_policy(s, env.cfg)
            states.append(s.clone())
            actions.append(u.clone())
            s = env.step(s, u)

    return torch.cat(states, dim=0), torch.cat(actions, dim=0)


# -----------------------------
# 3) Conditional diffusion for action model
# -----------------------------
class ConditionalDenoiser(nn.Module):
    def __init__(self, state_dim: int = 2, hidden: int = 128):
        super().__init__()
        # input: noisy action (1), state (2), time embedding (2)
        self.net = nn.Sequential(
            nn.Linear(1 + state_dim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def t_embed(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.sin(2.0 * math.pi * t), torch.cos(2.0 * math.pi * t)], dim=-1)

    def forward(self, noisy_action: torch.Tensor, state: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noisy_action, state, self.t_embed(t_norm)], dim=-1)
        return self.net(x)


class ActionDiffusion:
    def __init__(self, steps: int = 20, beta_start: float = 1e-4, beta_end: float = 2e-2, device: str = "cpu"):
        self.steps = steps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, a0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(a0)
        alpha_bar_t = self.alpha_bars[t_idx].unsqueeze(-1)
        at = torch.sqrt(alpha_bar_t) * a0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        return at, noise

    @torch.no_grad()
    def sample_action(
        self,
        model: nn.Module,
        state: torch.Tensor,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        batch = state.size(0)
        a = torch.randn(batch, 1, device=self.device)

        for t in reversed(range(self.steps)):
            t_norm = torch.full((batch, 1), t / (self.steps - 1), device=self.device)
            eps = model(a, state, t_norm)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]

            mean = (1.0 / torch.sqrt(alpha_t)) * (a - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps)
            if t > 0:
                a = mean + noise_scale * torch.sqrt(beta_t) * torch.randn_like(a)
            else:
                a = mean

        return a


# -----------------------------
# 4) Training + evaluation study case
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_diffusion_policy(
    model: nn.Module,
    diffusion: ActionDiffusion,
    env: InvertedPendulum,
    states: torch.Tensor,
    actions: torch.Tensor,
    device: str,
    epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    log_interval: int = 50,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[float],
    list[int],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    # Normalize states/actions to make training stable.
    s_mean = states.mean(dim=0, keepdim=True)
    s_std = states.std(dim=0, keepdim=True) + 1e-6
    a_mean = actions.mean(dim=0, keepdim=True)
    a_std = actions.std(dim=0, keepdim=True) + 1e-6

    s_norm = (states - s_mean) / s_std
    a_norm = (actions - a_mean) / a_std

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = states.size(0)
    loss_history: list[float] = []
    eval_epochs: list[int] = []
    expert_err_curve: list[float] = []
    diffusion_err_curve: list[float] = []
    expert_sr_curve: list[float] = []
    diffusion_sr_curve: list[float] = []

    for step in range(1, epochs + 1):
        idx = torch.randint(0, n, (batch_size,), device=states.device)
        s = s_norm[idx]
        a0 = a_norm[idx]

        t_idx = torch.randint(0, diffusion.steps, (batch_size,), device=states.device)
        at, noise = diffusion.q_sample(a0, t_idx)

        t_norm = (t_idx.float() / (diffusion.steps - 1)).unsqueeze(-1)
        pred_noise = model(at, s, t_norm)

        loss = F.mse_loss(pred_noise, noise)
        loss_history.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_interval == 0:
            print(f"Train step {step:4d} | loss = {loss.item():.6f}")
            e_err, e_sr = evaluate_policy(
                env,
                policy_name=f"Expert@{step}",
                episodes=8,
                horizon=100,
                policy_fn=lambda s: expert_policy(s, env.cfg),
                device=device,
                verbose=False,
            )
            d_err, d_sr = evaluate_policy(
                env,
                policy_name=f"Diffusion@{step}",
                episodes=8,
                horizon=100,
                policy_fn=lambda s: diffusion_policy_action(
                    model,
                    diffusion,
                    s,
                    s_mean,
                    s_std,
                    a_mean,
                    a_std,
                    env.cfg.u_max,
                    n_action_samples=6,
                ),
                device=device,
                verbose=False,
            )
            eval_epochs.append(step)
            expert_err_curve.append(e_err)
            diffusion_err_curve.append(d_err)
            expert_sr_curve.append(e_sr)
            diffusion_sr_curve.append(d_sr)
            print(
                f"  Eval@{step}: expert_err={e_err:.3f}, diffusion_err={d_err:.3f}, "
                f"expert_sr={100.0 * e_sr:.1f}%, diffusion_sr={100.0 * d_sr:.1f}%"
            )

    return (
        s_mean,
        s_std,
        a_mean,
        a_std,
        loss_history,
        eval_epochs,
        expert_err_curve,
        diffusion_err_curve,
        expert_sr_curve,
        diffusion_sr_curve,
    )


@torch.no_grad()
def diffusion_policy_action(
    model: nn.Module,
    diffusion: ActionDiffusion,
    state: torch.Tensor,
    s_mean: torch.Tensor,
    s_std: torch.Tensor,
    a_mean: torch.Tensor,
    a_std: torch.Tensor,
    u_max: float,
    n_action_samples: int = 1,
    sample_noise_scale: float = 0.35,
) -> torch.Tensor:
    # Average multiple sampled actions to reduce variance.
    s_norm = (state - s_mean) / s_std
    actions = []
    for _ in range(n_action_samples):
        a_norm = diffusion.sample_action(model, s_norm, noise_scale=sample_noise_scale)
        a = a_norm * a_std + a_mean
        actions.append(a)
    a = torch.stack(actions, dim=0).median(dim=0).values
    return torch.clamp(a, -u_max, u_max)


@torch.no_grad()
def evaluate_policy(
    env: InvertedPendulum,
    policy_name: str,
    episodes: int,
    horizon: int,
    policy_fn,
    device: str,
    verbose: bool = True,
) -> tuple[float, float]:
    final_abs_theta = []
    success_count = 0

    for _ in range(episodes):
        theta0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.9
        theta_dot0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 2.5
        s = torch.cat([theta0, theta_dot0], dim=-1)

        trajectory_theta = []
        for _ in range(horizon):
            u = policy_fn(s)
            s = env.step(s, u)
            trajectory_theta.append(torch.abs(s[:, 0]).item())

        avg_tail_error = sum(trajectory_theta[-40:]) / 40.0
        final_abs_theta.append(avg_tail_error)
        if avg_tail_error < 0.15:
            success_count += 1

    mean_err = sum(final_abs_theta) / len(final_abs_theta)
    success_rate = success_count / episodes

    if verbose:
        print(f"[{policy_name}] mean tail |theta| = {mean_err:.4f} rad")
        print(f"[{policy_name}] success rate      = {100.0 * success_rate:.1f}%")
    return mean_err, success_rate


@torch.no_grad()
def rollout_theta_trajectory(
    env: InvertedPendulum,
    policy_fn,
    init_state: torch.Tensor,
    horizon: int,
) -> list[float]:
    s = init_state.clone()
    theta_values = [s[:, 0].item()]
    for _ in range(horizon):
        u = policy_fn(s)
        s = env.step(s, u)
        theta_values.append(s[:, 0].item())
    return theta_values


def plot_curves(
    save_dir: Path,
    loss_history: list[float],
    eval_epochs: list[int],
    expert_err: list[float],
    diffusion_err: list[float],
    expert_sr: list[float],
    diffusion_sr: list[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plotting curves.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].plot(loss_history, color="#0072B2", linewidth=1.4)
    axes[0].set_title("Training Loss (MSE noise prediction)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(eval_epochs, expert_err, "-o", color="#009E73", label="Expert")
    axes[1].plot(eval_epochs, diffusion_err, "-o", color="#D55E00", label="Diffusion")
    axes[1].set_title("Mean Tail |theta| vs Epoch")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Radians (lower is better)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(eval_epochs, [100.0 * x for x in expert_sr], "-o", color="#009E73", label="Expert")
    axes[2].plot(eval_epochs, [100.0 * x for x in diffusion_sr], "-o", color="#D55E00", label="Diffusion")
    axes[2].set_title("Success Rate vs Epoch")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Success %")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    out_file = save_dir / "training_curves.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_theta_trajectories(
    save_dir: Path,
    dt: float,
    expert_theta: list[float],
    diffusion_theta: list[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip trajectory plot.")
        return

    t = [i * dt for i in range(len(expert_theta))]

    # Remove 2*pi jumps from wrapped angle for a smoother trajectory plot.
    def unwrap(theta_values: list[float]) -> list[float]:
        if not theta_values:
            return theta_values
        out = [theta_values[0]]
        offset = 0.0
        for i in range(1, len(theta_values)):
            delta = theta_values[i] - theta_values[i - 1]
            if delta > math.pi:
                offset -= 2.0 * math.pi
            elif delta < -math.pi:
                offset += 2.0 * math.pi
            out.append(theta_values[i] + offset)
        return out

    expert_unwrapped = unwrap(expert_theta)
    diffusion_unwrapped = unwrap(diffusion_theta)

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.plot(t, expert_unwrapped, color="#009E73", linewidth=2.0, label="Expert")
    ax.plot(t, diffusion_unwrapped, color="#D55E00", linewidth=2.0, label="Diffusion")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_title("Theta Trajectory Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("theta (rad)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_file = save_dir / "theta_trajectory.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def make_pendulum_gif(
    save_dir: Path,
    expert_theta: list[float],
    diffusion_theta: list[float],
) -> None:
    if not HAS_MATPLOTLIB or not HAS_IMAGEIO:
        print("matplotlib/imageio missing -> skip animation gif.")
        return

    length = 1.0
    frame_paths = []
    n = min(len(expert_theta), len(diffusion_theta))

    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
        for ax, th, title, color in [
            (axes[0], expert_theta[i], "Expert", "#009E73"),
            (axes[1], diffusion_theta[i], "Diffusion", "#D55E00"),
        ]:
            x = length * math.sin(th)
            y = length * math.cos(th)
            ax.plot([0, x], [0, y], color=color, linewidth=3)
            ax.scatter([x], [y], s=130, color=color)
            ax.scatter([0], [0], s=35, color="black")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.set_title(f"{title} | t={i}")

        fig.tight_layout()
        frame_path = save_dir / f"frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = save_dir / "pendulum_compare.gif"
    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, images, duration=0.03)
    print(f"Saved: {gif_path}")

    for path in frame_paths:
        path.unlink(missing_ok=True)


def main() -> None:
    set_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    print("Study case: inverted pendulum balance around upright (not full swing-up).")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir.resolve()}")

    env = InvertedPendulum(PendulumConfig(), device=device)

    print("\n[1/3] Generating expert dataset...")
    states, actions = generate_dataset(env, episodes=220, horizon=160, device=device)
    print(f"Dataset size: {states.size(0)} samples")

    print("\n[2/3] Training conditional diffusion policy...")
    model = ConditionalDenoiser(state_dim=2, hidden=128).to(device)
    diffusion = ActionDiffusion(steps=20, device=device)
    (
        s_mean,
        s_std,
        a_mean,
        a_std,
        loss_history,
        eval_epochs,
        expert_err_curve,
        diffusion_err_curve,
        expert_sr_curve,
        diffusion_sr_curve,
    ) = train_diffusion_policy(
        model,
        diffusion,
        env,
        states,
        actions,
        device,
        epochs=1000,
        batch_size=256,
        lr=1e-3,
        log_interval=200,
    )

    plot_curves(
        output_dir,
        loss_history,
        eval_epochs,
        expert_err_curve,
        diffusion_err_curve,
        expert_sr_curve,
        diffusion_sr_curve,
    )

    print("\n[3/3] Evaluating policies on random initial states...")
    evaluate_policy(
        env,
        policy_name="Expert",
        episodes=20,
        horizon=120,
        policy_fn=lambda s: expert_policy(s, env.cfg),
        device=device,
    )

    evaluate_policy(
        env,
        policy_name="Diffusion",
        episodes=20,
        horizon=120,
        policy_fn=lambda s: diffusion_policy_action(
            model,
            diffusion,
            s,
            s_mean,
            s_std,
            a_mean,
            a_std,
            env.cfg.u_max,
            n_action_samples=8,
            sample_noise_scale=0.25,
        ),
        device=device,
    )

    print("\n[Bonus] Plot theta trajectories on the same initial state...")
    init_state = torch.tensor([[0.75, 1.8]], device=device)
    expert_theta = rollout_theta_trajectory(
        env,
        lambda s: expert_policy(s, env.cfg),
        init_state,
        horizon=160,
    )
    diffusion_theta = rollout_theta_trajectory(
        env,
        lambda s: diffusion_policy_action(
            model,
            diffusion,
            s,
            s_mean,
            s_std,
            a_mean,
            a_std,
            env.cfg.u_max,
            n_action_samples=8,
            sample_noise_scale=0.25,
        ),
        init_state,
        horizon=160,
    )
    plot_theta_trajectories(output_dir, env.cfg.dt, expert_theta, diffusion_theta)
    make_pendulum_gif(output_dir, expert_theta, diffusion_theta)

    print("\nDone. This study case learns action distribution p(u | state) from expert trajectories.")


if __name__ == "__main__":
    main()
