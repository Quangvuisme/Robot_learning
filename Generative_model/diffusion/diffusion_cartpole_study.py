import argparse
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


@dataclass
class CartPoleConfig:
    g: float = 9.81
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    pole_half_length: float = 0.5
    cart_damping: float = 0.02
    pole_damping: float = 0.01
    dt: float = 0.02
    force_max: float = 30.0
    x_limit: float = 2.4
    theta_limit: float = 0.7
    x_dot_limit: float = 8.0
    theta_dot_limit: float = 12.0


class InvertedPendulumOnCart:
    def __init__(self, cfg: CartPoleConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device

    def _wrap_angle(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta + math.pi) % (2.0 * math.pi) - math.pi

    def _clamp_state(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(state[:, 0:1], -self.cfg.x_limit, self.cfg.x_limit)
        x_dot = torch.clamp(state[:, 1:2], -self.cfg.x_dot_limit, self.cfg.x_dot_limit)
        theta = self._wrap_angle(state[:, 2:3])
        theta_dot = torch.clamp(state[:, 3:4], -self.cfg.theta_dot_limit, self.cfg.theta_dot_limit)
        return torch.cat([x, x_dot, theta, theta_dot], dim=-1)

    def is_state_valid(self, state: torch.Tensor) -> torch.Tensor:
        theta = self._wrap_angle(state[:, 2:3])
        cond = (
            (torch.abs(state[:, 0:1]) <= self.cfg.x_limit)
            & (torch.abs(theta) <= self.cfg.theta_limit)
            & (torch.abs(state[:, 1:2]) <= self.cfg.x_dot_limit)
            & (torch.abs(state[:, 3:4]) <= self.cfg.theta_dot_limit)
        )
        return cond

    def sample_near_upright_state(self, device: str) -> torch.Tensor:
        x0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.18
        x_dot0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.45
        theta0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.14
        theta_dot0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.85
        return torch.cat([x0, x_dot0, theta0, theta_dot0], dim=-1)

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        state: [batch, 4] -> [x, x_dot, theta, theta_dot]
        theta = 0 means upright.
        action: [batch, 1] horizontal force in [-force_max, force_max]
        """
        x = state[:, 0:1]
        x_dot = state[:, 1:2]
        theta = state[:, 2:3]
        theta_dot = state[:, 3:4]

        force = torch.clamp(action, -self.cfg.force_max, self.cfg.force_max)

        total_mass = self.cfg.mass_cart + self.cfg.mass_pole
        polemass_length = self.cfg.mass_pole * self.cfg.pole_half_length

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        temp = (
            force
            + polemass_length * (theta_dot ** 2) * sin_t
            - self.cfg.cart_damping * x_dot
        ) / total_mass

        denom = self.cfg.pole_half_length * (
            4.0 / 3.0 - (self.cfg.mass_pole * (cos_t ** 2)) / total_mass
        )
        theta_ddot = (
            self.cfg.g * sin_t
            - cos_t * temp
            - self.cfg.pole_damping * theta_dot
        ) / denom

        x_ddot = temp - polemass_length * theta_ddot * cos_t / total_mass

        new_x_dot = x_dot + self.cfg.dt * x_ddot
        new_x = x + self.cfg.dt * new_x_dot
        new_theta_dot = theta_dot + self.cfg.dt * theta_ddot
        new_theta = self._wrap_angle(theta + self.cfg.dt * new_theta_dot)

        next_state = torch.cat([new_x, new_x_dot, new_theta, new_theta_dot], dim=-1)
        return self._clamp_state(next_state)


def expert_policy(state: torch.Tensor, cfg: CartPoleConfig) -> torch.Tensor:
    """
    Feedback-linearized controller around upright (theta=0).
    Designed specifically for this environment's equations in `InvertedPendulumOnCart.step`.
    """
    x = state[:, 0:1]
    x_dot = state[:, 1:2]
    theta = (state[:, 2:3] + math.pi) % (2.0 * math.pi) - math.pi
    theta_dot = state[:, 3:4]

    # Target angular acceleration via simple PD(+cart centering) in the linearized coordinates.
    k_th = 20.0
    k_thd = 7.0
    k_x = 2.0
    k_xd = 3.0
    theta_ddot_des = -(k_th * theta + k_thd * theta_dot + k_x * x + k_xd * x_dot)

    total_mass = cfg.mass_cart + cfg.mass_pole
    polemass_length = cfg.mass_pole * cfg.pole_half_length
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    denom = cfg.pole_half_length * (4.0 / 3.0 - (cfg.mass_pole * (cos_t**2)) / total_mass)

    # From env dynamics:
    #   theta_ddot = (g sin(theta) - cos(theta)*temp - pole_damping*theta_dot) / denom
    # Solve for temp needed to achieve theta_ddot_des.
    cos_safe = torch.where(
        torch.abs(cos_t) < 0.2,
        0.2 * torch.sign(cos_t + 1e-6),
        cos_t,
    )
    temp = (cfg.g * sin_t - cfg.pole_damping * theta_dot - denom * theta_ddot_des) / cos_safe

    # And temp = (force + polemass_length*theta_dot^2*sin(theta) - cart_damping*x_dot) / total_mass
    force = total_mass * temp - polemass_length * (theta_dot**2) * sin_t + cfg.cart_damping * x_dot

    return torch.clamp(force, -cfg.force_max, cfg.force_max)


def generate_dataset(
    env: InvertedPendulumOnCart,
    episodes: int,
    horizon: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = []
    actions = []

    for _ in range(episodes):
        s = env.sample_near_upright_state(device)

        for _ in range(horizon):
            if not env.is_state_valid(s).all():
                s = env.sample_near_upright_state(device)
                continue
            u = expert_policy(s, env.cfg)
            states.append(s.clone())
            actions.append(u.clone())
            s = env.step(s, u)

    if not states:
        raise RuntimeError("No valid expert samples were generated. Relax limits or tune expert gains.")

    return torch.cat(states, dim=0), torch.cat(actions, dim=0)


class ConditionalDenoiser(nn.Module):
    def __init__(self, state_dim: int = 4, hidden: int = 192):
        super().__init__()
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
    def __init__(
        self,
        steps: int = 24,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str = "cpu",
    ):
        self.steps = steps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(
        self,
        a0: torch.Tensor,
        t_idx: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        noise_scale: float = 0.35,
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


def set_seed(seed: int = 123) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def diffusion_policy_action(
    model: nn.Module,
    diffusion: ActionDiffusion,
    state: torch.Tensor,
    s_mean: torch.Tensor,
    s_std: torch.Tensor,
    a_mean: torch.Tensor,
    a_std: torch.Tensor,
    force_max: float,
    n_action_samples: int = 8,
    sample_noise_scale: float = 0.25,
) -> torch.Tensor:
    s_norm = (state - s_mean) / s_std
    actions = []
    for _ in range(n_action_samples):
        a_norm = diffusion.sample_action(model, s_norm, noise_scale=sample_noise_scale)
        a = a_norm * a_std + a_mean
        actions.append(a)
    a = torch.stack(actions, dim=0).median(dim=0).values
    return torch.clamp(a, -force_max, force_max)


@torch.no_grad()
def evaluate_policy(
    env: InvertedPendulumOnCart,
    policy_name: str,
    episodes: int,
    horizon: int,
    policy_fn,
    device: str,
    verbose: bool = True,
) -> tuple[float, float, float]:
    tail_theta_errors = []
    tail_x_errors = []
    success_count = 0

    for _ in range(episodes):
        s = env.sample_near_upright_state(device)

        theta_history = []
        x_history = []
        failed = False
        for _ in range(horizon):
            u = policy_fn(s)
            s = env.step(s, u)
            theta_abs = torch.abs(s[:, 2]).item()
            x_abs = torch.abs(s[:, 0]).item()
            theta_history.append(theta_abs)
            x_history.append(x_abs)
            if not env.is_state_valid(s).all().item():
                failed = True

        tail_theta = sum(theta_history[-40:]) / 40.0
        tail_x = sum(x_history[-40:]) / 40.0
        tail_theta_errors.append(tail_theta)
        tail_x_errors.append(tail_x)

        if (tail_theta < 0.10) and (tail_x < 0.20) and (not failed):
            success_count += 1

    mean_theta = sum(tail_theta_errors) / len(tail_theta_errors)
    mean_x = sum(tail_x_errors) / len(tail_x_errors)
    success_rate = success_count / episodes

    if verbose:
        print(f"[{policy_name}] mean tail |theta| = {mean_theta:.4f} rad")
        print(f"[{policy_name}] mean tail |x|     = {mean_x:.4f} m")
        print(f"[{policy_name}] success rate      = {100.0 * success_rate:.1f}%")

    return mean_theta, mean_x, success_rate


def train_diffusion_policy(
    model: nn.Module,
    diffusion: ActionDiffusion,
    env: InvertedPendulumOnCart,
    states: torch.Tensor,
    actions: torch.Tensor,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    log_interval: int,
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
    expert_curve: list[float] = []
    diffusion_curve: list[float] = []
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

            e_theta, _, e_sr = evaluate_policy(
                env,
                policy_name=f"Expert@{step}",
                episodes=8,
                horizon=120,
                policy_fn=lambda st: expert_policy(st, env.cfg),
                device=device,
                verbose=False,
            )
            d_theta, _, d_sr = evaluate_policy(
                env,
                policy_name=f"Diffusion@{step}",
                episodes=8,
                horizon=120,
                policy_fn=lambda st: diffusion_policy_action(
                    model,
                    diffusion,
                    st,
                    s_mean,
                    s_std,
                    a_mean,
                    a_std,
                    env.cfg.force_max,
                    n_action_samples=8,
                    sample_noise_scale=0.25,
                ),
                device=device,
                verbose=False,
            )

            eval_epochs.append(step)
            expert_curve.append(e_theta)
            diffusion_curve.append(d_theta)
            expert_sr_curve.append(e_sr)
            diffusion_sr_curve.append(d_sr)

            print(
                f"  Eval@{step}: expert_theta={e_theta:.3f}, diffusion_theta={d_theta:.3f}, "
                f"expert_sr={100.0 * e_sr:.1f}%, diffusion_sr={100.0 * d_sr:.1f}%"
            )

    return (
        s_mean,
        s_std,
        a_mean,
        a_std,
        loss_history,
        eval_epochs,
        expert_curve,
        diffusion_curve,
        expert_sr_curve,
        diffusion_sr_curve,
    )


@torch.no_grad()
def rollout_trajectory(
    env: InvertedPendulumOnCart,
    policy_fn,
    init_state: torch.Tensor,
    horizon: int,
) -> tuple[list[float], list[float]]:
    s = init_state.clone()
    xs = [s[:, 0].item()]
    thetas = [s[:, 2].item()]

    for _ in range(horizon):
        u = policy_fn(s)
        s = env.step(s, u)
        xs.append(s[:, 0].item())
        thetas.append(s[:, 2].item())

    return xs, thetas


def plot_curves(
    save_dir: Path,
    loss_history: list[float],
    eval_epochs: list[int],
    expert_theta: list[float],
    diffusion_theta: list[float],
    expert_sr: list[float],
    diffusion_sr: list[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plotting curves.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].plot(loss_history, color="#0072B2", linewidth=1.3)
    axes[0].set_title("Training Loss (MSE noise prediction)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(eval_epochs, expert_theta, "-o", color="#009E73", label="Expert")
    axes[1].plot(eval_epochs, diffusion_theta, "-o", color="#D55E00", label="Diffusion")
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
    axes[2].set_ylim(0.0, 105.0)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    out_file = save_dir / "training_curves_cartpole.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_trajectory(
    save_dir: Path,
    dt: float,
    expert_x: list[float],
    expert_theta: list[float],
    diffusion_x: list[float],
    diffusion_theta: list[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip trajectory plots.")
        return

    t = [i * dt for i in range(len(expert_theta))]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, expert_theta, color="#009E73", linewidth=2.0, label="Expert")
    axes[0].plot(t, diffusion_theta, color="#D55E00", linewidth=2.0, label="Diffusion")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("theta (rad)")
    axes[0].set_title("Pole Angle Trajectory")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(t, expert_x, color="#009E73", linewidth=2.0, label="Expert")
    axes[1].plot(t, diffusion_x, color="#D55E00", linewidth=2.0, label="Diffusion")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("x (m)")
    axes[1].set_title("Cart Position Trajectory")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "cartpole_trajectory.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def make_cartpole_gif(
    save_dir: Path,
    expert_x: list[float],
    expert_theta: list[float],
    diffusion_x: list[float],
    diffusion_theta: list[float],
) -> None:
    if not HAS_MATPLOTLIB or not HAS_IMAGEIO:
        print("matplotlib/imageio missing -> skip animation gif.")
        return

    frame_paths = []
    n = min(len(expert_x), len(diffusion_x))
    pole_len = 0.75

    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        for ax, x, th, title, color in [
            (axes[0], expert_x[i], expert_theta[i], "Expert", "#009E73"),
            (axes[1], diffusion_x[i], diffusion_theta[i], "Diffusion", "#D55E00"),
        ]:
            cart_w = 0.35
            cart_h = 0.20
            cart = plt.Rectangle((x - cart_w / 2.0, -cart_h / 2.0), cart_w, cart_h, color="#444444")
            ax.add_patch(cart)

            px = x + pole_len * math.sin(th)
            py = cart_h / 2.0 + pole_len * math.cos(th)
            ax.plot([x, px], [cart_h / 2.0, py], color=color, linewidth=3)
            ax.scatter([px], [py], s=110, color=color)

            ax.axhline(-0.25, color="black", linewidth=1.0)
            ax.set_xlim(-2.6, 2.6)
            ax.set_ylim(-0.55, 1.25)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.set_title(f"{title} | t={i}")

        fig.tight_layout()
        frame_path = save_dir / f"cartpole_frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = save_dir / "cartpole_compare.gif"
    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, images, duration=0.03)
    print(f"Saved: {gif_path}")

    for path in frame_paths:
        path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion policy study: inverted pendulum on cart")
    parser.add_argument("--episodes", type=int, default=260)
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on: {device}")
    print("Study case: inverted pendulum on cart, balance around upright.")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir.resolve()}")

    env = InvertedPendulumOnCart(CartPoleConfig(), device=device)

    print("\n[1/3] Generating expert dataset...")
    states, actions = generate_dataset(
        env,
        episodes=args.episodes,
        horizon=args.horizon,
        device=device,
    )
    print(f"Dataset size: {states.size(0)} samples")

    print("\n[2/3] Training conditional diffusion policy...")
    model = ConditionalDenoiser(state_dim=4, hidden=192).to(device)
    diffusion = ActionDiffusion(steps=24, device=device)

    (
        s_mean,
        s_std,
        a_mean,
        a_std,
        loss_history,
        eval_epochs,
        expert_curve,
        diffusion_curve,
        expert_sr_curve,
        diffusion_sr_curve,
    ) = train_diffusion_policy(
        model,
        diffusion,
        env,
        states,
        actions,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
    )

    plot_curves(
        output_dir,
        loss_history,
        eval_epochs,
        expert_curve,
        diffusion_curve,
        expert_sr_curve,
        diffusion_sr_curve,
    )

    print("\n[3/3] Evaluating policies on random initial states...")
    evaluate_policy(
        env,
        policy_name="Expert",
        episodes=20,
        horizon=140,
        policy_fn=lambda s: expert_policy(s, env.cfg),
        device=device,
    )

    evaluate_policy(
        env,
        policy_name="Diffusion",
        episodes=20,
        horizon=140,
        policy_fn=lambda s: diffusion_policy_action(
            model,
            diffusion,
            s,
            s_mean,
            s_std,
            a_mean,
            a_std,
            env.cfg.force_max,
            n_action_samples=8,
            sample_noise_scale=0.25,
        ),
        device=device,
    )

    print("\n[Bonus] Plot x/theta trajectories on same initial state...")
    init_state = torch.tensor([[0.3, 0.0, 0.28, 0.9]], device=device)

    expert_x, expert_theta = rollout_trajectory(
        env,
        lambda s: expert_policy(s, env.cfg),
        init_state,
        horizon=180,
    )
    diffusion_x, diffusion_theta = rollout_trajectory(
        env,
        lambda s: diffusion_policy_action(
            model,
            diffusion,
            s,
            s_mean,
            s_std,
            a_mean,
            a_std,
            env.cfg.force_max,
            n_action_samples=8,
            sample_noise_scale=0.25,
        ),
        init_state,
        horizon=180,
    )

    plot_trajectory(output_dir, env.cfg.dt, expert_x, expert_theta, diffusion_x, diffusion_theta)
    make_cartpole_gif(output_dir, expert_x, expert_theta, diffusion_x, diffusion_theta)

    print("\nDone. This study case learns p(force | state) for inverted pendulum on cart.")


if __name__ == "__main__":
    main()
