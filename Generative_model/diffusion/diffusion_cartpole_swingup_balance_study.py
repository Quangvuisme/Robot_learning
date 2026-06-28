import argparse
import csv
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

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


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
    # Swing-up typically needs more cart travel than classic balance-only constraints.
    x_limit: float = 7.0

    # Safety bounds used to reset invalid rollouts during dataset generation.
    theta_limit: float = math.pi
    x_dot_limit: float = 10.0
    theta_dot_limit: float = 14.0


class InvertedPendulumOnCart:
    def __init__(self, cfg: CartPoleConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device

    def _wrap_angle(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta + math.pi) % (2.0 * math.pi) - math.pi

    def _clamp_state(self, state: torch.Tensor) -> torch.Tensor:
        # Do NOT clamp x for swing-up: clamping makes trajectories stick to the boundary
        # and corrupts expert data. We only clamp velocities for numerical stability.
        x = state[:, 0:1]
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

    def sample_swingup_state(self, device: str) -> torch.Tensor:
        # Swing-up study typically starts near the stable downward equilibrium.
        x0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.25
        x_dot0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.6
        theta0 = math.pi + (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.35
        theta_dot0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 1.4
        s = torch.cat([x0, x_dot0, theta0, theta_dot0], dim=-1)
        return self._clamp_state(s)

    def sample_near_upright_state(self, device: str) -> torch.Tensor:
        # Small perturbations around the upright equilibrium.
        x0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.25
        x_dot0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.8
        theta0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.35
        theta_dot0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 2.5
        s = torch.cat([x0, x_dot0, theta0, theta_dot0], dim=-1)
        return self._clamp_state(s)

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
            + polemass_length * (theta_dot**2) * sin_t
            - self.cfg.cart_damping * x_dot
        ) / total_mass

        denom = self.cfg.pole_half_length * (4.0 / 3.0 - (self.cfg.mass_pole * (cos_t**2)) / total_mass)
        theta_ddot = (self.cfg.g * sin_t - cos_t * temp - self.cfg.pole_damping * theta_dot) / denom

        x_ddot = temp - polemass_length * theta_ddot * cos_t / total_mass

        new_x_dot = x_dot + self.cfg.dt * x_ddot
        new_x = x + self.cfg.dt * new_x_dot
        new_theta_dot = theta_dot + self.cfg.dt * theta_ddot
        new_theta = self._wrap_angle(theta + self.cfg.dt * new_theta_dot)

        next_state = torch.cat([new_x, new_x_dot, new_theta, new_theta_dot], dim=-1)
        return self._clamp_state(next_state)


def balance_controller(state: torch.Tensor, cfg: CartPoleConfig) -> torch.Tensor:
    """Feedback-linearized stabilizer (upright)."""
    x = state[:, 0:1]
    x_dot = state[:, 1:2]
    theta = (state[:, 2:3] + math.pi) % (2.0 * math.pi) - math.pi
    theta_dot = state[:, 3:4]

    # Target theta_ddot via PD(+cart centering)
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

    cos_safe = torch.where(
        torch.abs(cos_t) < 0.2,
        0.2 * torch.sign(cos_t + 1e-6),
        cos_t,
    )
    temp = (cfg.g * sin_t - cfg.pole_damping * theta_dot - denom * theta_ddot_des) / cos_safe

    force = total_mass * temp - polemass_length * (theta_dot**2) * sin_t + cfg.cart_damping * x_dot
    return torch.clamp(force, -cfg.force_max, cfg.force_max)


def swingup_controller(state: torch.Tensor, cfg: CartPoleConfig) -> torch.Tensor:
    """
    Heuristic energy shaping for cart-pole swing-up.
    This is not optimal, but produces usable expert data for the diffusion study.
    """
    x = state[:, 0:1]
    x_dot = state[:, 1:2]
    theta = (state[:, 2:3] + math.pi) % (2.0 * math.pi) - math.pi
    theta_dot = state[:, 3:4]

    # Pole energy relative to downward configuration.
    # With theta=0 (upright): E = 2*m*g*l
    # With theta=pi (downward): E = 0
    m = cfg.mass_pole
    l = cfg.pole_half_length
    e_des = 2.0 * m * cfg.g * l
    e = 0.5 * m * (l**2) * (theta_dot**2) + m * cfg.g * l * (torch.cos(theta) + 1.0)

    # Continuous energy shaping.
    # Standard form: u = k * (E - E_des) * theta_dot * cos(theta)
    # (works better with our theta convention where theta=0 is upright).
    k_e = 10.0
    k_x = 0.6
    k_xd = 1.0
    thd_cos = theta_dot * torch.cos(theta)
    u_energy = k_e * (e - e_des) * thd_cos
    u_center = -(k_x * x + k_xd * x_dot)

    # If stuck near downward with low speed, apply a brief kick to get moving.
    stuck_down = (torch.abs(theta) > 2.7) & (torch.abs(theta_dot) < 0.25)
    u_kick = torch.where(stuck_down, 10.0 * torch.sign(x_dot + 1e-3), torch.zeros_like(u_energy))

    force = u_energy + u_center + u_kick
    return torch.clamp(force, -cfg.force_max, cfg.force_max)


def expert_policy(state: torch.Tensor, cfg: CartPoleConfig) -> torch.Tensor:
    """Hybrid expert: swing-up far from upright, balance near upright."""
    theta = (state[:, 2:3] + math.pi) % (2.0 * math.pi) - math.pi
    theta_dot = state[:, 3:4]

    # Switch to stabilizer early enough to catch the pole.
    near_upright = (torch.abs(theta) < 0.55) & (torch.abs(theta_dot) < 3.5)

    u_balance = balance_controller(state, cfg)
    u_swing = swingup_controller(state, cfg)

    return torch.where(near_upright, u_balance, u_swing)


def generate_dataset(
    env: InvertedPendulumOnCart,
    episodes: int,
    horizon: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = []
    actions = []

    # Swing-up learns slower than balance; if we record full long horizons,
    # most samples become near-upright stabilization and the model never
    # learns to recover from far-from-upright states.
    # Keep some near-upright samples so the diffusion policy can learn the
    # stabilizer; too aggressive downsampling often breaks balance.
    keep_near_upright_prob = 0.45
    stabilized_steps_to_reset = 160

    for _ in range(episodes):
        s = env.sample_swingup_state(device)
        stable_count = 0

        for _ in range(horizon):
            if not env.is_state_valid(s).all():
                s = env.sample_swingup_state(device)
                stable_count = 0
                continue

            theta = (s[:, 2:3] + math.pi) % (2.0 * math.pi) - math.pi
            theta_dot = s[:, 3:4]
            near_upright = (torch.abs(theta) < 0.55) & (torch.abs(theta_dot) < 3.5)

            if near_upright.all().item():
                stable_count += 1
            else:
                stable_count = 0

            u = expert_policy(s, env.cfg)

            # Downsample the easy near-upright region.
            if (not near_upright.all().item()) or (random.random() < keep_near_upright_prob):
                states.append(s.clone())
                actions.append(u.clone())

            s = env.step(s, u)

            if stable_count >= stabilized_steps_to_reset:
                s = env.sample_swingup_state(device)
                stable_count = 0

    # Add a small amount of near-upright data to make balancing more reliable.
    balance_episodes = max(40, episodes // 6)
    balance_horizon = min(260, horizon)
    for _ in range(balance_episodes):
        s = env.sample_near_upright_state(device)
        for _ in range(balance_horizon):
            if not env.is_state_valid(s).all():
                s = env.sample_near_upright_state(device)
                continue
            u = expert_policy(s, env.cfg)
            states.append(s.clone())
            actions.append(u.clone())
            s = env.step(s, u)

    if not states:
        raise RuntimeError("No expert samples were generated.")

    return torch.cat(states, dim=0), torch.cat(actions, dim=0)


class ConditionalDenoiser(nn.Module):
    def __init__(self, state_dim: int = 5, hidden: int = 256):
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
        steps: int = 28,
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
        noise_scale: float = 0.25,
        sampling_steps: int | None = None,
    ) -> torch.Tensor:
        batch = state.size(0)
        a = torch.randn(batch, 1, device=self.device)

        if sampling_steps is None or sampling_steps >= self.steps:
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
        else:
            # Strided schedule with DDIM-style updates for faster inference.
            t_lin = torch.linspace(self.steps - 1, 0, sampling_steps, device=self.device)
            t_list = t_lin.round().long().tolist()
            t_schedule = sorted(set(t_list), reverse=True)
            if t_schedule[-1] != 0:
                t_schedule.append(0)

            for i, t in enumerate(t_schedule):
                t_norm = torch.full((batch, 1), t / (self.steps - 1), device=self.device)
                eps = model(a, state, t_norm)

                alpha_bar_t = self.alpha_bars[t]
                x0_pred = (a - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

                if t == 0:
                    a = x0_pred
                    break

                t_prev = t_schedule[i + 1]
                alpha_bar_prev = self.alpha_bars[t_prev]

                # DDIM with eta=noise_scale.
                eta = float(noise_scale)
                sigma = eta * torch.sqrt(
                    torch.clamp((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t), min=0.0)
                    * torch.clamp(1.0 - (alpha_bar_t / alpha_bar_prev), min=0.0)
                )
                coef_eps = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0))
                a = torch.sqrt(alpha_bar_prev) * x0_pred + coef_eps * eps + sigma * torch.randn_like(a)

        return a


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_features(state: torch.Tensor) -> torch.Tensor:
    """Feature map used by the diffusion policy.

    Using sin/cos(theta) avoids angle wrap discontinuities and makes the
    swing-up distribution easier to model.
    """
    x = state[:, 0:1]
    x_dot = state[:, 1:2]
    theta = state[:, 2:3]
    theta_dot = state[:, 3:4]
    return torch.cat([x, x_dot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)


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
    n_action_samples: int = 12,
    sample_noise_scale: float = 0.12,
    sampling_steps: int | None = None,
    action_gain: float = 1.0,
) -> torch.Tensor:
    s_feat = state_features(state)
    s_norm = (s_feat - s_mean) / s_std
    actions = []
    for _ in range(n_action_samples):
        a_norm = diffusion.sample_action(
            model,
            s_norm,
            noise_scale=sample_noise_scale,
            sampling_steps=sampling_steps,
        )
        a = a_norm * a_std + a_mean
        actions.append(a)
    a = torch.stack(actions, dim=0).median(dim=0).values
    a = action_gain * a
    return torch.clamp(a, -force_max, force_max)


@torch.no_grad()
def diffusion_hybrid_policy_action(
    model: nn.Module,
    diffusion: ActionDiffusion,
    state: torch.Tensor,
    s_mean: torch.Tensor,
    s_std: torch.Tensor,
    a_mean: torch.Tensor,
    a_std: torch.Tensor,
    cfg: CartPoleConfig,
    n_action_samples: int = 6,
    sample_noise_scale: float = 0.22,
    sampling_steps: int | None = 8,
    action_gain: float = 1.0,
) -> torch.Tensor:
    """Hybrid policy for fast, stable evaluation.

    Uses diffusion to swing up; switches to analytic balance controller near upright
    to avoid fragile long-horizon stabilization failures.
    """
    theta = (state[:, 2:3] + math.pi) % (2.0 * math.pi) - math.pi
    theta_dot = state[:, 3:4]
    near_upright = (torch.abs(theta) < 0.55) & (torch.abs(theta_dot) < 3.5)

    u_balance = balance_controller(state, cfg)
    u_diff = diffusion_policy_action(
        model,
        diffusion,
        state,
        s_mean,
        s_std,
        a_mean,
        a_std,
        cfg.force_max,
        n_action_samples=n_action_samples,
        sample_noise_scale=sample_noise_scale,
        sampling_steps=sampling_steps,
        action_gain=action_gain,
    )
    return torch.where(near_upright, u_balance, u_diff)


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

    tail_window = 60

    for _ in range(episodes):
        s = env.sample_swingup_state(device)

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
                break

        w = min(len(theta_history), tail_window)
        tail_theta = sum(theta_history[-w:]) / float(w)
        tail_x = sum(x_history[-w:]) / float(w)
        tail_theta_errors.append(tail_theta)
        tail_x_errors.append(tail_x)

        if (tail_theta < 0.20) and (tail_x < 0.55) and (not failed):
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
    eval_episodes: int,
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
    eval_steps: list[int] = []
    expert_theta_curve: list[float] = []
    diffusion_theta_curve: list[float] = []
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
                episodes=eval_episodes,
                horizon=360,
                policy_fn=lambda st: expert_policy(st, env.cfg),
                device=device,
                verbose=False,
            )
            d_theta, _, d_sr = evaluate_policy(
                env,
                policy_name=f"Diffusion@{step}",
                episodes=eval_episodes,
                horizon=360,
                policy_fn=lambda st: diffusion_hybrid_policy_action(
                    model,
                    diffusion,
                    st,
                    s_mean,
                    s_std,
                    a_mean,
                    a_std,
                    env.cfg,
                    n_action_samples=2,
                    sample_noise_scale=0.18,
                    sampling_steps=6,
                    action_gain=1.15,
                ),
                device=device,
                verbose=False,
            )

            eval_steps.append(step)
            expert_theta_curve.append(e_theta)
            diffusion_theta_curve.append(d_theta)
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
        eval_steps,
        expert_theta_curve,
        diffusion_theta_curve,
        expert_sr_curve,
        diffusion_sr_curve,
    )


def plot_curves(
    save_dir: Path,
    loss_history: list[float],
    eval_steps: list[int],
    expert_theta: list[float],
    diffusion_theta: list[float],
    expert_sr: list[float],
    diffusion_sr: list[float],
) -> None:
    # Always save numeric curves for later plotting, even if matplotlib isn't installed.
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "training_curves_cartpole_swingup.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("step,loss,expert_tail_theta,diffusion_tail_theta,expert_success,diffusion_success\n")
        # loss_history is per-step; eval curves are sparse.
        # We write rows aligned on eval steps; loss is last seen loss at that step.
        for i, step in enumerate(eval_steps):
            loss_val = loss_history[step - 1] if 0 <= (step - 1) < len(loss_history) else float("nan")
            f.write(
                f"{step},{loss_val},{expert_theta[i]},{diffusion_theta[i]},{expert_sr[i]},{diffusion_sr[i]}\n"
            )
    print(f"Saved: {csv_path}")

    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip saving PNG plot (CSV saved instead).")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].plot(loss_history, color="#0072B2", linewidth=1.3)
    axes[0].set_title("Training Loss (MSE noise prediction)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(eval_steps, expert_theta, "-o", color="#009E73", label="Expert")
    axes[1].plot(eval_steps, diffusion_theta, "-o", color="#D55E00", label="Diffusion")
    axes[1].set_title("Mean Tail |theta| vs Epoch")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Radians (lower is better)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(eval_steps, [100.0 * x for x in expert_sr], "-o", color="#009E73", label="Expert")
    axes[2].plot(eval_steps, [100.0 * x for x in diffusion_sr], "-o", color="#D55E00", label="Diffusion")
    axes[2].set_title("Success Rate vs Epoch")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Success %")
    axes[2].set_ylim(0.0, 105.0)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    out_file = save_dir / "training_curves_cartpole_swingup.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


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
    out_file = save_dir / "cartpole_swingup_trajectory.png"
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

    n = min(len(expert_x), len(diffusion_x))
    pole_len = 0.75

    max_abs_x = 0.0
    for v in expert_x:
        max_abs_x = max(max_abs_x, abs(v))
    for v in diffusion_x:
        max_abs_x = max(max_abs_x, abs(v))
    x_lim = max(2.6, max_abs_x + 0.6)


    gif_path = save_dir / "cartpole_swingup_compare.gif"

    # Fast path: render frames directly to numpy arrays (no temporary PNG files).
    if HAS_NUMPY:
        images: list[np.ndarray] = []
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
                ax.set_xlim(-x_lim, x_lim)
                ax.set_ylim(-0.55, 1.25)
                ax.set_aspect("equal")
                ax.grid(alpha=0.2)
                ax.set_title(f"{title} | t={i}")

            fig.tight_layout()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = buf.reshape(h, w, 3)
            images.append(img)
            plt.close(fig)

        imageio.mimsave(gif_path, images, duration=0.03)
        print(f"Saved: {gif_path}")
        return

    # Fallback: disk-based frames.
    frame_paths: list[Path] = []
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
            ax.set_xlim(-x_lim, x_lim)
            ax.set_ylim(-0.55, 1.25)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.set_title(f"{title} | t={i}")

        fig.tight_layout()
        frame_path = save_dir / f"cartpole_swingup_frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, images, duration=0.03)
    print(f"Saved: {gif_path}")

    for path in frame_paths:
        path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion policy study: cart-pole swing-up + balance")
    parser.add_argument("--episodes", type=int, default=450)
    parser.add_argument("--horizon", type=int, default=520)
    parser.add_argument("--epochs", type=int, default=2200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=8,
        help="Number of reverse diffusion steps for policy sampling (smaller=faster).",
    )
    parser.add_argument(
        "--diffusion-gain",
        type=float,
        default=1.15,
        help="Scale sampled diffusion force (can help swing-up if underpowered).",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid eval (use pure diffusion everywhere).",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip saving training curves (faster).")
    parser.add_argument("--skip-bonus", action="store_true", help="Skip trajectory plot + GIF (much faster).")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on: {device}")
    print("Study case: cart-pole swing-up then balance around upright.")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir.resolve()}")

    env = InvertedPendulumOnCart(CartPoleConfig(), device=device)

    print("\n[0/3] Quick expert sanity check...")
    evaluate_policy(
        env,
        policy_name="Expert",
        episodes=30,
        horizon=420,
        policy_fn=lambda s: expert_policy(s, env.cfg),
        device=device,
        verbose=True,
    )

    print("\n[1/3] Generating expert dataset...")
    states, actions = generate_dataset(
        env,
        episodes=args.episodes,
        horizon=args.horizon,
        device=device,
    )
    print(f"Dataset size: {states.size(0)} samples")

    # Train diffusion on a smoother feature representation (sin/cos of angle).
    states_feat = state_features(states)

    print("\n[2/3] Training conditional diffusion policy...")
    model = ConditionalDenoiser(state_dim=5, hidden=256).to(device)
    diffusion = ActionDiffusion(steps=28, device=device)

    (
        s_mean,
        s_std,
        a_mean,
        a_std,
        loss_history,
        eval_steps,
        expert_theta_curve,
        diffusion_theta_curve,
        expert_sr_curve,
        diffusion_sr_curve,
    ) = train_diffusion_policy(
        model,
        diffusion,
        env,
        states_feat,
        actions,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
        eval_episodes=args.eval_episodes,
    )

    if not args.skip_plots:
        plot_curves(
            output_dir,
            loss_history,
            eval_steps,
            expert_theta_curve,
            diffusion_theta_curve,
            expert_sr_curve,
            diffusion_sr_curve,
        )

    print("\n[3/3] Evaluating policies on random initial states...")
    evaluate_policy(
        env,
        policy_name="Expert",
        episodes=args.eval_episodes,
        horizon=420,
        policy_fn=lambda s: expert_policy(s, env.cfg),
        device=device,
        verbose=True,
    )

    evaluate_policy(
        env,
        policy_name="Diffusion",
        episodes=args.eval_episodes,
        horizon=420,
        policy_fn=(
            (lambda s: diffusion_policy_action(
                model,
                diffusion,
                s,
                s_mean,
                s_std,
                a_mean,
                a_std,
                env.cfg.force_max,
                n_action_samples=6,
                sample_noise_scale=0.22,
                sampling_steps=args.sampling_steps,
                action_gain=args.diffusion_gain,
            ))
            if args.no_hybrid
            else (lambda s: diffusion_hybrid_policy_action(
                model,
                diffusion,
                s,
                s_mean,
                s_std,
                a_mean,
                a_std,
                env.cfg,
                n_action_samples=6,
                sample_noise_scale=0.22,
                sampling_steps=args.sampling_steps,
                action_gain=args.diffusion_gain,
            ))
        ),
        device=device,
        verbose=True,
    )

    if not args.skip_bonus:
        print("\n[Bonus] Plot x/theta trajectories on the same initial state...")
        init_state = torch.tensor([[0.0, 0.0, math.pi - 0.15, 0.0]], device=device)

        expert_x, expert_theta = rollout_trajectory(
            env,
            lambda st: expert_policy(st, env.cfg),
            init_state,
            horizon=520,
        )
        diffusion_x, diffusion_theta = rollout_trajectory(
            env,
            lambda st: diffusion_policy_action(
                model,
                diffusion,
                st,
                s_mean,
                s_std,
                a_mean,
                a_std,
                env.cfg.force_max,
                n_action_samples=6,
                sample_noise_scale=0.22,
                sampling_steps=args.sampling_steps,
            ),
            init_state,
            horizon=520,
        )

        plot_trajectory(output_dir, env.cfg.dt, expert_x, expert_theta, diffusion_x, diffusion_theta)
        make_cartpole_gif(output_dir, expert_x, expert_theta, diffusion_x, diffusion_theta)

    print("\nDone. This study case learns p(force | state) for swing-up + balance.")


if __name__ == "__main__":
    main()
