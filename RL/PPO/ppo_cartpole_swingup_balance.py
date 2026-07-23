"""
PPO for CartPole Swing-up + Balance (no gym)

This example trains an on-policy Gaussian actor using Proximal Policy Optimization
(clipped surrogate objective) with a separate value baseline.
It includes plots and an optional GIF for a rollout.

Key difference from TRPO:
  - No conjugate gradient or Fisher-vector products.
  - The trust region is enforced by clipping the probability ratio.
  - Multiple minibatch epochs reuse each collected batch.
"""

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


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
    x_limit: float = 7.0
    theta_limit: float = math.pi
    x_dot_limit: float = 10.0
    theta_dot_limit: float = 14.0


class InvertedPendulumOnCart:
    def __init__(self, cfg: CartPoleConfig):
        self.cfg = cfg

    def _wrap_angle(self, theta: float) -> float:
        return (theta + math.pi) % (2.0 * math.pi) - math.pi

    def _clamp_state(self, state: np.ndarray) -> np.ndarray:
        x, x_dot, theta, theta_dot = state
        x_dot = float(np.clip(x_dot, -self.cfg.x_dot_limit, self.cfg.x_dot_limit))
        theta = self._wrap_angle(theta)
        theta_dot = float(np.clip(theta_dot, -self.cfg.theta_dot_limit, self.cfg.theta_dot_limit))
        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    def is_state_valid(self, state: np.ndarray) -> bool:
        x, x_dot, theta, theta_dot = state
        theta = self._wrap_angle(theta)
        return bool(
            abs(x) <= self.cfg.x_limit
            and abs(theta) <= self.cfg.theta_limit
            and abs(x_dot) <= self.cfg.x_dot_limit
            and abs(theta_dot) <= self.cfg.theta_dot_limit
        )

    def sample_swingup_state(self) -> np.ndarray:
        x0 = (np.random.rand() * 2.0 - 1.0) * 0.25
        x_dot0 = (np.random.rand() * 2.0 - 1.0) * 0.6
        theta0 = math.pi + (np.random.rand() * 2.0 - 1.0) * 0.35
        theta_dot0 = (np.random.rand() * 2.0 - 1.0) * 1.4
        return self._clamp_state(np.array([x0, x_dot0, theta0, theta_dot0], dtype=np.float32))

    def sample_near_upright_state(self) -> np.ndarray:
        x0 = (np.random.rand() * 2.0 - 1.0) * 0.25
        x_dot0 = (np.random.rand() * 2.0 - 1.0) * 0.8
        theta0 = (np.random.rand() * 2.0 - 1.0) * 0.35
        theta_dot0 = (np.random.rand() * 2.0 - 1.0) * 2.5
        return self._clamp_state(np.array([x0, x_dot0, theta0, theta_dot0], dtype=np.float32))

    def step(self, state: np.ndarray, action: float) -> np.ndarray:
        x, x_dot, theta, theta_dot = state
        force = float(np.clip(action, -self.cfg.force_max, self.cfg.force_max))

        total_mass = self.cfg.mass_cart + self.cfg.mass_pole
        polemass_length = self.cfg.mass_pole * self.cfg.pole_half_length

        sin_t = math.sin(theta)
        cos_t = math.cos(theta)

        temp = (
            force
            + polemass_length * (theta_dot**2) * sin_t
            - self.cfg.cart_damping * x_dot
        ) / total_mass

        denom = self.cfg.pole_half_length * (
            4.0 / 3.0 - (self.cfg.mass_pole * (cos_t**2)) / total_mass
        )
        theta_ddot = (self.cfg.g * sin_t - cos_t * temp - self.cfg.pole_damping * theta_dot) / denom

        x_ddot = temp - polemass_length * theta_ddot * cos_t / total_mass

        new_x_dot = x_dot + self.cfg.dt * x_ddot
        new_x = x + self.cfg.dt * new_x_dot
        new_theta_dot = theta_dot + self.cfg.dt * theta_ddot
        new_theta = self._wrap_angle(theta + self.cfg.dt * new_theta_dot)

        next_state = np.array([new_x, new_x_dot, new_theta, new_theta_dot], dtype=np.float32)
        return self._clamp_state(next_state)


# ---------------------------------------------------------------------------
# Features and reward
# ---------------------------------------------------------------------------


def state_features(state: np.ndarray) -> np.ndarray:
    x, x_dot, theta, theta_dot = state
    return np.array([x, x_dot, math.sin(theta), math.cos(theta), theta_dot], dtype=np.float32)


def reward_swingup_balance(state: np.ndarray) -> float:
    x, x_dot, theta, theta_dot = state
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    reward = 1.0 + math.cos(theta)
    reward -= 0.1 * (x**2)
    reward -= 0.01 * (x_dot**2 + theta_dot**2)
    if abs(theta) < 0.25 and abs(x) < 0.5:
        reward += 0.5
    return float(reward)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = torch.tanh(self.mean_head(x))
        log_std = self.log_std.clamp(-2.5, 0.25).expand_as(mean)
        return mean, log_std

    def sample_action(self, state: np.ndarray, device: torch.device) -> float:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.forward(state_t)
            std = log_std.exp()
            action = (mean + std * torch.randn_like(mean)).clamp(-1.0, 1.0)
        return float(action.cpu().numpy()[0, 0])

    def mean_action(self, state: np.ndarray, device: torch.device) -> float:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self.forward(state_t)
        return float(mean.cpu().numpy()[0, 0])

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(states)
        var = (2.0 * log_std).exp()
        log_p = -0.5 * (((actions - mean) ** 2) / (var + 1e-8) + 2.0 * log_std + math.log(2.0 * math.pi))
        return log_p.sum(dim=-1, keepdim=True)

    def entropy(self, states: torch.Tensor) -> torch.Tensor:
        _, log_std = self.forward(states)
        return (0.5 + 0.5 * math.log(2.0 * math.pi) + log_std).sum(dim=-1).mean()


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ---------------------------------------------------------------------------
# Batch collection with GAE
# ---------------------------------------------------------------------------


def collect_batch(
    env: InvertedPendulumOnCart,
    cfg: CartPoleConfig,
    policy: GaussianPolicy,
    value_net: ValueNetwork,
    device: torch.device,
    steps_per_iter: int,
    horizon: int,
    gamma: float,
    gae_lambda: float,
    start_episode_index: int,
) -> Tuple[Dict[str, np.ndarray], List[float], int]:
    trajectories = []
    episode_returns: List[float] = []
    steps_collected = 0
    episode_index = start_episode_index

    while steps_collected < steps_per_iter:
        if episode_index % 5 == 4:
            state = env.sample_near_upright_state()
        else:
            state = env.sample_swingup_state()

        states_traj: List[np.ndarray] = []
        actions_traj: List[np.ndarray] = []
        rewards_traj: List[float] = []
        next_states_traj: List[np.ndarray] = []
        dones_traj: List[float] = []
        episode_reward = 0.0

        for _ in range(horizon):
            features = state_features(state)
            action_norm = policy.sample_action(features, device)
            action_force = action_norm * cfg.force_max

            next_state = env.step(state, action_force)
            reward = reward_swingup_balance(next_state)
            done = not env.is_state_valid(next_state)

            states_traj.append(features)
            actions_traj.append(np.array([action_norm], dtype=np.float32))
            rewards_traj.append(float(reward))
            next_states_traj.append(state_features(next_state))
            dones_traj.append(float(done))

            state = next_state
            episode_reward += reward
            steps_collected += 1

            if done or steps_collected >= steps_per_iter:
                break

        trajectories.append(
            {
                "states": states_traj,
                "actions": actions_traj,
                "rewards": rewards_traj,
                "next_states": next_states_traj,
                "dones": dones_traj,
            }
        )
        episode_returns.append(float(episode_reward))
        episode_index += 1

    batch_states: List[np.ndarray] = []
    batch_actions: List[np.ndarray] = []
    batch_advantages: List[np.ndarray] = []
    batch_returns: List[np.ndarray] = []
    batch_old_log_probs: List[np.ndarray] = []

    for traj in trajectories:
        states_np = np.stack(traj["states"])
        actions_np = np.stack(traj["actions"])
        rewards_np = np.array(traj["rewards"], dtype=np.float32)
        next_states_np = np.stack(traj["next_states"])
        dones_np = np.array(traj["dones"], dtype=np.float32)

        with torch.no_grad():
            states_t = torch.as_tensor(states_np, dtype=torch.float32, device=device)
            next_states_t = torch.as_tensor(next_states_np, dtype=torch.float32, device=device)
            actions_t = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
            values = value_net(states_t).squeeze(-1).cpu().numpy()
            next_values = value_net(next_states_t).squeeze(-1).cpu().numpy()
            old_log_probs = policy.log_prob(states_t, actions_t).squeeze(-1).cpu().numpy()

        deltas = rewards_np + gamma * (1.0 - dones_np) * next_values - values
        advantages = np.zeros_like(rewards_np)
        gae = 0.0
        for t in reversed(range(len(rewards_np))):
            gae = deltas[t] + gamma * gae_lambda * (1.0 - dones_np[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        batch_states.append(states_np)
        batch_actions.append(actions_np)
        batch_advantages.append(advantages[:, None])
        batch_returns.append(returns[:, None])
        batch_old_log_probs.append(old_log_probs[:, None])

    states_out = np.concatenate(batch_states, axis=0).astype(np.float32)
    actions_out = np.concatenate(batch_actions, axis=0).astype(np.float32)
    advantages_out = np.concatenate(batch_advantages, axis=0).astype(np.float32)
    returns_out = np.concatenate(batch_returns, axis=0).astype(np.float32)
    old_log_probs_out = np.concatenate(batch_old_log_probs, axis=0).astype(np.float32)

    # Normalize advantages
    advantages_out = (advantages_out - advantages_out.mean()) / (advantages_out.std() + 1e-8)

    batch = {
        "states": states_out,
        "actions": actions_out,
        "advantages": advantages_out,
        "returns": returns_out,
        "old_log_probs": old_log_probs_out,
    }
    return batch, episode_returns, episode_index


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


def update_ppo(
    policy: GaussianPolicy,
    value_net: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_eps: float,
    ppo_epochs: int,
    minibatch_size: int,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> Tuple[float, float, float]:
    n = states.shape[0]
    indices = np.arange(n)

    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []

    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, minibatch_size):
            idx = indices[start : start + minibatch_size]
            mb_states = states[idx]
            mb_actions = actions[idx]
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]
            mb_old_log_probs = old_log_probs[idx]

            new_log_probs = policy.log_prob(mb_states, mb_actions)
            ratio = torch.exp(new_log_probs - mb_old_log_probs)

            # Clipped surrogate objective (maximize → negate for gradient descent)
            surr1 = ratio * mb_advantages
            surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_pred = value_net(mb_states)
            value_loss = F.mse_loss(value_pred, mb_returns)

            # Entropy bonus (maximize entropy → subtract from loss)
            entropy = policy.entropy(mb_states)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_net.parameters()), max_grad_norm
            )
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

    return (
        float(np.mean(policy_losses)),
        float(np.mean(value_losses)),
        float(np.mean(entropies)),
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_training_curves(
    save_dir: Path,
    reward_history: List[float],
    policy_loss_hist: List[float],
    value_loss_hist: List[float],
    entropy_hist: List[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip training plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(reward_history, color="#0072B2", linewidth=1.2)
    axes[0, 0].set_title("Average Episode Reward")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(policy_loss_hist, color="#009E73", linewidth=1.2)
    axes[0, 1].set_title("PPO Policy Loss (clipped surrogate)")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("-L_CLIP")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(value_loss_hist, color="#D55E00", linewidth=1.2)
    axes[1, 0].set_title("Value Loss")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(entropy_hist, color="#CC79A7", linewidth=1.2)
    axes[1, 1].set_title("Policy Entropy")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("H(pi)")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "ppo_training_curves.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def rollout_trajectory(
    env: InvertedPendulumOnCart,
    cfg: CartPoleConfig,
    policy: GaussianPolicy,
    device: torch.device,
    init_state: np.ndarray,
    horizon: int,
) -> Tuple[List[float], List[float]]:
    state = init_state.copy()
    xs = [state[0]]
    thetas = [state[2]]

    for _ in range(horizon):
        action_norm = policy.mean_action(state_features(state), device)
        state = env.step(state, action_norm * cfg.force_max)
        xs.append(state[0])
        thetas.append(state[2])

    return xs, thetas


def plot_trajectory(save_dir: Path, dt: float, xs: List[float], thetas: List[float]) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip trajectory plot.")
        return

    t = [i * dt for i in range(len(xs))]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(t, thetas, color="#D55E00", linewidth=1.8)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("theta (rad)")
    axes[0].set_title("Pole Angle Trajectory")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, xs, color="#009E73", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("x (m)")
    axes[1].set_title("Cart Position Trajectory")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "ppo_trajectory.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def create_cartpole_gif(save_dir: Path, xs: List[float], thetas: List[float]) -> None:
    if not HAS_MATPLOTLIB or not HAS_IMAGEIO:
        print("matplotlib/imageio not found -> skip GIF.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    pole_len = 0.75
    step_size = max(1, len(xs) // 120)

    frames = []
    for i in range(0, len(xs), step_size):
        fig, ax = plt.subplots(figsize=(6.8, 5.0))
        x = xs[i]
        theta = thetas[i]

        ax.plot([-7, 7], [0, 0], "k-", linewidth=2)
        ax.fill_between([-7, 7], -0.1, 0, color="gray", alpha=0.3)

        cart_w = 0.35
        cart_h = 0.20
        cart_x = [x - cart_w / 2, x + cart_w / 2, x + cart_w / 2, x - cart_w / 2, x - cart_w / 2]
        cart_y = [0, 0, cart_h, cart_h, 0]
        ax.plot(cart_x, cart_y, "b-", linewidth=2)
        ax.fill(cart_x, cart_y, color="#0072B2", alpha=0.5)

        pole_x = x + pole_len * math.sin(theta)
        pole_y = cart_h + pole_len * math.cos(theta)
        ax.plot([x, pole_x], [cart_h, pole_y], "r-", linewidth=2)
        ax.scatter([pole_x], [pole_y], s=70, color="#D55E00")

        ax.set_xlim([-7, 7])
        ax.set_ylim([-0.7, 1.6])
        ax.set_aspect("equal")
        ax.set_title(f"Step {i:04d}")
        ax.grid(alpha=0.2)

        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = buf.reshape(h, w, 3)
        frames.append(img)
        plt.close(fig)

    gif_path = save_dir / "ppo_cartpole.gif"
    imageio.mimsave(gif_path, frames, duration=0.05)
    print(f"Saved: {gif_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO for cartpole swing-up + balance")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--steps-per-iter", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CartPoleConfig()
    env = InvertedPendulumOnCart(cfg)

    policy = GaussianPolicy(state_dim=5, action_dim=1).to(device)
    value_net = ValueNetwork(state_dim=5).to(device)

    # PPO shares one optimizer for both actor and critic (common practice)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=args.lr
    )

    reward_history: List[float] = []
    policy_loss_hist: List[float] = []
    value_loss_hist: List[float] = []
    entropy_hist: List[float] = []
    episode_index = 0

    print(f"Running on: {device}")

    for iteration in range(1, args.iterations + 1):
        batch_np, episode_returns, episode_index = collect_batch(
            env=env,
            cfg=cfg,
            policy=policy,
            value_net=value_net,
            device=device,
            steps_per_iter=args.steps_per_iter,
            horizon=args.horizon,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            start_episode_index=episode_index,
        )

        states = torch.as_tensor(batch_np["states"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch_np["actions"], dtype=torch.float32, device=device)
        advantages = torch.as_tensor(batch_np["advantages"], dtype=torch.float32, device=device)
        returns = torch.as_tensor(batch_np["returns"], dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor(batch_np["old_log_probs"], dtype=torch.float32, device=device)

        p_loss, v_loss, entropy = update_ppo(
            policy=policy,
            value_net=value_net,
            optimizer=optimizer,
            states=states,
            actions=actions,
            advantages=advantages,
            returns=returns,
            old_log_probs=old_log_probs,
            clip_eps=args.clip_eps,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
        )

        avg_reward = float(np.mean(episode_returns)) if episode_returns else 0.0
        reward_history.append(avg_reward)
        policy_loss_hist.append(p_loss)
        value_loss_hist.append(v_loss)
        entropy_hist.append(entropy)

        if iteration % 10 == 0 or iteration == 1:
            print(
                f"Iter {iteration:4d} | avg_reward={avg_reward:8.2f} | "
                f"policy_loss={p_loss:8.4f} | value_loss={v_loss:8.4f} | "
                f"entropy={entropy:6.4f}"
            )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_plots:
        plot_training_curves(output_dir, reward_history, policy_loss_hist, value_loss_hist, entropy_hist)

    init_state = np.array([0.0, 0.0, math.pi - 0.2, 0.0], dtype=np.float32)
    xs, thetas = rollout_trajectory(env, cfg, policy, device, init_state, horizon=500)
    plot_trajectory(output_dir, cfg.dt, xs, thetas)

    if not args.skip_gif:
        create_cartpole_gif(output_dir, xs, thetas)

    print("Done. PPO trained for swing-up + balance.")


if __name__ == "__main__":
    main()
