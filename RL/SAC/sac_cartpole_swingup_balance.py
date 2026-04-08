"""
SAC for CartPole Swing-up + Balance (no gym)

This example trains a stochastic actor-critic with maximum-entropy objective.
It includes plots and an optional GIF for a rollout.
"""

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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

        denom = self.cfg.pole_half_length * (4.0 / 3.0 - (self.cfg.mass_pole * (cos_t**2)) / total_mass)
        theta_ddot = (self.cfg.g * sin_t - cos_t * temp - self.cfg.pole_damping * theta_dot) / denom

        x_ddot = temp - polemass_length * theta_ddot * cos_t / total_mass

        new_x_dot = x_dot + self.cfg.dt * x_ddot
        new_x = x + self.cfg.dt * new_x_dot
        new_theta_dot = theta_dot + self.cfg.dt * theta_ddot
        new_theta = self._wrap_angle(theta + self.cfg.dt * new_theta_dot)

        next_state = np.array([new_x, new_x_dot, new_theta, new_theta_dot], dtype=np.float32)
        return self._clamp_state(next_state)


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


class ReplayBuffer:
    def __init__(self, capacity: int = 200000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Tuple[np.ndarray, float, float, np.ndarray, float]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.stack(states)),
            torch.FloatTensor(np.array(actions)).unsqueeze(-1),
            torch.FloatTensor(np.array(rewards)).unsqueeze(-1),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class GaussianActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        action_limit: float = 1.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.action_limit = action_limit
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        if self.action_limit != 1.0:
            log_prob -= math.log(self.action_limit) * action.shape[-1]
        action = action * self.action_limit
        return action, log_prob, torch.tanh(mean) * self.action_limit


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_limit: float,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        alpha: float,
        auto_alpha: bool,
        target_entropy: float,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.target_entropy = target_entropy

        self.actor = GaussianActor(state_dim, action_dim, action_limit=action_limit).to(self.device)
        self.q1 = Critic(state_dim, action_dim).to(self.device)
        self.q2 = Critic(state_dim, action_dim).to(self.device)
        self.q1_target = Critic(state_dim, action_dim).to(self.device)
        self.q2_target = Critic(state_dim, action_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)

        if self.auto_alpha:
            self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=actor_lr)

    def select_action(self, state: np.ndarray, deterministic: bool) -> float:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                _, _, mean_action = self.actor.sample(state_t)
                action = mean_action
            else:
                action, _, _ = self.actor.sample(state_t)
        return float(action.cpu().numpy()[0, 0])

    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int) -> Tuple[float, float, float, float]:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target = rewards + self.gamma * (1.0 - dones) * q_next

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = F.smooth_l1_loss(q1_pred, target)
        q2_loss = F.smooth_l1_loss(q2_pred, target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        actions_pi, log_prob, _ = self.actor.sample(states)
        q1_pi = self.q1(states, actions_pi)
        q2_pi = self.q2(states, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_prob - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        entropy = float((-log_prob).mean().item())

        alpha_value = self.alpha
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = float(self.log_alpha.exp().item())
            alpha_value = self.alpha

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        critic_loss = float((q1_loss + q2_loss).item() * 0.5)
        return float(actor_loss.item()), critic_loss, entropy, alpha_value


def plot_training_curves(
    save_dir: Path,
    reward_history: List[float],
    actor_losses: List[float],
    critic_losses: List[float],
    entropy_hist: List[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip training plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(reward_history, color="#0072B2", linewidth=1.2)
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(actor_losses, color="#009E73", linewidth=1.2)
    axes[0, 1].set_title("Actor Loss")
    axes[0, 1].set_xlabel("Update")
    axes[0, 1].set_ylabel("alpha * log pi - Q")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(critic_losses, color="#D55E00", linewidth=1.2)
    axes[1, 0].set_title("Critic Loss")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(entropy_hist, color="#0072B2", linewidth=1.2)
    axes[1, 1].set_title("Policy Entropy (-log pi)")
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "sac_training_curves.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def rollout_trajectory(
    env: InvertedPendulumOnCart,
    agent: SACAgent,
    init_state: np.ndarray,
    horizon: int,
) -> Tuple[List[float], List[float]]:
    state = init_state.copy()
    xs = [state[0]]
    thetas = [state[2]]

    for _ in range(horizon):
        action = agent.select_action(state_features(state), deterministic=True)
        state = env.step(state, action)
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
    out_file = save_dir / "sac_trajectory.png"
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

    gif_path = save_dir / "sac_cartpole.gif"
    imageio.mimsave(gif_path, frames, duration=0.05)
    print(f"Saved: {gif_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAC for cartpole swing-up + balance")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", action="store_true")
    parser.add_argument("--target-entropy", type=float, default=-1.0)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    return parser.parse_args()


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

    cfg = CartPoleConfig()
    env = InvertedPendulumOnCart(cfg)

    agent = SACAgent(
        state_dim=5,
        action_dim=1,
        action_limit=cfg.force_max,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        target_entropy=args.target_entropy,
    )

    replay_buffer = ReplayBuffer(capacity=200000)
    reward_history: List[float] = []
    actor_losses: List[float] = []
    critic_losses: List[float] = []
    entropy_hist: List[float] = []

    global_step = 0

    print(f"Running on: {agent.device}")

    for episode in range(1, args.episodes + 1):
        if episode % 5 == 0:
            state = env.sample_near_upright_state()
        else:
            state = env.sample_swingup_state()

        episode_reward = 0.0
        done = False

        for _ in range(args.horizon):
            state_feat = state_features(state)
            if global_step < args.warmup_steps:
                action = float(np.random.uniform(-cfg.force_max, cfg.force_max))
            else:
                action = agent.select_action(state_feat, deterministic=False)

            next_state = env.step(state, action)
            reward = reward_swingup_balance(next_state)
            done = not env.is_state_valid(next_state)

            replay_buffer.push((state_feat, action, reward, state_features(next_state), float(done)))
            state = next_state
            episode_reward += reward
            global_step += 1

            if len(replay_buffer) >= args.batch_size and (global_step % args.update_every == 0):
                for _ in range(args.updates_per_step):
                    a_loss, c_loss, entropy, alpha_val = agent.train_step(replay_buffer, args.batch_size)
                    actor_losses.append(a_loss)
                    critic_losses.append(c_loss)
                    entropy_hist.append(entropy)

            if done:
                break

        reward_history.append(episode_reward)

        if episode % 25 == 0:
            avg_reward = float(np.mean(reward_history[-25:]))
            print(
                f"Episode {episode:4d} | avg_reward={avg_reward:7.2f} | "
                f"buffer={len(replay_buffer)}"
            )

    output_dir = Path("outputs")
    if not args.skip_plots:
        plot_training_curves(output_dir, reward_history, actor_losses, critic_losses, entropy_hist)

    init_state = np.array([0.0, 0.0, math.pi - 0.2, 0.0], dtype=np.float32)
    xs, thetas = rollout_trajectory(env, agent, init_state, horizon=500)
    plot_trajectory(output_dir, cfg.dt, xs, thetas)

    if not args.skip_gif:
        create_cartpole_gif(output_dir, xs, thetas)

    print("Done. SAC trained for swing-up + balance.")


if __name__ == "__main__":
    main()
