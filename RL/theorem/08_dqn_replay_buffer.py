"""
Example 08: DQN with Replay Buffer (Greedy, no gym)

Focus:
- Experience replay buffer
- Target network
- Greedy action selection (no exploration)
"""

import argparse
import math
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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


# ============================================================================
# CUSTOM CARTPOLE ENVIRONMENT (NO GYM DEPENDENCY)
# ============================================================================

class CartPoleEnv:
    """
    Minimal CartPole environment matching the classic control dynamics.

    Observation: [x, x_dot, theta, theta_dot]
    Actions: 0 = left, 1 = right
    Termination: |x| > 2.4 or |theta| > 12 degrees or max_steps reached
    Reward: 1.0 per step (like Gym CartPole-v1)
    """

    def __init__(self, max_steps: int = 500, seed: int | None = None) -> None:
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        self.x_threshold = 2.4
        self.theta_threshold_radians = 12.0 * math.pi / 180.0

        self.max_steps = max_steps
        self.np_random = np.random.RandomState(seed)
        self.state: np.ndarray | None = None
        self.steps = 0

    def seed(self, seed: int | None = None) -> None:
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.state is not None, "Call reset() before step()."
        assert action in (0, 1), "Action must be 0 (left) or 1 (right)."

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        truncated = self.steps >= self.max_steps
        reward = 1.0

        return self.state.copy(), reward, terminated, truncated, {}

    @property
    def observation_space_shape(self) -> tuple[int]:
        return (4,)

    @property
    def action_space_n(self) -> int:
        return 2


# ============================================================================
# DQN COMPONENTS
# ============================================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ReplayBuffer:
    def __init__(self, capacity: int = 50000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            torch.FloatTensor(np.stack(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(np.array(dones)),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        target_update_freq: int = 500,
        batch_size: int = 64,
        replay_capacity: int = 50000,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_current = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_network(next_states).max(dim=1)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = nn.functional.mse_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================


def train_dqn(
    episodes: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    log_interval: int,
) -> tuple[DQNAgent, List[float]]:
    env = CartPoleEnv(max_steps=500, seed=seed)
    state_dim = env.observation_space_shape[0]
    action_dim = env.action_space_n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    reward_history: List[float] = []
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            agent.train_step()

            episode_reward += reward
            state = next_state

        reward_history.append(episode_reward)

        if episode % log_interval == 0:
            avg_reward = float(np.mean(reward_history[-log_interval:]))
            print(
                f"Episode {episode:4d} | avg_reward={avg_reward:6.1f} | "
                f"buffer={len(agent.replay_buffer)}"
            )

    return agent, reward_history


def evaluate(agent: DQNAgent, episodes: int = 10, seed: int = 0) -> float:
    env = CartPoleEnv(max_steps=500, seed=seed)

    rewards: List[float] = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state

        rewards.append(ep_reward)

    return float(np.mean(rewards))


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(save_dir: Path, reward_history: List[float]) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip training plot.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
    episodes = range(1, len(reward_history) + 1)
    ax.plot(episodes, reward_history, color="#0072B2", linewidth=1.2, alpha=0.6)

    window = 50
    if len(reward_history) >= window:
        moving_avg = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(reward_history) + 1), moving_avg, color="#D55E00", linewidth=2.0)

    ax.set_title("DQN Replay Buffer (Greedy) - Episode Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)

    out_file = save_dir / "dqn_replay_buffer_rewards.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def rollout_trajectory(env: CartPoleEnv, agent: DQNAgent, num_steps: int = 500) -> dict:
    state, _ = env.reset()
    traj = {
        "x": [state[0]],
        "x_dot": [state[1]],
        "theta": [state[2]],
        "theta_dot": [state[3]],
        "actions": [],
    }

    for _ in range(num_steps):
        action = agent.select_action(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        traj["x"].append(next_state[0])
        traj["x_dot"].append(next_state[1])
        traj["theta"].append(next_state[2])
        traj["theta_dot"].append(next_state[3])
        traj["actions"].append(action)

        state = next_state
        if done:
            break

    return traj


def plot_trajectory(save_dir: Path, traj: dict, dt: float = 0.02) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip trajectory plot.")
        return

    t = [i * dt for i in range(len(traj["theta"]))]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

    axes[0, 0].plot(t, traj["theta"], color="#D55E00", linewidth=1.6)
    axes[0, 0].set_title("Pole Angle (theta)")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(t, traj["theta_dot"], color="#0072B2", linewidth=1.6)
    axes[0, 1].set_title("Pole Angular Velocity (theta_dot)")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(t, traj["x"], color="#009E73", linewidth=1.6)
    axes[1, 0].set_title("Cart Position (x)")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(t, traj["x_dot"], color="#CC79A7", linewidth=1.6)
    axes[1, 1].set_title("Cart Velocity (x_dot)")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "dqn_replay_buffer_trajectory.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def create_cartpole_animation(save_dir: Path, traj: dict) -> None:
    if not HAS_MATPLOTLIB or not HAS_IMAGEIO:
        print("matplotlib/imageio not found -> skip GIF.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    pole_len = 1.0
    cart_w = 0.4
    cart_h = 0.3
    step_size = max(1, len(traj["theta"]) // 120)

    frames = []
    for i in range(0, len(traj["theta"]), step_size):
        fig, ax = plt.subplots(figsize=(6.8, 5.0))
        x = traj["x"][i]
        theta = traj["theta"][i]

        ax.plot([-3, 3], [0, 0], "k-", linewidth=2)
        ax.fill_between([-3, 3], -0.1, 0, color="gray", alpha=0.3)

        cart_x = [x - cart_w / 2, x + cart_w / 2, x + cart_w / 2, x - cart_w / 2, x - cart_w / 2]
        cart_y = [0, 0, cart_h, cart_h, 0]
        ax.plot(cart_x, cart_y, "b-", linewidth=2)
        ax.fill(cart_x, cart_y, color="#0072B2", alpha=0.5)

        pole_x = x + pole_len * np.sin(theta)
        pole_y = cart_h + pole_len * np.cos(theta)
        ax.plot([x, pole_x], [cart_h, pole_y], "r-", linewidth=2)
        ax.scatter([pole_x], [pole_y], s=60, color="#D55E00")

        ax.set_xlim([-3, 3])
        ax.set_ylim([-0.5, 2.5])
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

    gif_path = save_dir / "dqn_replay_buffer.gif"
    imageio.mimsave(gif_path, frames, duration=0.05)
    print(f"Saved: {gif_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN with replay buffer (CartPole, no gym)")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("DQN with replay buffer (greedy, no gym)")
    save_dir = Path("outputs")
    agent, rewards = train_dqn(
        episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        log_interval=args.log_interval,
    )

    avg_reward = float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards))
    eval_reward = evaluate(agent, episodes=args.eval_episodes, seed=args.seed + 1)

    print(f"Final average reward (last window): {avg_reward:.1f}")
    print(f"Evaluation average reward: {eval_reward:.1f}")

    plot_training_curves(save_dir, rewards)
    env = CartPoleEnv(max_steps=500, seed=args.seed + 2)
    traj = rollout_trajectory(env, agent, num_steps=500)
    plot_trajectory(save_dir, traj)
    create_cartpole_animation(save_dir, traj)


if __name__ == "__main__":
    main()
