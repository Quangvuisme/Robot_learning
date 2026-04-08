"""
Study 03: Actor-Critic on CartPole (On-Policy vs Off-Policy)

- Uses a custom CartPole environment (no Gym dependency).
- Produces training curves, trajectory plots, and a GIF for each method.
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple

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
    import imageio.v2 as imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


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


class ActorNet(nn.Module):
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


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class QNet(nn.Module):
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
    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.data: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def add(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(transition)

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        idx = self.rng.choice(len(self.data), size=batch_size, replace=True)
        return [self.data[i] for i in idx]

    def __len__(self) -> int:
        return len(self.data)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def select_action_from_policy(actor: ActorNet, state: np.ndarray, device: torch.device) -> Tuple[int, float]:
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    logits = actor(state_t)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    return int(action.item()), float(dist.entropy().item())


def evaluate_policy(
    actor: ActorNet,
    env: CartPoleEnv,
    episodes: int,
    max_steps: int,
    device: torch.device,
) -> float:
    total_return = 0.0
    with torch.no_grad():
        for _ in range(episodes):
            state, _ = env.reset()
            episode_return = 0.0
            for _ in range(max_steps):
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                logits = actor(state_t)
                action = int(torch.argmax(logits, dim=1).item())
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                state = next_state
                if terminated or truncated:
                    break
            total_return += episode_return
    return total_return / float(episodes)


def train_on_policy_actor_critic(
    episodes: int,
    max_steps: int,
    actor_lr: float,
    critic_lr: float,
    gamma: float,
    seed: int,
    log_interval: int,
    device: torch.device,
) -> tuple[ActorNet, dict]:
    set_seed(seed)
    env = CartPoleEnv(max_steps=max_steps, seed=seed)

    state_dim = env.observation_space_shape[0]
    action_dim = env.action_space_n

    actor = ActorNet(state_dim, action_dim).to(device)
    critic = ValueNet(state_dim).to(device)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    reward_history: List[float] = []
    actor_loss_hist: List[float] = []
    critic_loss_hist: List[float] = []
    entropy_hist: List[float] = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        episode_return = 0.0

        for _ in range(max_steps):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = actor(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()

            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            episode_return += reward

            value = critic(state_t).squeeze(-1)
            with torch.no_grad():
                if done:
                    next_value = torch.tensor(0.0, device=device)
                else:
                    next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                    next_value = critic(next_state_t).squeeze(-1)
                target = reward + gamma * next_value
            td_error = target - value

            critic_loss = td_error.pow(2).mean()
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            actor_loss = -(log_prob * td_error.detach()).mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            actor_loss_hist.append(float(actor_loss.item()))
            critic_loss_hist.append(float(critic_loss.item()))
            entropy_hist.append(float(entropy.item()))

            state = next_state
            if done:
                break

        reward_history.append(float(episode_return))

        if episode % log_interval == 0:
            avg_return = float(np.mean(reward_history[-log_interval:]))
            print(f"[On-Policy] Episode {episode:4d} | avg_return={avg_return:6.1f}")

    metrics = {
        "reward": reward_history,
        "actor_loss": actor_loss_hist,
        "critic_loss": critic_loss_hist,
        "entropy": entropy_hist,
    }
    return actor, metrics


def train_off_policy_actor_critic(
    episodes: int,
    max_steps: int,
    actor_lr: float,
    critic_lr: float,
    gamma: float,
    seed: int,
    log_interval: int,
    behavior_epsilon: float,
    epsilon_decay: float,
    epsilon_min: float,
    buffer_size: int,
    batch_size: int,
    updates_per_step: int,
    tau: float,
    eval_interval: int,
    eval_episodes: int,
    device: torch.device,
) -> tuple[ActorNet, dict]:
    set_seed(seed)
    env = CartPoleEnv(max_steps=max_steps, seed=seed)
    eval_env = CartPoleEnv(max_steps=max_steps, seed=seed + 111)

    state_dim = env.observation_space_shape[0]
    action_dim = env.action_space_n

    actor = ActorNet(state_dim, action_dim).to(device)
    q_net = QNet(state_dim, action_dim).to(device)
    q_target = QNet(state_dim, action_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(q_net.parameters(), lr=critic_lr)

    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(buffer_size, rng)

    reward_history: List[float] = []
    eval_returns: List[float] = []
    eval_episodes_hist: List[int] = []
    actor_loss_hist: List[float] = []
    critic_loss_hist: List[float] = []
    entropy_hist: List[float] = []

    epsilon = behavior_epsilon

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        episode_return = 0.0

        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = int(rng.integers(action_dim))
            else:
                action, _ = select_action_from_policy(actor, state, device)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward

            buffer.add((state, action, reward, next_state, done))

            if len(buffer) >= batch_size:
                for _ in range(updates_per_step):
                    batch = buffer.sample(batch_size)
                    states = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=device)
                    actions = torch.as_tensor([b[1] for b in batch], dtype=torch.int64, device=device)
                    rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                    next_states = torch.as_tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=device)
                    dones = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=device)

                    q_pred = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        logits_next = actor(next_states)
                        pi_next = torch.softmax(logits_next, dim=1)
                        q_next = (pi_next * q_target(next_states)).sum(dim=1)
                        target = rewards + gamma * (1.0 - dones) * q_next

                    critic_loss = torch.mean((q_pred - target) ** 2)
                    critic_optim.zero_grad()
                    critic_loss.backward()
                    critic_optim.step()

                    logits = actor(states)
                    dist = torch.distributions.Categorical(logits=logits)
                    actions_pi = dist.sample()
                    log_probs = dist.log_prob(actions_pi)
                    q_actor = q_net(states).gather(1, actions_pi.unsqueeze(1)).squeeze(1)
                    actor_loss = -(log_probs * q_actor.detach()).mean()

                    actor_optim.zero_grad()
                    actor_loss.backward()
                    actor_optim.step()

                    for target_param, param in zip(q_target.parameters(), q_net.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

                    actor_loss_hist.append(float(actor_loss.item()))
                    critic_loss_hist.append(float(critic_loss.item()))
                    entropy_hist.append(float(dist.entropy().mean().item()))

            state = next_state
            if done:
                break

        reward_history.append(float(episode_return))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % eval_interval == 0:
            eval_return = evaluate_policy(actor, eval_env, eval_episodes, max_steps, device)
            eval_returns.append(float(eval_return))
            eval_episodes_hist.append(episode)
            avg_return = float(np.mean(reward_history[-eval_interval:]))
            print(
                f"[Off-Policy] Episode {episode:4d} | behavior_avg={avg_return:6.1f} "
                f"| eval_return={eval_return:6.1f} | eps={epsilon:.3f}"
            )

    metrics = {
        "reward": reward_history,
        "eval_return": eval_returns,
        "eval_episodes": eval_episodes_hist,
        "actor_loss": actor_loss_hist,
        "critic_loss": critic_loss_hist,
        "entropy": entropy_hist,
    }
    return actor, metrics


def plot_training_curves(save_dir: Path, metrics: dict, title_prefix: str) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    reward = metrics.get("reward", [])
    actor_loss = metrics.get("actor_loss", [])
    critic_loss = metrics.get("critic_loss", [])
    entropy = metrics.get("entropy", [])
    eval_return = metrics.get("eval_return", [])
    eval_episodes = metrics.get("eval_episodes", [])

    if reward:
        window = 50
        if len(reward) >= window:
            moving_avg = np.convolve(reward, np.ones(window) / window, mode="valid")
            axes[0, 0].plot(range(window, len(reward) + 1), moving_avg, color="#0072B2", linewidth=2.0)
        if eval_return and eval_episodes:
            axes[0, 0].plot(eval_episodes, eval_return, color="#D55E00", linewidth=1.6, marker="o", markersize=3)
        axes[0, 0].set_title("Episode Rewards Over Training")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.25)

    if actor_loss:
        axes[0, 1].plot(actor_loss, color="#009E73", linewidth=1.0)
        axes[0, 1].set_title("Actor Loss")
        axes[0, 1].set_xlabel("Update")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True, alpha=0.25)

    if critic_loss:
        axes[1, 0].plot(critic_loss, color="#D55E00", linewidth=1.0)
        axes[1, 0].set_title("Critic Loss")
        axes[1, 0].set_xlabel("Update")
        axes[1, 0].set_ylabel("MSE")
        axes[1, 0].grid(True, alpha=0.25)

    if entropy:
        axes[1, 1].plot(entropy, color="#0072B2", linewidth=1.0)
        axes[1, 1].set_title("Policy Entropy")
        axes[1, 1].set_xlabel("Update")
        axes[1, 1].set_ylabel("Entropy")
        axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(f"{title_prefix} Training Curves", fontsize=14)
    fig.tight_layout()
    out_file = save_dir / f"{title_prefix.lower().replace(' ', '_')}_training_curves.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def rollout_trajectory(env: CartPoleEnv, actor: ActorNet, max_steps: int, device: torch.device) -> dict:
    state, _ = env.reset()
    trajectory = {
        "x": [float(state[0])],
        "x_dot": [float(state[1])],
        "theta": [float(state[2])],
        "theta_dot": [float(state[3])],
        "actions": [],
    }

    episode_reward = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = actor(state_t)
            action = int(torch.argmax(logits, dim=1).item())
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            trajectory["x"].append(float(next_state[0]))
            trajectory["x_dot"].append(float(next_state[1]))
            trajectory["theta"].append(float(next_state[2]))
            trajectory["theta_dot"].append(float(next_state[3]))
            trajectory["actions"].append(action)

            state = next_state
            if terminated or truncated:
                break

    trajectory["episode_reward"] = float(episode_reward)
    return trajectory


def plot_trajectory(save_dir: Path, trajectory: dict, title_prefix: str, dt: float = 0.02) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plots.")
        return

    t = [i * dt for i in range(len(trajectory["theta"]))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(t, np.degrees(trajectory["theta"]), color="#D55E00", linewidth=2)
    axes[0, 0].axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    axes[0, 0].set_ylabel("Angle (degrees)")
    axes[0, 0].set_title("Pole Angle (theta)")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(t, trajectory["theta_dot"], color="#0072B2", linewidth=2)
    axes[0, 1].axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    axes[0, 1].set_ylabel("Angular Velocity (rad/s)")
    axes[0, 1].set_title("Pole Angular Velocity (theta_dot)")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(t, trajectory["x"], color="#009E73", linewidth=2)
    axes[1, 0].axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Position (m)")
    axes[1, 0].set_title("Cart Position (x)")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(t, trajectory["x_dot"], color="#CC79A7", linewidth=2)
    axes[1, 1].axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Velocity (m/s)")
    axes[1, 1].set_title("Cart Velocity (x_dot)")
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(f"{title_prefix} CartPole Trajectory", fontsize=14, y=1.02)
    fig.tight_layout()
    out_file = save_dir / f"{title_prefix.lower().replace(' ', '_')}_trajectory.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def create_cartpole_animation(save_dir: Path, trajectory: dict, title_prefix: str) -> None:
    if not HAS_MATPLOTLIB or not HAS_IMAGEIO:
        print("matplotlib or imageio not found -> skip GIF.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"{title_prefix.lower().replace(' ', '_')}_balance.gif"

    cart_width = 0.4
    cart_height = 0.3
    pole_len = 1.0

    frames = []
    step_size = max(1, len(trajectory["theta"]) // 120)

    for i in range(0, len(trajectory["theta"]), step_size):
        x = trajectory["x"][i]
        theta = trajectory["theta"][i]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Height (m)")
        ax.grid(True, alpha=0.25)

        track = plt.Rectangle((-3, -0.05), 6, 0.05, color="black", alpha=0.7)
        ax.add_patch(track)

        cart = plt.Rectangle((x - cart_width / 2, 0), cart_width, cart_height, color="#56B4E9")
        ax.add_patch(cart)

        pole_x = x
        pole_y = cart_height
        pole_end_x = pole_x + pole_len * math.sin(theta)
        pole_end_y = pole_y + pole_len * math.cos(theta)
        ax.plot([pole_x, pole_end_x], [pole_y, pole_end_y], color="red", linewidth=3)
        ax.plot(pole_end_x, pole_end_y, "o", color="#E69F00", markersize=10)

        ax.set_title(f"{title_prefix} CartPole | Step {i:04d}")

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(out_file, frames, fps=20)
    print(f"Saved: {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actor-Critic CartPole study")
    parser.add_argument("--mode", choices=["on", "off", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)

    parser.add_argument("--behavior-epsilon", type=float, default=0.3)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path("outputs")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if args.mode in ("on", "both"):
        print("\n[On-Policy Actor-Critic] Training...")
        on_actor, on_metrics = train_on_policy_actor_critic(
            episodes=args.episodes,
            max_steps=args.max_steps,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            seed=args.seed,
            log_interval=100,
            device=device,
        )
        plot_training_curves(save_dir, on_metrics, "On Policy")

        env = CartPoleEnv(max_steps=args.max_steps, seed=args.seed + 7)
        traj = rollout_trajectory(env, on_actor, args.max_steps, device)
        plot_trajectory(save_dir, traj, "On Policy")
        if not args.no_gif:
            create_cartpole_animation(save_dir, traj, "On Policy")

    if args.mode in ("off", "both"):
        print("\n[Off-Policy Actor-Critic] Training...")
        off_actor, off_metrics = train_off_policy_actor_critic(
            episodes=args.episodes,
            max_steps=args.max_steps,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            seed=args.seed,
            log_interval=100,
            behavior_epsilon=args.behavior_epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            updates_per_step=args.updates_per_step,
            tau=args.tau,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            device=device,
        )
        plot_training_curves(save_dir, off_metrics, "Off Policy")

        env = CartPoleEnv(max_steps=args.max_steps, seed=args.seed + 17)
        traj = rollout_trajectory(env, off_actor, args.max_steps, device)
        plot_trajectory(save_dir, traj, "Off Policy")
        if not args.no_gif:
            create_cartpole_animation(save_dir, traj, "Off Policy")


if __name__ == "__main__":
    main()
