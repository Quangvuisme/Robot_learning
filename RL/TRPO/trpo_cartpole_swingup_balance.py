"""
TRPO for CartPole Swing-up + Balance (no gym)

This example trains an on-policy Gaussian actor with a KL trust region and a
separate value baseline. It includes plots and an optional GIF for a rollout.
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


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = torch.tanh(self.mean(x))
        log_std = self.log_std.clamp(-2.5, 0.25).expand_as(mean)
        return mean, log_std

    def sample_action(self, state: np.ndarray, device: torch.device) -> float:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.forward(state_t)
            std = log_std.exp()
            action = mean + std * torch.randn_like(mean)
            action = action.clamp(-1.0, 1.0)
        return float(action.cpu().numpy()[0, 0])

    def mean_action(self, state: np.ndarray, device: torch.device) -> float:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self.forward(state_t)
        return float(mean.cpu().numpy()[0, 0])


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


def gaussian_log_prob(actions: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    var = torch.exp(2.0 * log_std)
    log_prob = -0.5 * (((actions - mean) ** 2) / (var + 1e-8) + 2.0 * log_std + math.log(2.0 * math.pi))
    return log_prob.sum(dim=-1, keepdim=True)


def gaussian_kl(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    old_var = torch.exp(2.0 * old_log_std)
    new_var = torch.exp(2.0 * new_log_std)
    kl = new_log_std - old_log_std
    kl += (old_var + (old_mean - new_mean) ** 2) / (2.0 * new_var + 1e-8)
    kl -= 0.5
    return kl.sum(dim=-1, keepdim=True)


def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(flat_params[offset : offset + numel].view_as(param))
        offset += numel


def flat_grad(
    outputs: torch.Tensor,
    model: nn.Module,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        outputs,
        tuple(model.parameters()),
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=False,
    )
    return torch.cat([grad.contiguous().view(-1) for grad in grads])


def conjugate_gradient(
    fisher_vector_product,
    b: torch.Tensor,
    cg_iters: int,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot_r = torch.dot(r, r)

    for _ in range(cg_iters):
        z = fisher_vector_product(p)
        denom = torch.dot(p, z) + 1e-8
        alpha = r_dot_r / denom
        x = x + alpha * p
        r = r - alpha * z
        new_r_dot_r = torch.dot(r, r)
        if new_r_dot_r <= residual_tol:
            break
        beta = new_r_dot_r / (r_dot_r + 1e-8)
        p = r + beta * p
        r_dot_r = new_r_dot_r
    return x


def surrogate_objective(
    policy: GaussianPolicy,
    states: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> torch.Tensor:
    mean, log_std = policy(states)
    log_probs = gaussian_log_prob(actions, mean, log_std)
    ratio = torch.exp(log_probs - old_log_probs)
    return (ratio * advantages).mean()


def mean_kl_divergence(
    policy: GaussianPolicy,
    states: torch.Tensor,
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
) -> torch.Tensor:
    mean, log_std = policy(states)
    return gaussian_kl(old_mean, old_log_std, mean, log_std).mean()


def fisher_vector_product(
    policy: GaussianPolicy,
    states: torch.Tensor,
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    vector: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    kl = mean_kl_divergence(policy, states, old_mean, old_log_std)
    kl_grad = flat_grad(kl, policy, retain_graph=True, create_graph=True)
    kl_grad_v = (kl_grad * vector).sum()
    hvp = flat_grad(kl_grad_v, policy, retain_graph=False, create_graph=False)
    return hvp.detach() + damping * vector


def update_policy_trpo(
    policy: GaussianPolicy,
    states: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    max_kl: float,
    cg_iters: int,
    damping: float,
    line_search_steps: int,
    backtrack_coeff: float,
) -> Tuple[float, float, bool]:
    with torch.no_grad():
        old_mean, old_log_std = policy(states)
        old_log_probs = gaussian_log_prob(actions, old_mean, old_log_std)

    objective = surrogate_objective(policy, states, actions, advantages, old_log_probs)
    grad = flat_grad(objective, policy, retain_graph=True).detach()

    if grad.norm().item() < 1e-10:
        return float(objective.item()), 0.0, False

    def fvp(vec: torch.Tensor) -> torch.Tensor:
        return fisher_vector_product(policy, states, old_mean, old_log_std, vec, damping)

    step_dir = conjugate_gradient(fvp, grad, cg_iters=cg_iters)
    fvp_step = fvp(step_dir)
    shs = 0.5 * torch.dot(step_dir, fvp_step)
    if shs.item() <= 0.0:
        return float(objective.item()), 0.0, False

    scale = torch.sqrt(torch.tensor((2.0 * max_kl) / (torch.dot(step_dir, fvp_step).item() + 1e-8)))
    full_step = step_dir * scale
    old_params = get_flat_params(policy)
    old_objective = float(objective.item())

    accepted = False
    final_objective = old_objective
    final_kl = 0.0

    for step in range(line_search_steps):
        step_frac = backtrack_coeff**step
        candidate = old_params + step_frac * full_step
        set_flat_params(policy, candidate)

        new_objective = surrogate_objective(policy, states, actions, advantages, old_log_probs)
        new_kl = mean_kl_divergence(policy, states, old_mean, old_log_std)

        if torch.isfinite(new_objective) and torch.isfinite(new_kl):
            if new_objective.item() > old_objective and new_kl.item() <= max_kl:
                accepted = True
                final_objective = float(new_objective.item())
                final_kl = float(new_kl.item())
                break

    if not accepted:
        set_flat_params(policy, old_params)

    return final_objective, final_kl, accepted


def update_value_function(
    value_net: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    returns: torch.Tensor,
    epochs: int,
    batch_size: int,
) -> float:
    indices = np.arange(states.shape[0])
    losses: List[float] = []

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            state_batch = states[batch_idx]
            return_batch = returns[batch_idx]

            value_pred = value_net(state_batch)
            loss = ((value_pred - return_batch) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

    if not losses:
        return 0.0
    return float(np.mean(losses))


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
    trajectories: List[Dict[str, List[np.ndarray]]] = []
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

        for step in range(horizon):
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

            if done or step == horizon - 1 or steps_collected >= steps_per_iter:
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

    for trajectory in trajectories:
        states_np = np.stack(trajectory["states"])
        actions_np = np.stack(trajectory["actions"])
        rewards_np = np.array(trajectory["rewards"], dtype=np.float32)
        next_states_np = np.stack(trajectory["next_states"])
        dones_np = np.array(trajectory["dones"], dtype=np.float32)

        with torch.no_grad():
            states_t = torch.as_tensor(states_np, dtype=torch.float32, device=device)
            next_states_t = torch.as_tensor(next_states_np, dtype=torch.float32, device=device)
            values = value_net(states_t).squeeze(-1).cpu().numpy()
            next_values = value_net(next_states_t).squeeze(-1).cpu().numpy()

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

    states = np.concatenate(batch_states, axis=0).astype(np.float32)
    actions = np.concatenate(batch_actions, axis=0).astype(np.float32)
    advantages = np.concatenate(batch_advantages, axis=0).astype(np.float32)
    returns = np.concatenate(batch_returns, axis=0).astype(np.float32)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch = {
        "states": states,
        "actions": actions,
        "advantages": advantages,
        "returns": returns,
    }
    return batch, episode_returns, episode_index


def plot_training_curves(
    save_dir: Path,
    reward_history: List[float],
    surrogate_hist: List[float],
    value_loss_hist: List[float],
    kl_hist: List[float],
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

    axes[0, 1].plot(surrogate_hist, color="#009E73", linewidth=1.2)
    axes[0, 1].set_title("Surrogate Objective")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("L(pi_new)")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(value_loss_hist, color="#D55E00", linewidth=1.2)
    axes[1, 0].set_title("Value Loss")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(kl_hist, color="#CC79A7", linewidth=1.2)
    axes[1, 1].set_title("Mean KL After Update")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("KL")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "trpo_training_curves.png"
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
    out_file = save_dir / "trpo_trajectory.png"
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

    gif_path = save_dir / "trpo_cartpole.gif"
    imageio.mimsave(gif_path, frames, duration=0.05)
    print(f"Saved: {gif_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRPO for cartpole swing-up + balance")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--steps-per-iter", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--value-epochs", type=int, default=40)
    parser.add_argument("--value-batch-size", type=int, default=256)
    parser.add_argument("--max-kl", type=float, default=0.01)
    parser.add_argument("--cg-iters", type=int, default=10)
    parser.add_argument("--cg-damping", type=float, default=0.1)
    parser.add_argument("--line-search-steps", type=int, default=10)
    parser.add_argument("--backtrack-coeff", type=float, default=0.7)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CartPoleConfig()
    env = InvertedPendulumOnCart(cfg)

    policy = GaussianPolicy(state_dim=5, action_dim=1).to(device)
    value_net = ValueNetwork(state_dim=5).to(device)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.value_lr)

    reward_history: List[float] = []
    surrogate_hist: List[float] = []
    value_loss_hist: List[float] = []
    kl_hist: List[float] = []
    accepted_updates = 0
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

        surrogate_value, mean_kl, accepted = update_policy_trpo(
            policy=policy,
            states=states,
            actions=actions,
            advantages=advantages,
            max_kl=args.max_kl,
            cg_iters=args.cg_iters,
            damping=args.cg_damping,
            line_search_steps=args.line_search_steps,
            backtrack_coeff=args.backtrack_coeff,
        )
        if accepted:
            accepted_updates += 1

        value_loss = update_value_function(
            value_net=value_net,
            optimizer=value_optimizer,
            states=states,
            returns=returns,
            epochs=args.value_epochs,
            batch_size=args.value_batch_size,
        )

        avg_reward = float(np.mean(episode_returns)) if episode_returns else 0.0
        reward_history.append(avg_reward)
        surrogate_hist.append(surrogate_value)
        value_loss_hist.append(value_loss)
        kl_hist.append(mean_kl)

        if iteration % 10 == 0 or iteration == 1:
            print(
                f"Iter {iteration:4d} | avg_reward={avg_reward:8.2f} | "
                f"surrogate={surrogate_value:8.4f} | kl={mean_kl:7.5f} | "
                f"accepted={accepted_updates}/{iteration}"
            )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_plots:
        plot_training_curves(output_dir, reward_history, surrogate_hist, value_loss_hist, kl_hist)

    init_state = np.array([0.0, 0.0, math.pi - 0.2, 0.0], dtype=np.float32)
    xs, thetas = rollout_trajectory(env, cfg, policy, device, init_state, horizon=500)
    plot_trajectory(output_dir, cfg.dt, xs, thetas)

    if not args.skip_gif:
        create_cartpole_gif(output_dir, xs, thetas)

    print("Done. TRPO trained for swing-up + balance.")


if __name__ == "__main__":
    main()