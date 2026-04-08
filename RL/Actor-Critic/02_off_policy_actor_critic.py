"""
Example 02: Off-Policy Actor-Critic on a tiny MDP.

- Behavior policy mu collects data into a replay buffer.
- Critic: Q(s,a) updated from replay data.
- Actor: updates use states from the buffer but actions sampled from current pi.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp = np.exp(z)
    return exp / np.sum(exp)


def step_env(state: int, action: int) -> Tuple[int, float, bool]:
    # States: 0..4, terminal at 0 (reward -1) and 4 (reward +1)
    delta = 1 if action == 1 else -1
    next_state = int(np.clip(state + delta, 0, 4))

    if next_state == 4:
        return next_state, 1.0, True
    if next_state == 0:
        return next_state, -1.0, True
    return next_state, 0.0, False


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.data: List[Tuple[int, int, float, int, bool]] = []

    def add(self, transition: Tuple[int, int, float, int, bool]) -> None:
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(transition)

    def sample(self, batch_size: int) -> List[Tuple[int, int, float, int, bool]]:
        idx = self.rng.choice(len(self.data), size=batch_size, replace=True)
        return [self.data[i] for i in idx]

    def __len__(self) -> int:
        return len(self.data)


def evaluate_policy(
    theta: np.ndarray,
    episodes: int,
    max_steps: int,
    rng: np.random.Generator,
) -> float:
    n_actions = theta.shape[1]
    total_return = 0.0

    for _ in range(episodes):
        state = 2
        episode_return = 0.0
        for _ in range(max_steps):
            if state in (0, 4):
                break
            pi = softmax(theta[state])
            action = int(rng.choice(n_actions, p=pi))
            next_state, reward, done = step_env(state, action)
            episode_return += reward
            state = next_state
            if done:
                break
        total_return += episode_return

    return total_return / float(episodes)


def run_off_policy_actor_critic(
    episodes: int,
    max_steps: int,
    actor_lr: float,
    critic_lr: float,
    gamma: float,
    behavior_pi0: float,
    buffer_size: int,
    batch_size: int,
    updates_per_step: int,
    eval_interval: int,
    eval_episodes: int,
    seed: int,
) -> tuple[List[float], List[float], List[int], List[float]]:
    rng = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 999)
    n_states = 5
    n_actions = 2

    theta = np.zeros((n_states, n_actions), dtype=np.float32)
    q_values = np.zeros((n_states, n_actions), dtype=np.float32)

    behavior_pi = np.array([behavior_pi0, 1.0 - behavior_pi0], dtype=np.float32)
    buffer = ReplayBuffer(buffer_size, rng)

    behavior_returns: List[float] = []
    eval_returns: List[float] = []
    eval_episodes_hist: List[int] = []
    p_right_hist: List[float] = []

    for episode in range(1, episodes + 1):
        state = 2
        episode_return = 0.0

        for _ in range(max_steps):
            if state in (0, 4):
                break

            action = int(rng.choice(n_actions, p=behavior_pi))
            next_state, reward, done = step_env(state, action)
            episode_return += reward

            buffer.add((state, action, reward, next_state, done))

            if len(buffer) >= batch_size:
                for _ in range(updates_per_step):
                    batch = buffer.sample(batch_size)

                    # Critic update (off-policy TD for Q)
                    for s_t, a_t, r_t, s_next, done_t in batch:
                        if done_t:
                            q_next = 0.0
                        else:
                            pi_next = softmax(theta[s_next])
                            q_next = float(np.dot(pi_next, q_values[s_next]))
                        target = r_t + gamma * q_next
                        q_values[s_t, a_t] += critic_lr * (target - q_values[s_t, a_t])

                    # Actor update (states from buffer, actions from current policy)
                    for s_t, _, _, _, _ in batch:
                        pi_s = softmax(theta[s_t])
                        a_pi = int(rng.choice(n_actions, p=pi_s))
                        onehot = np.zeros(n_actions, dtype=np.float32)
                        onehot[a_pi] = 1.0
                        grad_log_pi = onehot - pi_s
                        theta[s_t] += actor_lr * q_values[s_t, a_pi] * grad_log_pi

            state = next_state
            if done:
                break

        behavior_returns.append(float(episode_return))
        p_right_hist.append(float(softmax(theta[2])[1]))

        if episode % eval_interval == 0:
            eval_return = evaluate_policy(theta, eval_episodes, max_steps, rng_eval)
            eval_returns.append(float(eval_return))
            eval_episodes_hist.append(episode)

            avg_behavior = float(np.mean(behavior_returns[-eval_interval:]))
            print(
                f"Episode {episode:4d} | behavior_avg={avg_behavior:6.3f} "
                f"| eval_return={eval_return:6.3f} | pi_right={p_right_hist[-1]:.3f}"
            )

    return behavior_returns, eval_returns, eval_episodes_hist, p_right_hist


def plot_curves(
    save_dir: Path,
    behavior_returns: List[float],
    eval_returns: List[float],
    eval_episodes: List[int],
    p_right_hist: List[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    window = 50
    if len(behavior_returns) >= window:
        moving_avg = np.convolve(behavior_returns, np.ones(window) / window, mode="valid")
        axes[0].plot(range(window, len(behavior_returns) + 1), moving_avg, color="#0072B2", linewidth=2.0)
    axes[0].set_title("Behavior Return Moving Average")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].grid(alpha=0.25)

    if eval_returns:
        axes[1].plot(eval_episodes, eval_returns, color="#D55E00", linewidth=1.6, marker="o", markersize=3)
    axes[1].set_title("Eval Return (Current Policy)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Return")
    axes[1].grid(alpha=0.25)

    axes[2].plot(p_right_hist, color="#009E73", linewidth=1.6)
    axes[2].set_title("Policy Probability of Action=Right (state=2)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("pi(a=1)")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "actor_critic_off_policy.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Off-policy actor-critic with replay buffer")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--actor-lr", type=float, default=0.05)
    parser.add_argument("--critic-lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--behavior-pi0", type=float, default=0.8)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    behavior_returns, eval_returns, eval_episodes, p_right_hist = run_off_policy_actor_critic(
        episodes=args.episodes,
        max_steps=args.max_steps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        behavior_pi0=args.behavior_pi0,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )

    if eval_returns:
        print(f"Final eval return: {eval_returns[-1]:.3f}")
    print(f"Final pi_right (state=2): {p_right_hist[-1]:.3f}")

    if not args.no_plot:
        plot_curves(Path("outputs"), behavior_returns, eval_returns, eval_episodes, p_right_hist)


if __name__ == "__main__":
    main()
