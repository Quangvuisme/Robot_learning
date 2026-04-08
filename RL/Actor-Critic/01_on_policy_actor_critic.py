"""
Example 01: On-Policy Actor-Critic (TD(0)) on a tiny MDP.

- Policy: softmax per state (theta[s]).
- Critic: state value V(s).
- Update uses the on-policy TD error.
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


def run_on_policy_actor_critic(
    episodes: int,
    max_steps: int,
    actor_lr: float,
    critic_lr: float,
    gamma: float,
    seed: int,
    log_interval: int,
) -> tuple[List[float], List[float]]:
    rng = np.random.default_rng(seed)
    n_states = 5
    n_actions = 2

    theta = np.zeros((n_states, n_actions), dtype=np.float32)
    values = np.zeros(n_states, dtype=np.float32)

    returns: List[float] = []
    p_right_hist: List[float] = []

    for episode in range(1, episodes + 1):
        state = 2
        episode_return = 0.0

        for _ in range(max_steps):
            if state in (0, 4):
                break

            pi = softmax(theta[state])
            action = int(rng.choice(n_actions, p=pi))
            next_state, reward, done = step_env(state, action)
            episode_return += reward

            v_next = 0.0 if done else values[next_state]
            td_target = reward + gamma * v_next
            td_error = td_target - values[state]

            values[state] += critic_lr * td_error

            onehot = np.zeros(n_actions, dtype=np.float32)
            onehot[action] = 1.0
            grad_log_pi = onehot - pi
            theta[state] += actor_lr * td_error * grad_log_pi

            state = next_state
            if done:
                break

        returns.append(float(episode_return))
        p_right_hist.append(float(softmax(theta[2])[1]))

        if episode % log_interval == 0:
            avg_return = float(np.mean(returns[-log_interval:]))
            print(f"Episode {episode:4d} | avg_return={avg_return:6.3f} | pi_right={p_right_hist[-1]:.3f}")

    return returns, p_right_hist


def plot_curves(save_dir: Path, returns: List[float], p_right_hist: List[float]) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    window = 25
    if len(returns) >= window:
        moving_avg = np.convolve(returns, np.ones(window) / window, mode="valid")
        axes[0].plot(range(window, len(returns) + 1), moving_avg, color="#0072B2", linewidth=2.0)
    axes[0].set_title("Return Moving Average")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].grid(alpha=0.25)

    axes[1].plot(p_right_hist, color="#D55E00", linewidth=1.6)
    axes[1].set_title("Policy Probability of Action=Right (state=2)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("pi(a=1)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "actor_critic_on_policy.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-policy actor-critic (TD(0))")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--actor-lr", type=float, default=0.1)
    parser.add_argument("--critic-lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    returns, p_right_hist = run_on_policy_actor_critic(
        episodes=args.episodes,
        max_steps=args.max_steps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        seed=args.seed,
        log_interval=args.log_interval,
    )

    avg_return = float(np.mean(returns[-200:])) if len(returns) >= 200 else float(np.mean(returns))
    print(f"Final avg return: {avg_return:.3f}")
    print(f"Final pi_right (state=2): {p_right_hist[-1]:.3f}")

    if not args.no_plot:
        plot_curves(Path("outputs"), returns, p_right_hist)


if __name__ == "__main__":
    main()
