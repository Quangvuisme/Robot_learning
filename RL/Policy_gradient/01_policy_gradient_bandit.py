"""
Example 01: Policy Gradient (REINFORCE) on a 2-armed bandit.

This is the simplest setting with no state transitions.
We learn a softmax policy over two actions and maximize expected reward.
"""

import argparse
from pathlib import Path
from typing import List

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


def run_reinforce_bandit(
    steps: int,
    lr: float,
    p0: float,
    p1: float,
    baseline_beta: float,
    seed: int,
) -> tuple[List[float], List[float], List[float]]:
    rng = np.random.default_rng(seed)
    theta = np.zeros(2, dtype=np.float32)
    baseline = 0.0

    rewards: List[float] = []
    prob_action1: List[float] = []
    baseline_hist: List[float] = []

    probs = np.array([p0, p1], dtype=np.float32)

    for _ in range(steps):
        pi = softmax(theta)
        action = int(rng.choice(2, p=pi))
        reward = 1.0 if rng.random() < probs[action] else 0.0

        baseline = (1.0 - baseline_beta) * baseline + baseline_beta * reward
        advantage = reward - baseline

        onehot = np.zeros(2, dtype=np.float32)
        onehot[action] = 1.0
        grad_log_pi = onehot - pi

        theta += lr * advantage * grad_log_pi

        rewards.append(float(reward))
        prob_action1.append(float(pi[1]))
        baseline_hist.append(float(baseline))

    return rewards, prob_action1, baseline_hist


def plot_curves(save_dir: Path, rewards: List[float], prob_action1: List[float]) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(range(window, len(rewards) + 1), moving_avg, color="#0072B2", linewidth=2.0)
    axes[0].set_title("Reward Moving Average")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.25)

    axes[1].plot(prob_action1, color="#D55E00", linewidth=1.5)
    axes[1].set_title("Policy Probability of Action 1")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("pi(a=1)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "policy_gradient_bandit.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REINFORCE on a 2-armed bandit")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--p0", type=float, default=0.2, help="Reward probability for action 0")
    parser.add_argument("--p1", type=float, default=0.8, help="Reward probability for action 1")
    parser.add_argument("--baseline-beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rewards, prob_action1, _ = run_reinforce_bandit(
        steps=args.steps,
        lr=args.lr,
        p0=args.p0,
        p1=args.p1,
        baseline_beta=args.baseline_beta,
        seed=args.seed,
    )

    avg_reward = float(np.mean(rewards[-200:])) if len(rewards) >= 200 else float(np.mean(rewards))
    print(f"Final avg reward: {avg_reward:.3f}")
    print(f"Final pi(a=1): {prob_action1[-1]:.3f}")

    if not args.no_plot:
        plot_curves(Path("outputs"), rewards, prob_action1)


if __name__ == "__main__":
    main()
