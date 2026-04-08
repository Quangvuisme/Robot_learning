"""
Example 04: Off-Policy Policy Gradient (Bandit with Importance Sampling)

We optimize a target policy pi_theta using data sampled from a behavior policy mu.
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


def run_off_policy_pg(
    steps: int,
    lr: float,
    p0: float,
    p1: float,
    behavior_pi0: float,
    clip_rho: float | None,
    seed: int,
) -> tuple[List[float], List[float], List[float]]:
    rng = np.random.default_rng(seed)
    theta = np.zeros(2, dtype=np.float32)

    behavior_pi = np.array([behavior_pi0, 1.0 - behavior_pi0], dtype=np.float32)
    rewards: List[float] = []
    prob_action1: List[float] = []
    rho_hist: List[float] = []

    p_rewards = np.array([p0, p1], dtype=np.float32)

    for _ in range(steps):
        pi_t = softmax(theta)
        action = int(rng.choice(2, p=behavior_pi))
        reward = 1.0 if rng.random() < p_rewards[action] else 0.0

        rho = float(pi_t[action] / behavior_pi[action])
        if clip_rho is not None:
            rho = float(np.clip(rho, 0.0, clip_rho))

        onehot = np.zeros(2, dtype=np.float32)
        onehot[action] = 1.0
        grad_log_pi = onehot - pi_t

        theta += lr * rho * reward * grad_log_pi

        rewards.append(float(reward))
        prob_action1.append(float(pi_t[1]))
        rho_hist.append(float(rho))

    return rewards, prob_action1, rho_hist


def plot_curves(save_dir: Path, rewards: List[float], prob_action1: List[float], rho_hist: List[float]) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(range(window, len(rewards) + 1), moving_avg, color="#0072B2", linewidth=2.0)
    axes[0].set_title("Reward Moving Average (Behavior Data)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.25)

    axes[1].plot(prob_action1, color="#D55E00", linewidth=1.5)
    axes[1].set_title("Target Policy Prob of Action 1")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("pi_theta(a=1)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(rho_hist, color="#009E73", linewidth=1.2, alpha=0.8)
    axes[2].set_title("Importance Weight rho")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("rho")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "policy_gradient_off_policy.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Off-policy policy gradient (bandit)")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--p0", type=float, default=0.3)
    parser.add_argument("--p1", type=float, default=0.8)
    parser.add_argument("--behavior-pi0", type=float, default=0.8)
    parser.add_argument("--clip-rho", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rewards, prob_action1, rho_hist = run_off_policy_pg(
        steps=args.steps,
        lr=args.lr,
        p0=args.p0,
        p1=args.p1,
        behavior_pi0=args.behavior_pi0,
        clip_rho=args.clip_rho,
        seed=args.seed,
    )

    avg_reward = float(np.mean(rewards[-200:])) if len(rewards) >= 200 else float(np.mean(rewards))
    print(f"Final avg reward (behavior data): {avg_reward:.3f}")
    print(f"Final pi_theta(a=1): {prob_action1[-1]:.3f}")

    if not args.no_plot:
        plot_curves(Path("outputs"), rewards, prob_action1, rho_hist)


if __name__ == "__main__":
    main()
