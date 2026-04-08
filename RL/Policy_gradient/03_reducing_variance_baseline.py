"""
Example 03: Reducing Variance with a Baseline (Bandit)

We estimate the policy gradient for a fixed softmax policy and compare
variance with and without a baseline.
"""

import argparse
from pathlib import Path
from typing import Tuple

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


def sample_gradient(
    rng: np.random.Generator,
    pi: np.ndarray,
    p_rewards: np.ndarray,
    baseline: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    action = int(rng.choice(2, p=pi))
    reward = 1.0 if rng.random() < p_rewards[action] else 0.0

    onehot = np.zeros_like(pi)
    onehot[action] = 1.0
    grad_log_pi = onehot - pi

    grad_no_baseline = reward * grad_log_pi
    grad_with_baseline = (reward - baseline) * grad_log_pi
    return grad_no_baseline, grad_with_baseline, float(reward), float(action)


def run_variance_demo(
    samples: int,
    p0: float,
    p1: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    theta = np.zeros(2, dtype=np.float32)
    pi = softmax(theta)
    p_rewards = np.array([p0, p1], dtype=np.float32)

    expected_reward = float(np.dot(pi, p_rewards))

    grads_no_b = []
    grads_b = []
    rewards = []
    actions = []

    for _ in range(samples):
        g_no_b, g_b, r, a = sample_gradient(rng, pi, p_rewards, expected_reward)
        grads_no_b.append(g_no_b)
        grads_b.append(g_b)
        rewards.append(r)
        actions.append(a)

    return (
        np.array(grads_no_b),
        np.array(grads_b),
        np.array(rewards),
        np.array(actions),
        expected_reward,
    )


def plot_variance(save_dir: Path, grads_no_b: np.ndarray, grads_b: np.ndarray) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    g0 = grads_no_b[:, 1]
    g1 = grads_b[:, 1]
    lim = max(abs(g0).max(), abs(g1).max())
    bins = np.linspace(-lim, lim, 60)

    axes[0].hist(g0, bins=bins, color="#D55E00", alpha=0.7, density=True)
    axes[0].axvline(g0.mean(), color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Gradient Distribution (No Baseline)")
    axes[0].set_xlabel("grad component for action 1")
    axes[0].set_ylabel("density")
    axes[0].grid(alpha=0.25)

    axes[1].hist(g1, bins=bins, color="#0072B2", alpha=0.7, density=True)
    axes[1].axvline(g1.mean(), color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Gradient Distribution (With Baseline)")
    axes[1].set_xlabel("grad component for action 1")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(
        0.5,
        0.02,
        "Discrete spikes appear because reward and action are binary; baseline shrinks magnitude without changing mean.",
        ha="center",
        fontsize=9,
    )
    out_file = save_dir / "policy_gradient_baseline_variance.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Variance reduction with a baseline (bandit)")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--p0", type=float, default=0.3)
    parser.add_argument("--p1", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grads_no_b, grads_b, rewards, actions, expected_reward = run_variance_demo(
        samples=args.samples,
        p0=args.p0,
        p1=args.p1,
        seed=args.seed,
    )

    mean_no_b = grads_no_b.mean(axis=0)
    mean_b = grads_b.mean(axis=0)
    var_no_b = grads_no_b.var(axis=0)
    var_b = grads_b.var(axis=0)

    print(f"Policy pi = [0.5, 0.5]")
    print(f"Expected reward baseline b = {expected_reward:.3f}")
    print("Mean gradient (no baseline):", mean_no_b)
    print("Mean gradient (with baseline):", mean_b)
    print("Variance (no baseline):", var_no_b)
    print("Variance (with baseline):", var_b)

    if not args.no_plot:
        plot_variance(Path("outputs"), grads_no_b, grads_b)


if __name__ == "__main__":
    main()
