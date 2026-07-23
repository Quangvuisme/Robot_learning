"""
AMP (Adversarial Motion Priors) for CartPole Swing-up + Balance

This is the RL-based sibling of gan/gan_cartpole_swingup_balance_study.py. The GAN/GAIL
script backprops the discriminator's verdict directly into the action generator
(differentiable, works because the "environment" it fools is just the discriminator).
AMP instead assumes the physics step is NOT differentiable (true for most simulators):

  1. Discriminator D(s, s') scores *state transitions* (no action needed at all -
     matches AMP's original motivation of imitating mocap clips that have no action labels).
  2. D's score is converted into a scalar reward r_style(s,s') and combined with a
     (small, optional) task reward.
  3. The policy is a stochastic Gaussian actor trained with on-policy RL (PPO) against
     that combined reward - never backprop through D into the policy.

Discriminator uses the least-squares GAN (LSGAN) objective plus a zero-centered gradient
penalty on expert samples, exactly as in Peng et al. 2021, "AMP: Adversarial Motion
Priors for Stylized Physics-Based Character Control".
"""

import argparse
import importlib.util
import csv
import math
import random
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
    import imageio.v2 as imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def load_base_module():
    base_path = Path(__file__).resolve().parents[1] / "diffusion" / "diffusion_cartpole_swingup_balance_study.py"
    spec = importlib.util.spec_from_file_location("diffusion_cartpole_swingup_base", base_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class GaussianPolicy(nn.Module):
    """Stochastic actor. Same design as RL/PPO/ppo_cartpole_swingup_balance.py."""

    def __init__(self, state_dim: int, action_dim: int = 1, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(self, state_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state_feat)
        mean = torch.tanh(self.mean_head(x))
        log_std = self.log_std.clamp(-2.5, 0.25).expand_as(mean)
        return mean, log_std

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

    def forward(self, state_feat: torch.Tensor) -> torch.Tensor:
        return self.net(state_feat)


class AMPDiscriminator(nn.Module):
    """Scores a *transition* (s, s'), not a (state, action) pair - AMP's discriminator
    never needs actions, so it can imitate transitions that have no recorded action.
    Outputs a raw (unsquashed) score for the LSGAN objective: target +1 for expert,
    -1 for policy. Small last-layer init keeps early scores near 0 (D undecided)."""

    def __init__(self, state_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward_concat(self, transition: torch.Tensor) -> torch.Tensor:
        return self.net(transition)

    def forward(self, s_feat: torch.Tensor, sp_feat: torch.Tensor) -> torch.Tensor:
        return self.forward_concat(torch.cat([s_feat, sp_feat], dim=-1))


def style_reward_from_score(d_score: torch.Tensor) -> torch.Tensor:
    """AMP's least-squares style reward: r = clip(1 - 0.25*(D-1)^2, 0, 1).
    D=1 (expert-like) -> reward 1; D far from 1 in either direction -> reward -> 0."""
    return torch.clamp(1.0 - 0.25 * (d_score - 1.0) ** 2, min=0.0)


def reward_task(state: torch.Tensor) -> torch.Tensor:
    """Light task shaping (upright + centered). Kept small relative to style_weight so the
    discriminator - not hand tuning - supplies most of the swing-up/balance behavior."""
    x = state[:, 0:1]
    x_dot = state[:, 1:2]
    theta = (state[:, 2:3] + math.pi) % (2.0 * math.pi) - math.pi
    theta_dot = state[:, 3:4]
    reward = 1.0 + torch.cos(theta)
    reward = reward - 0.1 * (x**2) - 0.01 * (x_dot**2 + theta_dot**2)
    bonus = torch.where(
        (theta.abs() < 0.25) & (x.abs() < 0.5),
        torch.full_like(reward, 0.5),
        torch.zeros_like(reward),
    )
    return reward + bonus


# ---------------------------------------------------------------------------
# Expert transition dataset (state-only; no actions needed by the discriminator)
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_expert_transitions(env, base, episodes: int, horizon: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    s_list: List[torch.Tensor] = []
    sp_list: List[torch.Tensor] = []

    for _ in range(episodes):
        s = env.sample_swingup_state(device)
        for _ in range(horizon):
            if not env.is_state_valid(s).all():
                s = env.sample_swingup_state(device)
                continue
            u = base.expert_policy(s, env.cfg)
            s_next = env.step(s, u)
            s_list.append(base.state_features(s))
            sp_list.append(base.state_features(s_next))
            s = s_next

    # Extra near-upright coverage so the discriminator also knows what good balancing looks like.
    balance_episodes = max(30, episodes // 6)
    balance_horizon = min(220, horizon)
    for _ in range(balance_episodes):
        s = env.sample_near_upright_state(device)
        for _ in range(balance_horizon):
            if not env.is_state_valid(s).all():
                s = env.sample_near_upright_state(device)
                continue
            u = base.expert_policy(s, env.cfg)
            s_next = env.step(s, u)
            s_list.append(base.state_features(s))
            sp_list.append(base.state_features(s_next))
            s = s_next

    return torch.cat(s_list, dim=0), torch.cat(sp_list, dim=0)


# ---------------------------------------------------------------------------
# Policy action helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def amp_policy_action(
    policy: GaussianPolicy,
    base,
    state: torch.Tensor,
    force_max: float,
    deterministic: bool = True,
) -> torch.Tensor:
    feat = base.state_features(state)
    mean, log_std = policy(feat)
    if deterministic:
        action_norm = mean
    else:
        std = log_std.exp()
        action_norm = (mean + std * torch.randn_like(mean)).clamp(-1.0, 1.0)
    return torch.clamp(action_norm * force_max, -force_max, force_max)


# ---------------------------------------------------------------------------
# Rollout collection with GAE (reward = task + style, style from current D)
# ---------------------------------------------------------------------------


def collect_batch(
    env,
    base,
    policy: GaussianPolicy,
    value_net: ValueNetwork,
    discriminator: AMPDiscriminator,
    device: str,
    steps_per_iter: int,
    horizon: int,
    gamma: float,
    gae_lambda: float,
    task_weight: float,
    style_weight: float,
    start_episode_index: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], int]:
    trajectories = []
    ep_task, ep_style, ep_combined = [], [], []
    steps_collected = 0
    episode_index = start_episode_index

    while steps_collected < steps_per_iter:
        state = env.sample_near_upright_state(device) if episode_index % 5 == 4 else env.sample_swingup_state(device)

        feats, actions, rewards, next_feats, dones = [], [], [], [], []
        task_sum = 0.0
        style_sum = 0.0

        for _ in range(horizon):
            feat = base.state_features(state)
            with torch.no_grad():
                mean, log_std = policy(feat)
                std = log_std.exp()
                action_norm = (mean + std * torch.randn_like(mean)).clamp(-1.0, 1.0)
                force = action_norm * env.cfg.force_max

            next_state = env.step(state, force)
            next_feat = base.state_features(next_state)

            with torch.no_grad():
                d_score = discriminator(feat, next_feat)
                style_r = style_reward_from_score(d_score)
            task_r = reward_task(next_state)
            combined_r = task_weight * task_r + style_weight * style_r
            done = not bool(env.is_state_valid(next_state).all().item())

            feats.append(feat.squeeze(0).cpu())
            actions.append(action_norm.squeeze(0).cpu())
            rewards.append(float(combined_r.item()))
            next_feats.append(next_feat.squeeze(0).cpu())
            dones.append(1.0 if done else 0.0)

            task_sum += float(task_r.item())
            style_sum += float(style_r.item())

            state = next_state
            steps_collected += 1
            if done or steps_collected >= steps_per_iter:
                break

        trajectories.append({"feats": feats, "actions": actions, "rewards": rewards, "next_feats": next_feats, "dones": dones})
        n_steps = max(1, len(rewards))
        ep_task.append(task_sum / n_steps)
        ep_style.append(style_sum / n_steps)
        ep_combined.append(sum(rewards) / n_steps)
        episode_index += 1

    batch_feats, batch_actions, batch_adv, batch_ret, batch_old_logp = [], [], [], [], []
    disc_feats, disc_next_feats = [], []

    for traj in trajectories:
        states_t = torch.stack(traj["feats"]).to(device)
        actions_t = torch.stack(traj["actions"]).to(device)
        next_states_t = torch.stack(traj["next_feats"]).to(device)
        rewards_t = torch.tensor(traj["rewards"], dtype=torch.float32)
        dones_t = torch.tensor(traj["dones"], dtype=torch.float32)

        with torch.no_grad():
            values = value_net(states_t).squeeze(-1).cpu()
            next_values = value_net(next_states_t).squeeze(-1).cpu()
            old_log_probs = policy.log_prob(states_t, actions_t).squeeze(-1).cpu()

        deltas = rewards_t + gamma * (1.0 - dones_t) * next_values - values
        advantages = torch.zeros_like(rewards_t)
        gae = 0.0
        for t in reversed(range(len(rewards_t))):
            gae = deltas[t] + gamma * gae_lambda * (1.0 - dones_t[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        batch_feats.append(states_t.cpu())
        batch_actions.append(actions_t.cpu())
        batch_adv.append(advantages.unsqueeze(-1))
        batch_ret.append(returns.unsqueeze(-1))
        batch_old_logp.append(old_log_probs.unsqueeze(-1))
        disc_feats.append(states_t.cpu())
        disc_next_feats.append(next_states_t.cpu())

    states_out = torch.cat(batch_feats, dim=0).to(device)
    actions_out = torch.cat(batch_actions, dim=0).to(device)
    advantages_out = torch.cat(batch_adv, dim=0).to(device)
    returns_out = torch.cat(batch_ret, dim=0).to(device)
    old_log_probs_out = torch.cat(batch_old_logp, dim=0).to(device)
    advantages_out = (advantages_out - advantages_out.mean()) / (advantages_out.std() + 1e-8)

    batch = {
        "states": states_out,
        "actions": actions_out,
        "advantages": advantages_out,
        "returns": returns_out,
        "old_log_probs": old_log_probs_out,
        "policy_feats": torch.cat(disc_feats, dim=0).to(device),
        "policy_next_feats": torch.cat(disc_next_feats, dim=0).to(device),
    }
    stats = {
        "task_reward": float(np.mean(ep_task)),
        "style_reward": float(np.mean(ep_style)),
        "combined_reward": float(np.mean(ep_combined)),
    }
    return batch, stats, episode_index


# ---------------------------------------------------------------------------
# Discriminator update: LSGAN targets (+1 expert / -1 policy) + gradient penalty
# ---------------------------------------------------------------------------


def gradient_penalty(discriminator: AMPDiscriminator, s_expert: torch.Tensor, sp_expert: torch.Tensor) -> torch.Tensor:
    transition = torch.cat([s_expert, sp_expert], dim=-1).clone().requires_grad_(True)
    d_out = discriminator.forward_concat(transition)
    grad = torch.autograd.grad(outputs=d_out.sum(), inputs=transition, create_graph=True)[0]
    return grad.pow(2).sum(dim=-1).mean()


def update_discriminator(
    discriminator: AMPDiscriminator,
    opt_d: torch.optim.Optimizer,
    s_expert: torch.Tensor,
    sp_expert: torch.Tensor,
    s_policy: torch.Tensor,
    sp_policy: torch.Tensor,
    batch_size: int,
    updates: int,
    gp_weight: float,
) -> float:
    n_e = s_expert.size(0)
    n_p = s_policy.size(0)
    losses = []
    for _ in range(updates):
        idx_e = torch.randint(0, n_e, (batch_size,), device=s_expert.device)
        idx_p = torch.randint(0, n_p, (batch_size,), device=s_policy.device)

        se, spe = s_expert[idx_e], sp_expert[idx_e]
        sp_, spp = s_policy[idx_p], sp_policy[idx_p]

        d_real = discriminator(se, spe)
        d_fake = discriminator(sp_, spp)

        loss_real = F.mse_loss(d_real, torch.ones_like(d_real))
        loss_fake = F.mse_loss(d_fake, -torch.ones_like(d_fake))
        gp = gradient_penalty(discriminator, se, spe)
        loss = loss_real + loss_fake + gp_weight * gp

        opt_d.zero_grad()
        loss.backward()
        opt_d.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# PPO update (identical mechanics to RL/PPO/ppo_cartpole_swingup_balance.py)
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
    policy_losses, value_losses, entropies = [], [], []

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

            surr1 = ratio * mb_advantages
            surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred = value_net(mb_states)
            value_loss = F.mse_loss(value_pred, mb_returns)

            entropy = policy.entropy(mb_states)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_net.parameters()), max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

    return float(np.mean(policy_losses)), float(np.mean(value_losses)), float(np.mean(entropies))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_curves_amp(
    save_dir: Path,
    reward_hist: List[float],
    task_reward_hist: List[float],
    style_reward_hist: List[float],
    disc_loss_hist: List[float],
    eval_steps: List[int],
    expert_theta: List[float],
    amp_theta: List[float],
    expert_sr: List[float],
    amp_sr: List[float],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "amp_training_curves_cartpole_swingup.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "combined_reward", "task_reward", "style_reward", "disc_loss"])
        for i in range(len(reward_hist)):
            writer.writerow([i + 1, reward_hist[i], task_reward_hist[i], style_reward_hist[i], disc_loss_hist[i]])
    print(f"Saved: {csv_path}")

    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip PNG plot (CSV saved instead).")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(reward_hist, color="#0072B2", linewidth=1.2, label="combined")
    axes[0, 0].plot(task_reward_hist, color="#009E73", linewidth=1.0, alpha=0.7, label="task")
    axes[0, 0].plot(style_reward_hist, color="#D55E00", linewidth=1.0, alpha=0.7, label="style (AMP)")
    axes[0, 0].set_title("Per-step Reward (avg over episodes)")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(disc_loss_hist, color="#CC79A7", linewidth=1.2)
    axes[0, 1].set_title("Discriminator Loss (LSGAN + gradient penalty)")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(eval_steps, expert_theta, "-o", color="#009E73", label="Expert")
    axes[1, 0].plot(eval_steps, amp_theta, "-o", color="#D55E00", label="AMP")
    axes[1, 0].set_title("Mean Tail |theta| vs Iteration")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Radians (lower is better)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(eval_steps, [100.0 * x for x in expert_sr], "-o", color="#009E73", label="Expert")
    axes[1, 1].plot(eval_steps, [100.0 * x for x in amp_sr], "-o", color="#D55E00", label="AMP")
    axes[1, 1].set_title("Success Rate vs Iteration")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Success %")
    axes[1, 1].set_ylim(0.0, 105.0)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    fig.tight_layout()
    out_file = save_dir / "amp_training_curves_cartpole_swingup.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_trajectory_amp(
    save_dir: Path,
    dt: float,
    expert_x: List[float],
    expert_theta: List[float],
    amp_x: List[float],
    amp_theta: List[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip trajectory plot.")
        return

    t = [i * dt for i in range(len(expert_theta))]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, expert_theta, color="#009E73", linewidth=2.0, label="Expert")
    axes[0].plot(t, amp_theta, color="#D55E00", linewidth=2.0, label="AMP")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("theta (rad)")
    axes[0].set_title("Pole Angle Trajectory")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(t, expert_x, color="#009E73", linewidth=2.0, label="Expert")
    axes[1].plot(t, amp_x, color="#D55E00", linewidth=2.0, label="AMP")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("x (m)")
    axes[1].set_title("Cart Position Trajectory")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "amp_cartpole_swingup_trajectory.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def make_cartpole_gif_amp(
    save_dir: Path,
    expert_x: List[float],
    expert_theta: List[float],
    amp_x: List[float],
    amp_theta: List[float],
    follow_camera: bool = True,
    camera_half_width: float = 2.4,
) -> None:
    if not HAS_MATPLOTLIB or not HAS_IMAGEIO:
        print("matplotlib/imageio missing -> skip animation gif.")
        return

    n = min(len(expert_x), len(amp_x))
    pole_len = 0.75
    max_abs_x = max(max(abs(v) for v in expert_x), max(abs(v) for v in amp_x))
    x_lim = max(2.6, max_abs_x + 0.6)

    frame_paths = []
    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        for ax, x, th, title, color in [
            (axes[0], expert_x[i], expert_theta[i], "Expert", "#009E73"),
            (axes[1], amp_x[i], amp_theta[i], "AMP", "#D55E00"),
        ]:
            cart_w = 0.35
            cart_h = 0.20
            cart = plt.Rectangle((x - cart_w / 2.0, -cart_h / 2.0), cart_w, cart_h, color="#444444")
            ax.add_patch(cart)

            px = x + pole_len * math.sin(th)
            py = cart_h / 2.0 + pole_len * math.cos(th)
            ax.plot([x, px], [cart_h / 2.0, py], color=color, linewidth=3)
            ax.scatter([px], [py], s=110, color=color)

            ax.axhline(-0.25, color="black", linewidth=1.0)
            if follow_camera:
                ax.set_xlim(x - camera_half_width, x + camera_half_width)
            else:
                ax.set_xlim(-x_lim, x_lim)
            ax.set_ylim(-0.55, 1.25)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.set_title(f"{title} | t={i}")

        fig.tight_layout()
        frame_path = save_dir / f"amp_cartpole_swingup_frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = save_dir / "amp_cartpole_swingup_compare.gif"
    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, images, duration=0.03)
    print(f"Saved: {gif_path}")

    for path in frame_paths:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Args + main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AMP policy study: cart-pole swing-up + balance")
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--steps-per-iter", type=int, default=4096)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--lr-policy", type=float, default=3e-4)
    p.add_argument("--lr-disc", type=float, default=1e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=8)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--disc-updates", type=int, default=4)
    p.add_argument("--disc-batch-size", type=int, default=512)
    p.add_argument("--gp-weight", type=float, default=5.0)
    p.add_argument("--task-weight", type=float, default=0.3)
    p.add_argument("--style-weight", type=float, default=0.7)
    p.add_argument("--expert-episodes", type=int, default=150)
    p.add_argument("--expert-horizon", type=int, default=300)
    p.add_argument("--eval-episodes", type=int, default=6)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--global-camera", action="store_true")
    p.add_argument("--camera-half-width", type=float, default=2.4)
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--skip-gif", action="store_true")
    p.add_argument("--skip-bonus", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = load_base_module()

    print(f"Running on: {device}")
    print("Study case: cart-pole swing-up + balance with AMP (PPO policy + transition discriminator)")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir.resolve()}")

    env = base.InvertedPendulumOnCart(base.CartPoleConfig(), device=device)

    print("\n[1/3] Generating expert transitions (state-only, no actions needed by D)...")
    s_expert, sp_expert = generate_expert_transitions(env, base, args.expert_episodes, args.expert_horizon, device)
    print(f"Expert transitions: {s_expert.size(0)}")

    policy = GaussianPolicy(state_dim=5, action_dim=1).to(device)
    value_net = ValueNetwork(state_dim=5).to(device)
    discriminator = AMPDiscriminator(state_dim=5).to(device)

    ppo_optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=args.lr_policy)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc)

    reward_hist: List[float] = []
    task_reward_hist: List[float] = []
    style_reward_hist: List[float] = []
    disc_loss_hist: List[float] = []
    eval_steps: List[int] = []
    expert_theta_curve: List[float] = []
    amp_theta_curve: List[float] = []
    expert_sr_curve: List[float] = []
    amp_sr_curve: List[float] = []
    best_sr = -1.0
    best_theta = float("inf")
    best_state = {"policy": None, "value": None}
    episode_index = 0

    print("\n[2/3] Training AMP policy (rollout -> update D -> update policy via PPO)...")
    for iteration in range(1, args.iterations + 1):
        batch, stats, episode_index = collect_batch(
            env, base, policy, value_net, discriminator, device,
            steps_per_iter=args.steps_per_iter, horizon=args.horizon,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            task_weight=args.task_weight, style_weight=args.style_weight,
            start_episode_index=episode_index,
        )

        d_loss = update_discriminator(
            discriminator, disc_optimizer,
            s_expert, sp_expert, batch["policy_feats"], batch["policy_next_feats"],
            batch_size=args.disc_batch_size, updates=args.disc_updates, gp_weight=args.gp_weight,
        )

        p_loss, v_loss, entropy = update_ppo(
            policy, value_net, ppo_optimizer,
            batch["states"], batch["actions"], batch["advantages"], batch["returns"], batch["old_log_probs"],
            clip_eps=args.clip_eps, ppo_epochs=args.ppo_epochs, minibatch_size=args.minibatch_size,
            value_coef=args.value_coef, entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
        )

        reward_hist.append(stats["combined_reward"])
        task_reward_hist.append(stats["task_reward"])
        style_reward_hist.append(stats["style_reward"])
        disc_loss_hist.append(d_loss)

        if iteration % args.log_interval == 0 or iteration == 1:
            print(
                f"Iter {iteration:4d} | reward={stats['combined_reward']:7.3f} "
                f"(task={stats['task_reward']:6.3f}, style={stats['style_reward']:6.3f}) | "
                f"d_loss={d_loss:.4f} | p_loss={p_loss:.4f} | v_loss={v_loss:.4f} | entropy={entropy:.4f}"
            )

            e_theta, _, e_sr = base.evaluate_policy(
                env, policy_name=f"Expert@{iteration}", episodes=args.eval_episodes, horizon=360,
                policy_fn=lambda st: base.expert_policy(st, env.cfg), device=device, verbose=False,
            )
            a_theta, _, a_sr = base.evaluate_policy(
                env, policy_name=f"AMP@{iteration}", episodes=args.eval_episodes, horizon=360,
                policy_fn=lambda st: amp_policy_action(policy, base, st, env.cfg.force_max, deterministic=True),
                device=device, verbose=False,
            )

            eval_steps.append(iteration)
            expert_theta_curve.append(e_theta)
            amp_theta_curve.append(a_theta)
            expert_sr_curve.append(e_sr)
            amp_sr_curve.append(a_sr)
            print(f"  Eval@{iteration}: expert_theta={e_theta:.3f}, amp_theta={a_theta:.3f}, expert_sr={100.0*e_sr:.1f}%, amp_sr={100.0*a_sr:.1f}%")

            if (a_sr > best_sr) or (abs(a_sr - best_sr) < 1e-9 and a_theta < best_theta):
                best_sr = a_sr
                best_theta = a_theta
                best_state = {"policy": {k: v.clone() for k, v in policy.state_dict().items()},
                               "value": {k: v.clone() for k, v in value_net.state_dict().items()}}

    if best_state["policy"] is not None:
        policy.load_state_dict(best_state["policy"])
        value_net.load_state_dict(best_state["value"])

    if not args.skip_plots:
        plot_curves_amp(
            output_dir, reward_hist, task_reward_hist, style_reward_hist, disc_loss_hist,
            eval_steps, expert_theta_curve, amp_theta_curve, expert_sr_curve, amp_sr_curve,
        )

    print("\n[3/3] Evaluating policies on random initial states...")
    base.evaluate_policy(
        env, policy_name="Expert", episodes=args.eval_episodes, horizon=420,
        policy_fn=lambda s: base.expert_policy(s, env.cfg), device=device, verbose=True,
    )
    amp_policy_fn = lambda s: amp_policy_action(policy, base, s, env.cfg.force_max, deterministic=True)
    base.evaluate_policy(
        env, policy_name="AMP", episodes=args.eval_episodes, horizon=420,
        policy_fn=amp_policy_fn, device=device, verbose=True,
    )

    if not args.skip_bonus:
        print("\n[Bonus] Plot x/theta trajectories on the same initial state...")
        init_state = torch.tensor([[0.0, 0.0, torch.pi - 0.15, 0.0]], device=device)

        expert_x, expert_theta = base.rollout_trajectory(env, lambda st: base.expert_policy(st, env.cfg), init_state, horizon=520)
        amp_x, amp_theta = base.rollout_trajectory(env, amp_policy_fn, init_state, horizon=520)

        plot_trajectory_amp(output_dir, env.cfg.dt, expert_x, expert_theta, amp_x, amp_theta)
        if not args.skip_gif:
            make_cartpole_gif_amp(
                output_dir, expert_x, expert_theta, amp_x, amp_theta,
                follow_camera=(not args.global_camera), camera_half_width=args.camera_half_width,
            )

    print("\nDone. This study case learns a swing-up + balance controller with AMP: PPO policy + LSGAN transition discriminator reward.")


if __name__ == "__main__":
    main()
