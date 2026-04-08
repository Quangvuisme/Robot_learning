import argparse
import importlib.util
import csv
import math
import random
import copy
from pathlib import Path

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


class ConditionalGenerator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, z_dim: int = 12, hidden: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + z_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, z], dim=-1))


class ConditionalDiscriminator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_curves_gan(
    save_dir: Path,
    g_losses: list[float],
    d_losses: list[float],
    eval_steps: list[int],
    expert_theta: list[float],
    gan_theta: list[float],
    expert_sr: list[float],
    gan_sr: list[float],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "gan_training_curves_cartpole_swingup.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "gen_loss",
                "disc_loss",
                "expert_tail_theta",
                "gan_tail_theta",
                "expert_success",
                "gan_success",
            ]
        )
        for i, step in enumerate(eval_steps):
            g_val = g_losses[step - 1] if 0 <= (step - 1) < len(g_losses) else float("nan")
            d_val = d_losses[step - 1] if 0 <= (step - 1) < len(d_losses) else float("nan")
            writer.writerow([step, g_val, d_val, expert_theta[i], gan_theta[i], expert_sr[i], gan_sr[i]])
    print(f"Saved: {csv_path}")

    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip PNG plot (CSV saved instead).")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(g_losses, color="#0072B2", linewidth=1.3, label="Gen")
    axes[0].plot(d_losses, color="#D55E00", linewidth=1.2, label="Disc")
    axes[0].set_title("Training Loss (GAN)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("WGAN-GP loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(eval_steps, expert_theta, "-o", color="#009E73", label="Expert")
    axes[1].plot(eval_steps, gan_theta, "-o", color="#D55E00", label="GAN")
    axes[1].set_title("Mean Tail |theta| vs Epoch")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Radians (lower is better)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(eval_steps, [100.0 * x for x in expert_sr], "-o", color="#009E73", label="Expert")
    axes[2].plot(eval_steps, [100.0 * x for x in gan_sr], "-o", color="#D55E00", label="GAN")
    axes[2].set_title("Success Rate vs Epoch")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Success %")
    axes[2].set_ylim(0.0, 105.0)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    out_file = save_dir / "gan_training_curves_cartpole_swingup.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_trajectory_gan(
    save_dir: Path,
    dt: float,
    expert_x: list[float],
    expert_theta: list[float],
    gan_x: list[float],
    gan_theta: list[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip trajectory plot.")
        return

    t = [i * dt for i in range(len(expert_theta))]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, expert_theta, color="#009E73", linewidth=2.0, label="Expert")
    axes[0].plot(t, gan_theta, color="#D55E00", linewidth=2.0, label="GAN")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("theta (rad)")
    axes[0].set_title("Pole Angle Trajectory")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(t, expert_x, color="#009E73", linewidth=2.0, label="Expert")
    axes[1].plot(t, gan_x, color="#D55E00", linewidth=2.0, label="GAN")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("x (m)")
    axes[1].set_title("Cart Position Trajectory")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "gan_cartpole_swingup_trajectory.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def make_cartpole_gif_gan(
    save_dir: Path,
    expert_x: list[float],
    expert_theta: list[float],
    gan_x: list[float],
    gan_theta: list[float],
    follow_camera: bool = True,
    camera_half_width: float = 2.4,
) -> None:
    if not HAS_MATPLOTLIB or not HAS_IMAGEIO:
        print("matplotlib/imageio missing -> skip animation gif.")
        return

    n = min(len(expert_x), len(gan_x))
    pole_len = 0.75
    max_abs_x = max(max(abs(v) for v in expert_x), max(abs(v) for v in gan_x))
    x_lim = max(2.6, max_abs_x + 0.6)

    frame_paths = []
    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        for ax, x, th, title, color in [
            (axes[0], expert_x[i], expert_theta[i], "Expert", "#009E73"),
            (axes[1], gan_x[i], gan_theta[i], "GAN", "#D55E00"),
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
        frame_path = save_dir / f"gan_cartpole_swingup_frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = save_dir / "gan_cartpole_swingup_compare.gif"
    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, images, duration=0.03)
    print(f"Saved: {gif_path}")

    for path in frame_paths:
        path.unlink(missing_ok=True)


def downsample_near_upright(states: torch.Tensor, actions: torch.Tensor, keep_prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    if keep_prob >= 0.999:
        return states, actions
    theta = (states[:, 2:3] + torch.pi) % (2.0 * torch.pi) - torch.pi
    theta_dot = states[:, 3:4]
    near_upright = (torch.abs(theta) < 0.55) & (torch.abs(theta_dot) < 3.5)
    keep_mask = (~near_upright.squeeze(-1)) | (torch.rand(states.size(0), device=states.device) < keep_prob)
    return states[keep_mask], actions[keep_mask]


def pretrain_generator_bc(
    generator: ConditionalGenerator,
    states_feat: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
    lr: float,
    steps: int,
) -> None:
    if steps <= 0:
        return
    opt = torch.optim.Adam(generator.parameters(), lr=lr)
    n = states_feat.size(0)
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch_size,), device=states_feat.device)
        s = states_feat[idx]
        a = actions[idx]
        z = torch.zeros(s.size(0), generator.z_dim, device=s.device)
        pred = generator(s, z)
        loss = F.mse_loss(pred, a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(50, steps // 5) == 0:
            print(f"BC pretrain step {step:4d} | mse={loss.item():.6f}")


@torch.no_grad()
def collect_student_rollouts(
    env,
    base,
    generator: ConditionalGenerator,
    s_mean: torch.Tensor,
    s_std: torch.Tensor,
    a_mean: torch.Tensor,
    a_std: torch.Tensor,
    episodes: int,
    horizon: int,
    device: str,
    stochastic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    states_norm = []
    actions_norm = []

    for _ in range(episodes):
        s = env.sample_swingup_state(device)
        for _ in range(horizon):
            s_feat = base.state_features(s)
            s_n = (s_feat - s_mean) / s_std

            if stochastic:
                z = torch.randn(s.size(0), generator.z_dim, device=device)
            else:
                z = torch.zeros(s.size(0), generator.z_dim, device=device)

            a_n = generator(s_n, z)
            a = torch.clamp(a_n * a_std + a_mean, -env.cfg.force_max, env.cfg.force_max)
            a_exec_n = (a - a_mean) / a_std

            states_norm.append(s_n.detach().clone())
            actions_norm.append(a_exec_n.detach().clone())
            s = env.step(s, a)

    if not states_norm:
        return torch.empty(0, s_mean.size(-1), device=device), torch.empty(0, 1, device=device)
    return torch.cat(states_norm, dim=0), torch.cat(actions_norm, dim=0)


@torch.no_grad()
def gan_policy_action(
    generator: ConditionalGenerator,
    base,
    state: torch.Tensor,
    s_mean: torch.Tensor,
    s_std: torch.Tensor,
    a_mean: torch.Tensor,
    a_std: torch.Tensor,
    force_max: float,
    n_action_samples: int = 12,
    action_gain: float = 1.0,
    deterministic: bool = True,
) -> torch.Tensor:
    s_feat = base.state_features(state)
    s_norm = (s_feat - s_mean) / s_std
    actions = []
    draws = 1 if deterministic else n_action_samples
    for _ in range(draws):
        if deterministic:
            z = torch.zeros(state.size(0), generator.z_dim, device=state.device)
        else:
            z = torch.randn(state.size(0), generator.z_dim, device=state.device)
        a_norm = generator(s_norm, z)
        actions.append(a_norm * a_std + a_mean)
    a = torch.stack(actions, dim=0).median(dim=0).values
    a = action_gain * a
    return torch.clamp(a, -force_max, force_max)


@torch.no_grad()
def gan_hybrid_policy_action(
    generator: ConditionalGenerator,
    base,
    state: torch.Tensor,
    s_mean: torch.Tensor,
    s_std: torch.Tensor,
    a_mean: torch.Tensor,
    a_std: torch.Tensor,
    cfg,
    n_action_samples: int = 12,
    action_gain: float = 1.1,
    deterministic: bool = True,
) -> torch.Tensor:
    theta = (state[:, 2:3] + torch.pi) % (2.0 * torch.pi) - torch.pi
    theta_dot = state[:, 3:4]
    near_upright = (torch.abs(theta) < 0.55) & (torch.abs(theta_dot) < 3.5)

    u_balance = base.balance_controller(state, cfg)
    u_gan = gan_policy_action(
        generator,
        base,
        state,
        s_mean,
        s_std,
        a_mean,
        a_std,
        cfg.force_max,
        n_action_samples=n_action_samples,
        action_gain=action_gain,
        deterministic=deterministic,
    )
    return torch.where(near_upright, u_balance, u_gan)


def train_gail_policy(
    generator: ConditionalGenerator,
    discriminator: ConditionalDiscriminator,
    env,
    base,
    states: torch.Tensor,
    actions: torch.Tensor,
    device: str,
    epochs: int,
    batch_size: int,
    lr_g: float,
    lr_d: float,
    log_interval: int,
    eval_episodes: int,
    rollout_episodes: int,
    rollout_horizon: int,
    disc_updates: int,
    gen_updates: int,
    eval_action_gain: float,
    eval_use_hybrid: bool,
    bc_weight: float,
    stochastic_train: bool,
):
    states_feat = base.state_features(states)

    s_mean = states_feat.mean(dim=0, keepdim=True)
    s_std = states_feat.std(dim=0, keepdim=True) + 1e-6
    a_mean = actions.mean(dim=0, keepdim=True)
    a_std = actions.std(dim=0, keepdim=True) + 1e-6

    s_norm = (states_feat - s_mean) / s_std
    a_norm = (actions - a_mean) / a_std

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()
    n = states.size(0)
    g_losses = []
    d_losses = []
    eval_steps = []
    expert_theta_curve = []
    gan_theta_curve = []
    expert_sr_curve = []
    gan_sr_curve = []
    best_eval = float("inf")
    best_sr = -1.0
    best_state = copy.deepcopy(generator.state_dict())

    for step in range(1, epochs + 1):
        # Step 1: sample trajectories from student policy.
        s_student, a_student = collect_student_rollouts(
            env,
            base,
            generator,
            s_mean,
            s_std,
            a_mean,
            a_std,
            episodes=rollout_episodes,
            horizon=rollout_horizon,
            device=device,
            stochastic=stochastic_train,
        )

        if s_student.size(0) == 0:
            d_loss = torch.tensor(0.0, device=states.device)
            g_loss = torch.tensor(0.0, device=states.device)
            continue

        # Step 2: update discriminator (classify expert vs student).
        d_loss = torch.tensor(0.0, device=states.device)
        n_student = s_student.size(0)
        for _ in range(disc_updates):
            idx_e = torch.randint(0, n, (batch_size,), device=states.device)
            idx_s = torch.randint(0, n_student, (batch_size,), device=states.device)

            s_e = s_norm[idx_e]
            a_e = a_norm[idx_e]
            s_s = s_student[idx_s]
            a_s = a_student[idx_s]

            logits_e = discriminator(s_e, a_e)
            logits_s = discriminator(s_s, a_s)
            d_loss_real = bce(logits_e, torch.ones_like(logits_e))
            d_loss_fake = bce(logits_s, torch.zeros_like(logits_s))
            d_loss = d_loss_real + d_loss_fake

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        # Step 3: update student policy to fool discriminator.
        g_loss = torch.tensor(0.0, device=states.device)
        for _ in range(gen_updates):
            idx_s = torch.randint(0, n_student, (batch_size,), device=states.device)
            s = s_student[idx_s]

            z = torch.randn(batch_size, generator.z_dim, device=states.device)
            a_pred = generator(s, z)
            logits = discriminator(s, a_pred)
            adv_loss = bce(logits, torch.ones_like(logits))

            if bc_weight > 0.0:
                idx_e = torch.randint(0, n, (batch_size,), device=states.device)
                s_e = s_norm[idx_e]
                a_e = a_norm[idx_e]
                z0 = torch.zeros(batch_size, generator.z_dim, device=states.device)
                a_det = generator(s_e, z0)
                bc_loss = F.mse_loss(a_det, a_e)
            else:
                bc_loss = torch.tensor(0.0, device=states.device)

            g_loss = adv_loss + bc_weight * bc_loss

            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            opt_g.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        if step % log_interval == 0:
            print(f"Train step {step:4d} | g_loss={g_loss.item():.6f} d_loss={d_loss.item():.6f}")

            e_theta, _, e_sr = base.evaluate_policy(
                env,
                policy_name=f"Expert@{step}",
                episodes=eval_episodes,
                horizon=360,
                policy_fn=lambda st: base.expert_policy(st, env.cfg),
                device=device,
                verbose=False,
            )
            if eval_use_hybrid:
                eval_policy_fn = lambda st: gan_hybrid_policy_action(
                    generator,
                    base,
                    st,
                    s_mean,
                    s_std,
                    a_mean,
                    a_std,
                    env.cfg,
                    n_action_samples=4,
                    action_gain=eval_action_gain,
                    deterministic=True,
                )
            else:
                eval_policy_fn = lambda st: gan_policy_action(
                    generator,
                    base,
                    st,
                    s_mean,
                    s_std,
                    a_mean,
                    a_std,
                    env.cfg.force_max,
                    n_action_samples=6,
                    action_gain=eval_action_gain,
                    deterministic=True,
                )
            g_theta, _, g_sr = base.evaluate_policy(
                env,
                policy_name=f"GAN@{step}",
                episodes=eval_episodes,
                horizon=360,
                policy_fn=eval_policy_fn,
                device=device,
                verbose=False,
            )

            eval_steps.append(step)
            expert_theta_curve.append(e_theta)
            gan_theta_curve.append(g_theta)
            expert_sr_curve.append(e_sr)
            gan_sr_curve.append(g_sr)

            print(
                f"  Eval@{step}: expert_theta={e_theta:.3f}, gan_theta={g_theta:.3f}, "
                f"expert_sr={100.0 * e_sr:.1f}%, gan_sr={100.0 * g_sr:.1f}%"
            )

            # Select checkpoint by success first, then tail-theta.
            if (g_sr > best_sr) or (abs(g_sr - best_sr) < 1e-9 and g_theta < best_eval):
                best_sr = g_sr
                best_eval = g_theta
                best_state = copy.deepcopy(generator.state_dict())

    # Restore best EMA checkpoint for final rollout/evaluation.
    generator.load_state_dict(best_state)

    return (
        s_mean,
        s_std,
        a_mean,
        a_std,
        g_losses,
        d_losses,
        eval_steps,
        expert_theta_curve,
        gan_theta_curve,
        expert_sr_curve,
        gan_sr_curve,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GAN policy study: cart-pole swing-up + balance")
    p.add_argument("--episodes", type=int, default=160)
    p.add_argument("--horizon", type=int, default=360)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr-g", type=float, default=3e-4)
    p.add_argument("--lr-d", type=float, default=3e-4)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=6)
    p.add_argument("--rollout-episodes", type=int, default=20)
    p.add_argument("--rollout-horizon", type=int, default=220)
    p.add_argument("--disc-updates", type=int, default=3)
    p.add_argument("--gen-updates", type=int, default=2)
    p.add_argument("--gan-gain", type=float, default=1.1)
    p.add_argument("--keep-near-upright", type=float, default=0.3)
    p.add_argument("--bc-pretrain-steps", type=int, default=500)
    p.add_argument("--bc-weight", type=float, default=0.2)
    p.add_argument("--stochastic-train", action="store_true")
    p.add_argument("--stochastic-policy", action="store_true")
    p.add_argument("--global-camera", action="store_true")
    p.add_argument("--camera-half-width", type=float, default=2.4)
    p.add_argument("--max-samples", type=int, default=60000)
    p.add_argument("--skip-expert-check", action="store_true")
    p.add_argument("--no-hybrid", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--skip-bonus", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = load_base_module()

    print(f"Running on: {device}")
    print("Study case: cart-pole swing-up + balance with conditional GAN policy")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir.resolve()}")

    env = base.InvertedPendulumOnCart(base.CartPoleConfig(), device=device)

    if not args.skip_expert_check:
        print("\n[0/3] Quick expert sanity check...")
        base.evaluate_policy(
            env,
            policy_name="Expert",
            episodes=20,
            horizon=420,
            policy_fn=lambda s: base.expert_policy(s, env.cfg),
            device=device,
            verbose=True,
        )

    print("\n[1/3] Generating expert dataset...")
    states, actions = base.generate_dataset(
        env,
        episodes=args.episodes,
        horizon=args.horizon,
        device=device,
    )
    states, actions = downsample_near_upright(states, actions, keep_prob=args.keep_near_upright)
    if args.max_samples > 0 and states.size(0) > args.max_samples:
        idx = torch.randperm(states.size(0), device=states.device)[: args.max_samples]
        states = states[idx]
        actions = actions[idx]
    print(f"Dataset size: {states.size(0)} samples")

    print("\n[2/3] Training GAIL-style policy...")
    generator = ConditionalGenerator(state_dim=5, action_dim=1, z_dim=12, hidden=256).to(device)
    discriminator = ConditionalDiscriminator(state_dim=5, action_dim=1, hidden=256, dropout=0.0).to(device)

    states_feat = base.state_features(states)
    s_mean = states_feat.mean(dim=0, keepdim=True)
    s_std = states_feat.std(dim=0, keepdim=True) + 1e-6
    a_mean = actions.mean(dim=0, keepdim=True)
    a_std = actions.std(dim=0, keepdim=True) + 1e-6
    s_norm = (states_feat - s_mean) / s_std
    a_norm = (actions - a_mean) / a_std

    if args.bc_pretrain_steps > 0:
        print("\n[2a/3] BC pretrain generator...")
        pretrain_generator_bc(
            generator,
            s_norm,
            a_norm,
            batch_size=args.batch_size,
            lr=args.lr_g,
            steps=args.bc_pretrain_steps,
        )

    (
        s_mean,
        s_std,
        a_mean,
        a_std,
        g_losses,
        d_losses,
        eval_steps,
        expert_theta_curve,
        gan_theta_curve,
        expert_sr_curve,
        gan_sr_curve,
    ) = train_gail_policy(
        generator,
        discriminator,
        env,
        base,
        states,
        actions,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        log_interval=args.log_interval,
        eval_episodes=args.eval_episodes,
        rollout_episodes=args.rollout_episodes,
        rollout_horizon=args.rollout_horizon,
        disc_updates=args.disc_updates,
        gen_updates=args.gen_updates,
        eval_action_gain=args.gan_gain,
        eval_use_hybrid=(not args.no_hybrid),
        bc_weight=args.bc_weight,
        stochastic_train=args.stochastic_train,
    )

    if not args.skip_plots:
        plot_curves_gan(
            output_dir,
            g_losses,
            d_losses,
            eval_steps,
            expert_theta_curve,
            gan_theta_curve,
            expert_sr_curve,
            gan_sr_curve,
        )

    print("\n[3/3] Evaluating policies on random initial states...")
    base.evaluate_policy(
        env,
        policy_name="Expert",
        episodes=args.eval_episodes,
        horizon=420,
        policy_fn=lambda s: base.expert_policy(s, env.cfg),
        device=device,
        verbose=True,
    )

    if args.no_hybrid:
        gan_policy = lambda s: gan_policy_action(
            generator,
            base,
            s,
            s_mean,
            s_std,
            a_mean,
            a_std,
            env.cfg.force_max,
            n_action_samples=12,
            action_gain=args.gan_gain,
            deterministic=(not args.stochastic_policy),
        )
    else:
        gan_policy = lambda s: gan_hybrid_policy_action(
            generator,
            base,
            s,
            s_mean,
            s_std,
            a_mean,
            a_std,
            env.cfg,
            n_action_samples=12,
            action_gain=args.gan_gain,
            deterministic=(not args.stochastic_policy),
        )

    base.evaluate_policy(
        env,
        policy_name="GAN",
        episodes=args.eval_episodes,
        horizon=420,
        policy_fn=gan_policy,
        device=device,
        verbose=True,
    )

    if not args.skip_bonus:
        print("\n[Bonus] Plot x/theta trajectories on the same initial state...")
        init_state = torch.tensor([[0.0, 0.0, torch.pi - 0.15, 0.0]], device=device)

        expert_x, expert_theta = base.rollout_trajectory(
            env,
            lambda st: base.expert_policy(st, env.cfg),
            init_state,
            horizon=520,
        )
        gan_x, gan_theta = base.rollout_trajectory(
            env,
            gan_policy,
            init_state,
            horizon=520,
        )

        plot_trajectory_gan(output_dir, env.cfg.dt, expert_x, expert_theta, gan_x, gan_theta)
        make_cartpole_gif_gan(
            output_dir,
            expert_x,
            expert_theta,
            gan_x,
            gan_theta,
            follow_camera=(not args.global_camera),
            camera_half_width=args.camera_half_width,
        )

    print("\nDone. This study case learns p(force | state) with a conditional GAN.")


if __name__ == "__main__":
    main()
