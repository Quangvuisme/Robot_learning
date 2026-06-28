import argparse
import importlib.util
import math
import random
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
    base_path = Path(__file__).resolve().parents[1] / "diffusion" / "diffusion_cartpole_study.py"
    spec = importlib.util.spec_from_file_location("diffusion_cartpole_base", base_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class ConditionalVAE(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, latent_dim: int = 8, hidden: int = 192):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.enc_mu = nn.Linear(hidden, latent_dim)
        self.enc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        h = self.encoder(torch.cat([state, action], dim=-1))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(torch.cat([state, z], dim=-1))
        return recon, mu, logvar

    @torch.no_grad()
    def sample_action(self, state: torch.Tensor) -> torch.Tensor:
        z = torch.randn(state.size(0), self.latent_dim, device=state.device)
        return self.decoder(torch.cat([state, z], dim=-1))


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rename_output_file(output_dir: Path, old_name: str, new_name: str) -> None:
    src = output_dir / old_name
    dst = output_dir / new_name
    if src.exists():
        src.replace(dst)
        print(f"Saved: {dst}")


def plot_curves_vae(
    save_dir: Path,
    loss_history: list[float],
    eval_steps: list[int],
    expert_theta: list[float],
    vae_theta: list[float],
    expert_sr: list[float],
    vae_sr: list[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plotting curves.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(loss_history, color="#0072B2", linewidth=1.3)
    axes[0].set_title("Training Loss (VAE)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(eval_steps, expert_theta, "-o", color="#009E73", label="Expert")
    axes[1].plot(eval_steps, vae_theta, "-o", color="#D55E00", label="VAE")
    axes[1].set_title("Mean Tail |theta| vs Epoch")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Radians (lower is better)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(eval_steps, [100.0 * x for x in expert_sr], "-o", color="#009E73", label="Expert")
    axes[2].plot(eval_steps, [100.0 * x for x in vae_sr], "-o", color="#D55E00", label="VAE")
    axes[2].set_title("Success Rate vs Epoch")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Success %")
    axes[2].set_ylim(0.0, 105.0)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    out_file = save_dir / "vae_training_curves_cartpole.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_trajectory_vae(
    save_dir: Path,
    dt: float,
    expert_x: list[float],
    expert_theta: list[float],
    vae_x: list[float],
    vae_theta: list[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip trajectory plots.")
        return

    t = [i * dt for i in range(len(expert_theta))]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, expert_theta, color="#009E73", linewidth=2.0, label="Expert")
    axes[0].plot(t, vae_theta, color="#D55E00", linewidth=2.0, label="VAE")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("theta (rad)")
    axes[0].set_title("Pole Angle Trajectory")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(t, expert_x, color="#009E73", linewidth=2.0, label="Expert")
    axes[1].plot(t, vae_x, color="#D55E00", linewidth=2.0, label="VAE")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("x (m)")
    axes[1].set_title("Cart Position Trajectory")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "vae_cartpole_trajectory.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def make_cartpole_gif_vae(
    save_dir: Path,
    expert_x: list[float],
    expert_theta: list[float],
    vae_x: list[float],
    vae_theta: list[float],
) -> None:
    if not HAS_MATPLOTLIB or not HAS_IMAGEIO:
        print("matplotlib/imageio missing -> skip animation gif.")
        return

    frame_paths = []
    n = min(len(expert_x), len(vae_x))
    pole_len = 0.75
    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        for ax, x, th, title, color in [
            (axes[0], expert_x[i], expert_theta[i], "Expert", "#009E73"),
            (axes[1], vae_x[i], vae_theta[i], "VAE", "#D55E00"),
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
            ax.set_xlim(-2.6, 2.6)
            ax.set_ylim(-0.55, 1.25)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.set_title(f"{title} | t={i}")

        fig.tight_layout()
        frame_path = save_dir / f"vae_cartpole_frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = save_dir / "vae_cartpole_compare.gif"
    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, images, duration=0.03)
    print(f"Saved: {gif_path}")

    for path in frame_paths:
        path.unlink(missing_ok=True)


@torch.no_grad()
def vae_policy_action(
    model: ConditionalVAE,
    state: torch.Tensor,
    s_mean: torch.Tensor,
    s_std: torch.Tensor,
    a_mean: torch.Tensor,
    a_std: torch.Tensor,
    force_max: float,
    n_action_samples: int = 8,
) -> torch.Tensor:
    s_norm = (state - s_mean) / s_std
    actions = []
    for _ in range(n_action_samples):
        a_norm = model.sample_action(s_norm)
        actions.append(a_norm * a_std + a_mean)
    a = torch.stack(actions, dim=0).median(dim=0).values
    return torch.clamp(a, -force_max, force_max)


def train_vae_policy(
    model: ConditionalVAE,
    env,
    base,
    states: torch.Tensor,
    actions: torch.Tensor,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    beta_kl: float,
    log_interval: int,
):
    s_mean = states.mean(dim=0, keepdim=True)
    s_std = states.std(dim=0, keepdim=True) + 1e-6
    a_mean = actions.mean(dim=0, keepdim=True)
    a_std = actions.std(dim=0, keepdim=True) + 1e-6

    s_norm = (states - s_mean) / s_std
    a_norm = (actions - a_mean) / a_std

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = states.size(0)

    loss_history = []
    eval_steps = []
    expert_curve = []
    vae_curve = []
    expert_sr_curve = []
    vae_sr_curve = []

    for step in range(1, epochs + 1):
        idx = torch.randint(0, n, (batch_size,), device=states.device)
        s = s_norm[idx]
        a = a_norm[idx]

        recon, mu, logvar = model(s, a)
        recon_loss = F.mse_loss(recon, a)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta_kl * kl

        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_history.append(loss.item())

        if step % log_interval == 0:
            print(f"Train step {step:4d} | loss={loss.item():.6f} recon={recon_loss.item():.6f} kl={kl.item():.6f}")
            e_theta, _, e_sr = base.evaluate_policy(
                env,
                policy_name=f"Expert@{step}",
                episodes=8,
                horizon=120,
                policy_fn=lambda st: base.expert_policy(st, env.cfg),
                device=device,
                verbose=False,
            )
            v_theta, _, v_sr = base.evaluate_policy(
                env,
                policy_name=f"VAE@{step}",
                episodes=8,
                horizon=120,
                policy_fn=lambda st: vae_policy_action(
                    model,
                    st,
                    s_mean,
                    s_std,
                    a_mean,
                    a_std,
                    env.cfg.force_max,
                    n_action_samples=8,
                ),
                device=device,
                verbose=False,
            )

            eval_steps.append(step)
            expert_curve.append(e_theta)
            vae_curve.append(v_theta)
            expert_sr_curve.append(e_sr)
            vae_sr_curve.append(v_sr)
            print(
                f"  Eval@{step}: expert_theta={e_theta:.3f}, vae_theta={v_theta:.3f}, "
                f"expert_sr={100.0 * e_sr:.1f}%, vae_sr={100.0 * v_sr:.1f}%"
            )

    return (
        s_mean,
        s_std,
        a_mean,
        a_std,
        loss_history,
        eval_steps,
        expert_curve,
        vae_curve,
        expert_sr_curve,
        vae_sr_curve,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE policy study: cart-pole balance")
    p.add_argument("--episodes", type=int, default=260)
    p.add_argument("--horizon", type=int, default=180)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta-kl", type=float, default=1e-3)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = load_base_module()

    print(f"Running on: {device}")
    print("Study case: inverted pendulum on cart balance with conditional VAE policy")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir.resolve()}")

    env = base.InvertedPendulumOnCart(base.CartPoleConfig(), device=device)

    print("\n[1/3] Generating expert dataset...")
    states, actions = base.generate_dataset(
        env,
        episodes=args.episodes,
        horizon=args.horizon,
        device=device,
    )
    print(f"Dataset size: {states.size(0)} samples")

    print("\n[2/3] Training conditional VAE policy...")
    model = ConditionalVAE(state_dim=4, action_dim=1, latent_dim=8, hidden=192).to(device)

    (
        s_mean,
        s_std,
        a_mean,
        a_std,
        loss_history,
        eval_steps,
        expert_curve,
        vae_curve,
        expert_sr_curve,
        vae_sr_curve,
    ) = train_vae_policy(
        model,
        env,
        base,
        states,
        actions,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_kl=args.beta_kl,
        log_interval=args.log_interval,
    )

    plot_curves_vae(
        output_dir,
        loss_history,
        eval_steps,
        expert_curve,
        vae_curve,
        expert_sr_curve,
        vae_sr_curve,
    )

    print("\n[3/3] Evaluating policies on random initial states...")
    base.evaluate_policy(
        env,
        policy_name="Expert",
        episodes=20,
        horizon=140,
        policy_fn=lambda s: base.expert_policy(s, env.cfg),
        device=device,
    )
    base.evaluate_policy(
        env,
        policy_name="VAE",
        episodes=20,
        horizon=140,
        policy_fn=lambda s: vae_policy_action(
            model,
            s,
            s_mean,
            s_std,
            a_mean,
            a_std,
            env.cfg.force_max,
            n_action_samples=10,
        ),
        device=device,
    )

    print("\n[Bonus] Plot x/theta trajectories on same initial state...")
    init_state = torch.tensor([[0.3, 0.0, 0.28, 0.9]], device=device)
    expert_x, expert_theta = base.rollout_trajectory(
        env,
        lambda s: base.expert_policy(s, env.cfg),
        init_state,
        horizon=180,
    )
    vae_x, vae_theta = base.rollout_trajectory(
        env,
        lambda s: vae_policy_action(
            model,
            s,
            s_mean,
            s_std,
            a_mean,
            a_std,
            env.cfg.force_max,
            n_action_samples=10,
        ),
        init_state,
        horizon=180,
    )

    plot_trajectory_vae(output_dir, env.cfg.dt, expert_x, expert_theta, vae_x, vae_theta)
    make_cartpole_gif_vae(output_dir, expert_x, expert_theta, vae_x, vae_theta)

    print("\nDone. This study case learns p(force | state) with a conditional VAE.")


if __name__ == "__main__":
    main()
