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
    base_path = Path(__file__).resolve().parents[1] / "diffusion" / "diffusion_inverted_pendulum_study.py"
    spec = importlib.util.spec_from_file_location("diffusion_inverted_base", base_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class ConditionalVAE(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, latent_dim: int = 8, hidden: int = 128):
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

    def encode(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([state, action], dim=-1))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([state, z], dim=-1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(state, action)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(state, z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample_action(self, state: torch.Tensor) -> torch.Tensor:
        z = torch.randn(state.size(0), self.latent_dim, device=state.device)
        return self.decode(state, z)


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
    expert_err: list[float],
    vae_err: list[float],
    expert_sr: list[float],
    vae_sr: list[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plotting curves.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].plot(loss_history, color="#0072B2", linewidth=1.4)
    axes[0].set_title("Training Loss (VAE)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(eval_steps, expert_err, "-o", color="#009E73", label="Expert")
    axes[1].plot(eval_steps, vae_err, "-o", color="#D55E00", label="VAE")
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
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    out_file = save_dir / "vae_training_curves_pendulum.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_theta_trajectory_vae(save_dir: Path, dt: float, expert_theta: list[float], vae_theta: list[float]) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip trajectory plot.")
        return

    def unwrap(theta_values: list[float]) -> list[float]:
        if not theta_values:
            return theta_values
        out = [theta_values[0]]
        offset = 0.0
        for i in range(1, len(theta_values)):
            delta = theta_values[i] - theta_values[i - 1]
            if delta > math.pi:
                offset -= 2.0 * math.pi
            elif delta < -math.pi:
                offset += 2.0 * math.pi
            out.append(theta_values[i] + offset)
        return out

    t = [i * dt for i in range(len(expert_theta))]
    expert_u = unwrap(expert_theta)
    vae_u = unwrap(vae_theta)

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.plot(t, expert_u, color="#009E73", linewidth=2.0, label="Expert")
    ax.plot(t, vae_u, color="#D55E00", linewidth=2.0, label="VAE")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_title("Theta Trajectory Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("theta (rad)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_file = save_dir / "vae_theta_trajectory.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def make_pendulum_gif_vae(save_dir: Path, expert_theta: list[float], vae_theta: list[float]) -> None:
    if not HAS_MATPLOTLIB or not HAS_IMAGEIO:
        print("matplotlib/imageio missing -> skip animation gif.")
        return

    length = 1.0
    frame_paths = []
    n = min(len(expert_theta), len(vae_theta))
    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
        for ax, th, title, color in [
            (axes[0], expert_theta[i], "Expert", "#009E73"),
            (axes[1], vae_theta[i], "VAE", "#D55E00"),
        ]:
            x = length * math.sin(th)
            y = length * math.cos(th)
            ax.plot([0, x], [0, y], color=color, linewidth=3)
            ax.scatter([x], [y], s=130, color=color)
            ax.scatter([0], [0], s=35, color="black")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.set_title(f"{title} | t={i}")

        fig.tight_layout()
        frame_path = save_dir / f"vae_frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = save_dir / "vae_pendulum_compare.gif"
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
    u_max: float,
    n_action_samples: int = 8,
) -> torch.Tensor:
    s_norm = (state - s_mean) / s_std
    actions = []
    for _ in range(n_action_samples):
        a_norm = model.sample_action(s_norm)
        actions.append(a_norm * a_std + a_mean)
    a = torch.stack(actions, dim=0).median(dim=0).values
    return torch.clamp(a, -u_max, u_max)


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
    eval_episodes: int,
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
    expert_err_curve = []
    vae_err_curve = []
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

            e_err, e_sr = base.evaluate_policy(
                env,
                policy_name=f"Expert@{step}",
                episodes=eval_episodes,
                horizon=100,
                policy_fn=lambda st: base.expert_policy(st, env.cfg),
                device=device,
                verbose=False,
            )
            v_err, v_sr = base.evaluate_policy(
                env,
                policy_name=f"VAE@{step}",
                episodes=eval_episodes,
                horizon=100,
                policy_fn=lambda st: vae_policy_action(
                    model,
                    st,
                    s_mean,
                    s_std,
                    a_mean,
                    a_std,
                    env.cfg.u_max,
                    n_action_samples=8,
                ),
                device=device,
                verbose=False,
            )

            eval_steps.append(step)
            expert_err_curve.append(e_err)
            vae_err_curve.append(v_err)
            expert_sr_curve.append(e_sr)
            vae_sr_curve.append(v_sr)

            print(
                f"  Eval@{step}: expert_err={e_err:.3f}, vae_err={v_err:.3f}, "
                f"expert_sr={100.0 * e_sr:.1f}%, vae_sr={100.0 * v_sr:.1f}%"
            )

    return (
        s_mean,
        s_std,
        a_mean,
        a_std,
        loss_history,
        eval_steps,
        expert_err_curve,
        vae_err_curve,
        expert_sr_curve,
        vae_sr_curve,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE policy study: inverted pendulum balance")
    p.add_argument("--episodes", type=int, default=220)
    p.add_argument("--horizon", type=int, default=160)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta-kl", type=float, default=1e-3)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = load_base_module()

    print(f"Running on: {device}")
    print("Study case: inverted pendulum balance with conditional VAE policy")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir.resolve()}")

    env = base.InvertedPendulum(base.PendulumConfig(), device=device)

    print("\n[1/3] Generating expert dataset...")
    states, actions = base.generate_dataset(env, episodes=args.episodes, horizon=args.horizon, device=device)
    print(f"Dataset size: {states.size(0)} samples")

    print("\n[2/3] Training conditional VAE policy...")
    model = ConditionalVAE(state_dim=2, action_dim=1, latent_dim=8, hidden=128).to(device)

    (
        s_mean,
        s_std,
        a_mean,
        a_std,
        loss_history,
        eval_steps,
        expert_err_curve,
        vae_err_curve,
        expert_sr_curve,
        vae_sr_curve,
    ) = train_vae_policy(
        model,
        env,
        base,
        states,
        actions,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_kl=args.beta_kl,
        log_interval=args.log_interval,
        eval_episodes=args.eval_episodes,
    )

    plot_curves_vae(
        output_dir,
        loss_history,
        eval_steps,
        expert_err_curve,
        vae_err_curve,
        expert_sr_curve,
        vae_sr_curve,
    )

    print("\n[3/3] Evaluating policies on random initial states...")
    base.evaluate_policy(
        env,
        policy_name="Expert",
        episodes=20,
        horizon=120,
        policy_fn=lambda s: base.expert_policy(s, env.cfg),
        device=device,
    )

    base.evaluate_policy(
        env,
        policy_name="VAE",
        episodes=20,
        horizon=120,
        policy_fn=lambda s: vae_policy_action(
            model,
            s,
            s_mean,
            s_std,
            a_mean,
            a_std,
            env.cfg.u_max,
            n_action_samples=10,
        ),
        device=device,
    )

    print("\n[Bonus] Plot theta trajectories on the same initial state...")
    init_state = torch.tensor([[0.75, 1.8]], device=device)
    expert_theta = base.rollout_theta_trajectory(
        env,
        lambda s: base.expert_policy(s, env.cfg),
        init_state,
        horizon=160,
    )
    vae_theta = base.rollout_theta_trajectory(
        env,
        lambda s: vae_policy_action(
            model,
            s,
            s_mean,
            s_std,
            a_mean,
            a_std,
            env.cfg.u_max,
            n_action_samples=10,
        ),
        init_state,
        horizon=160,
    )

    plot_theta_trajectory_vae(output_dir, env.cfg.dt, expert_theta, vae_theta)
    make_pendulum_gif_vae(output_dir, expert_theta, vae_theta)

    print("\nDone. This study case learns p(torque | state) with a conditional VAE.")


if __name__ == "__main__":
    main()
