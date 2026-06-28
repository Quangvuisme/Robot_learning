import argparse
import csv
import importlib.util
import math
import random
from dataclasses import dataclass
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


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Normalizer:
    s_mean: torch.Tensor
    s_std: torch.Tensor
    a_mean: torch.Tensor
    a_std: torch.Tensor

    def norm_state(self, s_feat: torch.Tensor) -> torch.Tensor:
        return (s_feat - self.s_mean) / self.s_std

    def denorm_action(self, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm * self.a_std + self.a_mean

    def norm_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.a_mean) / self.a_std


class ACTConditionalVAETransformer(nn.Module):
    """ACT-style CVAE with transformer sequence modeling.

    Encoder: q(z | s, a_{t:t+K-1})
      - Takes state feature + expert action chunk, outputs latent (mu, logvar)
    Decoder: p(a_{t:t+K-1} | s, z)
      - Predicts an action chunk in parallel via transformer blocks.

    This is a lightweight approximation of ACT (no vision, no multi-cam).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_len: int,
        latent_dim: int = 16,
        d_model: int = 192,
        nhead: int = 6,
        num_layers_enc: int = 3,
        num_layers_dec: int = 4,
        dropout: float = 0.05,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.latent_dim = latent_dim

        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Linear(action_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed_enc = nn.Parameter(torch.zeros(1, 1 + chunk_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder_tf = nn.TransformerEncoder(enc_layer, num_layers=num_layers_enc)
        self.enc_mu = nn.Linear(d_model, latent_dim)
        self.enc_logvar = nn.Linear(d_model, latent_dim)

        self.z_embed = nn.Linear(latent_dim, d_model)
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, chunk_len, d_model))

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder_tf = nn.TransformerEncoder(dec_layer, num_layers=num_layers_dec)
        self.action_head = nn.Linear(d_model, action_dim)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed_enc, std=0.01)
        nn.init.normal_(self.pos_embed_dec, std=0.01)

    def encode(self, state_feat: torch.Tensor, action_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode q(z|s, a_chunk).

        state_feat: (B, state_dim)
        action_chunk: (B, K, action_dim)
        """
        b = state_feat.size(0)
        s_tok = self.state_embed(state_feat).unsqueeze(1)  # (B, 1, D)
        a_tok = self.action_embed(action_chunk)  # (B, K, D)

        cls = self.cls_token.expand(b, -1, -1) + s_tok
        x = torch.cat([cls, a_tok], dim=1) + self.pos_embed_enc
        h = self.encoder_tf(x)
        pooled = h[:, 0]
        return self.enc_mu(pooled), self.enc_logvar(pooled)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, state_feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode action chunk p(a_chunk | s, z).

        Returns: (B, K, action_dim)
        """
        ctx = self.state_embed(state_feat) + self.z_embed(z)  # (B, D)
        q = ctx.unsqueeze(1) + self.pos_embed_dec  # (B, K, D)
        y = self.decoder_tf(q)
        return self.action_head(y)

    def forward(self, state_feat: torch.Tensor, action_chunk: torch.Tensor):
        mu, logvar = self.encode(state_feat, action_chunk)
        z = self.reparameterize(mu, logvar)
        pred = self.decode(state_feat, z)
        return pred, mu, logvar

    @torch.no_grad()
    def sample_chunk(self, state_feat: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        b = state_feat.size(0)
        if deterministic:
            z = torch.zeros(b, self.latent_dim, device=state_feat.device)
        else:
            z = torch.randn(b, self.latent_dim, device=state_feat.device)
        return self.decode(state_feat, z)


def generate_chunk_dataset(
    env,
    base,
    episodes: int,
    horizon: int,
    chunk_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (state_feat_t, action_chunk_{t:t+K-1}) samples.

    We collect contiguous expert rollouts so action-chunks are meaningful.
    """

    state_feats = []
    action_chunks = []

    for _ in range(episodes):
        s = env.sample_swingup_state(device)
        states_ep = []
        actions_ep = []

        for _ in range(horizon):
            if not env.is_state_valid(s).all().item():
                break
            u = base.expert_policy(s, env.cfg)
            states_ep.append(s.clone())
            actions_ep.append(u.clone())
            s = env.step(s, u)

        if len(states_ep) < chunk_len:
            continue

        states_t = torch.cat(states_ep, dim=0)
        actions_t = torch.cat(actions_ep, dim=0)  # (T, action_dim)

        feats_t = base.state_features(states_t)  # (T, state_feat_dim)
        t_max = feats_t.size(0) - chunk_len + 1
        for t in range(t_max):
            state_feats.append(feats_t[t])
            action_chunks.append(actions_t[t : t + chunk_len])

    if not state_feats:
        raise RuntimeError("Generated empty dataset; try increasing --episodes or reducing --chunk-len")

    states_feat = torch.stack(state_feats, dim=0).to(device)
    action_chunks_t = torch.stack(action_chunks, dim=0).to(device)
    return states_feat, action_chunks_t


def make_normalizer(states_feat: torch.Tensor, action_chunks: torch.Tensor) -> Normalizer:
    s_mean = states_feat.mean(dim=0, keepdim=True)
    s_std = states_feat.std(dim=0, keepdim=True) + 1e-6

    a_mean = action_chunks.mean(dim=(0, 1), keepdim=True)  # (1, 1, action_dim)
    a_std = action_chunks.std(dim=(0, 1), keepdim=True) + 1e-6

    return Normalizer(s_mean=s_mean, s_std=s_std, a_mean=a_mean, a_std=a_std)


def temporal_ensemble_action(
    preds: list[tuple[int, torch.Tensor]],
    t: int,
    chunk_len: int,
    decay: float,
) -> torch.Tensor | None:
    """Combine overlapping chunk predictions for time t.

    preds: list of (start_t, chunk_actions) where chunk_actions is (K, action_dim)
    Returns action (1, action_dim) or None if no coverage.
    """

    actions = []
    weights = []
    for start_t, chunk in preds:
        age = t - start_t
        if 0 <= age < chunk_len:
            actions.append(chunk[age])
            weights.append(decay**age)

    if not actions:
        return None

    a = torch.stack(actions, dim=0)
    w = torch.tensor(weights, device=a.device, dtype=a.dtype).view(-1, 1)
    return (a * w).sum(dim=0, keepdim=True) / (w.sum() + 1e-8)


class ACTPolicy:
    def __init__(
        self,
        model: ACTConditionalVAETransformer,
        base,
        normalizer: Normalizer,
        cfg,
        chunk_len: int,
        ensemble_decay: float,
        stride: int,
        deterministic: bool,
        action_gain: float,
        use_hybrid: bool,
    ):
        self.model = model
        self.base = base
        self.normalizer = normalizer
        self.cfg = cfg
        self.chunk_len = chunk_len
        self.ensemble_decay = ensemble_decay
        self.stride = stride
        self.deterministic = deterministic
        self.action_gain = action_gain
        self.use_hybrid = use_hybrid
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self._preds: list[tuple[int, torch.Tensor]] = []

    @torch.no_grad()
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        theta = (state[:, 2:3] + math.pi) % (2.0 * math.pi) - math.pi
        theta_dot = state[:, 3:4]
        near_upright = (torch.abs(theta) < 0.55) & (torch.abs(theta_dot) < 3.5)

        if self.use_hybrid and near_upright.all().item():
            # Clear chunk cache to avoid stale ensemble when switching back.
            self._preds.clear()
            u = self.base.balance_controller(state, self.cfg)
            self.t += 1
            return u

        # Prune old predictions that can no longer cover current time.
        min_start = self.t - (self.chunk_len - 1)
        if min_start > 0:
            self._preds = [(st, ch) for (st, ch) in self._preds if st >= min_start]

        # Sample a new chunk on schedule (stride=1 enables maximal temporal ensemble).
        if (self.t % self.stride) == 0:
            s_feat = self.base.state_features(state)
            s_norm = self.normalizer.norm_state(s_feat)
            a_chunk_norm = self.model.sample_chunk(s_norm, deterministic=self.deterministic)  # (1, K, 1)
            a_chunk = self.normalizer.denorm_action(a_chunk_norm)  # (1, K, 1)
            a_chunk = a_chunk.squeeze(0)  # (K, 1)
            self._preds.append((self.t, a_chunk))

        a = temporal_ensemble_action(self._preds, self.t, self.chunk_len, self.ensemble_decay)
        if a is None:
            # Fallback: sample once and take first.
            s_feat = self.base.state_features(state)
            s_norm = self.normalizer.norm_state(s_feat)
            a_chunk_norm = self.model.sample_chunk(s_norm, deterministic=self.deterministic)
            a_chunk = self.normalizer.denorm_action(a_chunk_norm).squeeze(0)
            self._preds.append((self.t, a_chunk))
            a = a_chunk[0:1]

        u = self.action_gain * a
        u = torch.clamp(u, -self.cfg.force_max, self.cfg.force_max)
        self.t += 1
        return u


@torch.no_grad()
def evaluate_policy_stateful(
    env,
    policy_name: str,
    episodes: int,
    horizon: int,
    policy: ACTPolicy,
    device: str,
    verbose: bool = True,
) -> tuple[float, float, float]:
    tail_theta_errors = []
    tail_x_errors = []
    success_count = 0
    tail_window = 60

    for _ in range(episodes):
        policy.reset()
        s = env.sample_swingup_state(device)
        theta_history = []
        x_history = []
        failed = False

        for _ in range(horizon):
            u = policy(s)
            s = env.step(s, u)
            theta_abs = torch.abs(s[:, 2]).item()
            x_abs = torch.abs(s[:, 0]).item()
            theta_history.append(theta_abs)
            x_history.append(x_abs)
            if not env.is_state_valid(s).all().item():
                failed = True
                break

        w = min(len(theta_history), tail_window)
        tail_theta = sum(theta_history[-w:]) / float(w)
        tail_x = sum(x_history[-w:]) / float(w)
        tail_theta_errors.append(tail_theta)
        tail_x_errors.append(tail_x)

        if (tail_theta < 0.20) and (tail_x < 0.55) and (not failed):
            success_count += 1

    mean_theta = sum(tail_theta_errors) / len(tail_theta_errors)
    mean_x = sum(tail_x_errors) / len(tail_x_errors)
    success_rate = success_count / episodes

    if verbose:
        print(f"[{policy_name}] mean tail |theta| = {mean_theta:.4f} rad")
        print(f"[{policy_name}] mean tail |x|     = {mean_x:.4f} m")
        print(f"[{policy_name}] success rate      = {100.0 * success_rate:.1f}%")

    return mean_theta, mean_x, success_rate


def plot_curves_act(
    save_dir: Path,
    loss_history: list[float],
    eval_steps: list[int],
    expert_theta: list[float],
    act_theta: list[float],
    expert_sr: list[float],
    act_sr: list[float],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "act_training_curves_cartpole_swingup.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "expert_tail_theta", "act_tail_theta", "expert_success", "act_success"])
        for i, step in enumerate(eval_steps):
            loss_val = loss_history[step - 1] if 0 <= (step - 1) < len(loss_history) else float("nan")
            writer.writerow([step, loss_val, expert_theta[i], act_theta[i], expert_sr[i], act_sr[i]])
    print(f"Saved: {csv_path}")

    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip PNG plot (CSV saved instead).")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(loss_history, color="#0072B2", linewidth=1.3)
    axes[0].set_title("Training Loss (ACT-CVAE)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(eval_steps, expert_theta, "-o", color="#009E73", label="Expert")
    axes[1].plot(eval_steps, act_theta, "-o", color="#D55E00", label="ACT")
    axes[1].set_title("Mean Tail |theta| vs Epoch")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Radians (lower is better)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(eval_steps, [100.0 * x for x in expert_sr], "-o", color="#009E73", label="Expert")
    axes[2].plot(eval_steps, [100.0 * x for x in act_sr], "-o", color="#D55E00", label="ACT")
    axes[2].set_title("Success Rate vs Epoch")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Success %")
    axes[2].set_ylim(0.0, 105.0)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    out_file = save_dir / "act_training_curves_cartpole_swingup.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_trajectory_act(
    save_dir: Path,
    dt: float,
    expert_x: list[float],
    expert_theta: list[float],
    act_x: list[float],
    act_theta: list[float],
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip trajectory plot.")
        return

    t = [i * dt for i in range(len(expert_theta))]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, expert_theta, color="#009E73", linewidth=2.0, label="Expert")
    axes[0].plot(t, act_theta, color="#D55E00", linewidth=2.0, label="ACT")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("theta (rad)")
    axes[0].set_title("Pole Angle Trajectory")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(t, expert_x, color="#009E73", linewidth=2.0, label="Expert")
    axes[1].plot(t, act_x, color="#D55E00", linewidth=2.0, label="ACT")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("x (m)")
    axes[1].set_title("Cart Position Trajectory")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = save_dir / "act_cartpole_swingup_trajectory.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def make_cartpole_gif_act(
    save_dir: Path,
    expert_x: list[float],
    expert_theta: list[float],
    act_x: list[float],
    act_theta: list[float],
) -> None:
    if not HAS_MATPLOTLIB or not HAS_IMAGEIO:
        print("matplotlib/imageio missing -> skip animation gif.")
        return

    n = min(len(expert_x), len(act_x))
    pole_len = 0.75
    max_abs_x = max(max(abs(v) for v in expert_x), max(abs(v) for v in act_x))
    x_lim = max(2.6, max_abs_x + 0.6)

    frame_paths = []
    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        for ax, x, th, title, color in [
            (axes[0], expert_x[i], expert_theta[i], "Expert", "#009E73"),
            (axes[1], act_x[i], act_theta[i], "ACT", "#D55E00"),
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
            ax.set_xlim(-x_lim, x_lim)
            ax.set_ylim(-0.55, 1.25)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.set_title(f"{title} | t={i}")

        fig.tight_layout()
        frame_path = save_dir / f"act_cartpole_swingup_frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = save_dir / "act_cartpole_swingup_compare.gif"
    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, images, duration=0.03)
    print(f"Saved: {gif_path}")

    for path in frame_paths:
        path.unlink(missing_ok=True)


def train_act_policy(
    model: ACTConditionalVAETransformer,
    env,
    base,
    states_feat: torch.Tensor,
    action_chunks: torch.Tensor,
    normalizer: Normalizer,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    beta_kl: float,
    log_interval: int,
    eval_episodes: int,
    eval_horizon: int,
    eval_stride: int,
    eval_ensemble_decay: float,
    eval_action_gain: float,
    eval_use_hybrid: bool,
    eval_deterministic: bool,
):
    s_norm = normalizer.norm_state(states_feat)
    a_norm = normalizer.norm_action(action_chunks)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = states_feat.size(0)

    loss_history: list[float] = []
    eval_steps: list[int] = []
    expert_theta_curve: list[float] = []
    act_theta_curve: list[float] = []
    expert_sr_curve: list[float] = []
    act_sr_curve: list[float] = []

    for step in range(1, epochs + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)
        s_b = s_norm[idx]
        a_b = a_norm[idx]

        pred, mu, logvar = model(s_b, a_b)
        recon_loss = F.mse_loss(pred, a_b)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta_kl * kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        loss_history.append(loss.item())

        if step % log_interval == 0:
            print(
                f"Train step {step:4d} | loss={loss.item():.6f} recon={recon_loss.item():.6f} kl={kl.item():.6f}"
            )

            e_theta, _, e_sr = base.evaluate_policy(
                env,
                policy_name=f"Expert@{step}",
                episodes=eval_episodes,
                horizon=eval_horizon,
                policy_fn=lambda st: base.expert_policy(st, env.cfg),
                device=device,
                verbose=False,
            )

            policy = ACTPolicy(
                model=model,
                base=base,
                normalizer=normalizer,
                cfg=env.cfg,
                chunk_len=model.chunk_len,
                ensemble_decay=eval_ensemble_decay,
                stride=eval_stride,
                deterministic=eval_deterministic,
                action_gain=eval_action_gain,
                use_hybrid=eval_use_hybrid,
            )

            model.eval()
            a_theta, _, a_sr = evaluate_policy_stateful(
                env,
                policy_name=f"ACT@{step}",
                episodes=eval_episodes,
                horizon=eval_horizon,
                policy=policy,
                device=device,
                verbose=False,
            )
            model.train()

            eval_steps.append(step)
            expert_theta_curve.append(e_theta)
            act_theta_curve.append(a_theta)
            expert_sr_curve.append(e_sr)
            act_sr_curve.append(a_sr)

            print(
                f"  Eval@{step}: expert_theta={e_theta:.3f}, act_theta={a_theta:.3f}, "
                f"expert_sr={100.0 * e_sr:.1f}%, act_sr={100.0 * a_sr:.1f}%"
            )

    return (
        loss_history,
        eval_steps,
        expert_theta_curve,
        act_theta_curve,
        expert_sr_curve,
        act_sr_curve,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACT-style Transformer CVAE: cart-pole swing-up + balance")
    p.add_argument("--episodes", type=int, default=220)
    p.add_argument("--horizon", type=int, default=520)
    p.add_argument("--epochs", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta-kl", type=float, default=3e-4)
    p.add_argument("--log-interval", type=int, default=120)
    p.add_argument("--eval-episodes", type=int, default=6)
    p.add_argument("--eval-horizon", type=int, default=360)

    p.add_argument("--chunk-len", type=int, default=8)
    p.add_argument("--stride", type=int, default=1, help="Sample a new chunk every STRIDE steps (1 enables temporal ensemble).")
    p.add_argument("--ensemble-decay", type=float, default=0.75)

    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--enc-layers", type=int, default=3)
    p.add_argument("--dec-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)

    p.add_argument("--act-gain", type=float, default=1.25)
    p.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="Sample latent z~N(0,1) at inference (default uses deterministic z=0).",
    )
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
    print("Study case: cart-pole swing-up + balance with ACT-style transformer CVAE")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir.resolve()}")

    env = base.InvertedPendulumOnCart(base.CartPoleConfig(), device=device)

    print("\n[0/3] Quick expert sanity check...")
    base.evaluate_policy(
        env,
        policy_name="Expert",
        episodes=10,
        horizon=360,
        policy_fn=lambda s: base.expert_policy(s, env.cfg),
        device=device,
        verbose=True,
    )

    print("\n[1/3] Generating chunked expert dataset...")
    states_feat, action_chunks = generate_chunk_dataset(
        env,
        base,
        episodes=args.episodes,
        horizon=args.horizon,
        chunk_len=args.chunk_len,
        device=device,
    )
    print(f"Dataset size: {states_feat.size(0)} chunks | chunk_len={args.chunk_len}")

    normalizer = make_normalizer(states_feat, action_chunks)

    print("\n[2/3] Training ACT-style conditional VAE (transformer encoder/decoder)...")
    model = ACTConditionalVAETransformer(
        state_dim=states_feat.size(-1),
        action_dim=action_chunks.size(-1),
        chunk_len=args.chunk_len,
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers_enc=args.enc_layers,
        num_layers_dec=args.dec_layers,
        dropout=args.dropout,
    ).to(device)

    model.train()
    (
        loss_history,
        eval_steps,
        expert_theta_curve,
        act_theta_curve,
        expert_sr_curve,
        act_sr_curve,
    ) = train_act_policy(
        model,
        env,
        base,
        states_feat,
        action_chunks,
        normalizer,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_kl=args.beta_kl,
        log_interval=args.log_interval,
        eval_episodes=args.eval_episodes,
        eval_horizon=args.eval_horizon,
        eval_stride=args.stride,
        eval_ensemble_decay=args.ensemble_decay,
        eval_action_gain=args.act_gain,
        eval_use_hybrid=(not args.no_hybrid),
        eval_deterministic=(not args.stochastic_policy),
    )

    if not args.skip_plots:
        plot_curves_act(
            output_dir,
            loss_history,
            eval_steps,
            expert_theta_curve,
            act_theta_curve,
            expert_sr_curve,
            act_sr_curve,
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

    act_policy = ACTPolicy(
        model=model,
        base=base,
        normalizer=normalizer,
        cfg=env.cfg,
        chunk_len=args.chunk_len,
        ensemble_decay=args.ensemble_decay,
        stride=args.stride,
        deterministic=(not args.stochastic_policy),
        action_gain=args.act_gain,
        use_hybrid=(not args.no_hybrid),
    )

    model.eval()
    evaluate_policy_stateful(
        env,
        policy_name="ACT",
        episodes=args.eval_episodes,
        horizon=420,
        policy=act_policy,
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
            horizon=args.horizon,
        )

        act_policy.reset()
        act_x, act_theta = base.rollout_trajectory(
            env,
            lambda st: act_policy(st),
            init_state,
            horizon=args.horizon,
        )

        plot_trajectory_act(output_dir, env.cfg.dt, expert_x, expert_theta, act_x, act_theta)
        make_cartpole_gif_act(output_dir, expert_x, expert_theta, act_x, act_theta)

    print("\nDone. This study case learns p(action_chunk | state) with an ACT-style transformer CVAE.")


if __name__ == "__main__":
    main()
