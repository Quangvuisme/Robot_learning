"""
Diffusion policy variant: predict the clean action (x0) directly, not the injected noise.

Sibling of diffusion_cartpole_swingup_balance_study.py (the "epsilon-prediction" variant).
Same architecture (ConditionalDenoiser, reused from the base module), same forward noising
process (q_sample), same overall training loop shape - the only thing that changes is what the
network is trained to output, and the algebra used to walk the reverse (denoising) chain:

  epsilon-prediction (base module):  net(a_t, s, t) ~= epsilon   (the noise that was added)
  x0-prediction      (this file):    net(a_t, s, t) ~= a_0       (the clean expert action)

Both are valid parameterizations of the same diffusion model - epsilon and a_0 are related
deterministically given a_t:  a_0 = (a_t - sqrt(1-alpha_bar_t)*epsilon) / sqrt(alpha_bar_t).
See readme.md's "Two parameterizations" section for the full derivation and a worked example.

Everything except the diffusion math (env, expert controller, dataset generation, evaluation,
trajectory/GIF plotting) is reused from the base module via load_base_module(), the same pattern
used by gan_cartpole_swingup_balance_study.py and amp_cartpole_swingup_balance_study.py to reuse
the diffusion folder's environment code.
"""

import argparse
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_base_module():
    base_path = Path(__file__).resolve().with_name("diffusion_cartpole_swingup_balance_study.py")
    spec = importlib.util.spec_from_file_location("diffusion_cartpole_swingup_base", base_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class ActionDiffusionX0:
    """Same noise schedule and forward process as base.ActionDiffusion; the reverse process
    is rewritten around a directly-predicted a0 instead of a predicted epsilon."""

    def __init__(self, steps: int = 28, beta_start: float = 1e-4, beta_end: float = 2e-2, device: str = "cpu"):
        self.steps = steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(
        self, a0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(a0)
        alpha_bar_t = self.alpha_bars[t_idx].unsqueeze(-1)
        at = torch.sqrt(alpha_bar_t) * a0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        return at, noise

    @torch.no_grad()
    def sample_action(
        self,
        model,
        state: torch.Tensor,
        noise_scale: float = 0.25,
        sampling_steps: int | None = None,
    ) -> torch.Tensor:
        batch = state.size(0)
        a = torch.randn(batch, 1, device=self.device)

        if sampling_steps is None or sampling_steps >= self.steps:
            for t in reversed(range(self.steps)):
                t_norm = torch.full((batch, 1), t / (self.steps - 1), device=self.device)
                a0_pred = model(a, state, t_norm)  # network output IS the clean-action estimate

                if t == 0:
                    a = a0_pred
                    break

                alpha_t = self.alphas[t]
                alpha_bar_t = self.alpha_bars[t]
                alpha_bar_prev = self.alpha_bars[t - 1]
                beta_t = self.betas[t]

                # Mean of q(a_{t-1} | a_t, a0), with the true a0 replaced by a0_pred.
                coef_a0 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar_t)
                coef_at = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
                mean = coef_a0 * a0_pred + coef_at * a
                posterior_var = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t
                a = mean + noise_scale * torch.sqrt(posterior_var) * torch.randn_like(a)
        else:
            # Strided DDIM-style schedule: derive an equivalent epsilon from a0_pred so the
            # same DDIM update as the epsilon-prediction script can be reused verbatim.
            t_lin = torch.linspace(self.steps - 1, 0, sampling_steps, device=self.device)
            t_list = t_lin.round().long().tolist()
            t_schedule = sorted(set(t_list), reverse=True)
            if t_schedule[-1] != 0:
                t_schedule.append(0)

            for i, t in enumerate(t_schedule):
                t_norm = torch.full((batch, 1), t / (self.steps - 1), device=self.device)
                a0_pred = model(a, state, t_norm)

                if t == 0:
                    a = a0_pred
                    break

                alpha_bar_t = self.alpha_bars[t]
                eps_pred = (a - torch.sqrt(alpha_bar_t) * a0_pred) / torch.sqrt(1.0 - alpha_bar_t)

                t_prev = t_schedule[i + 1]
                alpha_bar_prev = self.alpha_bars[t_prev]

                eta = float(noise_scale)
                sigma = eta * torch.sqrt(
                    torch.clamp((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t), min=0.0)
                    * torch.clamp(1.0 - (alpha_bar_t / alpha_bar_prev), min=0.0)
                )
                coef_eps = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0))
                a = torch.sqrt(alpha_bar_prev) * a0_pred + coef_eps * eps_pred + sigma * torch.randn_like(a)

        return a


def train_diffusion_policy_x0(
    model,
    diffusion: ActionDiffusionX0,
    env,
    base,
    states: torch.Tensor,
    actions: torch.Tensor,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
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

    loss_history: list[float] = []
    eval_steps: list[int] = []
    expert_theta_curve: list[float] = []
    diffusion_theta_curve: list[float] = []
    expert_sr_curve: list[float] = []
    diffusion_sr_curve: list[float] = []

    for step in range(1, epochs + 1):
        idx = torch.randint(0, n, (batch_size,), device=states.device)
        s = s_norm[idx]
        a0 = a_norm[idx]

        t_idx = torch.randint(0, diffusion.steps, (batch_size,), device=states.device)
        at, _noise = diffusion.q_sample(a0, t_idx)
        t_norm = (t_idx.float() / (diffusion.steps - 1)).unsqueeze(-1)

        a0_pred = model(at, s, t_norm)
        loss = F.mse_loss(a0_pred, a0)  # <- the one-line difference from epsilon-prediction
        loss_history.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_interval == 0:
            print(f"Train step {step:4d} | loss = {loss.item():.6f}")

            e_theta, _, e_sr = base.evaluate_policy(
                env,
                policy_name=f"Expert@{step}",
                episodes=eval_episodes,
                horizon=360,
                policy_fn=lambda st: base.expert_policy(st, env.cfg),
                device=device,
                verbose=False,
            )
            d_theta, _, d_sr = base.evaluate_policy(
                env,
                policy_name=f"DiffusionX0@{step}",
                episodes=eval_episodes,
                horizon=360,
                policy_fn=lambda st: base.diffusion_hybrid_policy_action(
                    model,
                    diffusion,
                    st,
                    s_mean,
                    s_std,
                    a_mean,
                    a_std,
                    env.cfg,
                    n_action_samples=2,
                    sample_noise_scale=0.18,
                    sampling_steps=6,
                    action_gain=1.15,
                ),
                device=device,
                verbose=False,
            )

            eval_steps.append(step)
            expert_theta_curve.append(e_theta)
            diffusion_theta_curve.append(d_theta)
            expert_sr_curve.append(e_sr)
            diffusion_sr_curve.append(d_sr)
            print(
                f"  Eval@{step}: expert_theta={e_theta:.3f}, x0_theta={d_theta:.3f}, "
                f"expert_sr={100.0 * e_sr:.1f}%, x0_sr={100.0 * d_sr:.1f}%"
            )

    return (
        s_mean,
        s_std,
        a_mean,
        a_std,
        loss_history,
        eval_steps,
        expert_theta_curve,
        diffusion_theta_curve,
        expert_sr_curve,
        diffusion_sr_curve,
    )


def plot_curves_x0(
    save_dir: Path,
    loss_history: list[float],
    eval_steps: list[int],
    expert_theta: list[float],
    diffusion_theta: list[float],
    expert_sr: list[float],
    diffusion_sr: list[float],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "training_curves_cartpole_swingup.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("step,loss,expert_tail_theta,diffusion_tail_theta,expert_success,diffusion_success\n")
        for i, step in enumerate(eval_steps):
            loss_val = loss_history[step - 1] if 0 <= (step - 1) < len(loss_history) else float("nan")
            f.write(
                f"{step},{loss_val},{expert_theta[i]},{diffusion_theta[i]},{expert_sr[i]},{diffusion_sr[i]}\n"
            )
    print(f"Saved: {csv_path}")

    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip saving PNG plot (CSV saved instead).")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].plot(loss_history, color="#0072B2", linewidth=1.3)
    axes[0].set_title("Training Loss (MSE action/x0 prediction)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(eval_steps, expert_theta, "-o", color="#009E73", label="Expert")
    axes[1].plot(eval_steps, diffusion_theta, "-o", color="#D55E00", label="Diffusion (x0)")
    axes[1].set_title("Mean Tail |theta| vs Epoch")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Radians (lower is better)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(eval_steps, [100.0 * x for x in expert_sr], "-o", color="#009E73", label="Expert")
    axes[2].plot(eval_steps, [100.0 * x for x in diffusion_sr], "-o", color="#D55E00", label="Diffusion (x0)")
    axes[2].set_title("Success Rate vs Epoch")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Success %")
    axes[2].set_ylim(0.0, 105.0)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    out_file = save_dir / "training_curves_cartpole_swingup.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diffusion policy study (x0/action-prediction variant): cart-pole swing-up + balance"
    )
    parser.add_argument("--episodes", type=int, default=450)
    parser.add_argument("--horizon", type=int, default=520)
    parser.add_argument("--epochs", type=int, default=2200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=8,
        help="Number of reverse diffusion steps for policy sampling (smaller=faster).",
    )
    parser.add_argument(
        "--diffusion-gain",
        type=float,
        default=1.15,
        help="Scale sampled diffusion force (can help swing-up if underpowered).",
    )
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid eval (use pure diffusion everywhere).")
    parser.add_argument("--skip-plots", action="store_true", help="Skip saving training curves (faster).")
    parser.add_argument("--skip-bonus", action="store_true", help="Skip trajectory plot + GIF (much faster).")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_base_module()
    base.set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on: {device}")
    print("Study case: cart-pole swing-up + balance, diffusion x0-prediction (predict-action) variant.")

    # Separate subfolder so this doesn't clobber the epsilon-prediction script's outputs/ files.
    output_dir = Path("outputs") / "x0"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir.resolve()}")

    env = base.InvertedPendulumOnCart(base.CartPoleConfig(), device=device)

    print("\n[0/3] Quick expert sanity check...")
    base.evaluate_policy(
        env,
        policy_name="Expert",
        episodes=30,
        horizon=420,
        policy_fn=lambda s: base.expert_policy(s, env.cfg),
        device=device,
        verbose=True,
    )

    print("\n[1/3] Generating expert dataset...")
    states, actions = base.generate_dataset(env, episodes=args.episodes, horizon=args.horizon, device=device)
    print(f"Dataset size: {states.size(0)} samples")

    states_feat = base.state_features(states)

    print("\n[2/3] Training conditional diffusion policy (x0/action-prediction)...")
    model = base.ConditionalDenoiser(state_dim=5, hidden=256).to(device)
    diffusion = ActionDiffusionX0(steps=28, device=device)

    (
        s_mean,
        s_std,
        a_mean,
        a_std,
        loss_history,
        eval_steps,
        expert_theta_curve,
        diffusion_theta_curve,
        expert_sr_curve,
        diffusion_sr_curve,
    ) = train_diffusion_policy_x0(
        model,
        diffusion,
        env,
        base,
        states_feat,
        actions,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
        eval_episodes=args.eval_episodes,
    )

    if not args.skip_plots:
        plot_curves_x0(
            output_dir,
            loss_history,
            eval_steps,
            expert_theta_curve,
            diffusion_theta_curve,
            expert_sr_curve,
            diffusion_sr_curve,
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

    policy_fn = (
        (
            lambda s: base.diffusion_policy_action(
                model, diffusion, s, s_mean, s_std, a_mean, a_std, env.cfg.force_max,
                n_action_samples=6, sample_noise_scale=0.22,
                sampling_steps=args.sampling_steps, action_gain=args.diffusion_gain,
            )
        )
        if args.no_hybrid
        else (
            lambda s: base.diffusion_hybrid_policy_action(
                model, diffusion, s, s_mean, s_std, a_mean, a_std, env.cfg,
                n_action_samples=6, sample_noise_scale=0.22,
                sampling_steps=args.sampling_steps, action_gain=args.diffusion_gain,
            )
        )
    )
    base.evaluate_policy(
        env,
        policy_name="DiffusionX0",
        episodes=args.eval_episodes,
        horizon=420,
        policy_fn=policy_fn,
        device=device,
        verbose=True,
    )

    if not args.skip_bonus:
        print("\n[Bonus] Plot x/theta trajectories on the same initial state...")
        init_state = torch.tensor([[0.0, 0.0, math.pi - 0.15, 0.0]], device=device)

        expert_x, expert_theta = base.rollout_trajectory(
            env, lambda st: base.expert_policy(st, env.cfg), init_state, horizon=520
        )
        diffusion_x, diffusion_theta = base.rollout_trajectory(env, policy_fn, init_state, horizon=520)

        base.plot_trajectory(output_dir, env.cfg.dt, expert_x, expert_theta, diffusion_x, diffusion_theta)
        base.make_cartpole_gif(output_dir, expert_x, expert_theta, diffusion_x, diffusion_theta)

    print(
        "\nDone. This study case learns p(force | state) with a diffusion policy trained to "
        "predict the clean action (x0) directly, instead of predicting injected noise."
    )


if __name__ == "__main__":
    main()
