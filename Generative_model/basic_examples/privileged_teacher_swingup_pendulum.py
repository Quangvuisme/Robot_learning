import argparse
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


class PendulumSwingupEnv:
    """
    Privileged-state pendulum environment.

    State: s = [theta, theta_dot], where theta=0 is upright.
    Dynamics: theta_ddot = (g/l) * sin(theta) + u / (m*l^2) - d * theta_dot
    """

    def __init__(self, dt: float = 0.02, g: float = 9.81, l: float = 1.0, m: float = 1.0, damping: float = 0.05, u_max: float = 18.0):
        self.dt = dt
        self.g = g
        self.l = l
        self.m = m
        self.damping = damping
        self.u_max = u_max

    def wrap(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta + math.pi) % (2.0 * math.pi) - math.pi

    def step(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        th = s[:, 0:1]
        thd = s[:, 1:2]
        u = torch.clamp(u, -self.u_max, self.u_max)

        thdd = (self.g / self.l) * torch.sin(th) + u / (self.m * self.l * self.l) - self.damping * thd
        thd_next = thd + self.dt * thdd
        th_next = self.wrap(th + self.dt * thd_next)
        return torch.cat([th_next, thd_next], dim=-1)


@torch.no_grad()
def privileged_teacher_policy(s: torch.Tensor, env: PendulumSwingupEnv) -> torch.Tensor:
    """
    Teacher has privileged access to full state [theta, theta_dot].
    Uses energy-pumping far from upright and PD stabilization near upright.
    """
    th = env.wrap(s[:, 0:1])
    thd = s[:, 1:2]

    near_upright = (torch.abs(th) < 0.45) & (torch.abs(thd) < 2.6)

    # Balance controller with nonlinear cancellation.
    kp = 14.0
    kd = 4.0
    inv = env.m * env.l * env.l
    u_balance = inv * (-(env.g / env.l) * torch.sin(th) - kp * th - (kd - env.damping) * thd)

    # Energy shaping for dynamics theta_ddot = +(g/l) sin(theta) + u/(m l^2) - d theta_dot.
    # Natural energy: H = 0.5 * omega^2 + (g/l) * cos(theta)
    # Upright target energy (theta=0, omega=0): H* = g/l
    # Choose u proportional to (H* - H) * omega to inject/remove energy.
    h = 0.5 * (thd**2) + (env.g / env.l) * torch.cos(th)
    h_star = env.g / env.l
    k_e = 2.4
    u_swing = inv * (k_e * (h_star - h) * thd + env.damping * thd)

    u = torch.where(near_upright, u_balance, u_swing)
    return torch.clamp(u, -env.u_max, env.u_max)


def student_observation(s: torch.Tensor, prev_s: torch.Tensor, prev_u: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Student only sees partial observation:
    [sin(theta_t), cos(theta_t), sin(theta_{t-1}), cos(theta_{t-1}), dtheta_est, u_{t-1}]

    dtheta_est is finite-difference angular velocity from angle history.
    """
    th = s[:, 0:1]
    prev_th = prev_s[:, 0:1]
    dtheta_est = ((th - prev_th + math.pi) % (2.0 * math.pi) - math.pi) / max(dt, 1e-6)
    return torch.cat([torch.sin(th), torch.cos(th), torch.sin(prev_th), torch.cos(prev_th), dtheta_est, prev_u], dim=-1)


class TeacherPolicy(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class StudentPolicy(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_swingup_init(env: PendulumSwingupEnv, device: str) -> torch.Tensor:
    th0 = math.pi + (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.35
    thd0 = (torch.rand(1, 1, device=device) * 2.0 - 1.0) * 0.8
    s = torch.cat([th0, thd0], dim=-1)
    return torch.cat([env.wrap(s[:, 0:1]), s[:, 1:2]], dim=-1)


def generate_teacher_supervision_dataset(
    env: PendulumSwingupEnv,
    episodes: int,
    horizon: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_all = []
    act_all = []

    for _ in range(episodes):
        s = sample_swingup_init(env, device)

        for _ in range(horizon):
            u_oracle = privileged_teacher_policy(s, env)
            state_all.append(s.clone())
            act_all.append(u_oracle.clone())
            s = env.step(s, u_oracle)

    return torch.cat(state_all, dim=0), torch.cat(act_all, dim=0)


@torch.no_grad()
def teacher_model_action(
    teacher: TeacherPolicy,
    s: torch.Tensor,
    s_mean: torch.Tensor,
    s_std: torch.Tensor,
    a_mean: torch.Tensor,
    a_std: torch.Tensor,
    u_max: float,
) -> torch.Tensor:
    s_n = (s - s_mean) / s_std
    a_n = teacher(s_n)
    a = a_n * a_std + a_mean
    return torch.clamp(a, -u_max, u_max)


def train_teacher_model(
    teacher: TeacherPolicy,
    states: torch.Tensor,
    actions: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    s_mean = states.mean(dim=0, keepdim=True)
    s_std = states.std(dim=0, keepdim=True) + 1e-6
    a_mean = actions.mean(dim=0, keepdim=True)
    a_std = actions.std(dim=0, keepdim=True) + 1e-6

    s_n = (states - s_mean) / s_std
    a_n = (actions - a_mean) / a_std

    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    n = s_n.size(0)
    losses = []
    for _ in range(epochs):
        idx = torch.randint(0, n, (batch_size,), device=s_n.device)
        xb = s_n[idx]
        yb = a_n[idx]
        pred = teacher(xb)
        loss = F.mse_loss(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return s_mean, s_std, a_mean, a_std, losses


@torch.no_grad()
def generate_student_distill_dataset(
    env: PendulumSwingupEnv,
    teacher: TeacherPolicy,
    t_s_mean: torch.Tensor,
    t_s_std: torch.Tensor,
    t_a_mean: torch.Tensor,
    t_a_std: torch.Tensor,
    episodes: int,
    horizon: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_all = []
    act_all = []

    for _ in range(episodes):
        s = sample_swingup_init(env, device)
        prev_s = s.clone()
        prev_u = torch.zeros(1, 1, device=device)

        for _ in range(horizon):
            u_teacher = teacher_model_action(teacher, s, t_s_mean, t_s_std, t_a_mean, t_a_std, env.u_max)
            obs = student_observation(s, prev_s, prev_u, env.dt)

            obs_all.append(obs.clone())
            act_all.append(u_teacher.clone())

            s_next = env.step(s, u_teacher)
            prev_s = s
            prev_u = u_teacher
            s = s_next

    return torch.cat(obs_all, dim=0), torch.cat(act_all, dim=0)


@torch.no_grad()
def evaluate_policy(env: PendulumSwingupEnv, policy_fn, episodes: int, horizon: int, device: str) -> tuple[float, float]:
    tail_errs = []
    success = 0

    for _ in range(episodes):
        s = sample_swingup_init(env, device)

        prev_s = s.clone()
        prev_u = torch.zeros(1, 1, device=device)

        errs = []
        for _ in range(horizon):
            u = policy_fn(s, prev_s, prev_u)
            s_next = env.step(s, u)
            err = torch.abs(env.wrap(s_next[:, 0:1])).item()
            errs.append(err)
            prev_s = s
            prev_u = u
            s = s_next

        tail = sum(errs[-40:]) / 40.0
        tail_errs.append(tail)
        if tail < 0.25:
            success += 1

    return sum(tail_errs) / len(tail_errs), success / episodes


@torch.no_grad()
def rollout_theta(env: PendulumSwingupEnv, policy_fn, init_state: torch.Tensor, horizon: int):
    s = init_state.clone()
    prev_s = s.clone()
    prev_u = torch.zeros(1, 1, device=s.device)

    ths = [env.wrap(s[:, 0:1]).item()]
    for _ in range(horizon):
        u = policy_fn(s, prev_s, prev_u)
        s_next = env.step(s, u)
        ths.append(env.wrap(s_next[:, 0:1]).item())
        prev_s = s
        prev_u = u
        s = s_next
    return ths


def plot_results(
    out_dir: Path,
    teacher_losses: list[float],
    student_losses: list[float],
    eval_steps: list[int],
    teacher_err: list[float],
    student_err: list[float],
    teacher_sr: list[float],
    student_sr: list[float],
    dt: float,
    teacher_th: list[float],
    student_th: list[float],
):
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    axes[0].plot(teacher_losses, color="#0072B2", label="Teacher train")
    axes[0].plot(student_losses, color="#D55E00", label="Student train")
    axes[0].set_title("Teacher/Student Training Loss (MSE)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(eval_steps, teacher_err, "-o", color="#009E73", label="Teacher (privileged)")
    axes[1].plot(eval_steps, student_err, "-o", color="#D55E00", label="Student (partial obs)")
    axes[1].set_title("Mean Tail |theta|")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Radians (lower is better)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(eval_steps, [100.0 * x for x in teacher_sr], "-o", color="#009E73", label="Teacher")
    axes[2].plot(eval_steps, [100.0 * x for x in student_sr], "-o", color="#D55E00", label="Student")
    axes[2].set_title("Swing-up Success Rate")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Success %")
    axes[2].set_ylim(0, 105)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    curve_file = out_dir / "privileged_teacher_swingup_curves.png"
    fig.savefig(curve_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {curve_file}")

    t = [i * dt for i in range(len(teacher_th))]
    fig2, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(t, teacher_th, color="#009E73", linewidth=2.0, label="Teacher (privileged)")
    ax.plot(t, student_th, color="#D55E00", linewidth=2.0, label="Student (partial obs)")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_title("Single Swing-up Rollout")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("theta (rad)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig2.tight_layout()
    traj_file = out_dir / "privileged_teacher_swingup_rollout.png"
    fig2.savefig(traj_file, dpi=150)
    plt.close(fig2)
    print(f"Saved: {traj_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Privileged teacher example: swing-up pendulum")
    p.add_argument("--episodes", type=int, default=140)
    p.add_argument("--horizon", type=int, default=220)
    p.add_argument("--teacher-epochs", type=int, default=700)
    p.add_argument("--student-epochs", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--teacher-lr", type=float, default=1e-3)
    p.add_argument("--student-lr", type=float, default=1e-3)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on: {device}")
    print("Example: train privileged teacher then distill to student (pendulum swing-up)")

    env = PendulumSwingupEnv()
    out_dir = Path("outputs")

    print("\n[1/4] Generate supervision data for privileged teacher...")
    teacher_states, teacher_actions = generate_teacher_supervision_dataset(env, args.episodes, args.horizon, device)
    print(f"Teacher dataset size: {teacher_states.size(0)}")

    teacher = TeacherPolicy(hidden=128).to(device)
    print("\n[2/4] Train privileged teacher model...")
    t_s_mean, t_s_std, t_a_mean, t_a_std, teacher_losses = train_teacher_model(
        teacher,
        teacher_states,
        teacher_actions,
        epochs=args.teacher_epochs,
        batch_size=args.batch_size,
        lr=args.teacher_lr,
    )

    teacher_err, teacher_sr = evaluate_policy(
        env,
        policy_fn=lambda s, ps, pu: teacher_model_action(teacher, s, t_s_mean, t_s_std, t_a_mean, t_a_std, env.u_max),
        episodes=args.eval_episodes,
        horizon=args.horizon,
        device=device,
    )
    print(f"Teacher after training: tail={teacher_err:.3f}, success={100.0 * teacher_sr:.1f}%")

    print("\n[3/4] Distill dataset from trained teacher and train student...")
    obs, act = generate_student_distill_dataset(
        env,
        teacher,
        t_s_mean,
        t_s_std,
        t_a_mean,
        t_a_std,
        args.episodes,
        args.horizon,
        device,
    )
    print(f"Student distill dataset size: {obs.size(0)}")

    obs_mean = obs.mean(dim=0, keepdim=True)
    obs_std = obs.std(dim=0, keepdim=True) + 1e-6
    act_mean = act.mean(dim=0, keepdim=True)
    act_std = act.std(dim=0, keepdim=True) + 1e-6
    obs_n = (obs - obs_mean) / obs_std
    act_n = (act - act_mean) / act_std

    student = StudentPolicy(hidden=128).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.student_lr)

    student_losses = []
    eval_steps = []
    teacher_err_curve = []
    student_err_curve = []
    teacher_sr_curve = []
    student_sr_curve = []

    print("\nTrain student on distilled labels...")
    n = obs_n.size(0)
    for step in range(1, args.student_epochs + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=obs_n.device)
        xb = obs_n[idx]
        yb = act_n[idx]

        pred = student(xb)
        loss = F.mse_loss(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        student_losses.append(loss.item())

        if step % args.log_interval == 0:
            print(f"Train step {step:4d} | loss={loss.item():.6f}")

            teacher_err, teacher_sr = evaluate_policy(
                env,
                policy_fn=lambda s, ps, pu: teacher_model_action(teacher, s, t_s_mean, t_s_std, t_a_mean, t_a_std, env.u_max),
                episodes=args.eval_episodes,
                horizon=args.horizon,
                device=device,
            )

            student_err, student_sr = evaluate_policy(
                env,
                policy_fn=lambda s, ps, pu: torch.clamp(
                    student((student_observation(s, ps, pu, env.dt) - obs_mean) / obs_std) * act_std + act_mean,
                    -env.u_max,
                    env.u_max,
                ),
                episodes=args.eval_episodes,
                horizon=args.horizon,
                device=device,
            )

            eval_steps.append(step)
            teacher_err_curve.append(teacher_err)
            student_err_curve.append(student_err)
            teacher_sr_curve.append(teacher_sr)
            student_sr_curve.append(student_sr)

            print(
                f"  Eval@{step}: trained_teacher_tail={teacher_err:.3f}, student_tail={student_err:.3f}, "
                f"teacher_sr={100.0 * teacher_sr:.1f}%, student_sr={100.0 * student_sr:.1f}%"
            )

    print("\n[4/4] Plot diagnostics...")
    init_state = torch.tensor([[math.pi - 0.2, 0.0]], device=device)
    teacher_th = rollout_theta(
        env,
        lambda s, ps, pu: teacher_model_action(teacher, s, t_s_mean, t_s_std, t_a_mean, t_a_std, env.u_max),
        init_state,
        args.horizon,
    )
    student_th = rollout_theta(
        env,
        lambda s, ps, pu: torch.clamp(
            student((student_observation(s, ps, pu, env.dt) - obs_mean) / obs_std) * act_std + act_mean,
            -env.u_max,
            env.u_max,
        ),
        init_state,
        args.horizon,
    )

    plot_results(
        out_dir,
        teacher_losses,
        student_losses,
        eval_steps,
        teacher_err_curve,
        student_err_curve,
        teacher_sr_curve,
        student_sr_curve,
        env.dt,
        teacher_th,
        student_th,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
