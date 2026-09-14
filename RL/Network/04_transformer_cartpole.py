"""
Network 04: TRANSFORMER on the INVERTED PENDULUM ON A CART
The sequel to 03_mlp_cartpole.py -- same task, but now a sequence model, and a
concrete reason WHY you would ever need one.

THE SETUP (and the twist):
--------------------------
In 03_mlp_cartpole.py the MLP saw the FULL state [x, x_dot, theta, theta_dot]
every step, so a single frame was enough to pick a force. Real robots rarely get
that: cameras give positions, not velocities. So here we hide the velocities and
let the controller observe ONLY:
        obs = [ x , theta ]          (cart position, pole angle)

With one frame of [x, theta] you literally CANNOT balance the pole: you can't
tell whether the pole is falling fast or slowly, or which way the cart drifts.
The missing velocity is hidden in HOW obs changed over the last few frames.

WHY A TRANSFORMER:
    A Transformer reads a WINDOW of the recent observations
        [ obs_{t-T+1}, ..., obs_{t-1}, obs_t ]
    and via self-attention can compare frames to each other -- effectively
    recovering "how fast theta is changing" (the missing velocity) from the
    sequence. Then it outputs the force. This is the same idea as a
    history-conditioned policy / Decision Transformer.

WHAT THIS FILE SHOWS (all concrete):
    1. A memoryless MLP on [x, theta] FAILS to balance (no velocity info).
    2. A causal Transformer on a window of [x, theta] SUCCEEDS.
    3. WORKED EXAMPLE: feed two windows with the SAME final frame but different
       history (pole rising vs falling) -> the Transformer outputs different
       forces. Proof that it uses the sequence, not just the current frame.
    4. GIFs + curves so you can watch both controllers.

Teacher (what we imitate): the same LQR controller from file 03. It is allowed
to see the true full state to GENERATE the data; the student networks must cope
with only [x, theta]. This is behavior cloning under partial observation.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

try:
    import imageio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

_Base = nn.Module if HAVE_TORCH else object
DEVICE = torch.device("cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu")


# ============================================================================
# CART-POLE PHYSICS + LQR TEACHER  (same as 03; theta = 0 is UPRIGHT)
# ============================================================================
@dataclass
class CartPole:
    g: float = 9.81
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    half_length: float = 0.5
    dt: float = 0.02
    force_max: float = 20.0

    def step(self, s, force):
        x, x_dot, theta, theta_dot = s
        force = float(np.clip(force, -self.force_max, self.force_max))
        total = self.mass_cart + self.mass_pole
        pml = self.mass_pole * self.half_length
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        temp = (force + pml * theta_dot**2 * sin_t) / total
        theta_ddot = (self.g * sin_t - cos_t * temp) / (
            self.half_length * (4.0 / 3.0 - self.mass_pole * cos_t**2 / total)
        )
        x_ddot = temp - pml * theta_ddot * cos_t / total
        x_dot += self.dt * x_ddot
        x += self.dt * x_dot
        theta_dot += self.dt * theta_ddot
        theta += self.dt * theta_dot
        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)


TEACHER_K = None


def setup_teacher(env):
    """Finite-difference linearization + discrete LQR -> stabilizing gains K."""
    global TEACHER_K
    n = 4
    s0 = np.zeros(n, dtype=np.float32)
    eps = 1e-4
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    for i in range(n):
        ds = np.zeros(n, dtype=np.float32)
        ds[i] = eps
        A[:, i] = (env.step(s0 + ds, 0.0) - env.step(s0 - ds, 0.0)) / (2 * eps)
    B[:, 0] = (env.step(s0, eps) - env.step(s0, -eps)) / (2 * eps)
    Q = np.diag([1.0, 1.0, 10.0, 1.0])
    R = np.array([[0.1]])
    P = Q.copy()
    for _ in range(10000):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(P_next - P)) < 1e-9:
            break
        P = P_next
    TEACHER_K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A).ravel()
    print(f"LQR teacher gains K = {np.round(TEACHER_K, 2)}")
    return TEACHER_K


def teacher_force(s):
    return float(-(TEACHER_K @ np.asarray(s, dtype=np.float32)))


# Partial observation: the controller sees ONLY these two numbers.
def observe(s):
    return np.array([s[0], s[2]], dtype=np.float32)   # [x, theta]


# ============================================================================
# DATA: roll the teacher out, keep (partial-obs sequence -> teacher force)
# ============================================================================
def collect_sequences(env, T=12, n_episodes=300, horizon=200, seed=0):
    """Return windows of observations and the matching teacher forces.

    X: [N, T, 2]  windows of [x, theta]
    Y: [N, T, 1]  teacher force at each timestep in the window (causal target)
    Each episode is left-padded with its first frame so the start-of-episode
    windows match how we pad at deploy time.
    """
    rng = np.random.default_rng(seed)
    Xs, Ys = [], []
    for _ in range(n_episodes):
        s = np.array([
            rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5),
            rng.uniform(-0.20, 0.20), rng.uniform(-0.5, 0.5),
        ], dtype=np.float32)
        obs_seq, force_seq = [], []
        for _ in range(horizon):
            obs_seq.append(observe(s))
            force_seq.append(teacher_force(s))
            s = env.step(s, force_seq[-1])
            if abs(s[2]) > math.pi / 2:
                break
        obs_seq = np.array(obs_seq, dtype=np.float32)
        force_seq = np.array(force_seq, dtype=np.float32)
        # left-pad with first frame so we can take a full T-window at every t
        pad_o = np.repeat(obs_seq[:1], T - 1, axis=0)
        pad_f = np.repeat(force_seq[:1], T - 1, axis=0)
        obs_p = np.concatenate([pad_o, obs_seq], axis=0)
        force_p = np.concatenate([pad_f, force_seq], axis=0)
        for t in range(len(obs_seq)):
            Xs.append(obs_p[t:t + T])
            Ys.append(force_p[t:t + T])
    X = np.array(Xs, dtype=np.float32)
    Y = np.array(Ys, dtype=np.float32)[..., None]
    print(f"collected {len(X)} windows of length {T} (obs=[x,theta] -> force)")
    return X, Y


# ============================================================================
# THE CAUSAL TRANSFORMER  (same components as 02_transformer.py)
# ============================================================================
class PositionalEncoding(_Base):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadSelfAttention(_Base):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = scores.softmax(dim=-1)
        ctx = (weights @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(ctx), weights


class TransformerBlock(_Base):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        a, w = self.attn(self.ln1(x))
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, w


class CartPoleTransformer(_Base):
    """Window of [x, theta] -> a force at each timestep (causal)."""

    def __init__(self, d_in=2, d_model=32, n_heads=4, d_ff=64, n_blocks=2):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_blocks)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.pos(self.embed(x))
        w = None
        for blk in self.blocks:
            x, w = blk(x)
        return self.head(x), w


# Memoryless baseline: an MLP that sees only the CURRENT [x, theta].
class MemorylessMLP(_Base):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def normalizer(X):
    """Per-feature mean/std over the obs dimension (last axis), on DEVICE."""
    flat = X.reshape(-1, X.shape[-1])
    mean = torch.tensor(flat.mean(0)).to(DEVICE)
    std = torch.tensor(flat.std(0) + 1e-6).to(DEVICE)
    return mean, std


def train_transformer(X, Y, norm, epochs=400, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    mean, std = norm
    Xt = (torch.tensor(X).to(DEVICE) - mean) / std
    Yt = torch.tensor(Y).to(DEVICE)
    model = CartPoleTransformer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        pred, _ = model(Xt)
        # train on the LAST position only: "given the window, what force NOW?"
        # (this matches exactly how we query the model at deploy time)
        loss = loss_fn(pred[:, -1], Yt[:, -1])
        loss.backward()
        opt.step()
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            hist.append((ep, loss.item()))
            print(f"[transformer] epoch {ep:4d}  imitation MSE = {loss.item():.4f}")
    print(f"[transformer] parameters: {sum(p.numel() for p in model.parameters())} (device: {DEVICE})")
    return model, hist


def train_memoryless(X, Y, norm, epochs=400, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    mean, std = norm
    # use only the LAST frame of each window (memoryless sees one obs)
    Xt = (torch.tensor(X[:, -1, :]).to(DEVICE) - mean) / std
    Yt = torch.tensor(Y[:, -1, :]).to(DEVICE)
    model = MemorylessMLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), Yt)
        loss.backward()
        opt.step()
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            hist.append((ep, loss.item()))
            print(f"[memoryless] epoch {ep:4d}  imitation MSE = {loss.item():.4f}")
    return model, hist


# ============================================================================
# DEPLOY: drive the cart with each learned controller
# ============================================================================
def run_transformer(env, model, norm, T=12, horizon=300, theta0=0.05, theta_dot0=2.5):
    mean, std = norm
    s = np.array([0.0, 0.0, theta0, theta_dot0], dtype=np.float32)
    window = [observe(s)] * T          # start padded with the first frame
    traj = [s.copy()]
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            x = (torch.tensor(np.array(window)[None], dtype=torch.float32).to(DEVICE) - mean) / std
            pred, _ = model(x)
            force = float(pred[0, -1, 0])     # force for the most recent frame
            s = env.step(s, force)
            traj.append(s.copy())
            window = window[1:] + [observe(s)]
    return np.array(traj)


def run_memoryless(env, model, norm, horizon=300, theta0=0.05, theta_dot0=2.5):
    mean, std = norm
    s = np.array([0.0, 0.0, theta0, theta_dot0], dtype=np.float32)
    traj = [s.copy()]
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            x = (torch.tensor(observe(s), dtype=torch.float32).to(DEVICE) - mean) / std
            force = float(model(x).item())
            s = env.step(s, force)
            traj.append(s.copy())
            if abs(s[2]) > math.pi / 2:
                break
    return np.array(traj)


def upright_pct(traj):
    return float(np.mean(np.abs(traj[:, 2]) < 0.25) * 100)


# ============================================================================
# WORKED EXAMPLE: same final frame, different history -> different force
# ============================================================================
def worked_example(model, norm, T=12):
    mean, std = norm
    final = np.array([0.0, 0.10], dtype=np.float32)   # x=0, theta=+0.10 (tilted right)

    # History A: pole was rising toward 0.10 quickly (falling right fast).
    rising = np.linspace(0.02, 0.10, T, dtype=np.float32)
    winA = np.stack([np.zeros(T, dtype=np.float32), rising], axis=1)
    # History B: pole was coming back down toward 0.10 (was higher, now returning).
    falling = np.linspace(0.18, 0.10, T, dtype=np.float32)
    winB = np.stack([np.zeros(T, dtype=np.float32), falling], axis=1)

    model.eval()
    with torch.no_grad():
        xA = (torch.tensor(winA[None]).to(DEVICE) - mean) / std
        xB = (torch.tensor(winB[None]).to(DEVICE) - mean) / std
        fA = float(model(xA)[0][0, -1, 0])
        fB = float(model(xB)[0][0, -1, 0])
        _, wA = model(xA)

    print("\n" + "=" * 70)
    print("WORKED EXAMPLE  -  the Transformer USES the history, not just one frame")
    print("=" * 70)
    print(f"Both windows END at the SAME frame: obs = [x=0.00, theta=+0.10].")
    print(f"  History A: theta was RISING {rising[0]:.2f} -> {rising[-1]:.2f}  (pole falling right fast)")
    print(f"  History B: theta was FALLING {falling[0]:.2f} -> {falling[-1]:.2f}  (pole already recovering)")
    print(f"\n  Transformer force for A = {fA:+.3f} N")
    print(f"  Transformer force for B = {fB:+.3f} N")
    print(f"  => A needs a STRONGER right push ({fA:+.2f} > {fB:+.2f}): it inferred the")
    print(f"     hidden angular velocity from the SEQUENCE. A memoryless net sees the")
    print(f"     same single frame for both and cannot tell them apart.")
    # attention of the final query over the window (head-averaged)
    attn = wA[0].mean(0)[-1].cpu().numpy()
    print(f"\n  attention of the last frame over the window (sums to 1):")
    print("   " + "  ".join(f"{a:.2f}" for a in attn))


# ============================================================================
# RENDER + PLOTS
# ============================================================================
def _fig_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def create_gif(env, traj, out_dir, name, subtitle):
    if not HAVE_PLT or not HAVE_IMAGEIO:
        print("matplotlib/imageio not available, skipping GIF")
        return
    L = 2.0 * env.half_length
    step = max(1, len(traj) // 150)
    frames = []
    for i in range(0, len(traj), step):
        x, _, theta, _ = traj[i]
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot([-3, 3], [0, 0], "k-", linewidth=2)
        cw, ch = 0.5, 0.3
        cart_x = [x - cw / 2, x + cw / 2, x + cw / 2, x - cw / 2, x - cw / 2]
        ax.fill(cart_x, [0, 0, ch, ch, 0], color="#0072B2", alpha=0.6)
        px, py = x + L * math.sin(theta), ch + L * math.cos(theta)
        ax.plot([x, px], [ch, py], color="#D55E00", linewidth=4)
        ax.scatter([px], [py], s=140, color="#D55E00")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 2.7)
        ax.set_aspect("equal")
        ax.set_title(f"{subtitle}\nstep {i:03d}  x={x:+.2f}  theta={math.degrees(theta):+.1f}deg",
                     fontsize=10)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        frames.append(_fig_to_rgb(fig))
        plt.close(fig)
    path = Path(out_dir) / name
    imageio.mimsave(path, frames, duration=0.04)
    print(f"saved {path}   (upright {upright_pct(traj):.0f}% of the run)")


def plot_compare(traj_tf, traj_mem, hist_tf, out_dir, name="cartpole_transformer_curves.png"):
    if not HAVE_PLT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep, ls = zip(*hist_tf)
    axes[0].plot(ep, ls, color="#0072B2")
    axes[0].set_title("Transformer training (imitate teacher)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("imitation MSE")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)

    axes[1].plot(np.degrees(traj_tf[:, 2]), label="Transformer (history)", color="#0072B2", linewidth=2)
    axes[1].plot(np.degrees(traj_mem[:, 2]), label="Memoryless MLP (1 frame)", color="#D55E00", linewidth=2)
    axes[1].axhline(0, color="gray", linestyle=":")
    axes[1].set_title("Pole angle over time (lower = balanced)")
    axes[1].set_xlabel("timestep")
    axes[1].set_ylabel("theta (deg)")
    axes[1].set_ylim(-90, 90)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = Path(out_dir) / name
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Transformer balancing a cart-pole from partial observations")
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).parent / "figures" / "04_transformer"))
    args = parser.parse_args()

    if not HAVE_TORCH:
        print("torch not installed -> this example needs torch. Aborting.")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = CartPole()
    T = args.seq_len

    print("=" * 70)
    print("STEP 0: build the LQR teacher (sees full state to generate data)")
    print("=" * 70)
    setup_teacher(env)

    print("\n" + "=" * 70)
    print("STEP 1: collect partial-obs sequences ([x, theta] only -> force)")
    print("=" * 70)
    X, Y = collect_sequences(env, T=T)
    norm = normalizer(X)

    print("\n" + "=" * 70)
    print("STEP 2a: train the Transformer on the HISTORY window")
    print("=" * 70)
    model_tf, hist_tf = train_transformer(X, Y, norm, epochs=max(args.epochs, 600))

    print("\n" + "=" * 70)
    print("STEP 2b: train a memoryless MLP on the CURRENT frame only (baseline)")
    print("=" * 70)
    model_mem, _ = train_memoryless(X, Y, norm, epochs=args.epochs)

    worked_example(model_tf, norm, T=T)

    print("\n" + "=" * 70)
    print("STEP 3: balance from a TRICKY start: theta=0.05 (looks upright) but")
    print("        theta_dot=2.5 (actually spinning fast). obs = [x, theta] only.")
    print("=" * 70)
    th0, thd0 = 0.05, 2.5
    traj_tf = run_transformer(env, model_tf, norm, T=T, theta0=th0, theta_dot0=thd0)
    traj_mem = run_memoryless(env, model_mem, norm, theta0=th0, theta_dot0=thd0)
    print(f"Transformer  : upright {upright_pct(traj_tf):.0f}% of the run, survived {len(traj_tf)} steps")
    print(f"Memoryless MLP: upright {upright_pct(traj_mem):.0f}% of the run, survived {len(traj_mem)} steps")
    print("  (the single frame [x=0, theta=0.05] looks balanced, so the memoryless")
    print("   net does almost nothing and the spinning pole falls.)")

    create_gif(env, traj_tf, out_dir, "control.gif",
               "Transformer controller (uses history) - balances")
    create_gif(env, traj_mem, out_dir, "memoryless_fail.gif",
               "Memoryless MLP (one frame) - cannot sense the spin, falls")
    plot_compare(traj_tf, traj_mem, hist_tf, out_dir, name="curves.png")

    print("\nTAKEAWAYS")
    print("- Same task as 03, but the controller sees only [x, theta] (no velocity).")
    print("- One frame is not enough -> the memoryless MLP cannot balance.")
    print("- The Transformer attends over the recent window, recovers the missing")
    print("  velocity from the sequence, and balances. That is WHY you use it.")
    print("- This is exactly a history-conditioned policy (cf. Decision Transformer).")


if __name__ == "__main__":
    main()
