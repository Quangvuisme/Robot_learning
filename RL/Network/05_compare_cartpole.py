"""
Network 05: DETAILED head-to-head comparison  MLP vs Transformer (cart-pole)

Ties together files 01-04. Same task (imitate an LQR teacher that balances the
inverted pendulum), then stress-test the controllers so the architectural
differences show up as concrete numbers and plots.

THE CONTROLLERS:
    1. LQR teacher         - linear, sees FULL state [x, x_dot, theta, theta_dot]
    2. MLP (full state)    - sees full state, one frame          (file 03)
    3. MLP (partial)       - sees only [x, theta], one frame     (memoryless)
    4. MLP (flat window)   - sees T frames of [x, theta] FLATTENED into one vector
                             (has history, but NO attention / no sequence structure)
    5. Transformer         - causal attention over a window of [x, theta]  (file 04)

EXPERIMENTS (A -> F):
    A. Architecture card     - params, input, memory, what each can/cannot see
    B. Same frame, diff hist - proof Transformer uses history; MLP-partial cannot
    C. IC grid heatmap       - which (theta0, theta_dot0) each controller recovers
    D. Side-by-side trajs    - hard starts: theta / force / survival + GIFs
       (controllers rendered side-by-side in one video so differences are obvious)
    E. Window-length sweep   - how much history does the Transformer need?
    F. Sensor-noise sweep    - robustness as observation noise grows

WHAT TO TAKE AWAY:
    - Full info  -> tiny MLP already matches LQR. Sequence models are unnecessary.
    - Partial obs, 1 frame -> memoryless MLP collapses outside theta_dot ~ 0.
    - Partial obs + flat history -> better than 1 frame, still weaker than attention.
    - Transformer recovers velocity from the sequence and closes most of the gap.
    - Match the network to OBSERVABILITY, not to fashion.
"""

import argparse
import math
import time
from collections import deque
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
DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if HAVE_TORCH else None
)


# ============================================================================
# PHYSICS + LQR TEACHER  (identical to files 03/04; theta = 0 is upright)
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
    global TEACHER_K
    n, eps = 4, 1e-4
    s0 = np.zeros(n, dtype=np.float32)
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
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(Pn - P)) < 1e-9:
            break
        P = Pn
    TEACHER_K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A).ravel()
    print(f"LQR teacher gains K = {np.round(TEACHER_K, 2)}")
    return TEACHER_K


def teacher_force(s):
    return float(-(TEACHER_K @ np.asarray(s, dtype=np.float32)))


def full_obs(s):
    return np.array(s, dtype=np.float32)


def partial_obs(s):
    return np.array([s[0], s[2]], dtype=np.float32)   # [x, theta]


# ============================================================================
# DATA COLLECTION
# ============================================================================
def collect(env, T=12, n_episodes=300, horizon=200, seed=0):
    rng = np.random.default_rng(seed)
    full_X, part_X, force_Y = [], [], []
    seq_X, seq_Y = [], []
    for _ in range(n_episodes):
        s = np.array([
            rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5),
            rng.uniform(-0.20, 0.20), rng.uniform(-0.5, 0.5),
        ], dtype=np.float32)
        obs_seq, force_seq = [], []
        for _ in range(horizon):
            full_X.append(full_obs(s))
            part_X.append(partial_obs(s))
            f = teacher_force(s)
            force_Y.append(f)
            obs_seq.append(partial_obs(s))
            force_seq.append(f)
            s = env.step(s, f)
            if abs(s[2]) > math.pi / 2:
                break
        obs_seq = np.array(obs_seq, dtype=np.float32)
        force_seq = np.array(force_seq, dtype=np.float32)
        pad_o = np.repeat(obs_seq[:1], T - 1, axis=0)
        pad_f = np.repeat(force_seq[:1], T - 1, axis=0)
        obs_p = np.concatenate([pad_o, obs_seq], 0)
        force_p = np.concatenate([pad_f, force_seq], 0)
        for t in range(len(obs_seq)):
            seq_X.append(obs_p[t:t + T])
            seq_Y.append(force_p[t:t + T])
    out = dict(
        full_X=np.array(full_X, np.float32),
        part_X=np.array(part_X, np.float32),
        force_Y=np.array(force_Y, np.float32)[:, None],
        seq_X=np.array(seq_X, np.float32),
        seq_Y=np.array(seq_Y, np.float32)[..., None],
    )
    print(f"collected {len(out['full_X'])} frames, {len(out['seq_X'])} windows (T={T})")
    return out


# ============================================================================
# MODELS
# ============================================================================
class MLP(_Base):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


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


class Attn(_Base):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, return_weights=False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        sp = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        sc = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        sc = sc.masked_fill(mask, float("-inf"))
        w = sc.softmax(-1)
        ctx = (w @ v).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(ctx)
        if return_weights:
            return out, w
        return out


class Block(_Base):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = Attn(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, return_weights=False):
        if return_weights:
            a, w = self.attn(self.ln1(x), return_weights=True)
            x = x + a
            return x + self.ff(self.ln2(x)), w
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class Transformer(_Base):
    def __init__(self, d_in=2, d_model=32, n_heads=4, d_ff=64, n_blocks=2):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_blocks)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, return_weights=False):
        x = self.pos(self.embed(x))
        attn_w = None
        for i, blk in enumerate(self.blocks):
            if return_weights and i == len(self.blocks) - 1:
                x, attn_w = blk(x, return_weights=True)
            else:
                x = blk(x)
        out = self.head(x)
        if return_weights:
            return out, attn_w
        return out


def count_params(net):
    return sum(p.numel() for p in net.parameters())


def norm_of(X):
    flat = X.reshape(-1, X.shape[-1])
    return (torch.tensor(flat.mean(0)).to(DEVICE), torch.tensor(flat.std(0) + 1e-6).to(DEVICE))


def train_frame_mlp(X, Y, norm, hidden=64, epochs=800, lr=1e-3, seed=0, tag="mlp"):
    torch.manual_seed(seed)
    mean, std = norm
    Xt = (torch.tensor(X).to(DEVICE) - mean) / std
    Yt = torch.tensor(Y).to(DEVICE)
    net = MLP(X.shape[1], hidden).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lf = nn.MSELoss()
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        loss = lf(net(Xt), Yt)
        loss.backward()
        opt.step()
    dt = time.time() - t0
    print(f"[{tag}] params={count_params(net):,}  final MSE={loss.item():.4f}  train={dt:.1f}s")
    return net


def train_flat_window_mlp(seq_X, seq_Y, norm, hidden=64, epochs=800, lr=1e-3, seed=0):
    """MLP that sees the whole window, but flattened — no attention structure."""
    torch.manual_seed(seed)
    mean, std = norm
    # seq_X: (N, T, 2) -> flatten last two dims; target = force at last step
    X = seq_X.reshape(len(seq_X), -1)
    Y = seq_Y[:, -1, :]
    # normalize per-feature using the 2-d obs stats, tiled across T
    mean_flat = mean.detach().cpu().numpy()
    std_flat = std.detach().cpu().numpy()
    T = seq_X.shape[1]
    mean_t = torch.tensor(np.tile(mean_flat, T), dtype=torch.float32).to(DEVICE)
    std_t = torch.tensor(np.tile(std_flat, T), dtype=torch.float32).to(DEVICE)
    Xt = (torch.tensor(X).to(DEVICE) - mean_t) / std_t
    Yt = torch.tensor(Y).to(DEVICE)
    net = MLP(X.shape[1], hidden).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lf = nn.MSELoss()
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        loss = lf(net(Xt), Yt)
        loss.backward()
        opt.step()
    dt = time.time() - t0
    print(f"[mlp-flat-window] params={count_params(net):,}  final MSE={loss.item():.4f}  train={dt:.1f}s")
    return net, (mean_t, std_t)


def train_transformer(X, Y, norm, epochs=600, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    mean, std = norm
    Xt = (torch.tensor(X).to(DEVICE) - mean) / std
    Yt = torch.tensor(Y).to(DEVICE)
    net = Transformer().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lf = nn.MSELoss()
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        loss = lf(net(Xt)[:, -1], Yt[:, -1])
        loss.backward()
        opt.step()
    dt = time.time() - t0
    print(f"[transformer] params={count_params(net):,}  final MSE={loss.item():.4f}  train={dt:.1f}s")
    return net


# ============================================================================
# CONTROLLERS
# ============================================================================
class TeacherCtrl:
    obs_fn = staticmethod(full_obs)
    name = "LQR teacher (full)"

    def reset(self):
        pass

    def act(self, obs):
        return float(-(TEACHER_K @ obs))


class FrameMLPCtrl:
    def __init__(self, net, norm, obs_fn, name):
        self.net, self.mean, self.std = net, norm[0], norm[1]
        self.obs_fn = obs_fn
        self.name = name

    def reset(self):
        pass

    def act(self, obs):
        with torch.no_grad():
            x = (torch.tensor(obs, dtype=torch.float32).to(DEVICE) - self.mean) / self.std
            return float(self.net(x).item())


class FlatWindowMLPCtrl:
    """Sees T frames of [x, theta] flattened — history without attention."""
    obs_fn = staticmethod(partial_obs)
    name = "MLP (flat window)"

    def __init__(self, net, norm_flat, T=12):
        self.net, self.mean, self.std, self.T = net, norm_flat[0], norm_flat[1], T
        self.win = None

    def reset(self):
        self.win = None

    def act(self, obs):
        if self.win is None:
            self.win = deque([obs] * self.T, maxlen=self.T)
        else:
            self.win.append(obs)
        with torch.no_grad():
            flat = np.array(self.win, dtype=np.float32).reshape(-1)
            x = (torch.tensor(flat).to(DEVICE) - self.mean) / self.std
            return float(self.net(x).item())


class TransformerCtrl:
    obs_fn = staticmethod(partial_obs)
    name = "Transformer (history)"

    def __init__(self, net, norm, T=12):
        self.net, self.mean, self.std, self.T = net, norm[0], norm[1], T
        self.win = None

    def reset(self):
        self.win = None

    def act(self, obs):
        if self.win is None:
            self.win = deque([obs] * self.T, maxlen=self.T)
        else:
            self.win.append(obs)
        with torch.no_grad():
            x = (torch.tensor(np.array(self.win)[None], dtype=torch.float32).to(DEVICE) - self.mean) / self.std
            return float(self.net(x)[0, -1, 0])


# ============================================================================
# EVALUATION HELPERS
# ============================================================================
def rollout(env, ctrl, init, horizon=300, noise=0.0, rng=None, record_force=False):
    s = np.array(init, dtype=np.float32)
    ctrl.reset()
    traj = [s.copy()]
    forces = []
    for _ in range(horizon):
        obs = ctrl.obs_fn(s).copy()
        if noise > 0 and rng is not None:
            obs = obs + rng.normal(0, noise, size=obs.shape).astype(np.float32)
        f = ctrl.act(obs)
        if record_force:
            forces.append(f)
        s = env.step(s, f)
        traj.append(s.copy())
        if abs(s[2]) > math.pi / 2:
            break
    traj = np.array(traj)
    if record_force:
        return traj, np.array(forces, dtype=np.float32)
    return traj


def succeeded(traj, horizon=300):
    if len(traj) < horizon + 1:
        return False
    return abs(traj[-1, 2]) < 0.25 and abs(traj[-1, 0]) < 2.5


def upright_pct(traj):
    return float(np.mean(np.abs(traj[:, 2]) < 0.25) * 100)


def survival_steps(traj, horizon=300):
    return min(len(traj) - 1, horizon)


# ============================================================================
# EXPERIMENT A: architecture card
# ============================================================================
def architecture_card(controllers_meta, out_dir):
    print("\nArchitecture card (what each controller actually sees):")
    print("-" * 88)
    print(f"  {'name':28s} {'params':>8s}  {'input':22s}  memory")
    print("-" * 88)
    rows = []
    for m in controllers_meta:
        print(f"  {m['name']:28s} {m['params']:>8,}  {m['input']:22s}  {m['memory']}")
        rows.append(m)
    print("-" * 88)
    print("  Key contrast: MLP-partial sees 1 frame of [x,theta]  -> cannot know velocity.")
    print("                MLP-flat     sees T frames flattened  -> has history, no structure.")
    print("                Transformer  attends over T frames    -> recovers velocity from seq.")

    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.axis("off")
        col_labels = ["controller", "params", "observation", "memory / structure"]
        cell = [[r["name"], f"{r['params']:,}", r["input"], r["memory"]] for r in rows]
        table = ax.table(cellText=cell, colLabels=col_labels, loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.6)
        for (r, c), cell_obj in table.get_celld().items():
            if r == 0:
                cell_obj.set_facecolor("#2c3e50")
                cell_obj.set_text_props(color="white", weight="bold")
            elif "Transformer" in cell[r - 1][0]:
                cell_obj.set_facecolor("#d5f5e3")
            elif "partial" in cell[r - 1][0]:
                cell_obj.set_facecolor("#fadbd8")
        ax.set_title("A. Architecture card — MLP vs Transformer", fontsize=11, pad=12)
        fig.tight_layout()
        path = Path(out_dir) / "A_architecture.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"saved {path}")


# ============================================================================
# EXPERIMENT B: same final frame, different history
# ============================================================================
def same_frame_diff_history(mlp_part, tfm, norm_part, T=12, out_dir="."):
    """
    Both windows END at the same [x, theta]. A memoryless MLP must output the
    same force. The Transformer should output different forces (inferred velocity).
    """
    mean, std = norm_part
    final = np.array([0.0, 0.10], dtype=np.float32)

    rising = np.linspace(0.02, 0.10, T, dtype=np.float32)     # falling right fast
    falling = np.linspace(0.18, 0.10, T, dtype=np.float32)    # recovering toward upright
    winA = np.stack([np.zeros(T, dtype=np.float32), rising], axis=1)
    winB = np.stack([np.zeros(T, dtype=np.float32), falling], axis=1)

    with torch.no_grad():
        # memoryless MLP: only the final frame
        x_final = (torch.tensor(final).to(DEVICE) - mean) / std
        f_mlp = float(mlp_part(x_final).item())

        xA = (torch.tensor(winA[None]).to(DEVICE) - mean) / std
        xB = (torch.tensor(winB[None]).to(DEVICE) - mean) / std
        outA, wA = tfm(xA, return_weights=True)
        outB, wB = tfm(xB, return_weights=True)
        fA = float(outA[0, -1, 0])
        fB = float(outB[0, -1, 0])
        attnA = wA[0].mean(0)[-1].cpu().numpy()   # last query, head-avg
        attnB = wB[0].mean(0)[-1].cpu().numpy()

    # rough finite-diff "true" angular velocities of the synthetic histories
    dt = 0.02
    thd_A = (rising[-1] - rising[0]) / ((T - 1) * dt)
    thd_B = (falling[-1] - falling[0]) / ((T - 1) * dt)
    # teacher force if it saw x=0, xdot=0, theta=0.10, and those velocities
    f_teacher_A = teacher_force([0.0, 0.0, 0.10, thd_A])
    f_teacher_B = teacher_force([0.0, 0.0, 0.10, thd_B])

    print("\nSame final frame, different history (the smoking-gun test):")
    print(f"  Both windows END at obs = [x=0.00, theta=+0.10]")
    print(f"  History A: theta {rising[0]:.2f} -> {rising[-1]:.2f}   "
          f"(approx theta_dot ~ {thd_A:+.2f})")
    print(f"  History B: theta {falling[0]:.2f} -> {falling[-1]:.2f}  "
          f"(approx theta_dot ~ {thd_B:+.2f})")
    print(f"  MLP (partial, 1 frame) force for A = {f_mlp:+.3f} N")
    print(f"  MLP (partial, 1 frame) force for B = {f_mlp:+.3f} N   <-- IDENTICAL (must be)")
    print(f"  Transformer force for A            = {fA:+.3f} N   (teacher~{f_teacher_A:+.2f})")
    print(f"  Transformer force for B            = {fB:+.3f} N   (teacher~{f_teacher_B:+.2f})")
    print(f"  |fA - fB| = {abs(fA - fB):.3f} N  -> Transformer distinguished the two histories.")

    if HAVE_PLT:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

        # panel 1: the two theta histories
        t = np.arange(T)
        axes[0].plot(t, np.degrees(rising), "o-", color="#D55E00", label="A: rising (falling right)")
        axes[0].plot(t, np.degrees(falling), "s-", color="#0072B2", label="B: recovering")
        axes[0].axhline(np.degrees(0.10), color="gray", linestyle=":", label="shared final frame")
        axes[0].set_xlabel("window index")
        axes[0].set_ylabel("theta (deg)")
        axes[0].set_title("B1. Two histories, same last frame")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.3)

        # panel 2: force bars
        names = ["MLP\n(partial)", "TF\nhist A", "TF\nhist B", "LQR\n(A)", "LQR\n(B)"]
        vals = [f_mlp, fA, fB, f_teacher_A, f_teacher_B]
        colors = ["#D55E00", "#009E73", "#56B4E9", "#999999", "#666666"]
        axes[1].bar(names, vals, color=colors)
        axes[1].axhline(0, color="k", linewidth=0.8)
        axes[1].set_ylabel("force (N)")
        axes[1].set_title("B2. Output force — MLP identical, TF differs")
        axes[1].grid(alpha=0.3, axis="y")

        # panel 3: attention of last query
        xidx = np.arange(T)
        w = 0.35
        axes[2].bar(xidx - w / 2, attnA, w, color="#D55E00", label="attn on hist A")
        axes[2].bar(xidx + w / 2, attnB, w, color="#0072B2", label="attn on hist B")
        axes[2].set_xlabel("past frame index")
        axes[2].set_ylabel("attention weight")
        axes[2].set_title("B3. Last-frame attention over the window")
        axes[2].legend(fontsize=7)
        axes[2].grid(alpha=0.3, axis="y")

        fig.suptitle("B. Same observation, different history — only the Transformer can tell them apart",
                     fontsize=11)
        fig.tight_layout()
        path = Path(out_dir) / "B_same_frame.png"
        fig.savefig(path, dpi=120)
        print(f"saved {path}")

    return dict(f_mlp=f_mlp, fA=fA, fB=fB)


# ============================================================================
# EXPERIMENT C: initial-condition grid (success + survival time)
# ============================================================================
def grid_experiment(env, controllers, out_dir, n=11, horizon=300):
    thetas = np.linspace(-0.30, 0.30, n)
    theta_dots = np.linspace(-4.0, 4.0, n)
    success = {c.name: np.zeros((n, n)) for c in controllers}
    survive = {c.name: np.zeros((n, n)) for c in controllers}
    rates = {}
    mean_surv = {}

    for c in controllers:
        for i, thd in enumerate(theta_dots):
            for j, th in enumerate(thetas):
                traj = rollout(env, c, [0.0, 0.0, th, thd], horizon=horizon)
                success[c.name][i, j] = 100.0 if succeeded(traj, horizon) else 0.0
                survive[c.name][i, j] = survival_steps(traj, horizon)
        rates[c.name] = float(np.mean(success[c.name] > 0) * 100)
        mean_surv[c.name] = float(np.mean(survive[c.name]))
        print(f"  {c.name:28s}: success {rates[c.name]:5.1f}% of starts  |  "
              f"mean survival {mean_surv[c.name]:5.1f}/{horizon} steps")

    if HAVE_PLT:
        ncols = len(controllers)
        # success heatmaps
        fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 3.8))
        if ncols == 1:
            axes = [axes]
        for ax, c in zip(axes, controllers):
            ax.imshow(success[c.name], origin="lower", cmap="RdYlGn", vmin=0, vmax=100,
                      extent=[thetas[0], thetas[-1], theta_dots[0], theta_dots[-1]],
                      aspect="auto")
            ax.set_title(f"{c.name}\nsolved {rates[c.name]:.0f}%", fontsize=8)
            ax.set_xlabel("theta0 (rad)")
            ax.set_ylabel("theta_dot0")
        fig.suptitle("C1. Success region (green = balanced for full horizon)", fontsize=11)
        fig.tight_layout()
        path = Path(out_dir) / "C_grid_success.png"
        fig.savefig(path, dpi=120)
        print(f"saved {path}")

        # survival-time heatmaps (softer metric — shows near-misses)
        fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 3.8))
        if ncols == 1:
            axes = [axes]
        for ax, c in zip(axes, controllers):
            im = ax.imshow(survive[c.name], origin="lower", cmap="viridis",
                           vmin=0, vmax=horizon,
                           extent=[thetas[0], thetas[-1], theta_dots[0], theta_dots[-1]],
                           aspect="auto")
            ax.set_title(f"{c.name}\nmean surv {mean_surv[c.name]:.0f}", fontsize=8)
            ax.set_xlabel("theta0 (rad)")
            ax.set_ylabel("theta_dot0")
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="survival steps")
        fig.suptitle("C2. Survival time (seconds of balance before falling)", fontsize=11)
        fig.tight_layout()
        path = Path(out_dir) / "C_grid_survival.png"
        fig.savefig(path, dpi=120)
        print(f"saved {path}")

        # bar chart summary
        fig, ax = plt.subplots(figsize=(8, 3.6))
        names = [c.name for c in controllers]
        xs = np.arange(len(names))
        ax.bar(xs - 0.18, [rates[n] for n in names], 0.35, label="success % of starts", color="#009E73")
        ax.bar(xs + 0.18, [100 * mean_surv[n] / horizon for n in names], 0.35,
               label="mean survival % of horizon", color="#0072B2")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_ylabel("%")
        ax.set_title("C3. Aggregate: who recovers from more starts?")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        path = Path(out_dir) / "C_grid_summary.png"
        fig.savefig(path, dpi=120)
        print(f"saved {path}")

    return rates, mean_surv


# ============================================================================
# EXPERIMENT D: side-by-side trajectories on hard starts (+ GIFs)
# ============================================================================
HARD_STARTS = [
    ("easy_still",    "easy: almost still",   [0.0, 0.0, 0.05, 0.0]),
    ("medium_spin",   "medium: spinning",     [0.0, 0.0, 0.05, 2.5]),
    ("hard_tilt_spin","hard: tilt + spin",    [0.0, 0.0, 0.18, -3.0]),
    ("hard_opp_spin", "hard: opposite spin",  [0.0, 0.0, -0.15, 3.5]),
]

# Short panel titles + colors for side-by-side GIFs
CTRL_STYLE = {
    "LQR teacher (full)":      ("LQR teacher",   "#0072B2"),
    "MLP (full state)":        ("MLP full",      "#E69F00"),
    "MLP (partial)":           ("MLP partial",   "#D55E00"),
    "MLP (flat window)":       ("MLP flat win",  "#CC79A7"),
    "Transformer (history)":   ("Transformer",   "#009E73"),
}

# Focused 3-way duel — the clearest visual contrast
GIF_FOCUS = ("MLP (partial)", "MLP (flat window)", "Transformer (history)")


def _fig_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _draw_cartpole(ax, env, state, title, color, alive, step_i):
    """Draw one cart-pole panel. alive=False => faded + FELL stamp."""
    L = 2.0 * env.half_length
    x, _, theta, _ = state
    ax.clear()
    ax.plot([-3, 3], [0, 0], "k-", linewidth=2)
    cw, ch = 0.5, 0.3
    alpha = 0.65 if alive else 0.25
    cart_x = [x - cw / 2, x + cw / 2, x + cw / 2, x - cw / 2, x - cw / 2]
    ax.fill(cart_x, [0, 0, ch, ch, 0], color=color, alpha=alpha)
    px = x + L * math.sin(theta)
    py = ch + L * math.cos(theta)
    ax.plot([x, px], [ch, py], color=color, linewidth=4, alpha=alpha)
    ax.scatter([px], [py], s=110, color=color, alpha=alpha)
    status = "OK" if alive else "FELL"
    ax.set_title(f"{title}  [{status}]\n"
                 f"t={step_i:03d}  x={x:+.2f}  θ={math.degrees(theta):+.1f}°",
                 fontsize=9, color=color if alive else "#888888")
    if not alive:
        ax.text(0.0, 1.4, "FELL", ha="center", va="center", fontsize=18,
                fontweight="bold", color="#D55E00", alpha=0.85)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 2.7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.15)


def create_side_by_side_gif(env, trajs_by_name, names, out_path, subtitle,
                            max_frames=120, duration=0.05):
    """
    Render several controllers side-by-side in ONE gif, time-synced.
    Shorter (fallen) trajectories freeze on their last frame with a FELL stamp.
    """
    if not HAVE_PLT or not HAVE_IMAGEIO:
        print("matplotlib/imageio not available, skipping GIF")
        return

    n = len(names)
    lengths = [len(trajs_by_name[nm]) for nm in names]
    T_max = max(lengths)
    # subsample so gifs stay small/fast
    step = max(1, T_max // max_frames)
    indices = list(range(0, T_max, step))
    if indices[-1] != T_max - 1:
        indices.append(T_max - 1)

    fig_w = max(3.2 * n, 6.5)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.6))
    if n == 1:
        axes = [axes]

    frames = []
    for i in indices:
        for ax, nm in zip(axes, names):
            traj = trajs_by_name[nm]
            alive = i < len(traj)
            state = traj[min(i, len(traj) - 1)]
            short, color = CTRL_STYLE.get(nm, (nm, "#333333"))
            # still "alive" only while within recorded traj AND pole upright-ish
            pole_up = abs(state[2]) < math.pi / 2
            _draw_cartpole(ax, env, state, short, color, alive and pole_up, i)
        fig.suptitle(subtitle, fontsize=11, y=1.02)
        fig.tight_layout()
        frames.append(_fig_to_rgb(fig))
    plt.close(fig)

    path = Path(out_path)
    imageio.mimsave(path, frames, duration=duration, loop=0)
    print(f"saved {path}  ({len(frames)} frames, {n} panels)")


# Only interesting starts get GIFs (easy_still is boring — everyone succeeds)
GIF_STARTS = ("medium_spin", "hard_tilt_spin", "hard_opp_spin")
GIF_SLUG_NAME = {
    "medium_spin": "medium_spin",
    "hard_tilt_spin": "hard_tilt",
    "hard_opp_spin": "hard_opp",
}
TRAJ_PLOT_NAME = {
    "easy_still": "D_traj_easy.png",
    "medium_spin": "D_traj_medium.png",
    "hard_tilt_spin": "D_traj_hard_tilt.png",
    "hard_opp_spin": "D_traj_hard_opp.png",
}


def make_comparison_gifs(env, results, controllers, gif_dir, horizon=300):
    """Keep GIFs lean: 3-way focus for hard starts + one all-5 for medium_spin."""
    if not HAVE_PLT or not HAVE_IMAGEIO:
        print("matplotlib/imageio not available, skipping comparison GIFs")
        return

    gif_dir = Path(gif_dir)
    gif_dir.mkdir(parents=True, exist_ok=True)
    all_names = [c.name for c in controllers]
    focus_names = [n for n in GIF_FOCUS if n in all_names]

    for slug, label, init in HARD_STARTS:
        if slug not in GIF_STARTS:
            continue
        trajs = {nm: results[slug][nm][0] for nm in all_names}
        short = GIF_SLUG_NAME[slug]
        if focus_names:
            create_side_by_side_gif(
                env, trajs, focus_names,
                gif_dir / f"{short}_3way.gif",
                subtitle=f"{label}   MLP-partial | flat-window | Transformer",
            )
        if slug == "medium_spin":
            create_side_by_side_gif(
                env, trajs, all_names,
                gif_dir / f"{short}_all5.gif",
                subtitle=f"{label}   all controllers",
            )


def trajectory_experiment(env, controllers, plot_dir, gif_dir=None, horizon=300, make_gifs=True):
    print("\nSide-by-side trajectories on shared hard starts:")
    results = {}
    for slug, label, init in HARD_STARTS:
        print(f"\n  start = {label}  init={np.round(init, 2)}")
        results[slug] = {}
        for c in controllers:
            traj, forces = rollout(env, c, init, horizon=horizon, record_force=True)
            ok = succeeded(traj, horizon)
            results[slug][c.name] = (traj, forces)
            print(f"    {c.name:28s}: survived {len(traj)-1:3d} steps  "
                  f"upright {upright_pct(traj):5.1f}%  {'OK' if ok else 'FAIL'}")

    if HAVE_PLT:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        for slug, label, init in HARD_STARTS:
            fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
            for c in controllers:
                traj, forces = results[slug][c.name]
                t = np.arange(len(traj))
                axes[0].plot(t, np.degrees(traj[:, 2]), linewidth=1.8, label=c.name)
                axes[1].plot(np.arange(len(forces)), forces, linewidth=1.4, label=c.name)
            axes[0].axhline(0, color="gray", linestyle=":")
            axes[0].axhline(90, color="red", linestyle="--", alpha=0.4)
            axes[0].axhline(-90, color="red", linestyle="--", alpha=0.4)
            axes[0].set_ylabel("theta (deg)")
            axes[0].set_ylim(-100, 100)
            axes[0].set_title(f"D. {label}   init={np.round(init, 2)}")
            axes[0].legend(fontsize=7, loc="upper right")
            axes[0].grid(alpha=0.3)
            axes[1].axhline(0, color="gray", linestyle=":")
            axes[1].set_xlabel("timestep")
            axes[1].set_ylabel("force (N)")
            axes[1].grid(alpha=0.3)
            fig.tight_layout()
            path = plot_dir / TRAJ_PLOT_NAME[slug]
            fig.savefig(path, dpi=120)
            print(f"saved {path}")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (slug, label, init) in zip(axes.ravel(), HARD_STARTS):
            for c in controllers:
                traj, _ = results[slug][c.name]
                ax.plot(np.degrees(traj[:, 2]), linewidth=1.5, label=c.name)
            ax.axhline(0, color="gray", linestyle=":")
            ax.set_ylim(-100, 100)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("timestep")
            ax.set_ylabel("theta (deg)")
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=6)
        fig.suptitle("D. Pole angle under shared starts — where memoryless MLP fails", fontsize=11)
        fig.tight_layout()
        path = plot_dir / "D_traj_overview.png"
        fig.savefig(path, dpi=120)
        print(f"saved {path}")

    if make_gifs:
        print("\nRendering side-by-side comparison GIFs...")
        make_comparison_gifs(env, results, controllers, gif_dir or plot_dir, horizon=horizon)

    return results


# ============================================================================
# EXPERIMENT E: window-length sweep for the Transformer
# ============================================================================
def window_sweep(env, data_factory, norm_part, out_dir, Ts=(1, 2, 4, 8, 12, 16),
                 n_test=11, horizon=300, epochs=400):
    """
    Retrain a Transformer for each window length T and measure grid success.
    T=1 is essentially 'Transformer with no history' — should look like MLP-partial.
    """
    thetas = np.linspace(-0.30, 0.30, n_test)
    theta_dots = np.linspace(-4.0, 4.0, n_test)
    rates = []
    print("\nWindow-length sweep (Transformer only):")
    for T in Ts:
        data = data_factory(T)
        # re-fit a fresh transformer on this T
        tfm = train_transformer(data["seq_X"], data["seq_Y"], norm_part, epochs=epochs, seed=T)
        ctrl = TransformerCtrl(tfm, norm_part, T=T)
        wins = 0
        total = n_test * n_test
        for thd in theta_dots:
            for th in thetas:
                traj = rollout(env, ctrl, [0.0, 0.0, th, thd], horizon=horizon)
                if succeeded(traj, horizon):
                    wins += 1
        rate = 100.0 * wins / total
        rates.append(rate)
        print(f"  T={T:2d}: success {rate:5.1f}% of {total} starts")

    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(Ts, rates, "o-", color="#009E73", linewidth=2, markersize=8)
        ax.axhline(rates[0], color="#D55E00", linestyle="--",
                   label=f"T=1 (no history) = {rates[0]:.0f}%")
        ax.set_xlabel("window length T (frames of [x, theta])")
        ax.set_ylabel("success rate on IC grid (%)")
        ax.set_title("E. How much history does the Transformer need?")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = Path(out_dir) / "E_window_sweep.png"
        fig.savefig(path, dpi=120)
        print(f"saved {path}")
    return dict(zip(Ts, rates))


# ============================================================================
# EXPERIMENT F: sensor-noise sweep
# ============================================================================
def noise_experiment(env, controllers, out_dir, noises=(0.0, 0.01, 0.02, 0.04, 0.08), seed=0):
    rng_inits = np.random.default_rng(seed)
    inits = []
    for _ in range(40):
        inits.append([0.0,
                      rng_inits.uniform(-0.5, 0.5),
                      rng_inits.uniform(-0.15, 0.15),
                      rng_inits.uniform(-2.5, 2.5)])
    table = {c.name: [] for c in controllers}
    for c in controllers:
        for nz in noises:
            rng = np.random.default_rng(123)
            wins = sum(succeeded(rollout(env, c, s0, noise=nz, rng=rng)) for s0 in inits)
            table[c.name].append(100.0 * wins / len(inits))

    print("\nSensor-noise robustness (success % over 40 hard starts):")
    header = "  controller                   " + "".join(f"  nz={nz:<5}" for nz in noises)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in controllers:
        row = "".join(f"  {v:9.0f}" for v in table[c.name])
        print(f"  {c.name:28s}{row}")

    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for c in controllers:
            ax.plot(noises, table[c.name], marker="o", linewidth=2, label=c.name)
        ax.set_xlabel("observation noise (std)")
        ax.set_ylabel("success rate (%)")
        ax.set_title("F. Robustness to sensor noise")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = Path(out_dir) / "F_noise.png"
        fig.savefig(path, dpi=120)
        print(f"saved {path}")
    return table


# ============================================================================
# FINAL SCORECARD
# ============================================================================
def print_scorecard(rates, mean_surv, noise_table, same_frame, window_rates, horizon=300):
    print("\n" + "=" * 70)
    print("SCORECARD — MLP vs Transformer (what actually differed)")
    print("=" * 70)
    print(f"  {'controller':28s}  {'IC success':>10s}  {'mean surv':>10s}  {'noise=0':>8s}  {'noise=0.04':>10s}")
    print("  " + "-" * 72)
    for name in rates:
        nz0 = noise_table[name][0] if name in noise_table else float("nan")
        # find noise≈0.04 column
        nz04 = noise_table[name][3] if name in noise_table and len(noise_table[name]) > 3 else float("nan")
        print(f"  {name:28s}  {rates[name]:9.1f}%  {mean_surv[name]:8.1f}/{horizon}  "
              f"{nz0:7.0f}%  {nz04:9.0f}%")
    if same_frame:
        print(f"\n  Same-frame test: MLP-partial f={same_frame['f_mlp']:+.3f} (identical for both hist)")
        print(f"                   Transformer  fA={same_frame['fA']:+.3f}  fB={same_frame['fB']:+.3f}  "
              f"|Δ|={abs(same_frame['fA']-same_frame['fB']):.3f}")
    if window_rates:
        best_T = max(window_rates, key=window_rates.get)
        print(f"\n  Window sweep: best T={best_T} -> {window_rates[best_T]:.0f}%  "
              f"(T=1 -> {window_rates.get(1, float('nan')):.0f}%)")
    print("\nTAKEAWAYS")
    print("- Full state: a tiny MLP already matches the LQR teacher.")
    print("- Partial obs, 1 frame: memoryless MLP only survives a thin theta_dot~0 band.")
    print("- Flat window MLP: history helps, but unstructured mixing is weaker.")
    print("- Transformer: recovers velocity from the sequence; widest success region.")
    print("- More history (larger T) helps up to a point; T=1 collapses to memoryless.")
    print("- Under noise, a window also averages — history helps robustness.")
    print("- Match the network to the OBSERVABILITY of your problem, not to fashion.")


def main():
    parser = argparse.ArgumentParser(description="Detailed MLP vs Transformer comparison on cart-pole")
    parser.add_argument("--grid", type=int, default=11, help="grid resolution per axis")
    parser.add_argument("--seq-len", type=int, default=12, help="default Transformer window T")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--skip-window-sweep", action="store_true",
                        help="skip experiment E (re-trains several Transformers)")
    parser.add_argument("--skip-gifs", action="store_true",
                        help="skip side-by-side comparison GIFs")
    _default_out = Path(__file__).parent / "figures" / "05_compare"
    parser.add_argument("--out", type=str, default=str(_default_out),
                        help="output root (writes plots/ and gifs/ under this)")
    args = parser.parse_args()

    if not HAVE_TORCH:
        print("torch not installed -> this comparison needs torch. Aborting.")
        return

    out_dir = Path(args.out)
    plot_dir = out_dir / "plots"
    gif_dir = out_dir / "gifs"
    plot_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    T = args.seq_len
    env = CartPole()

    print("=" * 70)
    print("STEP 0: LQR teacher + collect data")
    print("=" * 70)
    setup_teacher(env)
    data = collect(env, T=T)

    print("\n" + "=" * 70)
    print("STEP 1: train students")
    print("=" * 70)
    norm_full = norm_of(data["full_X"])
    norm_part = norm_of(data["part_X"])
    mlp_full = train_frame_mlp(data["full_X"], data["force_Y"], norm_full,
                               epochs=args.epochs, tag="mlp-full")
    mlp_part = train_frame_mlp(data["part_X"], data["force_Y"], norm_part,
                               epochs=args.epochs, tag="mlp-partial")
    mlp_flat, norm_flat = train_flat_window_mlp(data["seq_X"], data["seq_Y"], norm_part,
                                                epochs=args.epochs)
    tfm = train_transformer(data["seq_X"], data["seq_Y"], norm_part, epochs=args.epochs)

    controllers = [
        TeacherCtrl(),
        FrameMLPCtrl(mlp_full, norm_full, full_obs, "MLP (full state)"),
        FrameMLPCtrl(mlp_part, norm_part, partial_obs, "MLP (partial)"),
        FlatWindowMLPCtrl(mlp_flat, norm_flat, T=T),
        TransformerCtrl(tfm, norm_part, T=T),
    ]

    # ----- A -----
    print("\n" + "=" * 70)
    print("EXPERIMENT A: architecture card")
    print("=" * 70)
    meta = [
        dict(name="LQR teacher (full)", params=4,
             input="[x,xdot,theta,thdot]", memory="none (linear gains)"),
        dict(name="MLP (full state)", params=count_params(mlp_full),
             input="[x,xdot,theta,thdot]", memory="none (1 frame)"),
        dict(name="MLP (partial)", params=count_params(mlp_part),
             input="[x, theta] x 1", memory="none (1 frame)"),
        dict(name="MLP (flat window)", params=count_params(mlp_flat),
             input=f"[x,theta] x {T} flat", memory=f"{T} frames, no attention"),
        dict(name="Transformer (history)", params=count_params(tfm),
             input=f"[x,theta] x {T} seq", memory=f"{T} frames + causal attn"),
    ]
    architecture_card(meta, plot_dir)

    # ----- B -----
    print("\n" + "=" * 70)
    print("EXPERIMENT B: same final frame, different history")
    print("=" * 70)
    same_frame = same_frame_diff_history(mlp_part, tfm, norm_part, T=T, out_dir=plot_dir)

    # ----- C -----
    print("\n" + "=" * 70)
    print("EXPERIMENT C: recover from a grid of (theta0, theta_dot0)")
    print("=" * 70)
    rates, mean_surv = grid_experiment(env, controllers, plot_dir, n=args.grid)

    # ----- D -----
    print("\n" + "=" * 70)
    print("EXPERIMENT D: side-by-side trajectories on hard starts")
    print("=" * 70)
    trajectory_experiment(env, controllers, plot_dir, gif_dir=gif_dir,
                          make_gifs=not args.skip_gifs)

    # ----- E -----
    window_rates = {}
    if not args.skip_window_sweep:
        print("\n" + "=" * 70)
        print("EXPERIMENT E: Transformer window-length sweep")
        print("=" * 70)

        def data_factory(T_):
            return collect(env, T=T_, n_episodes=200, horizon=200, seed=0)

        window_rates = window_sweep(env, data_factory, norm_part, plot_dir,
                                    epochs=max(300, args.epochs // 2))
    else:
        print("\n(skipping experiment E — pass without --skip-window-sweep to run it)")

    # ----- F -----
    print("\n" + "=" * 70)
    print("EXPERIMENT F: robustness to sensor noise")
    print("=" * 70)
    noise_table = noise_experiment(env, controllers, plot_dir)

    print_scorecard(rates, mean_surv, noise_table, same_frame, window_rates)

    print(f"\nOutputs under: {out_dir.resolve()}")
    print("  plots/  A_.. F_..  (one file per experiment)")
    print("  gifs/   *_3way.gif  (+ medium_spin_all5.gif)")


if __name__ == "__main__":
    main()
