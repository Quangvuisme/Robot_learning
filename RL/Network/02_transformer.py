"""
Network 02: Transformer / Self-Attention
From-scratch attention math + a PyTorch version, grounded in pendulum trajectories.

PURPOSE:
--------
An MLP (see 01_mlp.py) maps ONE state vector to an output. But control is
sequential: a pendulum's future depends on its recent HISTORY, not just the
current frame (think partial observability, momentum, sensor delay). A
Transformer is the network that learns "which past timesteps matter for
predicting now" via SELF-ATTENTION. Modern decision/trajectory transformers
turn RL itself into sequence modeling on exactly this kind of data.

THE PENDULUM CONNECTION:
------------------------
Instead of a single state we feed a short TRAJECTORY (a window of T steps):
    S = [ s_{t-T+1}, ..., s_{t-1}, s_t ]      each s = [cos th, sin th, th_dot]
and ask the network to predict the NEXT state s_{t+1}.
To do that well it must figure out, for each query timestep, which earlier
timesteps to "look at". That look-up is attention.

WHAT SELF-ATTENTION COMPUTES (the whole idea in 4 lines):
    For each token (timestep) we build three vectors by linear projection:
        Q = query  ("what am I looking for?")
        K = key    ("what do I offer?")
        V = value  ("what info do I carry?")
    Attention(Q,K,V) = softmax( Q Kᵀ / sqrt(d_k) ) V
        1. Q Kᵀ        -> similarity score of every token to every other token
        2. / sqrt(d_k) -> scale so softmax gradients stay healthy
        3. softmax     -> turn scores into weights that sum to 1 (a soft look-up)
        4. ... V       -> weighted average of values = the new token representation

WHY IT BEATS JUST FEEDING THE FLAT WINDOW TO AN MLP:
    - Attention weights are DATA-DEPENDENT: a fast-spinning pendulum attends to
      different history than a slow one. An MLP's weights are fixed once trained.
    - It handles variable-length / long context and learns relationships
      between any two timesteps directly (no info squeezed through one bottleneck).

THE OTHER PIECES OF A TRANSFORMER BLOCK:
    1. Positional encoding  - attention is order-agnostic by itself, so we ADD a
                              signal encoding "which timestep is this" to each token.
    2. Multi-head attention - run several attentions in parallel ("heads"); each
                              can specialize (one tracks angle, one tracks velocity).
    3. Residual + LayerNorm - x + Sublayer(x), normalized: stabilizes deep stacks.
    4. Feed-forward (an MLP!)- a per-token MLP after attention mixes the features.
    A "block" = [ Attention -> Add&Norm -> FeedForward -> Add&Norm ], stacked N times.

CAUSAL MASK:
    For prediction we must not let timestep t peek at the future. We mask the
    upper triangle of the score matrix to -inf before softmax, so each token
    attends only to itself and earlier tokens. This is what makes it a *causal*
    (decoder-style) transformer, the kind used by Decision Transformer & GPT.

REFERENCE:
    Vaswani et al. (2017), "Attention Is All You Need".
    Chen et al. (2021), "Decision Transformer: RL via Sequence Modeling".
"""

import argparse
import math
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

# Base class so the nn.Module subclasses below can be *defined* even when torch
# is absent (their torch code only runs on instantiation, which we guard with
# HAVE_TORCH). Lets the NumPy attention demo in Part 1 run on its own.
_Base = nn.Module if HAVE_TORCH else object


# ----------------------------------------------------------------------------
# Data: simulate pendulum trajectories, then learn to predict the next state
# ----------------------------------------------------------------------------
def simulate_pendulum(n_traj, T, dt=0.05, seed=0):
    """Roll out a free (unactuated) pendulum and return windows of observations.

    Dynamics:  theta_ddot = -(g/l) sin(theta) - damping * theta_dot
    Returns X: [n_traj, T, 3] inputs (cos, sin, theta_dot)
            Y: [n_traj, T, 3] the next-step observation for each timestep
    """
    rng = np.random.default_rng(seed)
    g, l, damping = 9.81, 1.0, 0.1
    X = np.zeros((n_traj, T, 3), dtype=np.float32)
    Y = np.zeros((n_traj, T, 3), dtype=np.float32)
    for i in range(n_traj):
        theta = rng.uniform(-math.pi, math.pi)
        theta_dot = rng.uniform(-3.0, 3.0)
        for t in range(T + 1):
            obs = np.array([math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
            if t < T:
                X[i, t] = obs
            if t > 0:
                Y[i, t - 1] = obs  # the observation one step later
            theta_ddot = -(g / l) * math.sin(theta) - damping * theta_dot
            theta_dot += theta_ddot * dt
            theta += theta_dot * dt
    return X, Y


# ============================================================================
# PART 1 - SELF-ATTENTION FROM SCRATCH IN NUMPY (see the matrices)
# ============================================================================
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)        # stability shift
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, causal=True):
    """Q,K,V: [T, d]. Returns output [T, d] and the attention weights [T, T].

    This single function is the mathematical heart of every Transformer.
    """
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / math.sqrt(d_k)            # [T, T] token-to-token affinity
    if causal:
        T = scores.shape[0]
        # mask upper triangle (future) so token t can't see t+1, t+2, ...
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)             # [T, T] each row sums to 1
    out = weights @ V                              # [T, d] weighted average of values
    return out, weights


def demo_attention_numpy(T=8, d=3, seed=0):
    """Show what attention does on ONE pendulum trajectory window."""
    X, _ = simulate_pendulum(n_traj=1, T=T, seed=seed)
    tokens = X[0]                                  # [T, 3]

    # In a real transformer Q,K,V come from LEARNED projections of the tokens.
    # Here we use random fixed projections just to expose the mechanism.
    rng = np.random.default_rng(seed)
    Wq = rng.normal(0, 0.5, size=(3, d))
    Wk = rng.normal(0, 0.5, size=(3, d))
    Wv = rng.normal(0, 0.5, size=(3, d))
    Q, K, V = tokens @ Wq, tokens @ Wk, tokens @ Wv

    out, weights = scaled_dot_product_attention(Q, K, V, causal=True)
    print("Attention weight matrix (row = query timestep, col = attended timestep):")
    print("Lower-triangular because of the causal mask; each row sums to 1.\n")
    for t in range(T):
        row = "  ".join(f"{w:.2f}" for w in weights[t])
        print(f"  t={t}:  {row}")
    print(f"\nattention output shape: {out.shape}  (one new vector per timestep)")
    return weights


# ============================================================================
# PART 2 - A REAL (TINY) CAUSAL TRANSFORMER IN PYTORCH
# ============================================================================
# Predicts s_{t+1} from s_{<=t}. This is the same architecture family used by
# Decision Transformer, just shrunk so it trains in seconds on a CPU.

class PositionalEncoding(_Base):
    """Inject timestep order into the tokens (the classic sinusoidal scheme)."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadSelfAttention(_Base):
    """Multi-head causal self-attention, written out (no nn.MultiheadAttention)."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # One big projection produces Q, K, V for all heads at once.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)                                  # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)                     # each [B, T, D]
        # split into heads: [B, T, D] -> [B, heads, T, d_head]
        def split(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,h,T,T]
        # causal mask: forbid attending to the future
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = scores.softmax(dim=-1)
        ctx = weights @ v                                  # [B, h, T, d_head]
        # merge heads back together
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(ctx), weights


class TransformerBlock(_Base):
    """Attention -> Add&Norm -> FeedForward -> Add&Norm (pre-norm variant)."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        a, weights = self.attn(self.ln1(x))
        x = x + a                       # residual connection
        x = x + self.ff(self.ln2(x))    # residual connection
        return x, weights


class PendulumTransformer(_Base):
    """Embed state -> +position -> N transformer blocks -> predict next state."""

    def __init__(self, d_in=3, d_model=32, n_heads=4, d_ff=64, n_blocks=2):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(d_model, d_in)   # predict next [cos, sin, theta_dot]

    def forward(self, x):
        x = self.pos(self.embed(x))
        last_weights = None
        for blk in self.blocks:
            x, last_weights = blk(x)
        return self.head(x), last_weights


def train_transformer(epochs=400, lr=1e-3, T=16, n_traj=512, seed=0):
    torch.manual_seed(seed)
    X_np, Y_np = simulate_pendulum(n_traj=n_traj, T=T, seed=seed)
    X = torch.tensor(X_np)
    Y = torch.tensor(Y_np)

    model = PendulumTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[transformer] learnable parameters: {n_params}")
    for epoch in range(epochs):
        opt.zero_grad()
        pred, _ = model(X)
        loss = loss_fn(pred, Y)         # predict next state at EVERY timestep
        loss.backward()
        opt.step()
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"[transformer] epoch {epoch:4d}  next-state MSE = {loss.item():.5f}")
    return model, (X, Y)


# ============================================================================
# VISUALIZATION - the attention map + prediction quality
# ============================================================================
def plot_attention(np_weights, model, data, out_dir):
    if not HAVE_PLT:
        print("matplotlib not available, skipping plot")
        return
    panels = []
    if np_weights is not None:
        panels.append(("NumPy causal attention", np_weights))
    if model is not None and HAVE_TORCH:
        X, _ = data
        with torch.no_grad():
            _, weights = model(X[:1])
        # average the learned attention over heads for one trajectory
        panels.append(("Learned attention (head-avg)", weights[0].mean(0).numpy()))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, W) in zip(axes, panels):
        im = ax.imshow(W, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("attended timestep (key)")
        ax.set_ylabel("query timestep")
        fig.colorbar(im, ax=ax)
    fig.suptitle("Self-attention weights on a pendulum trajectory (lower-triangular = causal)")
    fig.tight_layout()
    path = Path(out_dir) / "transformer_attention.png"
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Transformer explained on pendulum trajectories")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent))
    args = parser.parse_args()

    print("=" * 70)
    print("PART 1: Self-attention from scratch in NumPy")
    print("=" * 70)
    np_weights = demo_attention_numpy(T=8)

    model, data = None, None
    if HAVE_TORCH:
        print()
        print("=" * 70)
        print("PART 2: A tiny causal Transformer predicting the next pendulum state")
        print("=" * 70)
        model, data = train_transformer(epochs=args.epochs, T=args.seq_len)
    else:
        print("\n[torch not installed - skipping PyTorch section]")

    plot_attention(np_weights, model, data, args.out)

    print("\nTAKEAWAYS")
    print("- Attention = softmax(Q Kᵀ / sqrt(d)) V: a learned, data-dependent look-up.")
    print("- Q/K/V are projections of the tokens; the causal mask hides the future.")
    print("- Positional encoding restores order; multi-head lets features specialize.")
    print("- A block = attention + a per-token MLP, with residuals + LayerNorm.")
    print("- Feeding trajectories (not single states) is how RL becomes sequence modeling.")


if __name__ == "__main__":
    main()
