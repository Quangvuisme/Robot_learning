"""
Network 01: Multi-Layer Perceptron (MLP)
From-scratch math + a PyTorch version, grounded in the inverted pendulum.

PURPOSE:
--------
Every RL agent in this repo (DQN, DDPG, SAC, TRPO) is "a policy/value function
parameterized by a neural network". That network is almost always an MLP.
This file opens the black box and shows EXACTLY what an MLP computes, why it
can approximate the pendulum's value/policy, and how it learns.

THE PENDULUM CONNECTION:
------------------------
An inverted pendulum is observed as a state vector
    s = [cos(theta), sin(theta), theta_dot]     (3 numbers)
and we want a function that maps that state to something useful, e.g.
    - a torque to apply            (policy  pi(s) -> a)        [continuous]
    - a value estimate             (critic  V(s) -> scalar)
    - Q-values for each action     (Q(s) -> [q_left, q_right]) [discrete]

All three are just "vector in -> vector out". An MLP is the simplest network
that learns such a mapping. To make the learning *visible* without an RL loop,
this file trains an MLP on a known target function:

    target(s) = energy-shaping torque for swing-up
              = theta_dot * cos(theta)            (a classic swing-up signal)

We KNOW the right answer, so we can watch the MLP approximate it and measure
the error directly. The same network shape is what you would drop into an
actor or critic.

WHAT AN MLP IS (one line):
    A stack of (linear map -> nonlinearity), repeated.
    Linear alone = a single matrix = can only draw straight lines.
    Nonlinearity between layers = can bend, so it becomes a UNIVERSAL
    approximator (can fit essentially any smooth function given enough width).

ONE HIDDEN LAYER, WRITTEN OUT:
    Input    s in R^{d_in}
    Layer 1  h = act(W1 @ s + b1)        W1: [hidden, d_in],  b1: [hidden]
    Layer 2  y = W2 @ h + b2             W2: [d_out, hidden], b2: [d_out]
    Output   y in R^{d_out}

    "Parameters" = the numbers in W1, b1, W2, b2. Learning = nudging those
    numbers so y gets closer to the target. That nudge comes from gradient
    descent on a loss (here: mean squared error).

WHY THE NONLINEARITY MATTERS:
    Without act(), two stacked linear layers collapse:
        W2 @ (W1 @ s) = (W2 @ W1) @ s = W_combined @ s
    ...still just one linear map. The activation (ReLU/Tanh) is what gives the
    network the power to represent curved functions like swing-up energy.

REFERENCE:
    Cybenko (1989), Hornik (1991): universal approximation theorem.
"""

import argparse
import math
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # torch optional: the numpy section runs without it
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


# ----------------------------------------------------------------------------
# The target we want the MLP to learn (a known function of the pendulum state)
# ----------------------------------------------------------------------------
def pendulum_state(theta, theta_dot):
    """Map raw angle to the 3-D observation an RL agent actually sees."""
    return np.stack([np.cos(theta), np.sin(theta), theta_dot], axis=-1)


def target_torque(theta, theta_dot):
    """A hand-designed swing-up signal we'll ask the MLP to imitate.

    theta_dot * cos(theta) pumps energy in at the bottom and is the kind of
    smooth, nonlinear surface an MLP can fit but a single linear layer cannot.
    """
    return theta_dot * np.cos(theta)


# ============================================================================
# PART 1 - MLP FROM SCRATCH IN NUMPY (see every multiply and gradient)
# ============================================================================
# This is a 1-hidden-layer MLP: 3 inputs -> H hidden (tanh) -> 1 output.
# We implement the forward pass, the loss, and manual backprop so nothing is
# hidden behind a framework. This is literally what torch.nn does internally.

def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    # d/dx tanh(x) = 1 - tanh(x)^2
    return 1.0 - np.tanh(x) ** 2


class NumpyMLP:
    def __init__(self, d_in=3, hidden=64, d_out=1, seed=0):
        rng = np.random.default_rng(seed)
        # He-ish init: small random weights so signals don't explode.
        self.W1 = rng.normal(0, 1.0 / math.sqrt(d_in), size=(hidden, d_in))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1.0 / math.sqrt(hidden), size=(d_out, hidden))
        self.b2 = np.zeros(d_out)

    def forward(self, s):
        """s: [batch, d_in] -> y: [batch, d_out]. Cache activations for backprop."""
        self.s = s                              # [B, d_in]
        self.z1 = s @ self.W1.T + self.b1       # [B, H]   pre-activation
        self.h = tanh(self.z1)                  # [B, H]   hidden activation
        self.y = self.h @ self.W2.T + self.b2   # [B, d_out] output (linear)
        return self.y

    def backward(self, target):
        """Manual chain rule for MSE loss = mean((y - target)^2)."""
        B = self.s.shape[0]
        # dL/dy   (MSE derivative)
        dy = (2.0 / B) * (self.y - target)              # [B, d_out]
        # Output layer grads
        dW2 = dy.T @ self.h                             # [d_out, H]
        db2 = dy.sum(axis=0)                            # [d_out]
        # Backprop into hidden activation, then through tanh
        dh = dy @ self.W2                               # [B, H]
        dz1 = dh * tanh_grad(self.z1)                   # [B, H]
        # First layer grads
        dW1 = dz1.T @ self.s                            # [H, d_in]
        db1 = dz1.sum(axis=0)                           # [H]
        return dW1, db1, dW2, db2

    def step(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2


def train_numpy_mlp(epochs=6000, lr=0.01, hidden=64, seed=0):
    rng = np.random.default_rng(seed)
    # Sample random pendulum states across the full range.
    theta = rng.uniform(-math.pi, math.pi, size=2000)
    theta_dot = rng.uniform(-8.0, 8.0, size=2000)
    X = pendulum_state(theta, theta_dot)                      # [N, 3]
    Y = target_torque(theta, theta_dot)[:, None]             # [N, 1]

    net = NumpyMLP(d_in=3, hidden=hidden, d_out=1, seed=seed)
    history = []
    for epoch in range(epochs):
        y = net.forward(X)
        loss = np.mean((y - Y) ** 2)
        grads = net.backward(Y)
        net.step(grads, lr)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            history.append((epoch, float(loss)))
            print(f"[numpy MLP] epoch {epoch:5d}  MSE = {loss:.5f}")
    return net, history, (X, Y)


# ============================================================================
# PART 2 - THE SAME MLP IN PYTORCH (what you actually use in the RL files)
# ============================================================================
# nn.Sequential builds the identical (Linear -> activation -> Linear) stack.
# autograd computes the gradients we wrote by hand above; the optimizer is the
# same gradient-descent step. This is the exact pattern used by the actor and
# critic networks in DDPG / SAC / TRPO.

def build_torch_mlp(d_in=3, hidden=64, d_out=1):
    return nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, d_out),
    )


def train_torch_mlp(epochs=3000, lr=1e-3, hidden=64, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-math.pi, math.pi, size=2000)
    theta_dot = rng.uniform(-8.0, 8.0, size=2000)
    X = torch.tensor(pendulum_state(theta, theta_dot), dtype=torch.float32)
    Y = torch.tensor(target_torque(theta, theta_dot)[:, None], dtype=torch.float32)

    net = build_torch_mlp(3, hidden, 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(X), Y)
        loss.backward()              # autograd does our manual backprop
        opt.step()                   # optimizer does our manual .step
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            history.append((epoch, float(loss)))
            print(f"[torch MLP] epoch {epoch:5d}  MSE = {loss.item():.5f}")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"[torch MLP] total learnable parameters: {n_params}")
    return net, history


# ============================================================================
# VISUALIZATION - did the MLP learn the swing-up surface?
# ============================================================================
def plot_surface(net_numpy, net_torch, out_dir):
    if not HAVE_PLT:
        print("matplotlib not available, skipping plot")
        return
    # A grid over (theta, theta_dot) to render the learned surface vs. truth.
    th = np.linspace(-math.pi, math.pi, 120)
    thd = np.linspace(-8, 8, 120)
    TH, THD = np.meshgrid(th, thd)
    grid = pendulum_state(TH.ravel(), THD.ravel())
    truth = target_torque(TH.ravel(), THD.ravel()).reshape(TH.shape)

    pred_np = net_numpy.forward(grid).reshape(TH.shape)
    panels = [("Target (truth)", truth), ("NumPy MLP", pred_np)]
    if net_torch is not None and HAVE_TORCH:
        with torch.no_grad():
            pred_t = net_torch(torch.tensor(grid, dtype=torch.float32)).numpy()
        panels.append(("PyTorch MLP", pred_t.reshape(TH.shape)))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, Z) in zip(axes, panels):
        im = ax.pcolormesh(TH, THD, Z, shading="auto", cmap="coolwarm")
        ax.set_title(title)
        ax.set_xlabel("theta (rad)")
        ax.set_ylabel("theta_dot")
        fig.colorbar(im, ax=ax)
    fig.suptitle("MLP learning the swing-up torque surface  torque = theta_dot * cos(theta)")
    fig.tight_layout()
    path = Path(out_dir) / "mlp_surface.png"
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


def _fig_to_rgb(fig):
    """Render a matplotlib figure to an [H, W, 3] uint8 array.

    Uses buffer_rgba() (tostring_rgb() was removed in matplotlib 3.10).
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())   # [H, W, 4]
    return buf[..., :3].copy()                   # drop alpha -> RGB


def predict_torque(net, theta, theta_dot):
    """Run the trained MLP forward on one (theta, theta_dot) -> scalar torque."""
    s = pendulum_state(np.array([theta]), np.array([theta_dot]))   # [1, 3]
    if HAVE_TORCH and isinstance(net, nn.Module):
        with torch.no_grad():
            return float(net(torch.tensor(s, dtype=torch.float32)).item())
    return float(net.forward(s).ravel()[0])


def create_swingup_gif(net, out_dir, name="mlp_swingup.gif", gain=2.5, u_max=6.0,
                       steps=320, dt=0.02):
    """Let the LEARNED MLP torque act as a swing-up controller and animate it.

    theta = 0 is hanging straight down, theta = pi is balanced upright. Each
    frame: read state -> MLP outputs a torque -> apply it to the dynamics ->
    integrate. This is exactly the policy loop in the RL files, but the "policy"
    here is the tiny MLP we just trained to imitate the energy-shaping law.
    """
    if not HAVE_PLT or not HAVE_IMAGEIO:
        print("matplotlib/imageio not available, skipping GIF")
        return

    g, l, damping = 9.81, 1.0, 0.1
    theta, theta_dot = 0.15, 0.0          # start near the bottom, almost at rest
    L = 1.0
    frames, traj = [], []
    step_size = max(1, steps // 140)      # keep the GIF light

    for t in range(steps):
        u = float(np.clip(gain * predict_torque(net, theta, theta_dot), -u_max, u_max))
        # pendulum dynamics with the MLP torque as the control input
        theta_ddot = -(g / l) * math.sin(theta) - damping * theta_dot + u
        theta_dot += theta_ddot * dt
        theta += theta_dot * dt
        traj.append((theta, theta_dot, u))

        if t % step_size != 0:
            continue
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        # upright target (faint) and the pole
        ax.plot([0, 0], [0, L], color="green", linestyle="--", alpha=0.4, linewidth=2)
        bob_x, bob_y = L * math.sin(theta), -L * math.cos(theta)
        ax.plot([0, bob_x], [0, bob_y], color="#D55E00", linewidth=4)
        ax.scatter([0], [0], s=40, color="black")                 # pivot
        ax.scatter([bob_x], [bob_y], s=180, color="#0072B2")      # bob
        # torque indicator
        ax.set_title(f"step {t:03d}   MLP torque u = {u:+.2f}")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout()
        frames.append(_fig_to_rgb(fig))
        plt.close(fig)

    path = Path(out_dir) / name
    imageio.mimsave(path, frames, duration=0.04)
    upright = sum(1 for th, _, _ in traj if math.cos(th) < -0.7)
    print(f"saved {path}   (reached near-upright on {upright}/{len(traj)} steps)")


def main():
    parser = argparse.ArgumentParser(description="MLP explained on the inverted pendulum")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--numpy-epochs", type=int, default=6000)
    parser.add_argument("--torch-epochs", type=int, default=3000)
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent))
    args = parser.parse_args()

    print("=" * 70)
    print("PART 1: MLP from scratch in NumPy (manual forward + backprop)")
    print("=" * 70)
    net_np, _, _ = train_numpy_mlp(epochs=args.numpy_epochs, hidden=args.hidden)

    net_t = None
    if HAVE_TORCH:
        print()
        print("=" * 70)
        print("PART 2: Same MLP in PyTorch (autograd does the backprop for you)")
        print("=" * 70)
        net_t, _ = train_torch_mlp(epochs=args.torch_epochs, hidden=args.hidden)
    else:
        print("\n[torch not installed - skipping PyTorch section]")

    plot_surface(net_np, net_t, args.out)

    # Animate the pendulum being driven by the learned MLP torque.
    gif_net = net_t if net_t is not None else net_np
    create_swingup_gif(gif_net, args.out)

    print("\nTAKEAWAYS")
    print("- An MLP is just (Linear -> nonlinearity) stacked; params live in W,b.")
    print("- The nonlinearity is what lets it bend to fit swing-up energy.")
    print("- Training = gradient descent on a loss; backprop = the chain rule.")
    print("- Swap the target for a policy/value and this IS your RL network.")


if __name__ == "__main__":
    main()
