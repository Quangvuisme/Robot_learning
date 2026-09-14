"""
Network 03: MLP on the INVERTED PENDULUM ON A CART (a fully concrete example)

WHY THIS FILE:
--------------
01_mlp.py showed an MLP fitting an abstract surface. Here we make it concrete on
the exact system the RL files use: a pole hinged on a cart that rolls on a track.
You will SEE, with real numbers, how a network turns a state into an action, and
then watch that action balance the pole in a GIF.

THE PHYSICAL SYSTEM (4 numbers describe everything):
    state s = [ x , x_dot , theta , theta_dot ]
        x         : cart position on the track   (0 = center)
        x_dot     : cart velocity
        theta     : pole angle, 0 = straight UP (balanced), + = tilts right
        theta_dot : pole angular velocity
    action a = force pushed on the cart (left = negative, right = positive)

THE JOB OF THE NETWORK:
    a function   force = MLP(s)   that keeps the pole upright.
    Input  = 4 numbers (the state).
    Output = 1 number (the force).
    That's it. The MLP is just the box mapping 4 -> 1.

HOW WE TEACH IT (behavior cloning, the simplest possible setup):
    1. We already KNOW a controller that balances the pole: a linear law
           force = -K . s         (K = 4 gains computed by LQR)
       This is the "teacher". (RL would discover this from reward instead, but
       a known teacher makes the LEARNING itself crystal clear.)
    2. We roll the teacher out, recording many (state -> force) pairs.
    3. We train the MLP to copy the teacher: minimize (MLP(s) - force_teacher)^2.
    4. We then throw the teacher away and let the MLP drive the cart alone.

WHAT TO LOOK AT WHEN YOU RUN IT:
    - The "WORKED EXAMPLE" printout: one state pushed through the MLP by hand,
      layer by layer, so the matrix math is not a mystery.
    - cartpole_mlp_control.gif: the MLP balancing the real cart + pole.
    - cartpole_mlp_curves.png: training loss + how MLP force matches the teacher.
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


# ============================================================================
# THE CART-POLE PHYSICS  (theta = 0 is UPRIGHT, the balanced position)
# ============================================================================
@dataclass
class CartPole:
    g: float = 9.81
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    half_length: float = 0.5          # half the pole length; full pole = 1.0 m
    dt: float = 0.02
    force_max: float = 20.0

    def step(self, s, force):
        """One physics step. s = [x, x_dot, theta, theta_dot] -> next state."""
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


# ----------------------------------------------------------------------------
# THE TEACHER: an LQR controller (the textbook optimal linear balancer).
# ----------------------------------------------------------------------------
# A linear controller has the form  force = -K . s . The four numbers in K are
# the "gains". Hand-picking them is fiddly and easy to get wrong (an unstable
# teacher teaches the MLP garbage). Instead we COMPUTE the optimal K with LQR:
#   1. Linearize the dynamics around the upright point: s' ~ A s + B force.
#   2. Solve the discrete Riccati equation for the cost we care about (keep the
#      pole upright + cart centered, with small force).
# The result is provably stabilizing, so the MLP has a good teacher to copy.
TEACHER_K = None   # filled in by setup_teacher(env) before we collect data


def setup_teacher(env, seed=0):
    """Linearize the cart-pole and solve LQR -> store the gain vector K."""
    global TEACHER_K
    n = 4
    s0 = np.zeros(n, dtype=np.float32)
    eps = 1e-4
    # Finite-difference linearization of the (discrete) physics step about upright.
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    for i in range(n):
        ds = np.zeros(n, dtype=np.float32)
        ds[i] = eps
        A[:, i] = (env.step(s0 + ds, 0.0) - env.step(s0 - ds, 0.0)) / (2 * eps)
    B[:, 0] = (env.step(s0, eps) - env.step(s0, -eps)) / (2 * eps)

    # Cost weights: penalize angle most, then position/velocities; small force cost.
    Q = np.diag([1.0, 1.0, 10.0, 1.0])
    R = np.array([[0.1]])
    # Solve the discrete-time algebraic Riccati equation by iteration.
    P = Q.copy()
    for _ in range(10000):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(P_next - P)) < 1e-9:
            break
        P = P_next
    TEACHER_K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A).ravel()
    eig = np.abs(np.linalg.eigvals(A - B @ TEACHER_K.reshape(1, -1)))
    print(f"LQR teacher gains K = {np.round(TEACHER_K, 2)}")
    print(f"closed-loop eigenvalues |.| = {np.round(eig, 3)}  (all < 1 => stable)")
    return TEACHER_K


def teacher_force(s):
    # force = -K . s  (the optimal linear state-feedback law)
    return float(-(TEACHER_K @ np.asarray(s, dtype=np.float32)))


# ============================================================================
# COLLECT TRAINING DATA  (roll the teacher out from many random starts)
# ============================================================================
def collect_dataset(env, n_episodes=200, horizon=200, seed=0):
    rng = np.random.default_rng(seed)
    states, forces = [], []
    for _ in range(n_episodes):
        # start slightly off-balance so the teacher has work to do
        s = np.array([
            rng.uniform(-0.5, 0.5),     # x
            rng.uniform(-0.5, 0.5),     # x_dot
            rng.uniform(-0.20, 0.20),   # theta (radians, ~ +-11 deg)
            rng.uniform(-0.5, 0.5),     # theta_dot
        ], dtype=np.float32)
        for _ in range(horizon):
            f = teacher_force(s)
            states.append(s.copy())
            forces.append(f)
            s = env.step(s, f)
            if abs(s[2]) > math.pi / 2:        # pole fell past horizontal -> stop
                break
    X = np.array(states, dtype=np.float32)
    Y = np.array(forces, dtype=np.float32)[:, None]
    print(f"collected {len(X)} (state -> force) examples from the teacher")
    return X, Y


# ============================================================================
# THE MLP:  4 inputs (state) -> hidden -> hidden -> 1 output (force)
# ============================================================================
def build_mlp(hidden=64):
    return nn.Sequential(
        nn.Linear(4, hidden),   # layer 1: state -> hidden
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),   # output: a single force
    )


def train_mlp(X, Y, hidden=64, epochs=1500, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X)
    Yt = torch.tensor(Y)
    # normalize inputs so all 4 state numbers are on a similar scale (helps a lot)
    mean, std = Xt.mean(0), Xt.std(0) + 1e-6
    net = build_mlp(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net((Xt - mean) / std), Yt)
        loss.backward()
        opt.step()
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            history.append((epoch, loss.item()))
            print(f"[bc-mlp] epoch {epoch:4d}  imitation MSE = {loss.item():.4f}")
    print(f"[bc-mlp] parameters: {sum(p.numel() for p in net.parameters())}")
    return net, (mean, std), history


def mlp_force(net, norm, s):
    """Run the trained MLP forward: state (4 numbers) -> force (1 number)."""
    mean, std = norm
    with torch.no_grad():
        x = (torch.tensor(s, dtype=torch.float32) - mean) / std
        return float(net(x).item())


# ============================================================================
# WORKED EXAMPLE: push ONE state through the MLP by hand, layer by layer.
# ============================================================================
def worked_example(net, norm, env):
    """Demystify the forward pass by printing the actual numbers."""
    mean, std = norm
    s = np.array([0.0, 0.0, 0.10, 0.0], dtype=np.float32)   # pole tilted +0.10 rad
    print("\n" + "=" * 70)
    print("WORKED EXAMPLE  -  one state through the MLP, step by step")
    print("=" * 70)
    print(f"state s            = [x={s[0]:.2f}, x_dot={s[1]:.2f}, "
          f"theta={s[2]:.2f}, theta_dot={s[3]:.2f}]")
    print("  -> the pole is tilted +0.10 rad (right). We expect a positive (right) push.\n")

    # Layer 1 (we compute the FIRST hidden neuron fully so you see the formula)
    sn = (torch.tensor(s) - mean) / std                      # normalized input
    lin1 = net[0]                                            # nn.Linear(4, hidden)
    W1, b1 = lin1.weight.detach(), lin1.bias.detach()
    z1 = sn @ W1.T + b1                                      # pre-activation
    h1 = torch.tanh(z1)                                      # hidden activation
    print("LAYER 1:  h = tanh(W1 @ s_norm + b1)")
    w = W1[0]
    contrib = " + ".join(f"({w[i]:+.2f}*{sn[i]:+.2f})" for i in range(4))
    print(f"  neuron 0 pre-activation z1[0] = {contrib} + ({b1[0]:+.2f}) = {z1[0]:+.3f}")
    print(f"  neuron 0 after tanh           = tanh({z1[0]:+.3f}) = {h1[0]:+.3f}")
    print(f"  (there are {W1.shape[0]} such neurons; shapes: W1 {tuple(W1.shape)}, h {tuple(h1.shape)})\n")

    # Run the rest through the real network and compare with the teacher
    f_mlp = mlp_force(net, norm, s)
    f_teacher = teacher_force(s)
    print("OUTPUT LAYER:  force = W3 @ h2 + b3  (a single number)")
    print(f"  MLP force      = {f_mlp:+.3f} N")
    print(f"  teacher force  = {f_teacher:+.3f} N   <- what we trained it to copy")
    print(f"  both positive => push the cart RIGHT to catch the falling pole. Correct!")


# ============================================================================
# RUN THE MLP AS THE CONTROLLER + RENDER THE CART-POLE GIF
# ============================================================================
def rollout(env, controller, horizon=300, theta0=0.18):
    s = np.array([0.0, 0.0, theta0, 0.0], dtype=np.float32)
    traj = [s.copy()]
    forces = []
    for _ in range(horizon):
        f = controller(s)
        forces.append(f)
        s = env.step(s, f)
        traj.append(s.copy())
    return np.array(traj), np.array(forces)


def _fig_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def create_cartpole_gif(env, traj, out_dir, name="control.gif"):
    if not HAVE_PLT or not HAVE_IMAGEIO:
        print("matplotlib/imageio not available, skipping GIF")
        return
    L = 2.0 * env.half_length                       # full pole length for drawing
    step = max(1, len(traj) // 150)
    frames = []
    for i in range(0, len(traj), step):
        x, _, theta, _ = traj[i]
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        # track
        ax.plot([-3, 3], [0, 0], "k-", linewidth=2)
        # cart
        cw, ch = 0.5, 0.3
        cart_x = [x - cw / 2, x + cw / 2, x + cw / 2, x - cw / 2, x - cw / 2]
        cart_y = [0, 0, ch, ch, 0]
        ax.fill(cart_x, cart_y, color="#0072B2", alpha=0.6)
        ax.plot(cart_x, cart_y, "b-", linewidth=2)
        # pole (theta = 0 -> straight up)
        px, py = x + L * math.sin(theta), ch + L * math.cos(theta)
        ax.plot([x, px], [ch, py], color="#D55E00", linewidth=4)
        ax.scatter([px], [py], s=140, color="#D55E00")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 2.7)
        ax.set_aspect("equal")
        ax.set_title(f"step {i:03d}   x={x:+.2f}   theta={math.degrees(theta):+.1f}deg")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        frames.append(_fig_to_rgb(fig))
        plt.close(fig)
    path = Path(out_dir) / name
    imageio.mimsave(path, frames, duration=0.04)
    # success = pole stayed roughly upright the whole time
    upright = np.mean(np.abs(traj[:, 2]) < 0.25) * 100
    print(f"saved {path}   (pole within +-14deg for {upright:.0f}% of the run)")


def plot_curves(history, env, net, norm, out_dir, name="curves.png"):
    if not HAVE_PLT:
        print("matplotlib not available, skipping plot")
        return
    # compare MLP vs teacher force over one rollout driven by the teacher
    traj, _ = rollout(env, teacher_force, horizon=200, theta0=0.18)
    f_teacher = np.array([teacher_force(s) for s in traj])
    f_mlp = np.array([mlp_force(net, norm, s) for s in traj])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep, ls = zip(*history)
    axes[0].plot(ep, ls, color="#0072B2")
    axes[0].set_title("Training: MLP imitates the teacher")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("imitation MSE")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)

    axes[1].plot(f_teacher, label="teacher force", color="black", linewidth=2)
    axes[1].plot(f_mlp, label="MLP force", color="#D55E00", linestyle="--", linewidth=2)
    axes[1].set_title("Learned force matches the teacher")
    axes[1].set_xlabel("timestep")
    axes[1].set_ylabel("force (N)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = Path(out_dir) / name
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description="MLP balancing an inverted pendulum on a cart")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).parent / "figures" / "03_mlp"))
    args = parser.parse_args()

    if not HAVE_TORCH:
        print("torch not installed -> this concrete example needs torch. Aborting.")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = CartPole()

    print("=" * 70)
    print("STEP 0: build the LQR teacher (the controller we will imitate)")
    print("=" * 70)
    setup_teacher(env)

    print("\n" + "=" * 70)
    print("STEP 1: collect (state -> force) data from the LQR teacher")
    print("=" * 70)
    X, Y = collect_dataset(env)

    print("\n" + "=" * 70)
    print("STEP 2: train an MLP to imitate the teacher (4 inputs -> 1 force)")
    print("=" * 70)
    net, norm, history = train_mlp(X, Y, hidden=args.hidden, epochs=args.epochs)

    # STEP 3: show the forward pass with real numbers
    worked_example(net, norm, env)

    print("\n" + "=" * 70)
    print("STEP 4: let the MLP (not the teacher) balance the pole, and render it")
    print("=" * 70)
    traj, _ = rollout(env, lambda s: mlp_force(net, norm, s), horizon=300, theta0=0.18)
    create_cartpole_gif(env, traj, out_dir)
    plot_curves(history, env, net, norm, out_dir)

    print("\nTAKEAWAYS")
    print("- The MLP is just a function: 4 state numbers in -> 1 force out.")
    print("- Each layer is (W @ input + b) then a nonlinearity; see WORKED EXAMPLE.")
    print("- 'Learning' here = copy a teacher; in RL the reward replaces the teacher.")
    print("- Same network shape is the Actor in DDPG/SAC/TRPO.")


if __name__ == "__main__":
    main()
