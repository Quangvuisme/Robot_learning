import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def make_dataset(n: int, noise_std: float, device: str):
    x = torch.linspace(-2.0, 2.0, n, device=device).unsqueeze(-1)
    y = 2.0 * x + 1.0 + noise_std * torch.randn_like(x)
    return x, y


def train_linear_regression(x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float):
    model = nn.Linear(1, 1).to(x.device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    loss_history = []
    for step in range(1, epochs + 1):
        pred = model(x)
        loss = F.mse_loss(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        if step % max(1, epochs // 5) == 0:
            print(f"Step {step:4d} | MSE = {loss.item():.6f}")

    return model, loss_history


def maybe_plot(output_dir: Path, x: torch.Tensor, y: torch.Tensor, model: nn.Module, loss_history: list[float]):
    if not HAS_MATPLOTLIB:
        print("matplotlib not found -> skip plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        y_hat = model(x).cpu()

    x_cpu = x.cpu()
    y_cpu = y.cpu()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].scatter(x_cpu.numpy(), y_cpu.numpy(), s=12, alpha=0.6, label="Data")
    axes[0].plot(x_cpu.numpy(), y_hat.numpy(), color="#D55E00", linewidth=2.0, label="Linear fit")
    axes[0].set_title("Supervised Learning: y = 2x + 1 + noise")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(loss_history, color="#0072B2", linewidth=2.0)
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("MSE")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_file = output_dir / "supervised_basic_fit.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


def parse_args():
    p = argparse.ArgumentParser(description="Basic supervised learning example (linear regression).")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--noise-std", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on: {device}")
    print("Example: simplest supervised learning (linear regression)")

    x, y = make_dataset(args.n, args.noise_std, device)
    model, loss_history = train_linear_regression(x, y, epochs=args.epochs, lr=args.lr)

    with torch.no_grad():
        w = model.weight.item()
        b = model.bias.item()
        mse = F.mse_loss(model(x), y).item()

    print(f"Learned line: y = {w:.4f}x + {b:.4f}")
    print(f"Final MSE: {mse:.6f}")

    output_dir = Path("outputs")
    maybe_plot(output_dir, x, y, model, loss_history)


if __name__ == "__main__":
    main()
