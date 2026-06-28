
# Diffusion policy studies (pendulum + cart-pole)

This folder contains 3 main scripts that learn a policy with conditional diffusion from expert demonstrations.

## 1) Inverted pendulum (upright balance)

File: `diffusion_inverted_pendulum_study.py`

Goal: keep the pendulum near upright (theta ~ 0).

Quick run (sanity):

```bash
python3 diffusion_inverted_pendulum_study.py --epochs 800 --episodes 300 --seed 123
```

More thorough run:

```bash
python3 diffusion_inverted_pendulum_study.py --epochs 2000 --episodes 600 --seed 123
```

## 2) Cart-pole (balance-only quanh upright)

File: `diffusion_cartpole_study.py`

Goal: start near upright and keep balancing. This is easier than swing-up because it does not require injecting energy to lift the pole.

Quick run:

```bash
python3 diffusion_cartpole_study.py --epochs 1200 --episodes 500 --seed 123
```

More thorough run:

```bash
python3 diffusion_cartpole_study.py --epochs 2500 --episodes 900 --seed 123
```

## 3) Cart-pole (swing-up + balance)

File: `diffusion_cartpole_swingup_balance_study.py`

Goal: start near the stable downward equilibrium (theta ~ pi), swing up to upright, then balance.

Fast & effective run (recommended):

```bash
python3 diffusion_cartpole_swingup_balance_study.py \
	--episodes 220 --horizon 520 --epochs 5000 \
	--batch-size 4096 --lr 3e-4 --log-interval 30 \
	--eval-episodes 6 --sampling-steps 8 \
	--diffusion-gain 1.4 \
	--skip-bonus --seed 123
```

Important flags:

- `--sampling-steps`: number of reverse diffusion steps when sampling actions. Smaller is faster (e.g., 6–8) but may reduce quality.
- `--diffusion-gain`: scales the diffusion output force; useful if the diffusion policy is underpowered and cannot swing up.
- `--skip-bonus`: skips trajectory plotting + GIF (this part is very slow).
- `--skip-plots`: if enabled, do not save PNG curves; by default plots are saved (if matplotlib is installed).

## Outputs / Plots

All scripts write outputs into the `outputs/` directory (relative to the working directory where you run the command).

For swing-up + balance, after training you will get:

- `outputs/training_curves_cartpole_swingup.csv` (always)
- `outputs/training_curves_cartpole_swingup.png` (only if `matplotlib` is installed)

If you do not see the `.png` file:

```bash
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

If import fails, install it:

```bash
pip install matplotlib
```

