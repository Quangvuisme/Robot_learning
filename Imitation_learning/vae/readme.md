# VAE Policy Studies

This folder provides VAE-based versions of the three imitation-learning study cases.

## Files

- `vae_inverted_pendulum_study.py`: inverted pendulum upright balance.
- `vae_cartpole_study.py`: cart-pole balance around upright.
- `vae_cartpole_swingup_balance_study.py`: cart-pole swing-up + balance.
- `act_cartpole_swingup_balance_study.py`: ACT-style (Transformer + action chunking + temporal ensemble) swing-up + balance.

## Quick Runs

From `Imitation_learning`:

```bash
python3 vae/vae_inverted_pendulum_study.py
python3 vae/vae_cartpole_study.py
python3 vae/vae_cartpole_swingup_balance_study.py --episodes 220 --epochs 1300 --batch-size 16384 --log-interval 30

# ACT-style transformer CVAE (action chunking)
python3 vae/act_cartpole_swingup_balance_study.py --episodes 220 --epochs 1800 --chunk-len 8 --stride 1 --ensemble-decay 0.75
```

## Notes

- These scripts reuse the environment/expert dynamics from `diffusion/` and replace the policy learner with a conditional VAE.
- Outputs are saved to `outputs/` relative to your current working directory.
- For the swing-up case, use `--no-hybrid` to evaluate pure VAE everywhere; default uses hybrid near-upright stabilization for robustness.
- For the ACT-style script, action chunking is controlled by `--chunk-len`, and temporal ensembling by `--stride` + `--ensemble-decay`.

## Output Files (VAE-specific names)

- `vae_inverted_pendulum_study.py`
	- `outputs/vae_training_curves_pendulum.png`
	- `outputs/vae_theta_trajectory.png`
	- `outputs/vae_pendulum_compare.gif`
- `vae_cartpole_study.py`
	- `outputs/vae_training_curves_cartpole.png`
	- `outputs/vae_cartpole_trajectory.png`
	- `outputs/vae_cartpole_compare.gif`
- `vae_cartpole_swingup_balance_study.py`
	- `outputs/vae_training_curves_cartpole_swingup.csv`
	- `outputs/vae_training_curves_cartpole_swingup.png` (if matplotlib is installed)
	- `outputs/vae_cartpole_swingup_trajectory.png`
	- `outputs/vae_cartpole_swingup_compare.gif`
