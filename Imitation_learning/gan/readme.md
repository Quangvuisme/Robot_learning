# GAN Examples (Imitation Learning)

This folder mirrors the diffusion/VAE studies but uses a conditional GAN to learn $p(a \mid s)$.

Current swing-up implementation follows a GAIL-style loop:
- Step 1: sample student trajectories from current policy
- Step 2: train discriminator to classify expert vs student state-action pairs
- Step 3: update student policy to fool discriminator

For stability/speed, we still keep a short BC warm start before GAIL updates.

## Files
- gan_inverted_pendulum_study.py: inverted pendulum balance (torque control)
- gan_cartpole_study.py: cart-pole balance (force control)
- gan_cartpole_swingup_balance_study.py: cart-pole swing-up + balance (force control)

## Outputs
- outputs/gan_training_curves_pendulum.png
- outputs/gan_theta_trajectory.png
- outputs/gan_pendulum_compare.gif
- outputs/gan_training_curves_cartpole.png
- outputs/gan_cartpole_trajectory.png
- outputs/gan_cartpole_compare.gif
- outputs/gan_training_curves_cartpole_swingup.csv
- outputs/gan_training_curves_cartpole_swingup.png
- outputs/gan_cartpole_swingup_trajectory.png
- outputs/gan_cartpole_swingup_compare.gif

## Run
```bash
python3 gan_inverted_pendulum_study.py
python3 gan_cartpole_study.py
python3 gan_cartpole_swingup_balance_study.py
```

Recommended swing-up command (GAIL-style):
```bash
python3 gan_cartpole_swingup_balance_study.py \
	--skip-expert-check --max-samples 40000 --epochs 600 \
	--rollout-episodes 12 --rollout-horizon 180 \
	--disc-updates 3 --gen-updates 2 \
	--bc-pretrain-steps 500 --bc-weight 0.2 --skip-bonus
```

## Notes
- Generator/Student: $a = G(s, z)$, Discriminator: $D(s, a)$
- Swing-up script uses feature map from diffusion (`state_features`) and the same expert/environment.
- The discriminator loss is BCE expert-vs-student; policy is optimized to fool the discriminator.
