# Basic Teaching Examples

This folder contains minimal examples for quick learning and comparison.

## 1) Basic Supervised Learning

File: `supervised_learning_basic.py`

- Task: fit a line from synthetic data `y = 2x + 1 + noise`.
- Model: single linear layer.
- Loss: MSE.

Run:

```bash
python3 basic_examples/supervised_learning_basic.py
```

Output (if matplotlib is installed):

- `outputs/supervised_basic_fit.png`

## 2) Simple DAgger Example (compare with BC)

File: `dagger_simple.py`

- Environment: toy 1D control (`x_{t+1} = x_t + dt * u_t`).
- Expert: piecewise controller.
- BC: trained only on narrow state distribution near zero.
- DAgger: iteratively collects learner-visited states and queries expert labels.

Run:

```bash
python3 basic_examples/dagger_simple.py
```

Output (if matplotlib is installed):

- `outputs/dagger_vs_bc_toy.png`

You should typically see DAgger improve success over plain BC due to reduced covariate shift.

## 3) Double-Integrator DAgger (with GIF)

File: `dagger_double_integrator.py`

- System: double integrator, state `s = [x, v]`.
- Dynamics:
	- `x_{t+1} = x_t + dt * v_t`
	- `v_{t+1} = v_t + dt * u_t`
- Comparison: Behavior Cloning (BC) vs DAgger.
- Visualization: metric plot + animated GIF rollout.

Run:

```bash
python3 basic_examples/dagger_double_integrator.py
```

Outputs (if matplotlib/imageio installed):

- `outputs/double_integrator_bc_vs_dagger.png`
- `outputs/double_integrator_bc_vs_dagger.gif`

## 4) Privileged Teacher for Swing-up Pendulum

File: `privileged_teacher_swingup_pendulum.py`

- Stage 1 (teacher training):
	- Teacher model sees privileged full state `[theta, theta_dot]`.
	- Teacher is trained from an oracle privileged controller.
- Stage 2 (distillation):
	- Student is trained on labels produced by the trained teacher model.
	- Student only sees partial observations
	`[sin(theta_t), cos(theta_t), sin(theta_{t-1}), cos(theta_{t-1}), dtheta_est, u_{t-1}]`.
- Training objective: MSE imitation loss.

Run:

```bash
python3 basic_examples/privileged_teacher_swingup_pendulum.py
```

Outputs (if matplotlib installed):

- `outputs/privileged_teacher_swingup_curves.png`
- `outputs/privileged_teacher_swingup_rollout.png`
