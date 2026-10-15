# World-model comparison: trainable double-integrator toy

This workspace contains:

- `train_world_model_comparison.py` — real PyTorch optimization and evaluation on the same damped double-integrator task.
- `world_model_live_comparison.html` — live browser control for the four policies after loading training JSON.
- `world_model_results.json` — short sample run (1,000 update steps per method, 12 eval episodes) so you can open the HTML immediately. Regenerate it with the second command below.

## Four trainable methods

1. **JEPA-inspired**: latent next-state prediction loss plus a learned actor.
2. **DeepJEPA-inspired**: recurrent latent refinement, intermediate prediction losses and a learned continue gate.
3. **DreamerV3-inspired**: stochastic latent transition, reward prediction, value/target-value learning and actor regularization.
4. **Dreamer4-inspired**: action-conditioned latent denoising objective, transition prediction and a learned actor.

Important scientific scope note: these are deliberately small *method-inspired agents*, not faithful reproductions of the referenced papers or repositories. For instance, the actor policy in this minimal task is trained with a stabilizing teacher target while the world-model losses train the latent representation; therefore do not interpret the result as a benchmark of official DreamerV3/Dreamer4 vs JEPA/DeepJEPA.

## Run (uses the repo's shared `.venv`)

This lab has no environment of its own. It uses `Robot_learning/.venv`, which already has `torch` and `numpy`
(see the root `requirements.txt`). Do not create another venv here.

```bash
cd ~/quangvd7/Robot_learning
source .venv/bin/activate
cd World_model/world_model_training_lab

# fast smoke test (~10 s on GPU)
python train_world_model_comparison.py --steps 300 --batch 64 --eval-every 150 --eval-episodes 2 --out /tmp/wm_smoke.json

# the sample shipped in this folder (~25 s on GPU)
python train_world_model_comparison.py --steps 1000 --batch 256 --eval-every 500 --eval-episodes 12

# full run
python train_world_model_comparison.py --steps 20000 --batch 256 --eval-every 1000 --eval-episodes 12
```

`--out` defaults to `world_model_results.json` next to the script, whatever directory you run from.
CUDA is used when available; force a device with `--device cpu` or `--device cuda`.
Then open `world_model_live_comparison.html` in a browser and load the JSON with the file picker.
The page runs the exported `policy.encoder` and `policy.actor` weights in plain JavaScript (no server, no library).

## What the comparison does and does not show

- All four methods are scored on the **same 12 evaluation episodes** (`metadata.eval_seed`), at every checkpoint.
- All four actors are trained to clone the **same teacher** (`tanh(1.25*(target-x) - 1.1*v)`). Only the world-model
  loss that also shapes the encoder differs. Expect near-identical control quality (the 1,000-step sample gives
  returns between -223.1 and -223.4). Differences between methods in this task are small; do not read a ranking into them.
- The latent-dynamics, refinement-gate, critic and denoiser heads are trained and logged, but only `encoder` + `actor`
  are used to act, so they affect control only through the shared encoder.

## Fixes made to the original GPT-generated script

| Problem | Effect | Fix |
|---|---|---|
| `reset_done` flattened a `[B,1]` mask with `nonzero().reshape(-1)` | env 0 was reset every time any env finished | flatten the mask first |
| Eval seed was `seed + 100 + method_index + step` | each method was scored on different episodes, which made returns look very different (-144 to -482) | one eval seed for all methods and checkpoints |
| `eval_agent` called `seed_all` on the global RNG | every eval reset the training RNG stream | `isolated_rng` saves and restores RNG state around eval |
| `--out` was relative to the current directory | running from the repo root wrote the JSON there | default is next to the script |
| README told you to create a new venv | duplicated the repo environment | use the shared `.venv` |

## Original projects

- DreamerV3 repository: https://github.com/danijar/dreamerv3 (JAX implementation, its own configs and environment system).
- Dreamer4 repository: https://github.com/nicklashansen/dreamer4 (unofficial PyTorch implementation; its README describes a substantially larger training setup and GPU/data requirements).

Use the original repositories' own recipes and benchmarks when you need faithful original-model results.
