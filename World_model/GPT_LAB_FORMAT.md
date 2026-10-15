# Prompt template: generate a new world-model lab in the same format

Paste everything below the line into GPT, then replace the `<<...>>` fields.
It describes the exact format of `world_model_training_lab/` (the reference lab) plus the mistakes the first
GPT-generated version made, so the next lab runs in this repo without fixes.

---

## Task

Create a self-contained lab named `<<topic>>_lab` that trains and compares `<<N>>` methods on `<<task>>`.
Methods: `<<method 1>>`, `<<method 2>>`, ...
Plant / task: `<<state, action, dynamics, reward, episode end>>`.

## Environment (hard constraints)

- Run inside the repo's existing venv: `~/quangvd7/Robot_learning/.venv` (Python 3.10, torch with CUDA, numpy).
- Use only `torch`, `numpy` and the standard library. Do not add dependencies, create a venv, or tell me to `pip install`.
- The HTML must work offline from a `file://` URL: no CDN, no web fonts, no server, no external JS.
- English only in code, comments and docs.

## Deliverables: exactly these 4 files in one folder

```
<<topic>>_lab/
  train_<<topic>>_comparison.py     # training + evaluation + JSON export
  <<topic>>_live_comparison.html    # single-file dashboard (inline CSS and JS)
  <<topic>>_results.json            # small sample run so the HTML opens immediately
  README_train_<<topic>>.md         # how to run, what it shows, scope note
```

Give each file as a complete code block with its path. No placeholders, no "rest unchanged".

## train_*.py requirements

1. CLI (argparse): `--steps --batch --seed --eval-every --eval-episodes --out --device --log-every`.
   `--device auto` picks CUDA when available. `--out` defaults to a path next to the script
   (`Path(__file__).resolve().with_name(...)`), not the current directory.
2. One `Agent` class per method with the same interface: `name`, `act(obs) -> action in [-1, 1]`,
   `train_step(batch) -> dict of float metrics` (must include `loss`, `wm_loss`, `policy_loss`, `compute`).
   Shared pieces (plant, MLP, base agent) are defined once.
3. Plant is a batched torch class with `reset()`, `step(action) -> obs, reward, done`, `reset_done(done)`.
   Masks and indices are 1-D: flatten `[B,1]` tensors with `.reshape(-1)` before `nonzero()`.
4. Evaluation:
   - every method and every checkpoint is scored on the **same episodes** (one shared eval seed, stored in metadata);
   - evaluation must not change the training RNG: wrap it in a context manager that saves and restores
     `random`, `numpy`, `torch` and CUDA RNG state;
   - report `return_mean`, `return_std`, `mae_mean`, `success_rate`, and the first episode's trajectory.
5. The method must actually do the thing its name says. If the policy is cloned from a teacher shared by all
   methods, the comparison cannot rank them. Either train each policy through its own mechanism
   (for example from imagined rollouts of its own learned model), or say plainly in the README that the
   result is not a ranking.
6. Flush the JSON after every evaluation so a long run can be inspected while it trains.
7. Export weights so the browser can run the policy: every `nn.Linear` of `policy.encoder` and `policy.actor`
   as `{"weight": [[...]], "bias": [...]}`, plus `activation` and the observation names.

### JSON schema

```json
{
  "metadata": {"environment": "...", "seed": 42, "steps": 1000, "device": "cuda",
               "eval_seed": 1042, "eval_episodes": 12, "torch": "2.x", "note": "scope note"},
  "methods": {
    "<method name>": {
      "evaluations": [{"step": 500, "return_mean": 0.0, "return_std": 0.0, "mae_mean": 0.0,
                       "success_rate": 1.0, "trajectory": {"x": [], "v": [], "u": [], "target": 0.0}}],
      "training_log": [{"method": "...", "step": 250, "elapsed_sec": 1.2, "loss": 0.0, "wm_loss": 0.0,
                        "policy_loss": 0.0, "compute": 1.0}],
      "policy": {"encoder": [{"weight": [], "bias": []}], "actor": [{"weight": [], "bias": []}],
                 "activation": "silu", "observation": ["x", "v", "target_minus_x", "minus_v"]}
    }
  }
}
```

## HTML requirements

- One file. Loads the JSON through `<input type="file">`; shows an empty state until then.
- Runs the exported encoder and actor in JavaScript and steps the **same dynamics as the Python plant**
  (same equations, same action scaling and clamp, same observation order).
- One live panel per method, shared controls (target, initial state, limits, dt), start/pause/step/reset.
- Charts drawn on `<canvas>`: state tracking, training loss, evaluation return. A summary table per method.
- Responsive down to 400 px wide, no horizontal page scroll, visible keyboard focus.

## README requirements (same sections, same order)

1. Title and a list of the files.
2. The methods, one line each.
3. A scope note: these are small method-inspired agents, not reproductions of the papers.
4. Run commands that start with `cd ~/quangvd7/Robot_learning && source .venv/bin/activate`: smoke test, sample run, full run.
5. What the comparison does and does not show (state clearly if methods share a teacher).
6. Links to the original projects.

## Self-check before you answer

Run these yourself and paste the real output. Do not claim a check passed unless you ran it.

1. `python -m py_compile train_*.py` passes.
2. A 300-step smoke run finishes and writes valid JSON.
3. `reset_done` with only env 4 done changes only env 4.
4. Calling the evaluation function leaves `torch.rand` output unchanged for the same prior seed.
5. All methods report scores on identical episodes (same `eval_seed`).
6. The JS policy matches `torch` on 50 random observations to better than 1e-9
   (extract `activate()` from the HTML, run it with node, compare).
7. The HTML `<script>` parses (`new Function(src)` in node).

## Mistakes to avoid (all happened in the first lab)

- `done.nonzero().reshape(-1)` on a `[B,1]` mask resets env 0 whenever any env finishes.
- Eval seed depending on the method index or step: each method got different episodes and the table looked
  like a real ranking when it was noise.
- Calling the global `seed_all` inside evaluation: it silently resets the training RNG stream.
- Putting a `#` comment in the middle of a `;`-joined one-liner: it comments out the rest of the statements.
- Output path relative to the current directory; telling the user to create a new venv.
