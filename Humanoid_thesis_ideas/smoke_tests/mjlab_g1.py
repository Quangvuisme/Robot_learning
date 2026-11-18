import time, torch, warnings, io, sys
from contextlib import redirect_stdout
warnings.simplefilter("ignore")
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
import mjlab.tasks.velocity.config.g1.env_cfgs as V
import mjlab.tasks.tracking.config.g1.env_cfgs as T
print([n for n in dir(V) if n.startswith("unitree")], [n for n in dir(T) if n.startswith("unitree")])
cfg = V.unitree_g1_flat_env_cfg(); cfg.scene.num_envs = 4
with redirect_stdout(io.StringIO()): env = ManagerBasedRlEnv(cfg, device="cpu")
obs, _ = env.reset()
act_dim = env.action_manager.total_action_dim
print("obs groups", {k: tuple(v.shape) for k, v in obs.items()}, "act", act_dim)
t=time.time()
for i in range(50):
    obs, r, term, trunc, info = env.step(torch.zeros(4, act_dim))
dt=time.time()-t
print(f"50 steps x4 envs in {dt:.1f}s ; reward mean {r.mean():.3f}; terminated {int(term.sum())}")
env.close()
