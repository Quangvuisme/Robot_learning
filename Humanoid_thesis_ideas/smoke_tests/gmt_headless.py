"""Headless smoke test of the official GMT sim2sim (pretrained checkpoint) on CPU.
Stubs mujoco_viewer, runs N seconds per motion, reports fall + tracking error."""
import sys, types, os, json, time
import numpy as np
repo = sys.argv[1]; secs = float(sys.argv[2]); motions = sys.argv[3:]
class _V:
    def __init__(self,*a,**k): self.cam = types.SimpleNamespace(distance=0, lookat=None)
    def render(self): pass
    def close(self): pass
sys.modules['mujoco_viewer'] = types.SimpleNamespace(MujocoViewer=_V)
os.chdir(repo); sys.path.insert(0, repo)
import torch, mujoco
import sim2sim as S
out = {}
for m in motions:
    env = S.HumanoidEnv("assets/pretrained_checkpoints/pretrained.pt", f"assets/motions/{m}", "g1", device="cpu")
    env.sim_duration = secs
    n = int(secs/env.sim_dt); errs=[]; hz=[]; fell=None; t0=time.time()
    pd_target = env.default_dof_pos.copy()
    for i in range(n):
        dof_pos, dof_vel, quat, ang_vel = env.extract_data()
        if i % env.sim_decimation == 0:
            k = i // env.sim_decimation
            mimic = env._get_mimic_obs(k)
            rpy = S.quatToEuler(quat); v = dof_vel.copy(); v[[4,5,10,11]] = 0
            prop = np.concatenate([ang_vel*env.ang_vel_scale, rpy[:2], (dof_pos-env.default_dof_pos), v*env.dof_vel_scale, env.last_action])
            obs = np.concatenate([mimic, prop, np.array(env.proprio_history_buf).flatten()])
            with torch.no_grad(): a = env.policy_jit(torch.from_numpy(obs).float()[None]).numpy().squeeze()
            env.last_action = a.copy(); pd_target = np.clip(a,-10,10)*env.action_scale + env.default_dof_pos
            env.proprio_history_buf.append(prop)
            # reference at current time: first of tar_obs_steps is +1 step; use motion lib directly at t
            t = torch.tensor([k*env.control_dt]); 
            rp, rr, _, _, rdof, _ = env._motion_lib.calc_motion_frame(torch.zeros(1,dtype=torch.int), t)
            errs.append([float(np.abs(dof_pos - rdof.numpy().squeeze()).mean()), float(abs(env.data.qpos[2]-rp[0,2].item()))])
            hz.append(env.data.qpos[2])
            if env.data.qpos[2] < 0.35 and fell is None: fell = k*env.control_dt
        tau = np.clip((pd_target-dof_pos)*env.stiffness - dof_vel*env.damping, -env.torque_limits, env.torque_limits)
        env.data.ctrl = tau; mujoco.mj_step(env.model, env.data)
    e = np.array(errs)
    out[m] = dict(seconds=secs, fell_at=fell, mean_joint_err_rad=float(e[:,0].mean()), mean_root_h_err_m=float(e[:,1].mean()),
                  min_root_h=float(np.min(hz)), wall_s=round(time.time()-t0,1))
    print(m, out[m], flush=True)
json.dump(out, open(os.environ.get('OUT','/dev/null'),'w'), indent=1)
