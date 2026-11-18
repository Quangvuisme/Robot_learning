"""Reusable headless wrapper around the official GMT sim2sim (pretrained ckpt)."""
import sys, types, os, copy
from collections import deque
import numpy as np
class _V:
    def __init__(self,*a,**k): self.cam = types.SimpleNamespace(distance=0, lookat=None)
    def render(self): pass
    def close(self): pass
sys.modules['mujoco_viewer'] = types.SimpleNamespace(MujocoViewer=_V)
REPO = os.environ["GMT_REPO"]; os.chdir(REPO); sys.path.insert(0, REPO)
import torch, mujoco
import sim2sim as S
torch.set_num_threads(1)

class Tracker:
    def __init__(self, motion, latency_steps=0, payload_kg=0.0, friction=None, kp_scale=1.0):
        self.env = S.HumanoidEnv("assets/pretrained_checkpoints/pretrained.pt", f"assets/motions/{motion}", "g1", device="cpu")
        e = self.env; self.m, self.d = e.model, e.data
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if payload_kg: self.m.body_mass[torso] += payload_kg
        if friction is not None: self.m.geom_friction[:, 0] = friction
        self.kp = e.stiffness * kp_scale
        self.k = 0; self.latency = deque([np.zeros(23)]*(latency_steps+1), maxlen=latency_steps+1)
        self.pd_target = e.default_dof_pos.copy()
        # cache reference for the whole motion
        L = e._motion_lib._motion_lengths[0].item() if hasattr(e._motion_lib, "_motion_lengths") else 30.0
        self.motion_len = float(L)

    def ref(self, k):
        t = torch.tensor([min(k*self.env.control_dt, self.motion_len-1e-3)])
        rp, rr, _, _, rdof, _ = self.env._motion_lib.calc_motion_frame(torch.zeros(1, dtype=torch.int), t)
        return rp.numpy()[0], rdof.numpy()[0]

    def obs(self):
        e = self.env
        dof_pos = self.d.qpos[-23:].astype(np.float32); dof_vel = self.d.qvel[-23:].astype(np.float32)
        quat = self.d.sensor('orientation').data; ang = self.d.sensor('angular-velocity').data
        mimic = e._get_mimic_obs(self.k)
        rpy = S.quatToEuler(quat); v = dof_vel.copy(); v[[4,5,10,11]] = 0
        prop = np.concatenate([ang*e.ang_vel_scale, rpy[:2], dof_pos-e.default_dof_pos, v*e.dof_vel_scale, e.last_action])
        return np.concatenate([mimic, prop, np.array(e.proprio_history_buf).flatten()]), prop

    def policy(self, obs_batch):
        with torch.no_grad(): return self.env.policy_jit(torch.from_numpy(np.asarray(obs_batch)).float()).numpy()

    def apply(self, action, prop, push=None):
        """one 20 ms control step with the given raw action."""
        e = self.env
        e.last_action = action.copy(); e.proprio_history_buf.append(prop)
        self.latency.append(np.clip(action, -10, 10)*e.action_scale + e.default_dof_pos)
        target = self.latency[0]
        for _ in range(e.sim_decimation):
            if push is not None: self.d.xfrc_applied[self.pelvis, :3] = push
            q = self.d.qpos[-23:]; qd = self.d.qvel[-23:]
            self.d.ctrl[:] = np.clip((target-q)*self.kp - qd*e.damping, -e.torque_limits, e.torque_limits)
            mujoco.mj_step(self.m, self.d)
        self.d.xfrc_applied[self.pelvis, :3] = 0
        self.k += 1

    def snapshot(self):
        e = self.env
        return (self.d.qpos.copy(), self.d.qvel.copy(), self.d.act.copy(), self.d.time, e.last_action.copy(),
                copy.copy(e.proprio_history_buf), copy.copy(self.latency), self.k)
    def restore(self, s):
        e = self.env
        self.d.qpos[:], self.d.qvel[:] = s[0], s[1]; self.d.act[:] = s[2]; self.d.time = s[3]
        e.last_action = s[4].copy(); e.proprio_history_buf = copy.copy(s[5]); self.latency = copy.copy(s[6]); self.k = s[7]
        mujoco.mj_forward(self.m, self.d)

    def cost(self):
        rp, rdof = self.ref(self.k)
        quat = self.d.qpos[3:7]; up = 1 - 2*(quat[1]**2 + quat[2]**2)  # z of body z-axis
        h = self.d.qpos[2]
        return 5*abs(h-rp[2]) + 2*(1-up) + 0.5*np.abs(self.d.qpos[-23:]-rdof).mean() + (50.0 if h < 0.45 else 0.0)
