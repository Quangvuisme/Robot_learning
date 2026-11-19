"""Exact MJX gradients on the real Unitree G1 (Menagerie scene_mjx): explosion statistics and bias vs a finite-difference reference.
theta = constant 29-D joint-target offset. Each sample starts from the 'knees_bent' key with a random downward/lateral base velocity
(induces foot impacts). J = -sum_t [(z_pelvis - z0)^2 + 0.5*tilt^2] over H control steps (5 substeps each)."""
import os, sys, json, time
import jax, jax.numpy as jnp, numpy as np, mujoco
from mujoco import mjx
import mujoco.mjx._src.solver as _S
_ITERS = {"n": 5}
def _wl(cond, body, init):   # fixed-iteration, reverse-differentiable replacement for the solver's while_loop
    return _S._while_loop_scan(cond, body, init, _ITERS["n"])
import types as _t
_S.jax = _t.SimpleNamespace(**{k: getattr(jax, k) for k in dir(jax) if not k.startswith("__")})
_S.jax.lax = _t.SimpleNamespace(**{k: getattr(jax.lax, k) for k in dir(jax.lax) if not k.startswith("__")})
_S.jax.lax.while_loop = _wl
jax.config.update("jax_enable_x64", False)
XML = sys.argv[1]; OUT = sys.argv[2]; H = int(os.environ.get("H", 10)); SUB = 5
def build(timeconst):
    m = mujoco.MjModel.from_xml_path(XML)
    m.geom_solref[:, 0] = timeconst; m.pair_solref[:, 0] = timeconst
    return m
def make(m):
    mx = mjx.put_model(m); d0 = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d0, 1); mujoco.mj_forward(m, d0)
    dx0 = mjx.put_data(m, d0); ctrl0 = jnp.array(d0.ctrl); z0 = float(d0.qpos[2])
    def J(theta, v0):
        d = dx0.replace(qvel=dx0.qvel.at[:3].set(v0))
        def ctl(d, _):
            d = d.replace(ctrl=ctrl0 + theta)
            d = jax.lax.fori_loop(0, SUB, lambda i, dd: mjx.step(mx, dd), d)
            q = d.qpos[3:7]; tilt2 = q[1]**2 + q[2]**2
            return d, (d.qpos[2] - z0)**2 + 0.5*tilt2
        _, c = jax.lax.scan(ctl, d, None, length=H)
        return -c.sum()
    return J
res = {}
rng = np.random.default_rng(0)
for tc in [0.02, 0.005]:          # default-ish vs stiffer contact
    m = build(tc); _ITERS['n'] = int(m.opt.iterations); J = make(m)
    Jb = jax.jit(jax.vmap(J, in_axes=(None, 0)))
    gb = jax.jit(jax.vmap(jax.grad(J), in_axes=(None, 0)))
    theta = jnp.zeros(m.nu)
    def v0s(n, seed):
        r = np.random.default_rng(seed); return jnp.array(np.c_[r.normal(0, .3, n), r.normal(0, .3, n), -np.abs(r.normal(.6, .3, n))], jnp.float32)
    t0 = time.time(); Jb(theta, v0s(2, 0)).block_until_ready(); t_c = time.time() - t0
    # finite-difference reference on a large common sample
    VR = v0s(int(os.environ.get("BREF", 256)), 123); h = 2e-3; ref = np.zeros(m.nu)
    t0 = time.time()
    for i in range(m.nu):
        e = jnp.zeros(m.nu).at[i].set(h)
        ref[i] = float((Jb(theta + e, VR).mean() - Jb(theta - e, VR).mean()) / (2*h))
    t_fd = time.time() - t0
    VR2 = v0s(int(os.environ.get("BREF", 256)), 456); ref2 = np.zeros(m.nu)
    for i in range(m.nu):
        e = jnp.zeros(m.nu).at[i].set(h)
        ref2[i] = float((Jb(theta + e, VR2).mean() - Jb(theta - e, VR2).mean()) / (2*h))
    t0 = time.time(); g_ex = gb(theta, v0s(2, 1)).block_until_ready(); t_gc = time.time() - t0
    N, TR = 32, int(os.environ.get("TR", 8)); ex, cl, ratios, nonfinite = [], [], [], 0
    for tr in range(TR):
        G = np.array(gb(theta, v0s(N, 1000 + tr)))
        bad = ~np.isfinite(G).all(1); nonfinite += int(bad.sum()); G = G[~bad]
        nr = np.linalg.norm(G, axis=1); ratios.append(float(nr.max() / np.median(nr)))
        ex.append(G.mean(0)); sc = np.minimum(1.0, 3*np.median(nr)/(nr+1e-12)); cl.append((G*sc[:, None]).mean(0))
    def stats(L):
        S = np.array(L); mu = S.mean(0); c = [float(x@ref/(np.linalg.norm(x)*np.linalg.norm(ref)+1e-12)) for x in S]
        return dict(cos=round(float(np.mean(c)), 3), rel_bias=round(float(np.linalg.norm(mu-ref)/np.linalg.norm(ref)), 3), rel_std=round(float(np.linalg.norm(S.std(0))/np.linalg.norm(ref)), 3))
    res[f"timeconst={tc}"] = dict(H=H, ref_norm=round(float(np.linalg.norm(ref)), 4), ref_self_rel_err=round(float(np.linalg.norm(ref-ref2)/np.linalg.norm(ref)), 3),
        EXACT=stats(ex), CLIP=stats(cl), max_over_median=dict(mean=round(float(np.mean(ratios)), 2), max=round(float(np.max(ratios)), 2)), nonfinite_samples=nonfinite,
        seconds=dict(compile_fwd=round(t_c), fd_ref=round(t_fd), compile_grad=round(t_gc)))
    print(json.dumps({f"timeconst={tc}": res[f"timeconst={tc}"]}), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
