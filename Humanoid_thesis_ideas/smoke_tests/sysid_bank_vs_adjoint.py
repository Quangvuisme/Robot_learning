"""Gradient-estimator study on a stiff-contact hopper (torch, CPU).
Objective: J(theta) = E_{s0, action noise}[ sum_t r_t ] for a stochastic linear-feature policy.
Estimators at equal rollout budget N:
  LR      : likelihood-ratio (zeroth order, PPO-like) with mean baseline
  EXACT   : pathwise BPTT through the true simulator (SHAC-like, no critic)
  CLIP    : EXACT + per-sample median-relative clipping (NablaMimic-style)
  WM      : pathwise BPTT through a learned MLP dynamics model (SuperMimic-style)
  HYBRID  : EXACT, but outlier samples (norm > c * median) are replaced by their WM gradient (proposed)
Reference: central finite differences with common random numbers over 200k rollouts
(gradient of the expected return, including contact discontinuities; bias O(h^2))."""
import torch, json, sys, math, time
torch.manual_seed(0); torch.set_num_threads(4)
T, DT, SIG = 60, 0.01, 0.3
MASS, GAIN = 1.0, 1.0
def step(s, u, k):
    z, v, zr = s[:, 0], s[:, 1], s[:, 2]
    pen = torch.relu(-z)
    contact = k*pen - 2*math.sqrt(k)*v*(pen > 0)          # stiff spring-damper ground
    thrust = 30.0*torch.tanh(u)*(pen > 0)                  # leg can only push in contact
    a = -9.81 + (contact + GAIN*thrust)/MASS
    v2 = v + DT*a; z2 = z + DT*v2
    return torch.stack([z2, v2, zr], 1)
def feats(s): z, v, zr = s[:, 0], s[:, 1], s[:, 2]; return torch.stack([z, v, zr, zr - z, torch.ones_like(z), (z<0).float()], 1)
def reward(s): return -(s[:, 0] - s[:, 2])**2
def rollout(theta, s0, eps, k, dyn=None, keep_grad=True):
    s, R, logp = s0, 0., 0.
    for t in range(T):
        mu = feats(s) @ theta; u = mu + SIG*eps[:, t]
        logp = logp + (-(eps[:, t]**2)/2)  # placeholder (LR uses separate formula)
        s = step(s, u, k) if dyn is None else dyn(s, u)
        R = R + reward(s)
    return R
def sample_s0(n, g): return torch.stack([0.3+0.1*torch.rand(n, generator=g), torch.zeros(n), 0.4+0.3*torch.rand(n, generator=g)], 1)
def lr_grad(theta, n, k, g):
    s0 = sample_s0(n, g); eps = torch.randn(n, T, generator=g)
    th = theta.detach().requires_grad_(True)
    with torch.no_grad(): R = rollout(theta, s0, eps, k)
    # log-prob gradient: sum_t eps_t/SIG * dmu/dtheta, recompute mus without grad through dynamics
    s = s0; glp = torch.zeros(n, theta.numel())
    with torch.no_grad():
        for t in range(T):
            f = feats(s); glp += (eps[:, t:t+1]/SIG)*f; s = step(s, f@theta + SIG*eps[:, t], k)
    A = (R - R.mean())[:, None]
    return (A*glp).mean(0), R.mean().item()
def path_grads(theta, n, k, g, dyn=None):
    s0 = sample_s0(n, g); eps = torch.randn(n, T, generator=g)
    th = theta.detach().requires_grad_(True)
    R = rollout(th, s0, eps, k, dyn)
    G = torch.stack([torch.autograd.grad(R[i], th, retain_graph=True)[0] for i in range(n)])
    return G, R.mean().item()
class WM(torch.nn.Module):
    def __init__(s): super().__init__(); s.f = torch.nn.Sequential(torch.nn.Linear(4, 128), torch.nn.SiLU(), torch.nn.Linear(128, 128), torch.nn.SiLU(), torch.nn.Linear(128, 2))
    def forward(s, x, u): d = s.f(torch.cat([x[:, :2], x[:, 2:3], u[:, None]], 1)); return torch.stack([x[:, 0]+DT*d[:, 0], x[:, 1]+d[:, 1], x[:, 2]], 1)
def fit_wm(theta, k, g, n=512, iters=1500):
    s0 = sample_s0(n, g); eps = torch.randn(n, T, generator=g); X, U, Y = [], [], []
    with torch.no_grad():
        s = s0
        for t in range(T):
            u = feats(s)@theta + SIG*eps[:, t]; s2 = step(s, u, k); X.append(s); U.append(u); Y.append(s2); s = s2
    X, U, Y = torch.cat(X), torch.cat(U), torch.cat(Y)
    wm = WM(); opt = torch.optim.Adam(wm.parameters(), 1e-3)
    for i in range(iters):
        idx = torch.randint(0, len(X), (1024,), generator=g); p = wm(X[idx], U[idx])
        loss = ((p[:, :2]-Y[idx, :2])**2 / torch.tensor([1e-4, 1e-1])).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return wm, loss.item()
def clip(G, c=3.0):
    nrm = G.norm(dim=1); med = nrm.median(); sc = torch.clamp(c*med/(nrm+1e-12), max=1.0); return G*sc[:, None], (sc < 1).float().mean().item()
def hybrid(G, Gw, c=3.0):
    nrm = G.norm(dim=1); bad = nrm > c*nrm.median(); H = G.clone(); H[bad] = Gw[bad]; return H, bad.float().mean().item()
def cos(a, b): return (a@b/(a.norm()*b.norm()+1e-12)).item()

import json, sys
class Res(torch.nn.Module):   # residual on top of the exact nominal sim step
    def __init__(s): super().__init__(); s.f = torch.nn.Sequential(torch.nn.Linear(4,128), torch.nn.SiLU(), torch.nn.Linear(128,128), torch.nn.SiLU(), torch.nn.Linear(128,2))
    def forward(s, x, u):
        global MASS, GAIN
        m, g = MASS, GAIN; MASS, GAIN = 1.0, 1.0; y = step(x, u, KK); MASS, GAIN = m, g
        d = s.f(torch.cat([x[:, :2], x[:, 2:3], u[:, None]], 1))
        return torch.stack([y[:, 0] + DT*DT*d[:, 0], y[:, 1] + DT*d[:, 1], y[:, 2]], 1)
def collect(theta, k, g, n=512):
    s0 = sample_s0(n, g); eps = torch.randn(n, T, generator=g); X, U, Y = [], [], []
    with torch.no_grad():
        s = s0
        for t in range(T):
            u = feats(s)@theta + SIG*eps[:, t]; s2 = step(s, u, k); X.append(s); U.append(u); Y.append(s2); s = s2
    return torch.cat(X), torch.cat(U), torch.cat(Y)
def fit(model, X, U, Y, g, iters=1500):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for i in range(iters):
        idx = torch.randint(0, len(X), (1024,), generator=g); p = model(X[idx], U[idx])
        loss = ((p[:, :2]-Y[idx, :2])**2 / torch.tensor([1e-6, 1e-2])).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

import time
theta = torch.tensor([-2.0, -0.3, 2.0, 1.0, 0.2, 0.5]); out = {}
TRUE = (1.3, 0.6)
def one_step_err(Xw, Uw, Yw, m, gn):
    global MASS, GAIN
    MASS, GAIN = m, gn; y = step(Xw, Uw, KK); return ((y[:, 1]-Yw[:, 1])**2).mean(-1) if y.dim()>2 else ((y[:, 1]-Yw[:, 1])**2).mean()
for KK in [1e3, 1e4, 1e5]:
    MASS, GAIN = TRUE
    g = torch.Generator().manual_seed(3); X, U, Y = collect(theta, KK, g, n=8)   # one short "real" stream: 8 rollouts x 60 steps
    # keep only time-ordered transitions of rollout 0..7 interleaved as a stream
    row = {}
    for W in [10, 30, 100, 300]:
        Xw, Uw, Yw = X[:W*8:8][:W], U[:W*8:8][:W], Y[:W*8:8][:W]   # W transitions from rollout 0 (contiguous in time)
        Xw, Uw, Yw = X[0::8][:W], U[0::8][:W], Y[0::8][:W]
        # LLA-style bank: 33 x 33 grid, pick min windowed one-step error (no gradients)
        t0 = time.time(); best = (1e9, None)
        ms = torch.linspace(0.8, 1.8, 33); gs = torch.linspace(0.3, 1.2, 33)
        with torch.no_grad():
            for m in ms:
                for gn in gs:
                    e = one_step_err(Xw, Uw, Yw, float(m), float(gn)).item()
                    if e < best[0]: best = (e, (float(m), float(gn)))
        t_bank = time.time()-t0; mb, gb = best[1]
        # adjoint-gradient SysID on the same window (HALO-style), 300 Adam steps
        t0 = time.time(); p = torch.zeros(2, requires_grad=True); opt = torch.optim.Adam([p], 0.05)
        for i in range(300):
            MASS, GAIN = torch.exp(p[0]), torch.exp(p[1]); y = step(Xw, Uw, KK)
            loss = ((y[:, 1]-Yw[:, 1])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        t_grad = time.time()-t0; mg, gg = float(torch.exp(p[0])), float(torch.exp(p[1]))
        n_contact = int((Xw[:, 0] < 0).sum())
        row[f"W={W}"] = dict(contact_steps=n_contact,
            bank=dict(mass=round(mb, 3), gain=round(gb, 3), err=round(abs(mb-1.3)/1.3 + abs(gb-0.6)/0.6, 3), sims=33*33*W, s=round(t_bank, 2)),
            adjoint=dict(mass=round(mg, 3), gain=round(gg, 3), err=round(abs(mg-1.3)/1.3 + abs(gg-0.6)/0.6, 3), sims=300*W, s=round(t_grad, 2)))
    MASS, GAIN = TRUE
    out[f"k={KK:.0e}"] = row; print(f"k={KK:.0e}", json.dumps(row), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
