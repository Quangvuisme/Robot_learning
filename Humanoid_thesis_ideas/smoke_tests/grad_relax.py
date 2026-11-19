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
RHO = None  # backward-only relaxation width (None = exact)
def st(hard, soft): return hard + (soft - soft.detach())   # forward = hard, backward = d(soft)
def step(s, u, k):
    z, v, zr = s[:, 0], s[:, 1], s[:, 2]
    pen = torch.relu(-z); ind = (pen > 0).float()
    if RHO is not None:
        pen = st(pen, RHO*torch.nn.functional.softplus(-z/RHO)); ind = st(ind, torch.sigmoid(-z/RHO))
    contact = k*pen - 2*math.sqrt(k)*v*ind
    thrust = 30.0*torch.tanh(u)*ind
    a = -9.81 + contact + thrust
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
theta = torch.tensor([-2.0, -0.3, 2.0, 1.0, 0.2, 0.5]); N, TRIALS = 64, 20; out = {}
for k in [1e3, 1e4, 1e5]:
    RHO = None
    ref = torch.zeros(6); B, h = 200000, 0.02
    gr = torch.Generator().manual_seed(7); s0 = sample_s0(B, gr); eps = torch.randn(B, T, generator=gr)
    with torch.no_grad():
        for i in range(6):
            d = torch.zeros(6); d[i] = h
            ref[i] = (rollout(theta+d, s0, eps, k).mean() - rollout(theta-d, s0, eps, k).mean())/(2*h)
    row = {}
    for name, rho in [("EXACT", None), ("CLIP", None), ("RELAX rho=1e-3", 1e-3), ("RELAX rho=3e-3", 3e-3), ("RELAX rho=1e-2", 1e-2)]:
        RHO = rho; L = []
        for tr in range(TRIALS):
            G, _ = path_grads(theta, N, k, torch.Generator().manual_seed(1000+tr))
            if name == "CLIP": G, _ = clip(G)
            L.append(G.mean(0))
        S = torch.stack(L); m = S.mean(0)
        row[name] = dict(cos=round(sum(cos(x, ref) for x in L)/len(L), 3), rel_bias=round(((m-ref).norm()/ref.norm()).item(), 3), rel_std=round((S.std(0).norm()/ref.norm()).item(), 3))
    RHO = None; out[f"k={k:.0e}"] = row; print(f"k={k:.0e}", json.dumps(row), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
