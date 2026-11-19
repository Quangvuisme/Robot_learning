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

theta = torch.tensor([-2.0, -0.3, 2.0, 1.0, 0.2, 0.5]); N, TRIALS, out = 64, 20, {}
for KK in [1e3, 1e4, 1e5]:
    MASS, GAIN = 1.3, 0.6
    ref = torch.zeros(6); B, h = 200000, 0.02
    gr = torch.Generator().manual_seed(7); s0 = sample_s0(B, gr); eps = torch.randn(B, T, generator=gr)
    with torch.no_grad():
        for i in range(6):
            d = torch.zeros(6); d[i] = h
            ref[i] = (rollout(theta+d, s0, eps, KK).mean() - rollout(theta-d, s0, eps, KK).mean())/(2*h)
    g = torch.Generator().manual_seed(3); X, U, Y = collect(theta, KK, g)
    # structured residual: identify (mass, gain) by gradient descent through the exact sim step (HALO-style)
    p = torch.zeros(2, requires_grad=True); opt = torch.optim.Adam([p], 0.05)
    for i in range(400):
        MASS, GAIN = torch.exp(p[0]), torch.exp(p[1])
        idx = torch.randint(0, len(X), (2048,), generator=g); y = step(X[idx], U[idx], KK)
        loss = ((y[:, 1]-Y[idx, 1])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    m_hat, g_hat = float(torch.exp(p[0])), float(torch.exp(p[1]))
    L = []
    for tr in range(TRIALS):
        MASS, GAIN = m_hat, g_hat; G, _ = path_grads(theta, N, KK, torch.Generator().manual_seed(1000+tr)); L.append(G.mean(0))
    MASS, GAIN = 1.3, 0.6
    S = torch.stack(L); m = S.mean(0)
    row = {"identified_mass": round(m_hat, 3), "identified_gain": round(g_hat, 3), "SIM+IDENTIFIED_PARAMS": dict(cos=round(sum(cos(x, ref) for x in L)/len(L), 3), rel_bias=round(((m-ref).norm()/ref.norm()).item(), 3), rel_std=round((S.std(0).norm()/ref.norm()).item(), 3))}
    out[f"k={KK:.0e}"] = row; print(f"k={KK:.0e}", json.dumps(row), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
