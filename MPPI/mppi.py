"""MPPI (Model Predictive Path Integral) controller -- Algorithm 2 on the slide.

    Given:  F (transition model), K (samples), H (horizon), T=H (timesteps)
            (u_0, ..., u_{T-1}): warm-started control sequence
            Sigma, phi, q, lambda: cost/noise hyperparameters

    while task not done:
        x0 <- GetStateEstimate()
        for k in 0..K-1:
            sample eps^k_t ~ N(0, Sigma) for t in 0..T-1
            roll out x_t <- F(x_{t-1}, u_{t-1} + eps^k_{t-1})
            S(eps^k) = sum_t [ q(x_t) + lambda u_{t-1}^T Sigma^-1 eps^k_{t-1} ]
                       + phi(x_T)
        beta <- min_k S(eps^k)
        eta  <- sum_k exp(-1/lambda (S(eps^k) - beta))
        w_k  <- (1/eta) exp(-1/lambda (S(eps^k) - beta))
        u_t += sum_k w_k eps^k_t                     for all t
        SendToActuators(u_0)
        shift: u_{t-1} <- u_t  (receding horizon), u_{T-1} <- reinitialize

Everything below is a literal, vectorised (batch = K samples) implementation of
that algorithm on top of `cartpole.dynamics`, running entirely on the GPU when
one is available -- the same "massive parallelization" the slide advertises.
"""

import torch


class MPPIController:
    def __init__(self, dynamics_fn, cost_fn, terminal_cost_fn, nx, nu,
                 horizon=40, num_samples=8192, dt=0.02,
                 noise_sigma=1.0, lam=1.0, u_min=-15.0, u_max=15.0,
                 device=None, seed=None):
        self.F = dynamics_fn
        self.q = cost_fn                 # running state cost  q(x)
        self.phi = terminal_cost_fn      # terminal cost       phi(x_T)
        self.nx, self.nu = nx, nu
        self.H = horizon
        self.K = num_samples
        self.dt = dt
        self.lam = lam
        self.u_min, self.u_max = u_min, u_max

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sigma = torch.tensor(float(noise_sigma), device=self.device)
        self.sigma_inv = 1.0 / self.sigma

        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)

        # the warm-started nominal control sequence u_0 .. u_{H-1}
        self.u_seq = torch.zeros(self.H, device=self.device)

        self.last_rollouts = None   # kept around for plotting/animating
        self.last_weights = None

    @torch.no_grad()
    def command(self, x0):
        """One MPPI planning step from state x0 (shape (nx,)).  Returns the
        action to send to the actuator, u_0, and advances the warm start."""
        dev = self.device
        H, K = self.H, self.K
        x0 = x0.to(dev)

        # --- sample eps^k_t ~ N(0, Sigma), t=0..H-1, k=0..K-1 --------------
        eps = torch.normal(mean=0.0, std=float(self.sigma), size=(K, H),
                           generator=self.gen, device=dev)

        u_samp = self.u_seq.unsqueeze(0) + eps          # (K, H)
        u_samp = u_samp.clamp(self.u_min, self.u_max)

        # --- roll out x_t <- F(x_{t-1}, u_{t-1} + eps) ----------------------
        x = x0.unsqueeze(0).expand(K, self.nx).clone()  # (K, nx)
        costs = torch.zeros(K, device=dev)
        traj = torch.empty(K, H + 1, self.nx, device=dev)
        traj[:, 0] = x

        for t in range(H):
            u_t = u_samp[:, t]
            x = self.F(x, u_t, self.dt)
            traj[:, t + 1] = x
            # S += q(x_t) + lambda * u_{t-1}^T Sigma^-1 eps^k_{t-1}
            costs += self.q(x) + self.lam * self.u_seq[t] * self.sigma_inv * eps[:, t]

        costs += self.phi(x)   # phi(x_T)

        # --- importance weights: beta, eta, w_k -----------------------------
        beta = costs.min()
        w = torch.exp(-(costs - beta) / self.lam)
        eta = w.sum()
        w = w / eta

        # --- u_t += sum_k w_k eps^k_t  for all t ----------------------------
        self.u_seq = (self.u_seq + (w.unsqueeze(1) * eps).sum(dim=0)
                     ).clamp(self.u_min, self.u_max)

        self.last_rollouts = traj
        self.last_weights = w

        u0 = self.u_seq[0].clone()

        # --- receding horizon shift: u_{t-1} <- u_t, reinitialize u_{T-1} ---
        self.u_seq = torch.roll(self.u_seq, shifts=-1, dims=0)
        self.u_seq[-1] = 0.0

        return u0
