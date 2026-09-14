"""Time stepping: predict with the unconstrained Lagrangian, then project with
the unified least-action solve of `coupling.py`."""

import time

import numpy as np

from .coupling import unified_projection


class Simulation:
    GRAVITY_MODES = ("hydrostatic", "explicit")

    def __init__(self, grid, body=None, dt=1e-3, buoyancy=0.0, gravity=0.0,
                 gravity_mode="hydrostatic",
                 pre_step=None, solver_tol=1e-8, solver_maxiter=500):
        """
        gravity_mode
            "hydrostatic" -- gravity is *not* put on the fluid.  For a constant
                density fluid its hydrostatic pressure is known in closed form
                and its net effect on a submerged body is exactly Archimedes'
                force, so it is folded into the body's weight and the solver is
                left to handle only the dynamic pressure.  No numerical noise.
            "explicit" -- gravity is put on every fluid face and the body keeps
                its full weight.  Buoyancy then has to emerge from the discrete
                pressure multiplier.  It does; `sanity_checks` verifies that the
                two modes agree.
        """
        assert gravity_mode in self.GRAVITY_MODES
        self.grid = grid
        self.body = body
        self.dt = dt
        self.buoyancy = buoyancy
        self.gravity = gravity
        self.gravity_mode = gravity_mode
        self.pre_step = pre_step
        self.solver_tol = solver_tol
        self.solver_maxiter = solver_maxiter

        self.t = 0.0
        self.step_index = 0
        self.vr = body.v0.copy() if body is not None else None
        self.last = None
        self._lam = None
        self.history = {k: [] for k in
                        ("t", "q", "vr", "wrench", "ke_fluid", "ke_body",
                         "pe_body", "cg_iters", "cg_res")}

    # ------------------------------------------------------------------ step
    def step(self):
        g, dt = self.grid, self.dt

        if self.pre_step is not None:
            self.pre_step(self)

        # ---- L^f : transport + viscous dissipation -----------------------
        g.advect(dt)
        g.diffuse(dt)
        if self.gravity and self.gravity_mode == "explicit":
            # gravity on the fluid itself.  In a closed box the projection turns
            # this into the hydrostatic pressure field, and Archimedes' force on
            # an immersed body drops out of the same multiplier -- there is no
            # buoyancy term anywhere in this code.
            g.v -= dt * self.gravity
        if self.buoyancy:
            # thermal buoyancy on the y-faces, from the advected dye
            dye_v = np.zeros((g.nx, g.ny + 1))
            dye_v[:, 1:-1] = 0.5 * (g.dye[:, :-1] + g.dye[:, 1:])
            g.v += dt * self.buoyancy * dye_v
        g.enforce_walls()

        # ---- the coupled stationarity condition --------------------------
        # (the L^r prediction -- gravity, damping, F -- happens inside, because
        #  it has to use the same effective inertia A that the constraint sees)
        sol = unified_projection(g, self.body, self.vr, dt,
                                 gravity=self.gravity,
                                 hydrostatic=(self.gravity_mode == "hydrostatic"),
                                 tol=self.solver_tol, maxiter=self.solver_maxiter,
                                 lam0=self._lam)
        # the pressure moves slowly between steps: warm starting costs nothing
        # and typically halves the iteration count
        self._lam = sol.p
        self.last = sol

        if self.body is not None:
            self.vr = sol.vr
            self.body.integrate(dt, self.vr)

        self.t += dt
        self.step_index += 1
        return sol

    # ------------------------------------------------------------- recording
    def record(self):
        g, b, sol = self.grid, self.body, self.last
        h = self.history
        h["t"].append(self.t)
        if b is None:
            h["q"].append(np.zeros(0))
            h["vr"].append(np.zeros(0))
            h["wrench"].append(np.zeros(0))
            h["ke_body"].append(0.0)
            h["pe_body"].append(0.0)
        else:
            q = np.array([b.theta]) if b.ndof == 1 else np.array([*b.c, b.theta])
            h["q"].append(q)
            h["vr"].append(self.vr.copy())
            h["wrench"].append(sol.impulse / self.dt if sol is not None
                               else np.zeros(b.ndof))
            h["ke_body"].append(0.5 * float(self.vr @ b.M @ self.vr))
            h["pe_body"].append(b.potential_energy())
        h["ke_fluid"].append(g.kinetic_energy())
        h["cg_iters"].append(sol.iters if sol is not None else 0)
        h["cg_res"].append(sol.residual if sol is not None else 0.0)

    def run(self, nsteps, record_every=1, frame_cb=None, verbose=True):
        frames = []
        t0 = time.time()
        for k in range(nsteps):
            self.step()
            if k % record_every == 0:
                self.record()
                if frame_cb is not None:
                    frames.append(frame_cb(self))
            if verbose and (k + 1) % max(1, nsteps // 20) == 0:
                done = (k + 1) / nsteps
                el = time.time() - t0
                print(f"  [{100*done:5.1f}%] t={self.t:6.3f}s  "
                      f"cg={self.last.iters:3d}  "
                      f"eta={el*(1-done)/max(done,1e-9):5.1f}s", flush=True)
        for k in self.history:
            self.history[k] = np.array(self.history[k])
        return frames
