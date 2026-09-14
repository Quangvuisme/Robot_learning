"""The unified least-action step --- this file *is* the slide.

Over one time step the discrete action is stationary iff (v^f, v^r) solves

    minimize    1/2 (v^f - v^f*)^T M_f (v^f - v^f*)
    v^f, v^r  + 1/2 (v^r - v^r*)^T M_r (v^r - v^r*)

    subject to  D v^f      = 0        (incompressibility, multiplier p)
                c(v^f,v^r) = 0        (no-slip on the body, multiplier lambda)

where v^f*, v^r* are the "predicted" velocities after advection, viscosity,
gravity and the external forces F.  The two multipliers are exactly the two
red arrows at the bottom of the slide: `p` is the pressure that appears in
Navier-Stokes, `lambda` is the coupling force that appears in the manipulator
equation.

Eliminating the no-slip constraint
----------------------------------
`c(v^f, v^r) = 0` simply says "on the faces the body covers, the fluid moves
rigidly", i.e. v^f_S = J v^r.  Substituting that in closed form removes those
unknowns and leaves a *single* multiplier.  The quadratic term they carried is

    A = M_r (1 - rho_f/rho_s)  +  J^T M_S J.

Read it as "one continuum, two densities": the box is filled everywhere, with
density rho_s inside the body and rho_f outside, and the rigidity constraint
only says the inside moves as one piece.  The faces the body covers therefore
already contribute rho_f of inertia through `J^T M_S J`, so the body itself
must only bring the *excess* rho_s - rho_f.  Keeping those two pieces
consistent is what makes Archimedes' force come out exactly right instead of
approximately right; get it wrong and a cork sinks.  Splitting the
divergence into free faces F and body faces S, D v = D_F v_F + G v^r with
G = D_S J, the KKT system collapses to one symmetric positive semi-definite
solve for the pressure:

    (D_F M_f^-1 D_F^T  +  G A^-1 G^T) p  =  D_F v_F*  +  G v^r*
     \_________________/    \________/
      usual Poisson          rank-ndof coupling block

    v^f = v^f* - M_f^-1 D_F^T p          <- incompressible Navier-Stokes
    v^r = v^r* - A^-1 G^T p              <- manipulator equation

Both updates use the *same* multiplier, so momentum handed from one physics to
the other is conserved to solver tolerance, and the scheme is stable even when
the added mass dwarfs the body's own mass.  Nothing below is a drag model, a
buoyancy model or a contact model: lift, drag, buoyancy and added mass are all
just `p` doing its job.
"""

import numpy as np


def connected_components(fluid, open_u, open_v):
    """Label the fluid cells, where two cells are neighbours only if the face
    between them is free.  A body that seals against a wall really does cut the
    domain in two, and each piece then carries its own pressure constant.

    Plain neighbour propagation needs O(diameter) rounds; interleaving pointer
    jumping brings that down to a handful.
    """
    nx, ny = fluid.shape
    lab = np.where(fluid, np.arange(fluid.size).reshape(nx, ny), -1)
    au, av = open_u[1:nx, :], open_v[:, 1:ny]
    for _ in range(1000):
        prev = lab
        m = lab.copy()
        np.maximum(m[:-1, :], np.where(au, lab[1:, :], -1), out=m[:-1, :])
        np.maximum(m[1:, :], np.where(au, lab[:-1, :], -1), out=m[1:, :])
        np.maximum(m[:, :-1], np.where(av, lab[:, 1:], -1), out=m[:, :-1])
        np.maximum(m[:, 1:], np.where(av, lab[:, :-1], -1), out=m[:, 1:])
        m[~fluid] = -1
        f = m.ravel()
        for _ in range(4):
            gg = f.copy()
            ok = gg >= 0
            gg[ok] = f[gg[ok]]
            if np.array_equal(gg, f):
                break
            f = gg
        lab = f.reshape(nx, ny)
        if np.array_equal(lab, prev):
            break
    return lab


def dot(a, b):
    """np.vdot / np.linalg.norm route tiny arrays through a threaded BLAS and
    cost ~8 ms each here; einsum does the same 16k-element reduction in 15 us.
    The CG inner loop lives or dies on this."""
    return float(np.einsum("i,i->", a.reshape(-1), b.reshape(-1)))


def norm(a):
    return float(np.sqrt(dot(a, a)))


class NullProjector:
    """Projects out exactly the constants that S really annihilates.

    `1_c`, the indicator of one connected fluid component, is in the null space
    iff D_F^T 1_c = 0 (true inside any component) *and* G^T 1_c = 0, i.e. the
    body's rigid motion pushes no net flux into that component.  The second half
    fails precisely when the body seals a pocket against a wall -- there the
    pressure level is pinned by the trapped volume and projecting it away would
    delete a real part of the solution, leaving CG to stall forever on an
    inconsistent system.

    Everything is done with two bincounts so the cost does not grow with the
    number of components.
    """

    def __init__(self, lab, G):
        flat = lab.ravel()
        self.valid = flat >= 0
        _, inv = np.unique(flat[self.valid], return_inverse=True)
        self.inv = inv
        self.K = int(inv.max()) + 1 if inv.size else 0
        counts = np.bincount(inv, minlength=self.K).astype(float)
        keep = np.ones(self.K, dtype=bool)
        if G is not None and self.K:
            Gf = G.reshape(-1, G.shape[2])[self.valid]
            gs = np.stack([np.bincount(inv, weights=Gf[:, k], minlength=self.K)
                           for k in range(G.shape[2])], axis=1)
            keep = np.sqrt((gs ** 2).sum(1)) <= 1e-9 * (np.abs(G).sum() + 1e-300)
        self.coef = np.where(keep, 1.0 / np.maximum(counts, 1.0), 0.0)
        self.n_null = int(keep.sum())
        self.n_sealed = int(self.K - self.n_null)

    def __call__(self, x):
        xf = x.reshape(-1)
        xf[~self.valid] = 0.0
        if not self.K:
            return x
        s = np.bincount(self.inv, weights=xf[self.valid], minlength=self.K)
        xf[self.valid] -= (s * self.coef)[self.inv]
        return x


def _coarsen(f, ou, ov):
    """Halve the grid.  A coarse cell is fluid if any of its four children is,
    and a coarse face is open if any of the two fine faces behind it is."""
    nx, ny = f.shape
    NX, NY = nx // 2, ny // 2
    F = f.reshape(NX, 2, NY, 2).any(axis=(1, 3))
    OU = np.zeros((NX + 1, NY), dtype=bool)
    OV = np.zeros((NX, NY + 1), dtype=bool)
    if NX > 1:
        OU[1:NX, :] = ou[2:nx:2, :].reshape(NX - 1, NY, 2).any(axis=2)
    if NY > 1:
        OV[:, 1:NY] = ov[:, 2:ny:2].reshape(NX, 2, NY - 1).any(axis=1)
    return F, OU, OV


class MultigridPreconditioner:
    """Masked geometric multigrid V-cycle for the pressure Poisson block.

    Jacobi-preconditioned CG needs O(n) iterations on a Poisson problem -- about
    570 on a 128^2 grid -- because it has to move information across the whole
    grid one cell at a time.  A V-cycle moves it across in one pass, and the
    iteration count stops caring how fine the grid is.

    The coupling block `G A^-1 G^T` is deliberately left out: it has rank at
    most three and CG mops it up in a handful of extra iterations.  Restriction
    is exactly (1/4) P^T with P piecewise-constant prolongation, and the same
    number of pre- and post-smoothing sweeps, so the whole V-cycle is symmetric
    and positive definite -- which is what CG requires of a preconditioner.
    """

    def __init__(self, fluid, open_u, open_v, rho, min_size=8, omega=0.8,
                 nsmooth=2, ncoarse=40):
        self.rho, self.omega, self.nsmooth, self.ncoarse = rho, omega, nsmooth, ncoarse
        self.levels = []
        f, ou, ov, scale = fluid, open_u, open_v, 1.0
        while True:
            # .astype(float) FIRST: adding bool arrays is logical OR, which
            # would make every diagonal 1 instead of the face count
            diag = (ou[1:, :].astype(float) + ou[:-1, :]
                    + ov[:, 1:] + ov[:, :-1])
            diag /= rho * scale
            self.levels.append({
                "f": f, "ou": ou, "ov": ov, "scale": scale,
                "fl": f.astype(float),
                "dinv": np.where(diag > 0, 1.0 / np.maximum(diag, 1e-300), 0.0),
                "inv_n": 1.0 / max(1.0, float(f.sum())),
            })
            nx, ny = f.shape
            if min(nx, ny) <= min_size or nx % 2 or ny % 2:
                break
            f, ou, ov = _coarsen(f, ou, ov)
            # Galerkin consistency: with P piecewise constant over 2x2 blocks
            # and R = (1/4) P^T, one finds (1/4) P^T A_f P = stencil_c / 2 in
            # two dimensions -- so the stencil is divided by 2 per level, not by
            # the 4 that the "same operator on a coarser grid" reading suggests.
            # Using 4 makes every coarse correction twice too large and the
            # V-cycle diverges.
            scale *= 2.0

    def apply_A(self, lv, p):
        nx, ny = lv["f"].shape
        wu = np.zeros((nx + 1, ny))
        wv = np.zeros((nx, ny + 1))
        wu[1:nx, :] = p[:-1, :] - p[1:, :]
        wv[:, 1:ny] = p[:, :-1] - p[:, 1:]
        wu *= lv["ou"]
        wv *= lv["ov"]
        out = (wu[1:, :] - wu[:-1, :] + wv[:, 1:] - wv[:, :-1])
        out /= self.rho * lv["scale"]
        return out * lv["fl"]

    def _smooth(self, lv, e, r, n):
        for _ in range(n):
            e += self.omega * lv["dinv"] * (r - self.apply_A(lv, e))
        return e

    def _mean_free(self, lv, x):
        x -= (x.sum() * lv["inv_n"]) * lv["fl"]
        return x * lv["fl"]

    def _vcycle(self, l, r):
        lv = self.levels[l]
        e = np.zeros_like(r)
        if l == len(self.levels) - 1:
            r = self._mean_free(lv, r.copy())
            self._smooth(lv, e, r, self.ncoarse)
            return self._mean_free(lv, e)
        self._smooth(lv, e, r, self.nsmooth)
        res = (r - self.apply_A(lv, e)) * lv["fl"]
        nx, ny = lv["f"].shape
        rc = res.reshape(nx // 2, 2, ny // 2, 2).sum(axis=(1, 3)) * 0.25
        ec = self._vcycle(l + 1, rc)
        e += np.repeat(np.repeat(ec, 2, axis=0), 2, axis=1) * lv["fl"]
        self._smooth(lv, e, r, self.nsmooth)
        return e

    def __call__(self, r):
        return self._vcycle(0, r)


def pcg(Aop, b, precond, project, tol=1e-8, maxiter=500, x0=None):
    """Preconditioned conjugate gradient on a possibly singular SPD system."""
    if x0 is None:
        x = np.zeros_like(b)
        r = project(b.copy())
    else:
        x = x0.copy()
        r = project(b - Aop(x))
    b_norm = norm(r) + 1e-300
    z = project(precond(r))
    p = z.copy()
    rz = dot(r, z)
    it = 0
    for it in range(1, maxiter + 1):
        Ap = Aop(p)
        pAp = dot(p, Ap)
        if pAp <= 1e-300:
            break
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        if norm(r) / b_norm < tol:
            break
        z = project(precond(r))
        rz_new = dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, it, norm(r) / b_norm


class Masks:
    """Which faces are free fluid, which are carried by the body, which are wall."""

    def __init__(self, grid, solid_cell):
        nx, ny = grid.nx, grid.ny
        self.fluid_cell = ~solid_cell

        blocked_u = np.zeros((nx + 1, ny), dtype=bool)
        blocked_v = np.zeros((nx, ny + 1), dtype=bool)
        # a face is carried by the body as soon as either neighbouring cell is
        # inside it: that voxelised surface is where no-slip gets enforced
        blocked_u[1:nx, :] = solid_cell[:-1, :] | solid_cell[1:, :]
        blocked_v[:, 1:ny] = solid_cell[:, :-1] | solid_cell[:, 1:]

        self.wall_u = np.zeros((nx + 1, ny), dtype=bool)
        self.wall_v = np.zeros((nx, ny + 1), dtype=bool)
        self.wall_u[[0, nx], :] = True
        self.wall_v[:, [0, ny]] = True

        self.body_u = blocked_u
        self.body_v = blocked_v
        self.open_u = ~blocked_u & ~self.wall_u
        self.open_v = ~blocked_v & ~self.wall_v


class UnifiedSolve:
    """Result of one least-action step, kept around for plotting."""

    def __init__(self, p, vr, vr_star, impulse, A, iters, residual):
        self.p = p
        self.vr = vr
        self.vr_star = vr_star
        self.impulse = impulse          # generalised impulse the fluid gave the body
        self.A = A                      # effective inertia (body + displaced fluid)
        self.iters = iters
        self.residual = residual


def discrete_archimedes(grid, Jvs, gravity):
    """The hydrostatic part of the pressure multiplier, in closed form.

    For a constant density fluid the hydrostatic solution of the very same KKT
    system is p ~ -rho g y, and pushing that through `-G^T p` telescopes to

        F_k = rho_f g dx^2 sum_{body y-faces} J_v[face, k],

    i.e. Archimedes' force and torque, on exactly the voxelised body the solver
    sees.  Computing it here instead of loading gravity onto every fluid face
    gives identical physics without re-solving a large smooth pressure mode
    (and its discretisation noise) at every single step.
    """
    return grid.rho * gravity * grid.dx ** 2 * Jvs.sum(axis=0)


def unified_projection(grid, body, vr, dt, gravity=0.0, hydrostatic=False,
                       tol=1e-8, maxiter=500, lam0=None, multigrid=True):
    """Solve the coupled KKT system and write v^f, v^r back.

    `grid.u`, `grid.v` must already hold the predicted fluid velocity v^f*;
    `vr` is the rigid velocity at the start of the step.  Returns a
    `UnifiedSolve`; `grid.u/v` are overwritten with the real v^f.
    """
    nx, ny, dx, rho = grid.nx, grid.ny, grid.dx, grid.rho
    mF = rho * dx ** 2                       # mass sitting on one velocity sample

    solid_cell = (body.occupancy(grid) if body is not None
                  else np.zeros((nx, ny), dtype=bool))
    mk = Masks(grid, solid_cell)
    fluid = mk.fluid_cell.astype(float)

    ndof = 0 if body is None else body.ndof
    if ndof:
        Ju, Jv = body.jacobian(grid)

        # ---- A = excess body inertia + inertia of the fluid it displaces -----
        Jus, Jvs = Ju[mk.body_u], Jv[mk.body_v]
        M_disp = mF * (Jus.T @ Jus + Jvs.T @ Jvs)
        A = body.M * (1.0 - rho / body.rho_s) + M_disp
        Ainv = np.linalg.inv(A)

        # ---- L^r prediction: gravity, damping, any other external force -----
        F = body.ext_force(vr)
        if hydrostatic and gravity:
            F = F + discrete_archimedes(grid, Jvs, gravity)
        vr_star = vr + dt * (Ainv @ F)

        # ---- no-slip: the body's faces carry exactly the rigid velocity ------
        grid.u[mk.body_u] = Ju[mk.body_u] @ vr_star
        grid.v[mk.body_v] = Jv[mk.body_v] @ vr_star

        # ---- G = D_S J : how rigid motion pushes flux through each cell ------
        Jub = Ju * mk.body_u[:, :, None]
        Jvb = Jv * mk.body_v[:, :, None]
        G = dx * (Jub[1:, :, :] - Jub[:-1, :, :]
                  + Jvb[:, 1:, :] - Jvb[:, :-1, :])
        G *= mk.fluid_cell[:, :, None]
    else:
        Ainv = None
        G = None
        A = None
        vr_star = np.zeros(0)

    grid.u[mk.wall_u] = 0.0
    grid.v[mk.wall_v] = 0.0

    # ------------------------------------------------------ right hand side
    uo = grid.u * mk.open_u
    vo = grid.v * mk.open_v
    rhs = dx * (uo[1:, :] - uo[:-1, :] + vo[:, 1:] - vo[:, :-1])
    if ndof:
        rhs = rhs + G @ vr_star
    rhs *= fluid

    # ------------------------------------------------- the actual null space
    lab = connected_components(mk.fluid_cell, mk.open_u, mk.open_v)
    project = NullProjector(lab, G)

    # --------------------------------------------------- matrix-free operator
    def grad_over_mass(lam):
        """M_f^-1 D_F^T lam, living on the free faces."""
        wu = np.zeros((nx + 1, ny))
        wv = np.zeros((nx, ny + 1))
        wu[1:nx, :] = (lam[:-1, :] - lam[1:, :]) / (rho * dx)
        wv[:, 1:ny] = (lam[:, :-1] - lam[:, 1:]) / (rho * dx)
        wu *= mk.open_u
        wv *= mk.open_v
        return wu, wv

    def Aop(lam):
        wu, wv = grad_over_mass(lam)
        out = dx * (wu[1:, :] - wu[:-1, :] + wv[:, 1:] - wv[:, :-1])
        out *= fluid
        if ndof:
            out += G @ (Ainv @ np.einsum("ijk,ij->k", G, lam))
        return out

    if multigrid and min(nx, ny) >= 32:
        precond = MultigridPreconditioner(mk.fluid_cell, mk.open_u, mk.open_v, rho)
    else:
        # Jacobi diagonal: (#free faces)/rho  +  the coupling block's diagonal
        diag = (mk.open_u[1:, :].astype(float) + mk.open_u[:-1, :]
                + mk.open_v[:, 1:] + mk.open_v[:, :-1]) / rho
        if ndof:
            diag += np.einsum("ijk,kl,ijl->ij", G, Ainv, G)
        diag *= fluid
        dinv = np.where(diag > 1e-14, 1.0 / np.maximum(diag, 1e-14), 0.0)
        precond = lambda r: dinv * r        # noqa: E731

    lam, iters, residual = pcg(Aop, rhs, precond, project, tol=tol,
                               maxiter=maxiter, x0=lam0)

    # ------------------------------------------------------------- the update
    wu, wv = grad_over_mass(lam)
    grid.u -= wu
    grid.v -= wv

    if ndof:
        impulse = -np.einsum("ijk,ij->k", G, lam)      # fluid -> body
        vr_new = vr_star + Ainv @ impulse
        grid.u[mk.body_u] = Ju[mk.body_u] @ vr_new
        grid.v[mk.body_v] = Jv[mk.body_v] @ vr_new
    else:
        impulse = np.zeros(0)
        vr_new = vr_star

    grid.u[mk.wall_u] = 0.0
    grid.v[mk.wall_v] = 0.0

    return UnifiedSolve(lam, vr_new, vr_star, impulse, A, iters, residual)
