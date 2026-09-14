"""MAC-grid incompressible fluid --- the `L^f(q^f, v^f)` half of the slide.

Everything here is the *unconstrained* part of the action: advection (the
material derivative), viscous dissipation and external body forces.  The
constraint `div(v^f) = 0` is NOT enforced here; it is handed to `coupling.py`,
which solves it together with the rigid-body constraint in one KKT system.

Staggered (Marker-And-Cell) layout, i = x index, j = y index:

    u[i, j]  lives at ( i*dx      , (j+0.5)*dx )   shape (nx+1, ny)
    v[i, j]  lives at ( (i+0.5)*dx,  j*dx      )   shape (nx, ny+1)
    p[i, j]  lives at ( (i+0.5)*dx, (j+0.5)*dx )   shape (nx, ny)
"""

import numpy as np


def bilerp(F, gi, gj):
    """Bilinear sample of `F` at fractional grid coordinates (gi, gj)."""
    nx, ny = F.shape
    gi = np.clip(gi, 0.0, nx - 1 - 1e-7)
    gj = np.clip(gj, 0.0, ny - 1 - 1e-7)
    i0 = gi.astype(np.int64)
    j0 = gj.astype(np.int64)
    i1 = i0 + 1
    j1 = j0 + 1
    fi = gi - i0
    fj = gj - j0
    return ((1 - fi) * (1 - fj) * F[i0, j0]
            + fi * (1 - fj) * F[i1, j0]
            + (1 - fi) * fj * F[i0, j1]
            + fi * fj * F[i1, j1])


class MACGrid:
    def __init__(self, nx, ny, dx, rho=1.0, nu=1e-5):
        self.nx, self.ny, self.dx = nx, ny, dx
        self.rho, self.nu = rho, nu
        self.Lx, self.Ly = nx * dx, ny * dx

        self.u = np.zeros((nx + 1, ny))
        self.v = np.zeros((nx, ny + 1))
        self.dye = np.zeros((nx, ny))

        # cached sample positions
        iu, ju = np.meshgrid(np.arange(nx + 1), np.arange(ny), indexing="ij")
        self.Xu, self.Yu = iu * dx, (ju + 0.5) * dx
        iv, jv = np.meshgrid(np.arange(nx), np.arange(ny + 1), indexing="ij")
        self.Xv, self.Yv = (iv + 0.5) * dx, jv * dx
        ic, jc = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        self.Xc, self.Yc = (ic + 0.5) * dx, (jc + 0.5) * dx

    # ---------------------------------------------------------------- sampling
    def sample_u(self, X, Y):
        return bilerp(self.u, X / self.dx, Y / self.dx - 0.5)

    def sample_v(self, X, Y):
        return bilerp(self.v, X / self.dx - 0.5, Y / self.dx)

    def sample_c(self, F, X, Y):
        return bilerp(F, X / self.dx - 0.5, Y / self.dx - 0.5)

    def velocity_at(self, X, Y):
        return self.sample_u(X, Y), self.sample_v(X, Y)

    # ------------------------------------------------------------ advection
    def _backtrace(self, X, Y, dt):
        """Second order (midpoint) semi-Lagrangian backtrace."""
        u1, v1 = self.velocity_at(X, Y)
        Xm, Ym = X - 0.5 * dt * u1, Y - 0.5 * dt * v1
        u2, v2 = self.velocity_at(Xm, Ym)
        return X - dt * u2, Y - dt * v2

    def advect(self, dt):
        """Transport u, v and the dye field along the flow (the `(v.grad)v` term)."""
        Xb, Yb = self._backtrace(self.Xu, self.Yu, dt)
        new_u = self.sample_u(Xb, Yb)
        Xb, Yb = self._backtrace(self.Xv, self.Yv, dt)
        new_v = self.sample_v(Xb, Yb)
        Xb, Yb = self._backtrace(self.Xc, self.Yc, dt)
        new_dye = self.sample_c(self.dye, Xb, Yb)
        self.u, self.v, self.dye = new_u, new_v, new_dye

    # ------------------------------------------------------------- viscosity
    @staticmethod
    def _lap(F):
        P = np.pad(F, 1, mode="edge")
        return P[2:, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] + P[1:-1, :-2] - 4.0 * F

    def diffuse(self, dt):
        """Explicit viscous step, sub-cycled to stay inside the stability limit."""
        if self.nu <= 0.0:
            return
        a = self.nu * dt / self.dx ** 2
        n = max(1, int(np.ceil(a / 0.2)))
        ak = a / n
        for _ in range(n):
            self.u += ak * self._lap(self.u)
            self.v += ak * self._lap(self.v)

    # ------------------------------------------------------------ diagnostics
    def cell_velocity(self):
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        return uc, vc

    def vorticity(self):
        """omega = dv/dx - du/dy, evaluated at cell centres."""
        uc, vc = self.cell_velocity()
        wy = np.gradient(uc, self.dx, axis=1)
        wx = np.gradient(vc, self.dx, axis=0)
        return wx - wy

    def kinetic_energy(self):
        m = self.rho * self.dx ** 2
        return 0.5 * m * (np.sum(self.u ** 2) + np.sum(self.v ** 2))

    def enforce_walls(self):
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0
