"""Rigid bodies --- the `L^r(q^r, v^r)` half of the slide.

Each body exposes exactly what the unified KKT solve in `coupling.py` needs:

    ndof                   number of generalised velocities
    M                      (ndof, ndof) mass matrix  -> `M(q^r)` in the slide
    ext_force(dt)          gravity + damping         -> `G(q^r)` and `F^T q`
    occupancy(grid)        which cells the body covers
    jacobian(grid)         dc/dv^r : maps v^r to the grid-face velocities the
                           coupling constraint `c(v^f, v^r) = 0` pins down
    integrate(dt, vr)      advance the configuration q^r
"""

import numpy as np


class RigidBody:
    ndof = 0
    M = None
    rho_s = 1.0

    def ext_force(self, vr):
        """Generalised external force: the body's full weight plus damping.

        Never buoyancy --- that is the pressure multiplier's job.
        """
        raise NotImplementedError

    def occupancy(self, grid):
        raise NotImplementedError

    def jacobian(self, grid):
        raise NotImplementedError

    def integrate(self, dt, vr):
        raise NotImplementedError

    def polygon(self):
        raise NotImplementedError

    def potential_energy(self):
        return 0.0


class PendulumPaddle(RigidBody):
    """A flat blade hinged at a fixed pivot.  One DOF: the joint angle.

    theta = 0 hangs straight down.  The blade occupies s in [s0, s1] along the
    body axis and |t| <= hw across it.
    """

    ndof = 1

    def __init__(self, pivot, s0, s1, hw, rho_s=8.0, g=9.81,
                 theta0=0.0, omega0=0.0, joint_damping=0.0):
        self.pivot = np.asarray(pivot, dtype=float)
        self.s0, self.s1, self.hw = s0, s1, hw
        self.rho_s = rho_s
        self.g = g
        self.theta = float(theta0)
        self.joint_damping = joint_damping

        area = (s1 - s0) * (2 * hw)
        self.mass = rho_s * area
        self.sc = 0.5 * (s0 + s1)                      # centre of mass offset
        # I = rho_s * int (s^2 + t^2) ds dt   about the pivot
        self.I = rho_s * ((s1 ** 3 - s0 ** 3) / 3.0 * 2 * hw
                          + (s1 - s0) * (2 * hw ** 3) / 3.0)
        self.M = np.array([[self.I]])
        self.v0 = np.array([float(omega0)])

    # body axis: theta = 0 points along -y (straight down)
    def _frame(self):
        e = np.array([np.sin(self.theta), -np.cos(self.theta)])   # along blade
        n = np.array([np.cos(self.theta), np.sin(self.theta)])    # across blade
        return e, n

    def _inside(self, X, Y):
        e, n = self._frame()
        px, py = X - self.pivot[0], Y - self.pivot[1]
        s = px * e[0] + py * e[1]
        t = px * n[0] + py * n[1]
        return (s >= self.s0) & (s <= self.s1) & (np.abs(t) <= self.hw)

    def occupancy(self, grid):
        return self._inside(grid.Xc, grid.Yc)

    def jacobian(self, grid):
        """v(p) = thetadot * z_hat x (p - pivot)."""
        Ju = np.empty((grid.nx + 1, grid.ny, 1))
        Jv = np.empty((grid.nx, grid.ny + 1, 1))
        Ju[:, :, 0] = -(grid.Yu - self.pivot[1])
        Jv[:, :, 0] = (grid.Xv - self.pivot[0])
        return Ju, Jv

    def ext_force(self, vr):
        tau = -self.mass * self.g * self.sc * np.sin(self.theta)
        tau -= self.joint_damping * vr[0]
        return np.array([tau])

    def integrate(self, dt, vr):
        self.theta += dt * vr[0]

    def potential_energy(self):
        e, _ = self._frame()
        y_com = self.pivot[1] + self.sc * e[1]
        return self.mass * self.g * y_com

    def polygon(self):
        e, n = self._frame()
        p = self.pivot
        return np.array([p + self.s0 * e + self.hw * n,
                         p + self.s1 * e + self.hw * n,
                         p + self.s1 * e - self.hw * n,
                         p + self.s0 * e - self.hw * n])


class FreePlate(RigidBody):
    """A free rectangular plate: three DOF, v^r = (vx, vy, omega).

    Nothing here knows about buoyancy or drag --- both fall out of the pressure
    multiplier produced by the coupled solve.
    """

    ndof = 3

    def __init__(self, centre, half_len, half_thk, rho_s=3.0, g=9.81,
                 theta0=0.0, vel0=(0.0, 0.0, 0.0)):
        self.c = np.asarray(centre, dtype=float)
        self.a, self.b = half_len, half_thk
        self.theta = float(theta0)
        self.rho_s = rho_s
        self.g = g
        self.mass = rho_s * (2 * half_len) * (2 * half_thk)
        self.I = self.mass * (half_len ** 2 + half_thk ** 2) / 3.0
        self.M = np.diag([self.mass, self.mass, self.I])
        self.v0 = np.asarray(vel0, dtype=float)

    def _frame(self):
        e = np.array([np.cos(self.theta), np.sin(self.theta)])
        n = np.array([-np.sin(self.theta), np.cos(self.theta)])
        return e, n

    def _inside(self, X, Y):
        e, n = self._frame()
        px, py = X - self.c[0], Y - self.c[1]
        s = px * e[0] + py * e[1]
        t = px * n[0] + py * n[1]
        return (np.abs(s) <= self.a) & (np.abs(t) <= self.b)

    def occupancy(self, grid):
        return self._inside(grid.Xc, grid.Yc)

    def jacobian(self, grid):
        """v(p) = (vx, vy) + omega * z_hat x (p - c)."""
        Ju = np.zeros((grid.nx + 1, grid.ny, 3))
        Jv = np.zeros((grid.nx, grid.ny + 1, 3))
        Ju[:, :, 0] = 1.0
        Ju[:, :, 2] = -(grid.Yu - self.c[1])
        Jv[:, :, 1] = 1.0
        Jv[:, :, 2] = (grid.Xv - self.c[0])
        return Ju, Jv

    def ext_force(self, vr):
        return np.array([0.0, -self.mass * self.g, 0.0])

    def integrate(self, dt, vr):
        self.c = self.c + dt * vr[:2]
        self.theta += dt * vr[2]

    def potential_energy(self):
        return self.mass * self.g * self.c[1]

    def polygon(self):
        e, n = self._frame()
        return np.array([self.c + self.a * e + self.b * n,
                         self.c - self.a * e + self.b * n,
                         self.c - self.a * e - self.b * n,
                         self.c + self.a * e - self.b * n])
