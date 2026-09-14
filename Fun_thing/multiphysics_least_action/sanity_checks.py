"""Numerical sanity checks for the unified least-action solver.

Run:  python -m multiphysics_least_action.sanity_checks
"""

import numpy as np

from .bodies import FreePlate, PendulumPaddle
from .coupling import Masks, unified_projection
from .fluid import MACGrid
from .sim import Simulation


def _divergence(grid, mk):
    uo = grid.u * (mk.open_u | mk.body_u)
    vo = grid.v * (mk.open_v | mk.body_v)
    d = (uo[1:, :] - uo[:-1, :] + vo[:, 1:] - vo[:, :-1]) / grid.dx
    return d * mk.fluid_cell


def check_divergence_free():
    """After the projection the fluid cells must be divergence free."""
    rng = np.random.default_rng(0)
    g = MACGrid(48, 48, 1 / 48, rho=1.0, nu=0.0)
    g.u = rng.normal(size=g.u.shape)
    g.v = rng.normal(size=g.v.shape)
    body = PendulumPaddle((0.5, 0.75), 0.05, 0.35, 0.03, theta0=0.7)
    sol = unified_projection(g, body, np.array([1.3]), 1e-3)
    mk = Masks(g, body.occupancy(g))
    d = _divergence(g, mk)
    scale = np.abs(g.u).max() / g.dx
    print(f"  max |div v| = {np.abs(d).max():.3e}   (velocity scale {scale:.2e},"
          f" cg {sol.iters} its, res {sol.residual:.1e})")
    assert np.abs(d).max() < 1e-6 * scale


def check_no_slip():
    """Faces covered by the body must carry exactly the rigid velocity."""
    rng = np.random.default_rng(1)
    g = MACGrid(48, 48, 1 / 48, rho=1.0)
    g.u = rng.normal(size=g.u.shape)
    g.v = rng.normal(size=g.v.shape)
    body = FreePlate((0.5, 0.5), 0.15, 0.03, theta0=0.4)
    sol = unified_projection(g, body, np.array([0.3, -0.7, 2.0]), 1e-3)
    mk = Masks(g, body.occupancy(g))
    Ju, Jv = body.jacobian(g)
    eu = np.abs(g.u[mk.body_u] - Ju[mk.body_u] @ sol.vr).max()
    ev = np.abs(g.v[mk.body_v] - Jv[mk.body_v] @ sol.vr).max()
    print(f"  max no-slip violation = {max(eu, ev):.3e}")
    assert max(eu, ev) < 1e-12


def check_pendulum_in_vacuum():
    """With a near-massless fluid the blade must reduce to the ideal pendulum."""
    g = MACGrid(16, 16, 1 / 16, rho=1e-8, nu=0.0)
    body = PendulumPaddle((0.5, 0.8), 0.05, 0.45, 0.02, rho_s=8.0,
                          theta0=0.10, g=9.81)
    sim = Simulation(g, body, dt=5e-4)
    ts, th = [], []
    for _ in range(9000):
        sim.step()
        ts.append(sim.t)
        th.append(body.theta)
    th, ts = np.array(th), np.array(ts)
    # small-angle period of a compound pendulum: T = 2 pi sqrt(I / (m g d))
    T_exact = 2 * np.pi * np.sqrt(body.I / (body.mass * body.g * body.sc))
    s = np.sign(th)
    cross = ts[:-1][(s[:-1] < 0) & (s[1:] >= 0)]
    T_num = np.mean(np.diff(cross)) if len(cross) > 2 else np.nan
    err = abs(T_num - T_exact) / T_exact
    print(f"  period: numeric {T_num:.4f}s vs analytic {T_exact:.4f}s "
          f"({100*err:.2f}% error)")
    assert err < 0.02


def _drop(ratio, mode, nsteps=140, dt=1.5e-3):
    g = MACGrid(64, 64, 1 / 64, rho=1.0, nu=1e-4)
    plate = FreePlate((0.5, 0.5), 0.10, 0.05, rho_s=ratio, g=9.81)
    sim = Simulation(g, plate, dt=dt, gravity=9.81, gravity_mode=mode)
    for _ in range(nsteps):
        sim.step()
    return plate.c[1] - 0.5, sim


def check_buoyancy():
    """Nothing in this code models buoyancy: a cork may only rise because the
    pressure multiplier says so."""
    res = {}
    for ratio in (0.5, 1.0, 2.0):
        for mode in ("hydrostatic", "explicit"):
            res[ratio, mode] = _drop(ratio, mode)[0]
        print(f"  rho_s/rho_f={ratio:>3}:  dy_hydrostatic={res[ratio,'hydrostatic']:+.4f}"
              f"   dy_explicit={res[ratio,'explicit']:+.4f}")
    assert res[0.5, "hydrostatic"] > 0.01, "a cork must rise"
    assert res[2.0, "hydrostatic"] < -0.01, "a stone must sink"
    assert abs(res[1.0, "explicit"]) < 0.003, "neutral density must hover"
    for ratio in (0.5, 2.0):
        a, b = res[ratio, "hydrostatic"], res[ratio, "explicit"]
        rel = abs(a - b) / abs(a)
        print(f"    modes agree to {100*rel:.1f}% at rho_s/rho_f={ratio}")
        assert rel < 0.2, "the two gravity treatments must give the same physics"


def check_initial_acceleration():
    """Released from rest the plate must obey

        (A + m_added) a = -(rho_s - rho_f) V g

    with A the inertia the solver carries explicitly and m_added > 0 the
    classical added mass, which nothing in the code models -- it can only come
    from the dynamic part of the pressure multiplier."""
    for mode in ("hydrostatic", "explicit"):
        g = MACGrid(96, 96, 1 / 96, rho=1.0, nu=0.0)
        plate = FreePlate((0.5, 0.5), 0.12, 0.06, rho_s=3.0, g=9.81)
        sim = Simulation(g, plate, dt=1e-4, gravity=9.81, gravity_mode=mode)
        sim.step()
        a_num = sim.vr[1] / sim.dt
        A = sim.last.A[1, 1]
        V = (2 * plate.a) * (2 * plate.b)
        F = -(plate.rho_s - g.rho) * V * plate.g
        a_noadd = F / A
        m_added = F / a_num - A
        # a slender plate pushed broadside: m_added ~ pi rho a^2 per unit span,
        # inflated here because the box walls are only ~4 half-widths away
        m_ref = np.pi * g.rho * plate.a ** 2
        print(f"  [{mode:11}] a(0) = {a_num:+.4f} m/s^2 "
              f"(no added mass: {a_noadd:+.4f}, free fall: {-plate.g:+.2f}); "
              f"implied m_added = {m_added:.4f} vs pi.rho.a^2 = {m_ref:.4f}")
        assert a_noadd < a_num < 0, "added mass must slow the plate, not reverse it"
        assert m_added > 0.5 * m_ref, "added mass is implausibly small"
        if mode == "hydrostatic":
            a_hydro = a_num
    assert abs(a_num - a_hydro) / abs(a_hydro) < 0.05, \
        "the two gravity treatments must agree"


def check_energy_decays():
    """Viscosity and vortex shedding may only take energy out of the system."""
    g = MACGrid(64, 64, 1 / 64, rho=1.0, nu=2e-4)
    body = PendulumPaddle((0.5, 0.78), 0.06, 0.42, 0.02, rho_s=6.0,
                          theta0=1.2, g=9.81)
    sim = Simulation(g, body, dt=1.2e-3)
    sim.run(500, record_every=10, verbose=False)
    h = sim.history
    E = h["ke_fluid"] + h["ke_body"] + h["pe_body"]
    E -= E.min()
    growth = np.max(np.diff(E)) / max(E[0] - E[-1], 1e-12)
    print(f"  total energy {E[0]:.4f} -> {E[-1]:.4f}, "
          f"largest single-frame rise {100*growth:.2f}% of the loss")
    assert E[-1] < E[0]
    assert growth < 0.05


def main():
    checks = [("divergence free", check_divergence_free),
              ("no-slip constraint", check_no_slip),
              ("pendulum limit (L^r alone)", check_pendulum_in_vacuum),
              ("initial acceleration", check_initial_acceleration),
              ("buoyancy emerges from p", check_buoyancy),
              ("energy never grows", check_energy_decays)]
    for name, fn in checks:
        print(f"[ ] {name}")
        fn()
        print(f"[x] {name}\n")
    print("all checks passed")


if __name__ == "__main__":
    main()
