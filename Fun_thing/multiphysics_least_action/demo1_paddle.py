"""Demo 1 --- the headline: a hinged blade dropped into still water.

Both physics are live and neither knows about the other except through the
multiplier.  The blade falls, shreds the fluid into a vortex pair, and the
fluid then pushes back hard enough to stop it well short of where a dry
pendulum would swing.  Nothing in the code says "drag".

    python -m multiphysics_least_action.demo1_paddle [--quick]
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from .bodies import PendulumPaddle
from .fluid import MACGrid
from .render import (ACCENT, ACCENT2, ACCENT3, DYE_CMAP, FG, MUTED, VORT_CMAP,
                     draw_body, save_movie, style_timeaxis, use_dark_style)
from .sim import Simulation

OUT = os.path.join(os.path.dirname(__file__), "out")
PIVOT = (0.5, 0.55)


def build(nx=128):
    grid = MACGrid(nx, nx, 1.0 / nx, rho=1.0, nu=1.5e-5)
    # horizontal dye bands: they make the stirring legible
    band = np.floor(grid.Yc * 11.0) % 2 == 0
    grid.dye[band] = 1.0

    # the blade sweeps a disc of radius s1 about the pivot; keep that disc clear
    # of the walls, or it seals a pocket of incompressible fluid and locks up
    blade = PendulumPaddle(pivot=PIVOT, s0=0.05, s1=0.38, hw=0.015,
                           rho_s=6.0, g=9.81, theta0=2.4)
    return grid, blade


def dry_pendulum(blade, dt, n):
    """The same blade with no fluid at all, for the comparison trace."""
    th, om = blade.theta, 0.0
    out = []
    for _ in range(n):
        om += dt * (-blade.mass * blade.g * blade.sc * np.sin(th)) / blade.I
        th += dt * om
        out.append(th)
    return np.array(out)


def main(quick=False):
    use_dark_style()
    nx = 64 if quick else 128
    nsteps = 500 if quick else 2400
    every = 10 if quick else 6
    dt = 2e-3 if quick else 1e-3

    grid, blade = build(nx)
    theta0 = blade.theta
    sim = Simulation(grid, blade, dt=dt, gravity=9.81)

    print(f"demo1: {nx}x{nx} grid, {nsteps} steps, dt={dt}")
    frames = sim.run(nsteps, record_every=every,
                     frame_cb=lambda s: (s.grid.vorticity().astype(np.float32),
                                         s.grid.dye.copy().astype(np.float32),
                                         s.body.polygon()))
    h = sim.history

    # reference: the same blade swinging in vacuum
    ref = PendulumPaddle(pivot=PIVOT, s0=0.05, s1=0.38, hw=0.015,
                         rho_s=6.0, g=9.81, theta0=theta0)
    th_dry = dry_pendulum(ref, dt, nsteps)[::every][:len(h["t"])]

    vmax = float(np.percentile(np.abs(frames[len(frames) // 2][0]), 99.5)) or 1.0
    tmax = float(h["t"][-1])
    theta = h["q"][:, 0]
    torque = h["wrench"][:, 0]
    E_f = h["ke_fluid"]
    E_b = h["ke_body"] + h["pe_body"] - np.min(h["pe_body"])

    # ------------------------------------------------------------- figure
    fig = plt.figure(figsize=(11.2, 6.3))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.32, 1.0],
                          left=0.035, right=0.965, top=0.885, bottom=0.085,
                          wspace=0.22, hspace=0.42)
    axf = fig.add_subplot(gs[:, 0])
    axf.set_xticks([]); axf.set_yticks([])
    axf.set_aspect("equal")
    axf.set_xlim(0, grid.Lx); axf.set_ylim(0, grid.Ly)
    for s in axf.spines.values():
        s.set_color("#2a3040")

    im_w = axf.imshow(frames[0][0].T, origin="lower", extent=[0, 1, 0, 1],
                      cmap=VORT_CMAP, vmin=-vmax, vmax=vmax, interpolation="bilinear")
    im_d = axf.imshow(frames[0][1].T, origin="lower", extent=[0, 1, 0, 1],
                      cmap=DYE_CMAP, vmin=0, vmax=1.0, interpolation="bilinear")
    poly = draw_body(axf, frames[0][2])
    axf.plot(*blade.pivot, "o", color="#ffd866", ms=5, zorder=6)
    clock = axf.text(0.022, 0.965, "", transform=axf.transAxes, va="top",
                     fontsize=11, color=FG, family="monospace")
    axf.text(0.022, 0.035, "vorticity + dye", transform=axf.transAxes,
             fontsize=8, color=MUTED)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(h["t"], np.degrees(th_dry), color=MUTED, lw=1.1, ls="--",
             label="in vacuum")
    l1, = ax1.plot([], [], color=ACCENT, lw=1.8, label="in fluid")
    ax1.set_ylim(np.degrees(min(th_dry.min(), theta.min())) - 12,
                 np.degrees(max(th_dry.max(), theta.max())) + 12)
    style_timeaxis(ax1, r"joint angle  $\theta$  [deg]", tmax)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.set_title(r"$\mathcal{L}^r$ : the blade", loc="left", color=FG)

    ax2 = fig.add_subplot(gs[1, 1])
    l2, = ax2.plot([], [], color=ACCENT2, lw=1.6)
    ax2.axhline(0, color=MUTED, lw=0.7)
    pad = 0.12 * (np.abs(torque).max() + 1e-9)
    ax2.set_ylim(torque.min() - pad, torque.max() + pad)
    style_timeaxis(ax2, r"$-G^{\!\top}\!\lambda$   [N$\cdot$m]", tmax)
    ax2.set_title("coupling torque from the multiplier", loc="left", color=FG)

    ax3 = fig.add_subplot(gs[2, 1])
    l3a, = ax3.plot([], [], color=ACCENT3, lw=1.5, label="fluid KE")
    l3b, = ax3.plot([], [], color=ACCENT, lw=1.5, label="blade KE+PE")
    ax3.set_ylim(0, 1.08 * max(E_f.max(), E_b.max()))
    style_timeaxis(ax3, "energy [J]", tmax)
    ax3.set_xlabel("time [s]")
    ax3.legend(loc="upper right", fontsize=8, ncol=2)
    ax3.set_title("the blade's energy becomes the fluid's", loc="left", color=FG)

    fig.suptitle("Multiphysics as Optimization  —  one least-action problem, "
                 "solved once per step", x=0.035, ha="left", fontsize=13,
                 color=FG, weight="bold")
    fig.text(0.035, 0.925,
             r"min $\int \mathcal{L}^f(q^f,v^f) + \mathcal{L}^r(q^r,v^r) + F^{\!\top}q^f\,dt$"
             r"   s.t.  $\nabla\!\cdot v^f = 0$,   $c(v^f,v^r)=0$",
             ha="left", fontsize=10, color=MUTED)

    def update(i):
        w, d, pg = frames[i]
        im_w.set_data(w.T)
        im_d.set_data(d.T)
        poly.set_xy(pg)
        clock.set_text(f"t = {h['t'][i]:5.2f} s")
        k = i + 1
        l1.set_data(h["t"][:k], np.degrees(theta[:k]))
        l2.set_data(h["t"][:k], torque[:k])
        l3a.set_data(h["t"][:k], E_f[:k])
        l3b.set_data(h["t"][:k], E_b[:k])

    save_movie(fig, update, len(frames), os.path.join(OUT, "demo1_paddle"),
               fps=30, dpi=100, gif_fps=16, gif_width=780)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
