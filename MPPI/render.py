"""Render the MPPI cartpole run: cart-pole animation + rollout fan + cost/angle
traces, saved as mp4 and a compressed gif."""

import os
import shutil
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle, Circle

BG = "#0b0e14"
FG = "#e6e9ef"
MUTED = "#8b93a7"
ACCENT = "#57c7ff"
ACCENT2 = "#ff7b72"
ACCENT3 = "#7ee787"
GOLD = "#ffd866"


def use_dark_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
        "axes.edgecolor": MUTED, "axes.labelcolor": FG, "text.color": FG,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#20242e",
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    })


def mp4_to_gif(mp4, gif, fps=15, width=720):
    if shutil.which("ffmpeg") is None:
        return None
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,"
         f"split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
         f"[s1][p]paletteuse=dither=bayer:bayer_scale=4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", vf,
          "-loop", "0", gif]
    subprocess.run(cmd, check=True)
    print(f"    wrote {gif}  ({os.path.getsize(gif)/1e6:.1f} MB)")
    return gif


def animate(result, out_stem, fps_render=30, speed=1.0, dpi=105,
           gif_fps=15, gif_width=720, tail_frames=None):
    params = result["params"]
    hist_x = result["hist_x"]
    hist_u = result["hist_u"]
    dt = result["dt"]
    snaps = result["plan_snapshots"]
    steps = len(hist_u)
    l = params.l

    n_frames = steps if tail_frames is None else min(steps, tail_frames)
    start = steps - n_frames
    frame_dt = dt / speed
    fps = min(fps_render, max(1, round(1.0 / frame_dt)))
    stride = max(1, round((1.0 / fps) / dt))
    idxs = list(range(start, steps, stride))

    t_arr = np.arange(steps + 1) * dt
    theta = hist_x[:, 2]
    dtheta = ((theta - np.pi + np.pi) % (2 * np.pi)) - np.pi  # error to upright
    pos = hist_x[:, 0]

    snap_ticks = np.array([s[0] for s in snaps])

    def nearest_snap(k):
        j = int(np.argmin(np.abs(snap_ticks - k)))
        return snaps[j]

    fig = plt.figure(figsize=(11.4, 6.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0],
                          left=0.045, right=0.965, top=0.88, bottom=0.09,
                          wspace=0.24, hspace=0.48)
    axc = fig.add_subplot(gs[:, 0])
    rail = params.rail_limit
    axc.set_xlim(-rail - 0.4, rail + 0.4)
    axc.set_ylim(-1.15 * l - 0.3, 1.15 * l + 0.35)
    axc.set_aspect("equal")
    axc.set_xticks([]); axc.set_yticks([])
    for s in axc.spines.values():
        s.set_color("#2a3040")
    axc.axhline(0, color="#2a3040", lw=2, zorder=1)
    axc.axvline(-rail, color="#3a4050", lw=1, ls="--", zorder=1)
    axc.axvline(rail, color="#3a4050", lw=1, ls="--", zorder=1)

    fan_lines = [axc.plot([], [], color=ACCENT, lw=0.7, alpha=0.15, zorder=2)[0]
                for _ in range(120)]

    cart_w, cart_h = 0.32, 0.18
    cart_patch = Rectangle((-cart_w / 2, -cart_h / 2), cart_w, cart_h,
                           facecolor="#f2f4f8", edgecolor=GOLD, linewidth=1.4,
                           zorder=5)
    axc.add_patch(cart_patch)
    pole_line, = axc.plot([], [], color="#f2f4f8", lw=4, solid_capstyle="round",
                          zorder=6)
    bob = Circle((0, 0), 0.05, facecolor=ACCENT2, edgecolor="none", zorder=7)
    axc.add_patch(bob)
    pivot_dot, = axc.plot([], [], "o", color=GOLD, ms=4, zorder=8)

    clock = axc.text(0.02, 0.96, "", transform=axc.transAxes, va="top",
                     fontsize=11, color=FG, family="monospace")
    axc.text(0.02, 0.04, "faint lines: live MPPI rollouts (subset)",
             transform=axc.transAxes, fontsize=8, color=MUTED)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax1.plot(t_arr[1:], np.degrees(dtheta[1:]), color="#3a4050", lw=1.0)
    l1, = ax1.plot([], [], color=ACCENT, lw=1.8)
    ax1.set_ylim(-185, 185)
    ax1.set_ylabel("angle error [deg]")
    ax1.set_title("pole angle vs. upright", loc="left", color=FG)
    ax1.grid(True, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax2.axhline(rail, color="#3a4050", lw=0.8, ls=":")
    ax2.axhline(-rail, color="#3a4050", lw=0.8, ls=":")
    ax2.plot(t_arr[1:], pos[1:], color="#3a4050", lw=1.0)
    l2, = ax2.plot([], [], color=ACCENT3, lw=1.8)
    ax2.set_ylim(-rail - 0.3, rail + 0.3)
    ax2.set_ylabel("cart pos [m]")
    ax2.set_title("cart position on the rail", loc="left", color=FG)
    ax2.grid(True, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    ax3 = fig.add_subplot(gs[2, 1])
    ax3.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax3.plot(t_arr[:-1], hist_u, color="#3a4050", lw=1.0)
    l3, = ax3.plot([], [], color=ACCENT2, lw=1.6)
    ax3.set_ylim(-params.force_limit * 1.1, params.force_limit * 1.1)
    ax3.set_ylabel("force u [N]")
    ax3.set_xlabel("time [s]")
    ax3.set_title(r"applied force $u_0$ (receding horizon)", loc="left", color=FG)
    ax3.grid(True, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)

    fig.suptitle("MPPI (Model Predictive Path Integral) — cartpole swing-up",
                x=0.045, ha="left", fontsize=13, color=FG, weight="bold")
    fig.text(0.045, 0.925,
             rf"$K$={result['samples']}  samples, $H$={result['horizon']}  "
             rf"steps, {'GPU' if result['device']=='cuda' else 'CPU'}"
             rf"  ({1000*result['total_time']/steps:.1f} ms / replan)",
             ha="left", fontsize=10, color=MUTED)

    def draw_pole(k, ax_target_line, ax_target_bob, ax_target_pivot):
        p, _, th, _ = hist_x[k]
        tip_x = p + l * np.sin(th)
        tip_y = -l * np.cos(th)
        ax_target_line.set_data([p, tip_x], [0, tip_y])
        ax_target_bob.center = (tip_x, tip_y)
        ax_target_pivot.set_data([p], [0])
        cart_patch.set_x(p - cart_w / 2)

    def update(frame_i):
        k = idxs[frame_i]
        draw_pole(k, pole_line, bob, pivot_dot)
        clock.set_text(f"t = {k*dt:5.2f} s")

        sk, traj, w = nearest_snap(k)
        wn = w / (w.max() + 1e-12)
        for i, ln in enumerate(fan_lines):
            if i < traj.shape[0]:
                p_t = traj[i, :, 0]
                th_t = traj[i, :, 2]
                tip_x = p_t + l * np.sin(th_t)
                tip_y = -l * np.cos(th_t)
                ln.set_data(tip_x, tip_y)
                ln.set_alpha(0.05 + 0.35 * wn[i])
            else:
                ln.set_data([], [])

        l1.set_data(t_arr[1:k + 1], np.degrees(dtheta[1:k + 1]))
        l2.set_data(t_arr[1:k + 1], pos[1:k + 1])
        l3.set_data(t_arr[:k], hist_u[:k])
        return []

    writer = FFMpegWriter(fps=fps, bitrate=3600, metadata={"title": "mppi_cartpole"})
    mp4 = out_stem + ".mp4"
    os.makedirs(os.path.dirname(mp4) or ".", exist_ok=True)
    with writer.saving(fig, mp4, dpi=dpi):
        for i in range(len(idxs)):
            update(i)
            writer.grab_frame()
    print(f"    wrote {mp4}  ({len(idxs)} frames @ {fps} fps)")
    plt.close(fig)
    mp4_to_gif(mp4, out_stem + ".gif", fps=gif_fps, width=gif_width)
    return mp4
