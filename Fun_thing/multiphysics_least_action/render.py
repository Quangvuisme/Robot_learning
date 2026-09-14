"""Figure / movie helpers.  mp4 via matplotlib's ffmpeg writer, gif via an
ffmpeg palette pass (much smaller and cleaner than PillowWriter)."""

import os
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap

BG = "#0b0e14"
FG = "#e6e9ef"
MUTED = "#8b93a7"
ACCENT = "#57c7ff"
ACCENT2 = "#ff7b72"
ACCENT3 = "#7ee787"

VORT_CMAP = LinearSegmentedColormap.from_list(
    "vort", ["#1b3a63", "#2f6f9f", "#0b0e14", "#0b0e14", "#a8412f", "#e8613c"])

DYE_CMAP = LinearSegmentedColormap.from_list(
    "dye", [(0, (1, 1, 1, 0.0)), (1, (1, 1, 1, 0.85))])


def use_dark_style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": FG,
        "text.color": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": "#20242e",
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": False,
    })


def save_movie(fig, update, nframes, stem, fps=30, dpi=110, gif=True,
               gif_fps=15, gif_width=640):
    """Render `nframes` through `update(i)` into <stem>.mp4 (+ <stem>.gif)."""
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    mp4 = stem + ".mp4"
    writer = FFMpegWriter(fps=fps, bitrate=3600,
                          metadata={"title": os.path.basename(stem)})
    with writer.saving(fig, mp4, dpi=dpi):
        for i in range(nframes):
            update(i)
            writer.grab_frame()
    print(f"    wrote {mp4}")
    if gif:
        mp4_to_gif(mp4, stem + ".gif", fps=gif_fps, width=gif_width)
    return mp4


def mp4_to_gif(mp4, gif, fps=15, width=640):
    if shutil.which("ffmpeg") is None:
        print("    (ffmpeg not found, skipping gif)")
        return None
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,"
          f"split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
          f"[s1][p]paletteuse=dither=bayer:bayer_scale=4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", vf,
           "-loop", "0", gif]
    subprocess.run(cmd, check=True)
    size = os.path.getsize(gif) / 1e6
    print(f"    wrote {gif}  ({size:.1f} MB)")
    return gif


def draw_body(ax, poly, face="#f2f4f8", edge="#ffd866", lw=1.4, z=5):
    patch = plt.Polygon(poly, closed=True, facecolor=face, edgecolor=edge,
                        linewidth=lw, zorder=z)
    ax.add_patch(patch)
    return patch


def style_timeaxis(ax, ylabel, tmax):
    ax.set_xlim(0, tmax)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
