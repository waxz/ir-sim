"""Generate per-model animations showing robot trajectory and wheel behavior.

Each animation shows:
  - Top-left: robot trajectory (GT vs FK vs Actual)
  - Top-right: wheel angular velocities over time
  - Bottom-left: steering angles (delta_actual vs delta_cmd, for steer models)
  - Bottom-right: FK error and Cmd tracking error over time

Usage::

    python animate_wheels.py              # generate GIFs for all 6 models
    python animate_wheels.py --models diff mecanum  # specific models
    python animate_wheels.py --fps 20     # frame rate (default 15)
    python animate_wheels.py --step 5     # animate every Nth step (default 5)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

matplotlib.use("Agg")

DATA_FILE = Path(__file__).parent.parent.parent / "sim_data.json"
OUT_DIR = Path(__file__).parent / "animations"

MODEL_KEYS = [
    "diff",
    "mecanum",
    "ackermann",
    "forklift",
    "dual_steer",
    "quad_steer",
]

# Wheel geometries (body-frame positions, relative units) for each model
WHEEL_POSITIONS = {
    "diff": [(-0.08, 0.08), (-0.08, -0.08)],  # left, right
    "mecanum": [(0.15, 0.15), (0.15, -0.15), (-0.15, 0.15), (-0.15, -0.15)],
    "ackermann": [(0.15, 0.12), (0.15, -0.12), (-0.15, 0.12), (-0.15, -0.12)],
    "forklift": [(0.2, 0.12), (0.2, -0.12), (-0.2, 0.0)],
    "dual_steer": [(0.4, 0.0), (-0.4, 0.0)],  # FC, RC
    "quad_steer": [(0.15, 0.15), (0.15, -0.15), (-0.15, 0.15), (-0.15, -0.15)],
}

WHEEL_COLORS = {
    "diff": ["#4C72B0", "#DD8452"],
    "mecanum": ["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    "ackermann": ["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    "forklift": ["#4C72B0", "#DD8452", "#55A868"],
    "dual_steer": ["#4C72B0", "#DD8452"],
    "quad_steer": ["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
}

STEER_MODELS = {"ackermann", "forklift", "dual_steer", "quad_steer"}


def load_data(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def draw_robot(ax, x, y, theta, model_key: str, scale: float = 1.0, alpha: float = 0.8):
    """Draw a simple robot body + wheel markers at given pose."""
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def rotate(lx, ly):
        return x + scale * (cos_t * lx - sin_t * ly), y + scale * (
            sin_t * lx + cos_t * ly
        )

    # Body rectangle
    hw, hl = 0.10 * scale, 0.12 * scale
    corners_body = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
    corners_rot = [rotate(cx, cy) for cx, cy in corners_body]
    poly = plt.Polygon(
        corners_rot,
        closed=True,
        fill=True,
        facecolor="#AAAACC",
        edgecolor="#444466",
        alpha=alpha,
        linewidth=1.0,
    )
    ax.add_patch(poly)

    # Direction arrow
    fx, fy = rotate(hl * 0.8, 0)
    ax.annotate(
        "",
        xy=(fx, fy),
        xytext=(x, y),
        arrowprops={"arrowstyle": "->", "color": "#222244", "lw": 1.5},
    )

    # Wheel squares
    for (wx, wy), col in zip(
        WHEEL_POSITIONS[model_key], WHEEL_COLORS[model_key], strict=True
    ):
        wsx, wsy = rotate(wx * scale * 0.7, wy * scale * 0.7)
        wheel = plt.Circle((wsx, wsy), 0.025 * scale, color=col, alpha=0.9, zorder=5)
        ax.add_patch(wheel)


def make_animation(model_key: str, mdata: dict, fps: int, step: int) -> Path:
    name = mdata["name"]
    times = mdata["times"]
    gt = np.array(mdata["gt_traj"])  # (N+1, 3)
    fk = np.array(mdata["enc_traj"])  # (N+1, 3)
    fk_err = np.array(mdata["fk_error"]) * 1000  # mm
    cmd_err = np.array(mdata["cmd_error"]) * 1000  # mm
    wheel_omega = mdata["wheel_omega"]
    wheel_delta = mdata.get("wheel_delta", {})
    wheel_delta_cmd = mdata.get("wheel_delta_cmd", {})
    cmd_labels = mdata["cmd_labels"]
    cmds = np.array(mdata["cmds"])

    wnames = list(wheel_omega.keys())
    colors = WHEEL_COLORS.get(model_key, plt.cm.tab10.colors[: len(wnames)])
    is_steer = model_key in STEER_MODELS

    # Subsample frames
    n_steps = len(times)
    frames = list(range(0, n_steps, step))

    # --- Figure layout ---------------------------------------------------------
    fig = plt.figure(figsize=(13, 8), dpi=90)
    fig.patch.set_facecolor("#1A1A2E")

    ax_traj = fig.add_axes([0.04, 0.45, 0.40, 0.50])  # top-left: trajectory
    ax_omega = fig.add_axes([0.55, 0.55, 0.42, 0.38])  # top-right: wheel omegas
    ax_delta = fig.add_axes([0.55, 0.08, 0.42, 0.38])  # bottom-right: delta or cmd
    ax_err = fig.add_axes([0.04, 0.08, 0.40, 0.30])  # bottom-left: error

    for ax in [ax_traj, ax_omega, ax_delta, ax_err]:
        ax.set_facecolor("#0D0D1A")
        ax.tick_params(colors="#BBBBCC", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    title_kw = {"color": "#DDDDFF", "fontsize": 9, "fontweight": "bold", "pad": 4}
    label_kw = {"color": "#AAAACC", "fontsize": 7}

    # --- Static full-run paths (dimmed) ----------------------------------------
    ax_traj.plot(gt[:, 0], gt[:, 1], lw=0.8, color="#4455AA", alpha=0.4, label="GT")
    ax_traj.plot(fk[:, 0], fk[:, 1], lw=0.8, color="#44AA88", alpha=0.4, label="FK enc")

    # Axis limits
    all_x = np.concatenate([gt[:, 0], fk[:, 0]])
    all_y = np.concatenate([gt[:, 1], fk[:, 1]])
    xm = (all_x.max() + all_x.min()) / 2
    ym = (all_y.max() + all_y.min()) / 2
    xr = max((all_x.max() - all_x.min()) * 0.6, 0.3)
    yr = max((all_y.max() - all_y.min()) * 0.6, 0.3)
    rng = max(xr, yr)
    ax_traj.set_xlim(xm - rng, xm + rng)
    ax_traj.set_ylim(ym - rng, ym + rng)
    ax_traj.set_aspect("equal")
    ax_traj.set_title(f"{name} - Trajectory", **title_kw)
    ax_traj.set_xlabel("x (m)", **label_kw)
    ax_traj.set_ylabel("y (m)", **label_kw)
    ax_traj.legend(
        fontsize=6,
        loc="upper left",
        facecolor="#0D0D1A",
        labelcolor="#CCCCEE",
        framealpha=0.7,
    )

    # --- Wheel omega panel (full-run lines, all wheels) -------------------------
    omega_arrays = {nm: np.array(wheel_omega[nm]) for nm in wnames}
    for nm, col in zip(wnames, colors, strict=False):
        ax_omega.plot(times, omega_arrays[nm], lw=0.8, color=col, alpha=0.5)
    ax_omega.set_title("Wheel ω (rad/s)", **title_kw)
    ax_omega.set_xlabel("t (s)", **label_kw)
    ax_omega.set_ylabel("ω (rad/s)", **label_kw)
    patches = [
        mpatches.Patch(color=col, label=nm)
        for nm, col in zip(wnames, colors, strict=False)
    ]
    ax_omega.legend(
        handles=patches,
        fontsize=6,
        loc="upper right",
        facecolor="#0D0D1A",
        labelcolor="#CCCCEE",
        framealpha=0.7,
    )

    # --- Delta / cmd panel ------------------------------------------------------
    if is_steer and wheel_delta:
        # Use the first key present in wheel_delta (may differ from wnames which tracks omega)
        first_steer = next(iter(wheel_delta))
        delta_arr = np.array(wheel_delta[first_steer])
        ax_delta.plot(
            times, delta_arr, lw=0.8, color="#FF4466", alpha=0.6, label="δ actual"
        )
        if wheel_delta_cmd and first_steer in wheel_delta_cmd:
            delta_cmd_arr = np.array(wheel_delta_cmd[first_steer])
            ax_delta.plot(
                times, delta_cmd_arr, lw=0.8, color="#FFAA44", alpha=0.5, label="δ cmd"
            )
        ax_delta.set_title(f"Steer angle delta - {first_steer} (rad)", **title_kw)
        ax_delta.set_ylabel("δ (rad)", **label_kw)
    else:
        # Show command profile instead
        for j, lbl in enumerate(cmd_labels):
            ax_delta.plot(times, cmds[:, j], lw=0.8, alpha=0.6, label=lbl)
        ax_delta.set_title("Command profile", **title_kw)
        ax_delta.set_ylabel("cmd", **label_kw)
    ax_delta.set_xlabel("t (s)", **label_kw)
    ax_delta.legend(
        fontsize=6,
        loc="upper right",
        facecolor="#0D0D1A",
        labelcolor="#CCCCEE",
        framealpha=0.7,
    )

    # --- Error panel ------------------------------------------------------------
    ax_err.plot(times, fk_err, lw=0.8, color="#44CCFF", alpha=0.5, label="FK enc error")
    ax_err.plot(
        times, cmd_err, lw=0.8, color="#FF6688", alpha=0.5, label="Cmd track error"
    )
    ax_err.set_title("Position Error", **title_kw)
    ax_err.set_xlabel("t (s)", **label_kw)
    ax_err.set_ylabel("error (mm)", **label_kw)
    ax_err.legend(
        fontsize=6,
        loc="upper left",
        facecolor="#0D0D1A",
        labelcolor="#CCCCEE",
        framealpha=0.7,
    )

    # --- Time indicator lines (updated per frame) ------------------------------
    t_line_omega = ax_omega.axvline(0, color="#FFFFFF", lw=0.8, alpha=0.6)
    t_line_delta = ax_delta.axvline(0, color="#FFFFFF", lw=0.8, alpha=0.6)
    t_line_err = ax_err.axvline(0, color="#FFFFFF", lw=0.8, alpha=0.6)

    # Moving robot marker on trajectory
    (robot_gt_dot,) = ax_traj.plot([], [], "o", ms=7, color="#7788FF", zorder=10)
    (robot_fk_dot,) = ax_traj.plot([], [], "s", ms=5, color="#44FF99", zorder=10)

    # Wheel omega current markers
    omega_dots = {
        nm: ax_omega.plot([], [], "o", ms=4, color=col, zorder=10)[0]
        for nm, col in zip(wnames, colors, strict=False)
    }

    # Time text
    time_text = ax_traj.text(
        0.02,
        0.96,
        "",
        transform=ax_traj.transAxes,
        color="#EEEEFF",
        fontsize=8,
        va="top",
    )

    fig.suptitle(
        f"IR-SIM Wheel Layout: {name}",
        color="#EEEEFF",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )

    def update(frame_idx: int):
        fi = frames[frame_idx]
        t_now = times[fi]

        # Trajectory dots
        robot_gt_dot.set_data([gt[fi + 1, 0]], [gt[fi + 1, 1]])
        robot_fk_dot.set_data([fk[fi + 1, 0]], [fk[fi + 1, 1]])

        # Time lines
        for tl in [t_line_omega, t_line_delta, t_line_err]:
            tl.set_xdata([t_now, t_now])

        # Wheel omega markers
        for nm in wnames:
            omega_dots[nm].set_data([t_now], [omega_arrays[nm][fi]])

        time_text.set_text(f"t = {t_now:.2f} s")
        return [
            robot_gt_dot,
            robot_fk_dot,
            t_line_omega,
            t_line_delta,
            t_line_err,
            time_text,
            *omega_dots.values(),
        ]

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=True
    )

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{model_key}.gif"
    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=90)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=MODEL_KEYS)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--step",
        type=int,
        default=8,
        help="Animate every Nth simulation step (default 8)",
    )
    args = parser.parse_args()

    print(f"Loading data from {DATA_FILE}")
    data = load_data(DATA_FILE)

    # data["models"] is a list of dicts; build a slug->dict map from model names
    name_to_key = {
        "Differential Drive": "diff",
        "Mecanum Drive": "mecanum",
        "Ackermann (Car-like)": "ackermann",
        "Forklift": "forklift",
        "Dual Steer": "dual_steer",
        "Quad Steer": "quad_steer",
    }
    models_by_key = {
        name_to_key[m["name"]]: m for m in data["models"] if m["name"] in name_to_key
    }

    paths = []
    for mk in args.models:
        if mk not in models_by_key:
            print(f"  [skip] {mk} not in data")
            continue
        print(f"Animating {mk} ...")
        p = make_animation(mk, models_by_key[mk], fps=args.fps, step=args.step)
        paths.append(p)

    print(f"\nDone. {len(paths)} GIF(s) written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
