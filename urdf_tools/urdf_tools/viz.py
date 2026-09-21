"""Visualization: 2D floor plan, 3D wireframe, kinematic tree."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .geometry import (
    box_wireframe,
    cylinder_wireframe,
    pose_to_matrix,
    sphere_wireframe,
    world_transform,
)
from .parser import Geometry, Robot

# ── helpers ───────────────────────────────────────────────────────────────────


def _link_world_blocks(
    robot: Robot, use_collision: bool = True
) -> list[tuple[np.ndarray, Geometry]]:
    """Return (world_T 4×4, Geometry) for every geometry block in the robot."""
    parent_map = robot.parent_map()
    joint_T = {j.child: pose_to_matrix(j.pose.xyz, j.pose.rpy) for j in robot.joints}

    results: list[tuple[np.ndarray, Geometry]] = []
    for link in robot.links:
        T_world = world_transform(link.name, parent_map, joint_T)
        blocks = link.collisions if use_collision else link.visuals
        if not blocks:
            blocks = link.visuals if use_collision else link.collisions
        for block in blocks:
            if block.geometry is None or block.geometry.type == "mesh":
                continue
            T_local = pose_to_matrix(block.pose.xyz, block.pose.rpy)
            results.append((T_world @ T_local, block.geometry))
    return results


# ── 2D floor plan ─────────────────────────────────────────────────────────────


def _xy_footprint(T: np.ndarray, geom: Geometry) -> list[tuple[str, object]]:
    """Top-down XY footprint of a geometry in world frame."""
    patches: list[tuple[str, object]] = []
    if geom.type == "box":
        sx, sy = geom.size[0] / 2, geom.size[1] / 2
        local = np.array(
            [
                [-sx, -sy, 0, 1],
                [sx, -sy, 0, 1],
                [sx, sy, 0, 1],
                [-sx, sy, 0, 1],
            ]
        ).T
        world_xy = (T @ local)[:2, :].T  # (4, 2)
        patches.append(("polygon", world_xy))
    elif geom.type in ("cylinder", "sphere"):
        cx, cy = T[0, 3], T[1, 3]
        patches.append(("circle", (cx, cy, geom.radius)))
    return patches


def plot_floor_plan(
    robot: Robot,
    *,
    scan_ranges: np.ndarray | None = None,
    scan_angles: np.ndarray | None = None,
    sensor_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ax: plt.Axes | None = None,
    save: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Draw top-down 2D floor plan with optional LiDAR scan overlay."""
    fig, ax = plt.subplots(figsize=(10, 7)) if ax is None else (ax.get_figure(), ax)
    ax.set_facecolor("#f0f2f5")
    ax.set_aspect("equal")

    for T, geom in _link_world_blocks(robot):
        for kind, params in _xy_footprint(T, geom):
            if kind == "polygon":
                ax.add_patch(
                    plt.Polygon(
                        params,
                        fc="#94a3b8",
                        ec="#475569",
                        lw=0.8,
                        alpha=0.65,
                    )
                )
            elif kind == "circle":
                cx, cy, r = params
                ax.add_patch(
                    plt.Circle(
                        (cx, cy),
                        r,
                        fc="#94a3b8",
                        ec="#475569",
                        lw=0.8,
                        alpha=0.65,
                    )
                )

    if scan_ranges is not None and scan_angles is not None:
        sx, sy, sth = sensor_pose
        rmax = float(np.max(scan_ranges)) * 1.05
        hit = scan_ranges < rmax
        ex = sx + scan_ranges * np.cos(sth + scan_angles)
        ey = sy + scan_ranges * np.sin(sth + scan_angles)
        for i in range(0, len(scan_angles), 4):
            if hit[i]:
                ax.plot([sx, ex[i]], [sy, ey[i]], color="#ef4444", lw=0.15, alpha=0.2)
        ax.scatter(
            ex[hit], ey[hit], s=2, color="#ef4444", zorder=4, label=f"hits {hit.sum()}"
        )
        ax.scatter([sx], [sy], s=70, color="#1d4ed8", zorder=5, label="sensor")
        ax.legend(fontsize=8, loc="upper right")

    ax.autoscale_view()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Floor Plan — {robot.name}", fontweight="bold")
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
        print(f"  Saved → {save}")
    if show:
        plt.show()
    return fig


# ── 3D wireframe ──────────────────────────────────────────────────────────────


def plot_3d(
    robot: Robot,
    *,
    cloud: np.ndarray | None = None,
    ax=None,
    save: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Draw 3D wireframe for all link geometries."""
    fig = plt.figure(figsize=(10, 8)) if ax is None else ax.get_figure()
    ax3 = fig.add_subplot(111, projection="3d") if ax is None else ax
    ax3.set_facecolor("#f0f2f5")

    for T, geom in _link_world_blocks(robot):
        if geom.type == "box":
            corners, edges = box_wireframe(geom.size)
        elif geom.type == "cylinder":
            corners, edges = cylinder_wireframe(geom.radius, geom.length)
        elif geom.type == "sphere":
            corners, edges = sphere_wireframe(geom.radius)
        else:
            continue
        h = np.column_stack([corners, np.ones(len(corners))])
        w = (T @ h.T).T[:, :3]
        for i, j in edges:
            p0, p1 = w[i], w[j]
            ax3.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color="#334155",
                lw=0.6,
                alpha=0.8,
            )

    if cloud is not None and len(cloud):
        ax3.scatter(
            cloud[:, 0],
            cloud[:, 1],
            cloud[:, 2],
            s=0.5,
            c=cloud[:, 2],
            cmap="plasma",
            alpha=0.5,
        )

    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")
    ax3.set_title(f"3D View — {robot.name}", fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
        print(f"  Saved → {save}")
    if show:
        plt.show()
    return fig


# ── kinematic tree ────────────────────────────────────────────────────────────

_JOINT_COLORS = {
    "fixed": "#6366f1",
    "continuous": "#10b981",
    "revolute": "#f59e0b",
    "prismatic": "#ef4444",
    "floating": "#8b5cf6",
    "planar": "#ec4899",
}


def plot_kinematic_tree(
    robot: Robot,
    *,
    ax: plt.Axes | None = None,
    save: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Draw kinematic tree as a BFS box-and-arrow diagram."""
    children = robot.children_map()
    root = robot.root_link() or robot.links[0].name

    # BFS layout
    by_depth: dict[int, list[str]] = {}
    queue = [(root, 0)]
    visited: set[str] = set()
    while queue:
        name, depth = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)
        by_depth.setdefault(depth, []).append(name)
        for kid in children.get(name, []):
            queue.append((kid, depth + 1))
    for lnk in robot.links:
        if lnk.name not in visited:
            by_depth.setdefault(0, []).append(lnk.name)

    final_pos: dict[str, tuple[float, float]] = {}
    for d, names in by_depth.items():
        n = len(names)
        for i, name in enumerate(names):
            final_pos[name] = (d, i - (n - 1) / 2.0)

    max_d = max(by_depth.keys()) if by_depth else 0
    w = max(8.0, (max_d + 1) * 2.5)
    h = max(5.0, len(robot.links) * 0.7 + 2)
    fig, ax_t = plt.subplots(figsize=(w, h)) if ax is None else (ax.get_figure(), ax)
    ax_t.set_facecolor("#f8f9fb")
    ax_t.set_aspect("equal")
    ax_t.axis("off")

    for j in robot.joints:
        if j.parent not in final_pos or j.child not in final_pos:
            continue
        xp, yp = final_pos[j.parent]
        xc, yc = final_pos[j.child]
        col = _JOINT_COLORS.get(j.type, "#888")
        ax_t.annotate(
            "",
            xy=(xc - 0.42, yc),
            xytext=(xp + 0.42, yp),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4),
        )
        ax_t.text(
            (xp + xc) / 2,
            (yp + yc) / 2 + 0.08,
            j.name,
            fontsize=6.5,
            color=col,
            ha="center",
            va="bottom",
            style="italic",
        )

    for name, (x, y) in final_pos.items():
        is_root = name == root
        ax_t.add_patch(
            mpatches.FancyBboxPatch(
                (x - 0.40, y - 0.22),
                0.80,
                0.44,
                boxstyle="round,pad=0.05",
                lw=1.2,
                edgecolor="#94a3b8",
                facecolor="#1d4ed8" if is_root else "#e2e8f0",
            )
        )
        ax_t.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if is_root else "#1e293b",
            fontweight="bold" if is_root else "normal",
        )

    legend_entries = [
        mpatches.Patch(color=c, label=t)
        for t, c in _JOINT_COLORS.items()
        if any(j.type == t for j in robot.joints)
    ]
    if legend_entries:
        ax_t.legend(
            handles=legend_entries,
            fontsize=7,
            loc="upper right",
            title="Joint type",
            title_fontsize=7.5,
            framealpha=0.85,
        )

    yvals = [y for _, y in final_pos.values()]
    ax_t.set_xlim(-0.6, max_d + 0.6)
    ax_t.set_ylim(min(yvals) - 0.6, max(yvals) + 0.6)
    ax_t.set_title(
        f"Kinematic Tree — {robot.name}\n"
        f"{len(robot.links)} links · {len(robot.joints)} joints",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
        print(f"  Saved → {save}")
    if show:
        plt.show()
    return fig
