"""Example 05 — Subscribe to sensor topics: real-time 2D and 3D LiDAR viewers.

Requires shmbridge installed and 06_irsim_bridge.py (or 04_pub_sensors.py) running.

Usage:
    python 05_sub_viewer.py                     # text print mode
    python 05_sub_viewer.py --live2d            # 2D real-time LiDAR (no world)
    python 05_sub_viewer.py --live3d            # 3D real-time LiDAR (no world)

    # With world and robot URDF overlays
    python 05_sub_viewer.py --live2d \\
        --world models/warehouse_world.urdf \\
        --robot models/robot_diff.urdf

    python 05_sub_viewer.py --live3d \\
        --world models/warehouse_world.urdf \\
        --robot models/robot_diff.urdf

    # Legacy alias
    python 05_sub_viewer.py --live              # same as --live2d
    python 05_sub_viewer.py --count 20          # stop after 20 polls (text mode)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from urdf_tools.pubsub import SensorSubscriber

SHM_NAME = "/urdf_tools_sensors"

# ── shared style ──────────────────────────────────────────────────────────────

_BG = "#0f172a"  # page / figure background  (slate-900)
_PANE = "#0d1526"  # 3-D pane  (slightly darker)
_GRID = "#1e293b"  # grid lines                 (slate-800)
_TEXT = "#94a3b8"  # axis labels / tick text    (slate-400)
_WORLD_FC = "#1e3a5f"  # world geometry fill
_WORLD_EC = "#2d5fa0"  # world geometry edge
_ROBOT_C = "#3b82f6"  # robot marker / wireframe  (blue-500)
_HDG_C = "#60a5fa"  # heading arrow             (blue-400)
_SCAN_CMAP = "plasma"
_RAY_C = "#ef4444"  # scan ray colour           (red-500)
_SCAN_HIT_EC = "#f97316"  # scan hit edge          (orange-500)


def _style_fig(fig) -> None:
    fig.patch.set_facecolor(_BG)


def _style_ax2d(ax, title: str = "") -> None:
    ax.set_facecolor(_BG)
    ax.set_aspect("equal")
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.set_xlabel("X (m)", color=_TEXT, fontsize=9)
    ax.set_ylabel("Y (m)", color=_TEXT, fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, lw=0.4, alpha=0.7, zorder=0)
    if title:
        ax.set_title(title, color="white", fontweight="bold", fontsize=10)


def _style_ax3d(ax3, title: str = "") -> None:
    ax3.set_facecolor(_PANE)
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    for a in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        a.pane.set_edgecolor(_GRID)
        a.line.set_color(_GRID)
    ax3.tick_params(colors=_TEXT, labelsize=7)
    ax3.set_xlabel("X (m)", color=_TEXT, fontsize=9)
    ax3.set_ylabel("Y (m)", color=_TEXT, fontsize=9)
    ax3.set_zlabel("Z (m)", color=_TEXT, fontsize=9)
    if title:
        ax3.set_title(title, color="white", fontweight="bold", fontsize=10)


# ── URDF overlay helpers ───────────────────────────────────────────────────────


def _draw_urdf_2d(ax, robot, *, alpha: float = 0.55, root_T=None) -> None:
    """Draw URDF geometry as 2-D patches onto *ax* (static, called once)."""
    import matplotlib.pyplot as plt

    from urdf_tools.viz import _link_world_blocks, _xy_footprint

    for T, geom, _rgba in _link_world_blocks(robot, root_T=root_T):
        for kind, params, _ in _xy_footprint(T, geom, _rgba):
            if kind == "polygon":
                ax.add_patch(
                    plt.Polygon(
                        params,
                        fc=_WORLD_FC,
                        ec=_WORLD_EC,
                        lw=0.8,
                        alpha=alpha,
                        zorder=2,
                    )
                )
            elif kind == "circle":
                cx, cy, r = params
                ax.add_patch(
                    plt.Circle(
                        (cx, cy),
                        r,
                        fc=_WORLD_FC,
                        ec=_WORLD_EC,
                        lw=0.8,
                        alpha=alpha,
                        zorder=2,
                    )
                )


def _world_bounds_2d(robot) -> tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) of the world URDF geometry."""
    from urdf_tools.viz import _link_world_blocks, _xy_footprint

    xs, ys = [], []
    for T, geom, rgba in _link_world_blocks(robot):
        for kind, params, _ in _xy_footprint(T, geom, rgba):
            if kind == "polygon":
                pts = np.asarray(params)
                xs.extend(pts[:, 0])
                ys.extend(pts[:, 1])
            elif kind == "circle":
                cx, cy, r = params
                xs += [cx - r, cx + r]
                ys += [cy - r, cy + r]
    if not xs:
        return -15, 15, -15, 15
    pad = max((max(xs) - min(xs)) * 0.05, 1.0)
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def _draw_urdf_3d_static(ax3, robot) -> list[np.ndarray]:
    """Draw world URDF wireframe onto ax3 once; return world-space corners.

    Edge colour is derived from each geometry's URDF material rgba so that
    different element types (walls, columns, racks, beams) are visually distinct.
    """
    from urdf_tools.geometry import box_wireframe, cylinder_wireframe, sphere_wireframe
    from urdf_tools.viz import _link_world_blocks

    all_pts: list[np.ndarray] = []
    for T, geom, rgba in _link_world_blocks(robot):
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
        all_pts.append(w)
        # Blend material RGB toward the dark theme so each element type has a
        # distinct but coherent tint: e.g. amber racks stay warm, steel columns
        # stay cool-grey, walls stay neutral blue-grey.
        r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
        mat_a = float(rgba[3]) if len(rgba) > 3 else 1.0
        ec = (r * 0.55 + 0.08, g * 0.55 + 0.10, b * 0.55 + 0.15)
        wire_a = min(mat_a * 0.65, 0.75)
        lw = 0.55 if geom.type == "box" else 0.65
        for i, j in edges:
            p0, p1 = w[i], w[j]
            ax3.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=ec,
                lw=lw,
                alpha=wire_a,
            )
    return all_pts


def _robot_wireframe_segments(robot):
    """Pre-compute per-link homogeneous corner matrices + edge indices."""
    from urdf_tools.geometry import box_wireframe, cylinder_wireframe, sphere_wireframe
    from urdf_tools.viz import _link_world_blocks

    segs = []
    for _T, geom, rgba in _link_world_blocks(robot):
        if geom.type == "box":
            corners, edges = box_wireframe(geom.size)
        elif geom.type == "cylinder":
            corners, edges = cylinder_wireframe(geom.radius, geom.length)
        elif geom.type == "sphere":
            corners, edges = sphere_wireframe(geom.radius)
        else:
            continue
        h = np.column_stack([corners, np.ones(len(corners))])
        ec = tuple(max(0.0, c - 0.05) for c in rgba[:3])
        segs.append((h, edges, ec))
    return segs


# ── 2D real-time viewer ────────────────────────────────────────────────────────


def live2d(
    sub: SensorSubscriber,
    world_robot=None,
    overlay_robot=None,
) -> None:
    """Real-time 2D LiDAR viewer: world overlay + robot pose + scan."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    RMAX = 12.0
    RAY_STEP = 6  # draw every N-th beam as a ray line

    fig, ax = plt.subplots(figsize=(8, 8))
    _style_fig(fig)
    _style_ax2d(ax, "2D LiDAR — shmbridge")
    fig.tight_layout(pad=1.2)

    # Static world geometry
    if world_robot is not None:
        _draw_urdf_2d(ax, world_robot, alpha=0.55)
        xmin, xmax, ymin, ymax = _world_bounds_2d(world_robot)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        RMAX = min(max(xmax - xmin, ymax - ymin) * 0.4, 30.0)
    else:
        ax.set_xlim(-RMAX, RMAX)
        ax.set_ylim(-RMAX, RMAX)

    # Range rings (muted dashed circles)
    ring_step = max(1, int(RMAX / 4))
    for r in range(ring_step, int(RMAX) + 1, ring_step):
        ax.add_patch(
            plt.Circle((0, 0), r, fc="none", ec=_GRID, lw=0.5, ls="--", alpha=0.5)
        )

    # Animated artists
    scan_scat = ax.scatter(
        [],
        [],
        s=4,
        c=[],
        cmap=_SCAN_CMAP,
        vmin=0,
        vmax=RMAX,
        zorder=5,
        edgecolors="none",
    )
    ray_lines = [
        ax.plot([], [], color=_RAY_C, lw=0.3, alpha=0.18, zorder=3)[0]
        for _ in range(360 // RAY_STEP + 10)
    ]
    robot_circle = plt.Circle((0, 0), 0.2, fc=_ROBOT_C, ec="white", lw=0.8, zorder=7)
    ax.add_patch(robot_circle)
    (hdg_line,) = ax.plot([], [], color=_HDG_C, lw=2.0, zorder=8)

    state = {"pose": (0.0, 0.0, 0.0), "dynamic_xlim": world_robot is None}

    def update(_frame):
        scan = sub.read_scan()
        odom = sub.read_odom()
        if odom:
            state["pose"] = (odom.x, odom.y, odom.theta)
        x0, y0, th = state["pose"]

        # Robot circle + heading
        robot_circle.center = (x0, y0)
        hdg_line.set_data(
            [x0, x0 + 0.6 * math.cos(th)],
            [y0, y0 + 0.6 * math.sin(th)],
        )

        # Dynamic axis follow (when no world urdf)
        if state["dynamic_xlim"]:
            ax.set_xlim(x0 - RMAX, x0 + RMAX)
            ax.set_ylim(y0 - RMAX, y0 + RMAX)

        # Scan
        if scan and scan.ranges:
            rng = np.asarray(scan.ranges, dtype=np.float32)
            a = scan.angle_min + np.arange(len(rng)) * scan.angle_increment
            hit = rng < scan.range_max * 0.999
            hx = x0 + rng[hit] * np.cos(th + a[hit])
            hy = y0 + rng[hit] * np.sin(th + a[hit])
            pts = np.column_stack([hx, hy]) if len(hx) else np.empty((0, 2))
            scan_scat.set_offsets(pts)
            scan_scat.set_array(rng[hit])

            # Ray lines
            ray_idx = np.arange(0, len(rng), RAY_STEP)
            for k, li in enumerate(ray_lines):
                if k < len(ray_idx):
                    i = ray_idx[k]
                    ex = x0 + rng[i] * math.cos(th + float(a[i]))
                    ey = y0 + rng[i] * math.sin(th + float(a[i]))
                    li.set_data([x0, ex], [y0, ey])
                else:
                    li.set_data([], [])
        else:
            scan_scat.set_offsets(np.empty((0, 2)))
            for li in ray_lines:
                li.set_data([], [])

        return (scan_scat, robot_circle, hdg_line, *ray_lines)

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
    _ = ani
    plt.show()


# ── 3D real-time viewer ────────────────────────────────────────────────────────


def live3d(
    sub: SensorSubscriber,
    world_robot=None,
    overlay_robot=None,
) -> None:
    """Real-time 3D viewer: world wireframe + animated robot + LiDAR scan cloud."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    from urdf_tools.geometry import pose_to_matrix as _p2m

    fig = plt.figure(figsize=(11, 9))
    _style_fig(fig)
    ax3 = fig.add_subplot(111, projection="3d")
    _style_ax3d(ax3, "3D LiDAR — shmbridge")
    fig.tight_layout(pad=1.0)

    # Static world wireframe
    world_pts: list[np.ndarray] = []
    if world_robot is not None:
        world_pts = _draw_urdf_3d_static(ax3, world_robot)

    # Pre-build robot wireframe line objects (updated per-frame)
    robot_segs = _robot_wireframe_segments(overlay_robot) if overlay_robot else []
    robot_lines: list[tuple] = []
    for h, edges, ec in robot_segs:
        seg_lines = []
        for i, j in edges:
            (ln,) = ax3.plot([], [], [], color=ec, lw=1.1, alpha=0.95)
            seg_lines.append((ln, h, i, j))
        robot_lines.append(seg_lines)

    # Scan scatter — use a mutable holder so the scatter is removed and recreated
    # each frame.  Mutating _offsets3d + set_facecolors on the same object fails
    # in animation: matplotlib re-computes face colours from the stale _A array
    # (set at init via c=[scalar]) and overrides whatever set_facecolors wrote.
    # A fresh scatter every frame carries only the colours we provide.
    scan_holder: list = [None]  # scan_holder[0] is the current scatter or None
    robot_dot = ax3.scatter([], [], [], s=90, c=[_ROBOT_C], zorder=6, marker="^")

    # Axis limits — use per-axis tight bounds so the z-range tracks the actual
    # warehouse height (~6 m) rather than being inflated to match XY extent.
    if world_pts:
        pts = np.vstack(world_pts)
        xmin, ymin, zmin = pts.min(0)
        xmax, ymax, zmax = pts.max(0)
        xpad = (xmax - xmin) * 0.03 + 1.0
        ypad = (ymax - ymin) * 0.03 + 1.0
        ax3.set_xlim(xmin - xpad, xmax + xpad)
        ax3.set_ylim(ymin - ypad, ymax + ypad)
        ax3.set_zlim(-0.5, zmax + 0.5)
        ax3.view_init(elev=32, azim=-52)
    else:
        ax3.set_xlim(-12, 12)
        ax3.set_ylim(-12, 12)
        ax3.set_zlim(-1, 8)
        ax3.view_init(elev=32, azim=-52)

    scan_rmax = [12.0]  # mutable container for current range_max
    state = {"pose": (0.0, 0.0, 0.0)}

    def update(_frame):
        scan = sub.read_scan()
        odom = sub.read_odom()
        if odom:
            state["pose"] = (odom.x, odom.y, odom.theta)
        x0, y0, th = state["pose"]

        # Move robot wireframe
        root_T = _p2m([x0, y0, 0.0], [0.0, 0.0, th])
        for seg_lines in robot_lines:
            for ln, h, i, j in seg_lines:
                w = (root_T @ h.T).T[:, :3]
                ln.set_data_3d(
                    [w[i, 0], w[j, 0]], [w[i, 1], w[j, 1]], [w[i, 2], w[j, 2]]
                )

        robot_dot._offsets3d = ([x0], [y0], [0.12])

        # Remove previous scan scatter before replacing it
        if scan_holder[0] is not None:
            try:
                scan_holder[0].remove()
            except Exception:
                pass
            scan_holder[0] = None

        # Scan cloud in world frame — fresh scatter each frame avoids the
        # matplotlib colormap-override bug on animated Axes3D scatter objects.
        if scan and scan.ranges:
            scan_rmax[0] = scan.range_max
            rng = np.asarray(scan.ranges, dtype=np.float32)
            a = scan.angle_min + np.arange(len(rng)) * scan.angle_increment
            hit = rng < scan.range_max * 0.999
            if hit.any():
                lx = x0 + rng[hit] * np.cos(th + a[hit])
                ly = y0 + rng[hit] * np.sin(th + a[hit])
                lz = np.full(hit.sum(), 0.05, dtype=np.float32)
                norm_c = np.clip(rng[hit] / max(scan_rmax[0], 1e-6), 0.0, 1.0)
                scan_holder[0] = ax3.scatter(
                    lx,
                    ly,
                    lz,
                    s=4.0,
                    c=plt.cm.plasma(norm_c),
                    alpha=0.90,
                    edgecolors="none",
                    depthshade=False,
                    zorder=4,
                )

        artists = [robot_dot] + [
            ln for seg_lines in robot_lines for ln, *_ in seg_lines
        ]
        if scan_holder[0] is not None:
            artists.append(scan_holder[0])
        return artists

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
    _ = ani
    plt.show()


# ── text mode ─────────────────────────────────────────────────────────────────


def text_mode(sub: SensorSubscriber, count: int) -> None:
    n = 0
    try:
        while True:
            scan = sub.read_scan()
            imu = sub.read_imu()
            odom = sub.read_odom()
            if scan:
                hits = sum(1 for r in scan.ranges if r < scan.range_max)
                print(
                    f"  [scan] t={scan.stamp:.3f}"
                    f"  beams={len(scan.ranges)}  hits={hits}"
                )
            if imu:
                print(
                    f"  [imu]  t={imu.stamp:.3f}"
                    f"  acc={[f'{v:.2f}' for v in imu.linear_acceleration]}"
                )
            if odom:
                print(
                    f"  [odom] t={odom.stamp:.3f}"
                    f"  x={odom.x:.2f}  y={odom.y:.2f}  θ={odom.theta:.2f}"
                )
            n += 1
            if count and n >= count:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped.")


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Subscribe to sensor topics and visualise in 2D or 3D"
    )
    ap.add_argument("--shm", default=SHM_NAME, help="Shared memory segment name")
    ap.add_argument(
        "--count", type=int, default=0, help="Stop after N polls (text mode only)"
    )
    ap.add_argument(
        "--live2d",
        "--live",
        dest="live2d",
        action="store_true",
        help="Open real-time 2D LiDAR viewer",
    )
    ap.add_argument(
        "--live3d",
        action="store_true",
        help="Open real-time 3D LiDAR viewer",
    )
    ap.add_argument(
        "--world",
        default=None,
        metavar="URDF",
        help="World URDF drawn as static background overlay",
    )
    ap.add_argument(
        "--robot",
        default=None,
        metavar="URDF",
        help="Robot URDF model to animate at odometry pose",
    )
    ap.add_argument(
        "--timeout", type=float, default=10000.0, help="Attach timeout in ms"
    )
    args = ap.parse_args()

    world_robot = None
    overlay_robot = None
    if args.world or args.robot:
        from urdf_tools.parser import parse_urdf

        if args.world:
            world_robot = parse_urdf(args.world)
            print(f"World : {world_robot.name!r}  links={len(world_robot.links)}")
        if args.robot:
            overlay_robot = parse_urdf(args.robot)
            print(f"Robot : {overlay_robot.name!r}  links={len(overlay_robot.links)}")

    print(f"Attaching to shm={args.shm!r}  (timeout={args.timeout:.0f} ms) …")
    sub = SensorSubscriber(args.shm, timeout_ms=args.timeout)
    try:
        sub.attach()
    except TimeoutError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print("Attached.  Ctrl+C to stop.\n")

    try:
        if args.live3d:
            live3d(sub, world_robot=world_robot, overlay_robot=overlay_robot)
        elif args.live2d:
            live2d(sub, world_robot=world_robot, overlay_robot=overlay_robot)
        else:
            text_mode(sub, args.count)
    finally:
        sub.detach()


if __name__ == "__main__":
    main()
