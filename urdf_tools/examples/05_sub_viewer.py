"""Example 05 — Subscribe to sensor topics and display a live LiDAR scan.

Requires shmbridge installed and 04_pub_sensors.py (or urdf-tools publish) running.

Usage:
    python 05_sub_viewer.py            # text print mode
    python 05_sub_viewer.py --live     # 2D matplotlib live scan viewer
    python 05_sub_viewer.py --live3d   # 3D live viewer (world + robot + lidar)
    python 05_sub_viewer.py --live3d --world models/warehouse_world.urdf \\
        --robot models/robot_diff.urdf
    python 05_sub_viewer.py --count 20 # stop after 20 polls (text mode)
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


def live_viewer(sub: SensorSubscriber) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    RMAX = 25.0
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-RMAX, RMAX)
    ax.set_ylim(-RMAX, RMAX)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.set_title("Live LiDAR — shmbridge", fontweight="bold", color="white")
    ax.tick_params(colors="#475569")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e293b")

    scat = ax.scatter([], [], s=1.5, c=[], cmap="plasma", vmin=0, vmax=RMAX, zorder=3)
    robot_dot = ax.scatter([0], [0], s=100, c="#3b82f6", zorder=5, marker="D")
    (hdg,) = ax.plot([], [], color="#60a5fa", lw=1.5, zorder=4)
    state = {"pose": (0.0, 0.0, 0.0)}

    def update(_frame):
        scan = sub.read_scan()
        odom = sub.read_odom()
        if odom:
            state["pose"] = (odom.x, odom.y, odom.theta)
        x0, y0, th = state["pose"]
        robot_dot.set_offsets([[x0, y0]])
        hdg.set_data([x0, x0 + 1.2 * math.cos(th)], [y0, y0 + 1.2 * math.sin(th)])
        if scan and scan.ranges:
            rng = np.array(scan.ranges, dtype=np.float32)
            a = scan.angle_min + np.arange(len(rng)) * scan.angle_increment
            hit = rng < scan.range_max
            xs = x0 + rng[hit] * np.cos(th + a[hit])
            ys = y0 + rng[hit] * np.sin(th + a[hit])
            scat.set_offsets(np.column_stack([xs, ys]) if len(xs) else np.empty((0, 2)))
            scat.set_array(rng[hit])
        return scat, robot_dot, hdg

    ani = animation.FuncAnimation(fig, update, interval=60, blit=True)
    _ = ani
    plt.tight_layout()
    plt.show()


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
                    f"  [scan] t={scan.stamp:.3f}  beams={len(scan.ranges)}  hits={hits}"
                )
            if imu:
                print(
                    f"  [imu]  t={imu.stamp:.3f}  acc={[f'{v:.2f}' for v in imu.linear_acceleration]}"
                )
            if odom:
                print(
                    f"  [odom] t={odom.stamp:.3f}  x={odom.x:.2f}  y={odom.y:.2f}  θ={odom.theta:.2f}"
                )
            n += 1
            if count and n >= count:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped.")


def live_viewer_3d(
    sub: SensorSubscriber,
    world_robot=None,
    overlay_robot=None,
) -> None:
    """3D live viewer: world wireframe + robot pose + LiDAR scan cloud."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    from urdf_tools.geometry import (
        box_wireframe,
        cylinder_wireframe,
        sphere_wireframe,
    )
    from urdf_tools.geometry import (
        pose_to_matrix as _p2m,
    )

    fig = plt.figure(figsize=(11, 9))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    for ax in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        ax.pane.set_edgecolor("#1e293b")
    ax3.tick_params(colors="#475569")
    ax3.set_xlabel("X (m)", color="#94a3b8")
    ax3.set_ylabel("Y (m)", color="#94a3b8")
    ax3.set_zlabel("Z (m)", color="#94a3b8")
    ax3.set_title("3D Live View — shmbridge", fontweight="bold", color="white")

    # Draw static world wireframe once
    from urdf_tools.viz import _link_world_blocks as _lwb

    world_pts: list[np.ndarray] = []
    if world_robot is not None:
        for T, geom, rgba in _lwb(world_robot):
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
            world_pts.append(w)
            ec = [max(0, c - 0.15) for c in rgba[:3]]
            for i, j in edges:
                p0, p1 = w[i], w[j]
                ax3.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]],
                    color=ec,
                    lw=0.6,
                    alpha=0.5,
                )

    # Pre-compute robot link blocks (positions relative to robot origin)
    robot_blocks = list(_lwb(overlay_robot)) if overlay_robot is not None else []
    robot_lines = []
    for _T, geom, rgba in robot_blocks:
        if geom.type == "box":
            corners, edges = box_wireframe(geom.size)
        elif geom.type == "cylinder":
            corners, edges = cylinder_wireframe(geom.radius, geom.length)
        elif geom.type == "sphere":
            corners, edges = sphere_wireframe(geom.radius)
        else:
            robot_lines.append(None)
            continue
        h = np.column_stack([corners, np.ones(len(corners))])
        ec = [max(0, c - 0.15) for c in rgba[:3]]
        segs = []
        for i, j in edges:
            (ln,) = ax3.plot([], [], [], color=ec, lw=1.0, alpha=0.9)
            segs.append((ln, h, i, j))
        robot_lines.append(segs)

    scan_scat = ax3.scatter([], [], [], s=1.0, c=[], cmap="plasma", vmin=-1.0, vmax=2.0)
    robot_dot = ax3.scatter([], [], [], s=80, c="#3b82f6", zorder=6, marker="^")

    # Axis limits
    all_pts = world_pts.copy()
    if all_pts:
        pts = np.vstack(all_pts)
        mins, maxs = pts.min(0), pts.max(0)
        rng = maxs - mins
        mr = max(rng.max() * 0.5, 5.0)
        mids = (mins + maxs) * 0.5
        ax3.set_xlim(mids[0] - mr, mids[0] + mr)
        ax3.set_ylim(mids[1] - mr, mids[1] + mr)
        ax3.set_zlim(mids[2] - mr, mids[2] + mr)
    else:
        ax3.set_xlim(-15, 15)
        ax3.set_ylim(-15, 15)
        ax3.set_zlim(-1, 10)

    state = {"pose": (0.0, 0.0, 0.0)}

    def update(_frame):
        scan = sub.read_scan()
        odom = sub.read_odom()
        if odom:
            state["pose"] = (odom.x, odom.y, odom.theta)
        x0, y0, th = state["pose"]

        # Move robot wireframe
        root_T = _p2m([x0, y0, 0.0], [0.0, 0.0, th])
        for segs_or_none in robot_lines:
            if segs_or_none is None:
                continue
            for ln, h, i, j in segs_or_none:
                w = (root_T @ h.T).T[:, :3]
                p0, p1 = w[i], w[j]
                ln.set_data_3d([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]])

        robot_dot._offsets3d = ([x0], [y0], [0.1])

        # Lidar scan → world frame
        if scan and scan.ranges:
            rng = np.array(scan.ranges, dtype=np.float32)
            a = scan.angle_min + np.arange(len(rng)) * scan.angle_increment
            hit = rng < scan.range_max
            lx = x0 + rng[hit] * np.cos(th + a[hit])
            ly = y0 + rng[hit] * np.sin(th + a[hit])
            lz = np.zeros(hit.sum())
            pts_w = np.column_stack([lx, ly, lz]) if len(lx) else np.empty((0, 3))
            scan_scat._offsets3d = (pts_w[:, 0], pts_w[:, 1], pts_w[:, 2])
            scan_scat.set_array(lz)

        return scan_scat, robot_dot

    ani = animation.FuncAnimation(fig, update, interval=80, blit=False)
    _ = ani
    plt.tight_layout()
    plt.show()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Subscribe to sensor topics")
    ap.add_argument("--shm", default=SHM_NAME)
    ap.add_argument(
        "--count", type=int, default=0, help="Stop after N polls (0=forever)"
    )
    ap.add_argument(
        "--live", action="store_true", help="Open 2D matplotlib live scan viewer"
    )
    ap.add_argument(
        "--live3d", action="store_true", help="Open 3D live viewer (world+robot+lidar)"
    )
    ap.add_argument(
        "--world",
        default=None,
        metavar="URDF",
        help="World URDF for 3D viewer background",
    )
    ap.add_argument(
        "--robot",
        default=None,
        metavar="URDF",
        help="Robot URDF to animate in 3D viewer",
    )
    ap.add_argument(
        "--timeout", type=float, default=10000.0, help="Attach timeout in ms"
    )
    args = ap.parse_args()

    print(f"Attaching to shm={args.shm!r}  (timeout={args.timeout:.0f}ms) …")
    sub = SensorSubscriber(args.shm, timeout_ms=args.timeout)
    try:
        sub.attach()
    except TimeoutError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print("Attached.  Ctrl+C to stop.\n")

    try:
        if args.live3d:
            world_robot = None
            overlay_robot = None
            if args.world:
                from urdf_tools.parser import parse_urdf

                world_robot = parse_urdf(args.world)
                print(f"World: {world_robot.name!r}  links={len(world_robot.links)}")
            if args.robot:
                from urdf_tools.parser import parse_urdf

                overlay_robot = parse_urdf(args.robot)
                print(
                    f"Robot: {overlay_robot.name!r}  links={len(overlay_robot.links)}"
                )
            live_viewer_3d(sub, world_robot=world_robot, overlay_robot=overlay_robot)
        elif args.live:
            live_viewer(sub)
        else:
            text_mode(sub, args.count)
    finally:
        sub.detach()


if __name__ == "__main__":
    main()
