"""Example 05 — Subscribe to sensor topics and display a live LiDAR scan.

Requires shmbridge installed and 04_pub_sensors.py (or urdf-tools publish) running.

Usage:
    python 05_sub_viewer.py            # text print mode
    python 05_sub_viewer.py --live     # matplotlib live scan viewer
    python 05_sub_viewer.py --count 20 # stop after 20 polls
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


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Subscribe to sensor topics")
    ap.add_argument("--shm", default=SHM_NAME)
    ap.add_argument(
        "--count", type=int, default=0, help="Stop after N polls (0=forever)"
    )
    ap.add_argument(
        "--live", action="store_true", help="Open matplotlib live scan viewer"
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
        if args.live:
            live_viewer(sub)
        else:
            text_mode(sub, args.count)
    finally:
        sub.detach()


if __name__ == "__main__":
    main()
