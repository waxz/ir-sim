"""Example 06 — IR-SIM → shmbridge publisher bridge.

Runs an IR-SIM environment with a LiDAR-equipped robot and publishes real
sensor data (scan, odometry, IMU) via shmbridge shared memory so that
05_sub_viewer.py can visualise it in real-time.

Requires:
    pip install ir-sim      (or: cd ir-sim && pip install -e .)
    pip install shmbridge   (or: cd ir-sim/shmbridge && pip install -e .)

Usage:
    # Terminal 1 — run the simulation bridge
    python 06_irsim_bridge.py

    # Terminal 2 — 2D live scan viewer
    python 05_sub_viewer.py --live

    # Terminal 2 — 3D live viewer with world + robot wireframe
    python 05_sub_viewer.py --live3d \\
        --world models/robot_diff.urdf \\
        --robot models/robot_diff.urdf

    # Run headless for N steps then exit
    python 06_irsim_bridge.py --steps 500 --no-render
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from urdf_tools.pubsub import Imu, LaserScan, Odometry, SensorPublisher

SHM_NAME = "/urdf_tools_sensors"
WORLD_YAML = Path(__file__).parent / "irsim_world.yaml"


def _robot_xytheta(robot) -> tuple[float, float, float]:
    st = robot.state
    return float(st[0, 0]), float(st[1, 0]), float(st[2, 0])


def _robot_vel(robot) -> tuple[float, float]:
    vel = robot.velocity
    v = float(vel[0, 0]) if vel.size > 0 else 0.0
    omega = float(vel[-1, 0]) if vel.size > 1 else 0.0
    return v, omega


def main() -> None:
    ap = argparse.ArgumentParser(description="IR-SIM shmbridge publisher")
    ap.add_argument("--yaml", default=str(WORLD_YAML), help="World YAML path")
    ap.add_argument("--shm", default=SHM_NAME, help="Shared memory segment name")
    ap.add_argument("--steps", type=int, default=0, help="Max steps (0 = infinite)")
    ap.add_argument("--no-render", action="store_true", help="Run headless")
    args = ap.parse_args()

    import irsim

    env = irsim.make(args.yaml, headless=args.no_render)
    robot = env.robot_list[0]
    has_lidar = hasattr(robot, "lidar") and robot.lidar is not None

    if not has_lidar:
        print(
            "[warn] Robot has no lidar sensor — only odometry will be published.",
            file=sys.stderr,
        )

    print(f"[bridge] shm={args.shm!r}  render={not args.no_render}")
    print("[bridge] Ctrl+C to stop.\n")

    step = 0
    t0 = time.time()

    with SensorPublisher(args.shm) as pub:
        try:
            while True:
                env.step()
                if not args.no_render:
                    env.render(0.001)

                sim_time = time.time() - t0
                x, y, theta = _robot_xytheta(robot)
                v, omega = _robot_vel(robot)

                # --- Odometry ---
                pub.publish_odom(
                    Odometry(
                        stamp=sim_time,
                        x=x,
                        y=y,
                        theta=theta,
                        vx=v * math.cos(theta),
                        vy=v * math.sin(theta),
                        omega=omega,
                    )
                )

                # --- LiDAR scan ---
                if has_lidar:
                    scan_dict = robot.get_lidar_scan()
                    ranges = scan_dict.get("ranges")
                    if ranges is not None and len(ranges):
                        rng = np.asarray(ranges, dtype=np.float32).ravel()
                        pub.publish_scan(
                            LaserScan(
                                stamp=sim_time,
                                angle_min=float(scan_dict.get("angle_min", -math.pi)),
                                angle_max=float(scan_dict.get("angle_max", math.pi)),
                                angle_increment=float(
                                    scan_dict.get(
                                        "angle_increment",
                                        2 * math.pi / len(rng),
                                    )
                                ),
                                range_min=float(scan_dict.get("range_min", 0.1)),
                                range_max=float(scan_dict.get("range_max", 8.0)),
                                ranges=rng.tolist(),
                            )
                        )

                # --- IMU (simulated: gravity + yaw-rate) ---
                pub.publish_imu(
                    Imu(
                        stamp=sim_time,
                        linear_acceleration=[0.0, 0.0, 9.81],
                        angular_velocity=[0.0, 0.0, omega],
                        orientation_rpy=[0.0, 0.0, theta],
                    )
                )

                step += 1
                print(
                    f"\r  step={step:5d}  t={sim_time:6.1f}s"
                    f"  ({x:5.2f},{y:5.2f})  θ={theta:.2f}  ω={omega:.3f}",
                    end="",
                    flush=True,
                )

                if env.done():
                    print(f"\n[bridge] goal reached at step={step}")
                    break
                if args.steps and step >= args.steps:
                    print(f"\n[bridge] max steps ({args.steps}) reached")
                    break

        except KeyboardInterrupt:
            print("\n[bridge] stopped.")
        finally:
            env.end()


if __name__ == "__main__":
    main()
