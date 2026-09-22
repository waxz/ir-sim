"""Example 06 — IR-SIM → shmbridge publisher bridge.

Runs an IR-SIM environment with a LiDAR-equipped robot and publishes real
sensor data (scan, odometry, IMU) via shmbridge shared memory so that
05_sub_viewer.py can visualise it in real-time.

Requires:
    pip install ir-sim        (or: cd ir-sim && pip install -e .)
    pip install shmbridge     (or: cd ir-sim/shmbridge && pip install -e .)
    pip install irsim-devices (or: cd ir-sim/irsim_devices && pip install -e .)

Usage:
    # Basic — 10×10 m world with 4 yaml obstacles (irsim built-in lidar)
    python 06_irsim_bridge.py

    # Warehouse — irsim_devices raycaster against warehouse URDF geometry
    python 06_irsim_bridge.py \\
        --yaml irsim_warehouse.yaml \\
        --world models/warehouse_world.urdf

    # Terminal 2 — 3D live web viewer
    python 05_sub_viewer.py --live3d \\
        --world models/warehouse_world.urdf \\
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
    ap.add_argument(
        "--world",
        default=None,
        metavar="URDF",
        help="World URDF — geometry injected as static obstacles for lidar raycasting",
    )
    ap.add_argument(
        "--lidar-height",
        type=float,
        default=0.30,
        metavar="M",
        help="Scan-plane height for URDF Z filter (default 0.30 m)",
    )
    args = ap.parse_args()

    import irsim

    env = irsim.make(args.yaml, headless=args.no_render)

    # irsim_devices lidar (standalone, raycasts against URDF scene)
    dev_lidar = None
    dev_lidar3d = None
    if args.world:
        from urdf_tools.irsim_compat import urdf_to_scene_2d
        from urdf_tools.parser import parse_urdf

        world_robot = parse_urdf(args.world)
        scene = urdf_to_scene_2d(world_robot, lidar_height=args.lidar_height)

        from irsim_devices.sensors import Lidar2D as DevLidar2D

        robot_tmp = env.robot_list[0]
        irsim_lidar = getattr(robot_tmp, "lidar", None)
        x0, y0, th0 = _robot_xytheta(robot_tmp)
        dev_lidar = DevLidar2D(
            state=np.array([x0, y0, th0], dtype=np.float64),
            range_min=irsim_lidar.range_min if irsim_lidar else 0.1,
            range_max=irsim_lidar.range_max if irsim_lidar else 20.0,
            angle_range=irsim_lidar.angle_range if irsim_lidar else 6.2832,
            number=irsim_lidar.number if irsim_lidar else 360,
        )
        dev_lidar.set_scene(scene)
        print(
            f"[bridge] irsim_devices lidar — {len(scene)} scene objects"
            f" from {args.world!r}  (lidar_height={args.lidar_height} m)"
        )

        # 3D lidar — Embree BVH against the full URDF mesh
        try:
            from irsim_devices.models.urdf_loader import load_urdf
            from irsim_devices.sensors.lidar3d_embree import EmbreeLidar3D

            urdf_model = load_urdf(args.world, use_collision=False)
            dev_lidar3d = EmbreeLidar3D(
                state=np.array([x0, y0, th0], dtype=np.float64),
                obj_id=-2,
                profile="vlp16",
                sensor_height=args.lidar_height,
            )
            dev_lidar3d.build_embree_scene(urdf_model.vertices, urdf_model.triangles)
            print(
                f"[bridge] EmbreeLidar3D vlp16 — {len(urdf_model.triangles)} tris"
                f" ({len(urdf_model.vertices)} verts)"
            )
        except Exception as exc:
            print(f"[warn] EmbreeLidar3D unavailable: {exc}", file=sys.stderr)
            dev_lidar3d = None

    robot = env.robot_list[0]
    has_lidar = dev_lidar is not None or (
        hasattr(robot, "lidar") and robot.lidar is not None
    )

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
                    if dev_lidar is not None:
                        # irsim_devices raycaster against URDF scene
                        dev_lidar.step(np.array([x, y, theta], dtype=np.float64))
                        scan_dict = dev_lidar.get_scan()
                    else:
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

                # --- 3D LiDAR cloud ---
                if dev_lidar3d is not None:
                    dev_lidar3d.step(np.array([x, y, theta], dtype=np.float64))
                    pts3d = dev_lidar3d.scan
                    if pts3d is not None and len(pts3d):
                        pub.publish_cloud3d(pts3d, stamp=sim_time)

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
