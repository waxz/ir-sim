"""
open3d_sim.py — IR-SIM + Scene3D + Open3D real-time 3D LiDAR visualisation.

Two systems run in lock-step:
  • irsim (YAML-driven)  — robot kinematics, collision, navigation behaviour
  • Scene3D (Embree BVH) — 3D geometry matching the YAML obstacles, VLP-16 raycasting

Open3D Visualizer shows:
  • Static 3D scene (coloured meshes)
  • Dynamic LiDAR point cloud (height-coloured, updated every step)
  • Robot sphere and heading arrow (updated every step)

Requirements:
    pip install ir-sim[lidar3d]   # open3d + embree

Usage:
    python usage/28open3d_lidar3d/open3d_sim.py
    python usage/28open3d_lidar3d/open3d_sim.py --headless     # save PNG, no window
    python usage/28open3d_lidar3d/open3d_sim.py --profile os64
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

# ── ir-sim ──────────────────────────────────────────────────────────────────
import irsim

try:
    import open3d as o3d

    from irsim.world.env3d import Scene3D
except ImportError as exc:
    raise SystemExit(
        "open3d is required.  Install with:  pip install ir-sim[lidar3d]"
    ) from exc

# ── constants ────────────────────────────────────────────────────────────────
SENSOR_HEIGHT = 0.8  # LiDAR mount height above ground (m)
ROBOT_RADIUS = 0.25  # matches world.yaml
WORLD_HALF = 10.0  # half-extent of the 20x20 m world
WALL_H = 3.0  # outer wall height (m)
WALL_T = 0.15  # wall thickness (m)

HERE = os.path.dirname(__file__)


# ── Scene3D builder ──────────────────────────────────────────────────────────


def build_scene() -> Scene3D:
    """
    Build the 3D geometry that matches world.yaml obstacles.

    Outer boundary walls + 3D versions of every YAML obstacle.
    The 2D footprint of each 3D object should cover the corresponding
    irsim shape so that LiDAR hits correlate with actual collisions.
    """
    s = Scene3D()

    # Ground
    s.add_ground(
        -WORLD_HALF, WORLD_HALF, -WORLD_HALF, WORLD_HALF, color=(0.28, 0.32, 0.22)
    )

    # Outer boundary walls (match irsim's implicit world boundary)
    s.add_wall(
        [-WORLD_HALF, -WORLD_HALF],
        [WORLD_HALF, -WORLD_HALF],
        height=WALL_H,
        thickness=WALL_T,
    )
    s.add_wall(
        [WORLD_HALF, -WORLD_HALF],
        [WORLD_HALF, WORLD_HALF],
        height=WALL_H,
        thickness=WALL_T,
    )
    s.add_wall(
        [WORLD_HALF, WORLD_HALF],
        [-WORLD_HALF, WORLD_HALF],
        height=WALL_H,
        thickness=WALL_T,
    )
    s.add_wall(
        [-WORLD_HALF, WORLD_HALF],
        [-WORLD_HALF, -WORLD_HALF],
        height=WALL_H,
        thickness=WALL_T,
    )

    # Pillar near centre (1x1 m base, 2.5 m tall)
    s.add_box(
        center=[0.0, 0.0, 1.25],
        size=[1.0, 1.0, 2.5],
        color=(0.55, 0.50, 0.45),
        label="pillar",
    )

    # Crate (1.5x0.8 m base, 1.0 m tall)
    s.add_box(
        center=[4.0, -3.0, 0.5],
        size=[1.5, 0.8, 1.0],
        color=(0.45, 0.32, 0.18),
        label="crate",
    )

    # Barrier wall segment (4 m long, 1.5 m tall)
    s.add_wall(
        [-2.0, 2.0],
        [2.0, 2.0],
        height=1.5,
        thickness=0.2,
        color=(0.62, 0.56, 0.50),
        label="barrier",
    )

    # Tall column (cylinder-like via a box, r~0.4 m, 0.8x0.8 m footprint, 2.8 m tall)
    s.add_box(
        center=[-4.0, 4.0, 1.4],
        size=[0.8, 0.8, 2.8],
        color=(0.50, 0.55, 0.60),
        label="column",
    )

    s.build()
    return s


# ── Open3D helpers ───────────────────────────────────────────────────────────


def _color_by_height(pts: np.ndarray) -> np.ndarray:
    """Map z-values to a blue→green→yellow color ramp. Returns (N,3) float64."""
    z = pts[:, 2]
    z_lo, z_hi = float(z.min()), float(z.max())
    t = (z - z_lo) / max(z_hi - z_lo, 1e-6)
    colors = np.zeros((len(pts), 3))
    colors[:, 0] = np.clip(2 * t - 0.5, 0, 1)  # R: rises above 0.25
    colors[:, 1] = np.clip(1 - np.abs(2 * t - 1), 0, 1)  # G: peak at mid
    colors[:, 2] = np.clip(1 - 2 * t, 0, 1)  # B: falls from 1
    return colors


def _make_arrow_mesh(
    origin: np.ndarray, direction: np.ndarray, length: float = 0.6
) -> o3d.geometry.TriangleMesh:
    """Return a small arrow mesh pointing from origin along direction."""
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.06,
        cone_radius=0.12,
        cylinder_height=length * 0.7,
        cone_height=length * 0.3,
    )
    arrow.paint_uniform_color([1.0, 0.55, 0.0])
    # Default arrow points along +Z; rotate to direction
    d = direction / (np.linalg.norm(direction) + 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    ax_norm = np.linalg.norm(axis)
    if ax_norm > 1e-9:
        axis /= ax_norm
        angle = math.acos(float(np.clip(np.dot(z, d), -1, 1)))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(origin)
    return arrow


def _scene_meshes(scene: Scene3D) -> list[o3d.geometry.TriangleMesh]:
    """Extract coloured legacy meshes from a Scene3D for static display."""
    meshes = []
    idx = 0
    for rec in scene._grounds:
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(rec.vertices.astype(np.float64))
        m.triangles = o3d.utility.Vector3iVector(rec.faces)
        m.paint_uniform_color(list(rec.color))
        m.compute_vertex_normals()
        meshes.append(m)
        idx += 1
    for rec in scene._boxes:
        m = scene._meshes[idx]
        mc = o3d.geometry.TriangleMesh(m)  # copy
        mc.paint_uniform_color(list(rec.color))
        mc.compute_vertex_normals()
        meshes.append(mc)
        idx += 1
    return meshes


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="IR-SIM Open3D 3D LiDAR demo")
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Skip interactive window; save one PNG per episode end",
    )
    ap.add_argument(
        "--profile",
        default="vlp16",
        choices=list(Scene3D.PROFILES.keys()),
        help="LiDAR profile (default: vlp16)",
    )
    ap.add_argument(
        "--steps", type=int, default=2000, help="Max simulation steps (default: 2000)"
    )
    ap.add_argument(
        "--range",
        type=float,
        default=20.0,
        help="LiDAR range_max in metres (default: 20)",
    )
    args = ap.parse_args()

    # ── build Scene3D ────────────────────────────────────────────────────────
    print("Building Scene3D …")
    scene = build_scene()
    n_channels, n_beams, e_lo, e_hi = Scene3D.PROFILES[args.profile]
    print(
        f"  Embree BVH ready.  Profile: {args.profile} "
        f"({n_channels} ch x {n_beams} beams, "
        f"elev [{e_lo}°, {e_hi}°])"
    )

    # ── irsim environment ────────────────────────────────────────────────────
    yaml_path = os.path.join(HERE, "world.yaml")
    env = irsim.make(yaml_path, headless=True)
    robot = env.robot_list[0]
    print("IR-SIM environment started.")

    # ── Open3D setup ─────────────────────────────────────────────────────────
    if not args.headless:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="IR-SIM 3D LiDAR", width=1280, height=720)

        # Static scene geometry
        for m in _scene_meshes(scene):
            vis.add_geometry(m)

        # Goal marker (green sphere)
        goal_x, goal_y = 7.0, 7.0
        goal_sphere = o3d.geometry.TriangleMesh.create_sphere(0.35)
        goal_sphere.paint_uniform_color([0.1, 0.85, 0.2])
        goal_sphere.compute_vertex_normals()
        goal_sphere.translate([goal_x, goal_y, 0.35])
        vis.add_geometry(goal_sphere)

        # Dynamic point cloud
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        # Robot sphere (updated per step)
        robot_sphere = o3d.geometry.TriangleMesh.create_sphere(ROBOT_RADIUS)
        robot_sphere.paint_uniform_color([0.9, 0.2, 0.15])
        robot_sphere.compute_vertex_normals()
        robot_sphere.translate(
            [float(robot.state[0, 0]), float(robot.state[1, 0]), ROBOT_RADIUS]
        )
        vis.add_geometry(robot_sphere)

        # Heading arrow (rebuilt each step via remove + add)
        arrow_geom: o3d.geometry.TriangleMesh | None = None

        # Camera: look down from above at a slight angle
        vc = vis.get_view_control()
        vc.set_zoom(0.45)
        vc.set_front([0.3, -0.6, 0.75])
        vc.set_up([0, 0, 1])
        vc.set_lookat([0, 0, 0])

    # ── simulation loop ──────────────────────────────────────────────────────
    prev_x = float(robot.state[0, 0])
    prev_y = float(robot.state[1, 0])
    episode = 0

    print(f"Running {args.steps} steps (Ctrl-C to stop) …")

    for step in range(args.steps):
        # Reset on episode end
        if env.done():
            episode += 1
            env.reset()
            prev_x = float(robot.state[0, 0])
            prev_y = float(robot.state[1, 0])
            print(f"  episode {episode} done at step {step}")

        # Step simulation
        env.step()

        # Robot state
        rx = float(robot.state[0, 0])
        ry = float(robot.state[1, 0])
        rth = float(robot.state[2, 0])

        # Cast 3D LiDAR
        origin_3d = [rx, ry, SENSOR_HEIGHT]
        pts = scene.cast_3d_lidar(origin_3d, profile=args.profile, range_max=args.range)

        if step % 50 == 0:
            print(f"  step {step:4d}  pos=({rx:.2f},{ry:.2f})  lidar hits={len(pts):,}")

        if args.headless:
            continue

        # ── update point cloud ───────────────────────────────────────────────
        if len(pts) > 0:
            pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(_color_by_height(pts))
        else:
            pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        vis.update_geometry(pcd)

        # ── update robot sphere ──────────────────────────────────────────────
        robot_sphere.translate([rx - prev_x, ry - prev_y, 0.0])
        vis.update_geometry(robot_sphere)
        prev_x, prev_y = rx, ry

        # ── rebuild heading arrow ────────────────────────────────────────────
        if arrow_geom is not None:
            vis.remove_geometry(arrow_geom, reset_bounding_box=False)
        arrow_geom = _make_arrow_mesh(
            np.array([rx, ry, ROBOT_RADIUS * 2 + 0.1]),
            np.array([math.cos(rth), math.sin(rth), 0.0]),
        )
        vis.add_geometry(arrow_geom, reset_bounding_box=False)

        # ── render ───────────────────────────────────────────────────────────
        if not vis.poll_events():
            print("Window closed.")
            break
        vis.update_renderer()

    # ── cleanup ──────────────────────────────────────────────────────────────
    env.end()
    if not args.headless:
        vis.destroy_window()
    print("Done.")


if __name__ == "__main__":
    main()
