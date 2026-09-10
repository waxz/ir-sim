"""
open3d_sim.py — IR-SIM + Scene3D + Open3D 3D LiDAR demo
               with 2D LiDAR / IMU / encoder / motor sensors,
               keyboard driving, and Foxglove Studio integration.

Systems running in lock-step:
  irsim          — YAML-driven robot kinematics, collision, dash behaviour
  Scene3D        — Embree BVH 3D geometry, VLP-16/OS-64/OS-128 raycasting
  FoxgloveBridge — WebSocket publisher  (ws://localhost:8765)
  KeyboardControl — pynput global listener for manual driving

Published Foxglove topics:
  /irsim/pose     robot pose
  /irsim/lidar2d  2D 360-deg LiDAR scan
  /irsim/lidar3d  3D LiDAR point cloud
  /irsim/imu      IMU (angular velocity + linear acceleration)
  /irsim/encoder  wheel encoder ticks / angles
  /irsim/motor    motor command vs actual velocities
  /irsim/scene    3D robot marker (cylinder + arrow)
  /irsim/map      static occupancy grid

Keyboard (Open3D GLFW callbacks — focus the 3D window):
  w / s    forward / backward
  a / d    turn left / right
  x        toggle keyboard <-> auto (dash) control
  space    pause / resume
  r        reset to start
  esc      quit

  Headless mode falls back to pynput for keyboard input.

Requirements:
    pip install ir-sim[lidar3d,keyboard]    # open3d + embree + pynput

Foxglove Studio:
    Open ws://localhost:8765

Usage:
    python usage/28open3d_lidar3d/open3d_sim.py
    python usage/28open3d_lidar3d/open3d_sim.py --headless
    python usage/28open3d_lidar3d/open3d_sim.py --profile os64
    python usage/28open3d_lidar3d/open3d_sim.py --no-foxglove
    python usage/28open3d_lidar3d/open3d_sim.py --no-keyboard
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
from shapely.geometry import Point

import irsim

try:
    import open3d as o3d

    from irsim.world.env3d import Scene3D
except ImportError as exc:
    raise SystemExit(
        "open3d / embree is required.  Install with:  pip install ir-sim[lidar3d]"
    ) from exc

try:
    from irsim.util.foxglove_bridge import FoxgloveBridge

    _FOXGLOVE_OK = True
except ImportError:
    _FOXGLOVE_OK = False

try:
    from irsim.gui.keyboard_control import KeyboardControl

    _KEYBOARD_OK = True
except ImportError:
    _KEYBOARD_OK = False

# ── constants ─────────────────────────────────────────────────────────────────
SENSOR_HEIGHT = 0.8  # LiDAR mount height (m)
ROBOT_RADIUS = 0.25  # matches world.yaml
WORLD_HALF = 10.0
WALL_H = 3.0
WALL_T = 0.15
FOXGLOVE_PORT = 8765

HERE = os.path.dirname(__file__)


# ── Scene3D builder ───────────────────────────────────────────────────────────


def build_scene() -> Scene3D:
    """Build 3D geometry that mirrors world.yaml obstacle footprints."""
    s = Scene3D()

    s.add_ground(
        -WORLD_HALF, WORLD_HALF, -WORLD_HALF, WORLD_HALF, color=(0.28, 0.32, 0.22)
    )

    for a, b in [
        ([-WORLD_HALF, -WORLD_HALF], [WORLD_HALF, -WORLD_HALF]),
        ([WORLD_HALF, -WORLD_HALF], [WORLD_HALF, WORLD_HALF]),
        ([WORLD_HALF, WORLD_HALF], [-WORLD_HALF, WORLD_HALF]),
        ([-WORLD_HALF, WORLD_HALF], [-WORLD_HALF, -WORLD_HALF]),
    ]:
        s.add_wall(a, b, height=WALL_H, thickness=WALL_T)

    s.add_box(
        center=[0.0, 0.0, 1.25],
        size=[1.0, 1.0, 2.5],
        color=(0.55, 0.50, 0.45),
        label="pillar",
    )
    s.add_box(
        center=[4.0, -3.0, 0.5],
        size=[1.5, 0.8, 1.0],
        color=(0.45, 0.32, 0.18),
        label="crate",
    )
    s.add_wall(
        [-2.0, 2.0],
        [2.0, 2.0],
        height=1.5,
        thickness=0.2,
        color=(0.62, 0.56, 0.50),
        label="barrier",
    )
    s.add_box(
        center=[-4.0, 4.0, 1.4],
        size=[0.8, 0.8, 2.8],
        color=(0.50, 0.55, 0.60),
        label="column",
    )

    s.build()
    return s


# ── occupancy grid ────────────────────────────────────────────────────────────


def build_occupancy_grid(env, resolution: float = 0.2):
    """Return (grid, origin_xy) from obstacle Shapely geometries."""
    cols = int(2 * WORLD_HALF / resolution)
    rows = int(2 * WORLD_HALF / resolution)
    grid = np.zeros((rows, cols), dtype=np.float32)
    xs = np.linspace(-WORLD_HALF + resolution / 2, WORLD_HALF - resolution / 2, cols)
    ys = np.linspace(-WORLD_HALF + resolution / 2, WORLD_HALF - resolution / 2, rows)
    for obs in env.obstacle_list:
        geom = obs.geometry
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                if geom.contains(Point(x, y)):
                    grid[i, j] = 100.0
    return grid, np.array([-WORLD_HALF, -WORLD_HALF])


# ── Open3D helpers ────────────────────────────────────────────────────────────


def _color_by_height(pts: np.ndarray) -> np.ndarray:
    z = pts[:, 2]
    t = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    c = np.zeros((len(pts), 3))
    c[:, 0] = np.clip(2 * t - 0.5, 0, 1)
    c[:, 1] = np.clip(1 - np.abs(2 * t - 1), 0, 1)
    c[:, 2] = np.clip(1 - 2 * t, 0, 1)
    return c


def _make_arrow_mesh(
    origin: np.ndarray, direction: np.ndarray, length: float = 0.6
) -> o3d.geometry.TriangleMesh:
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.06,
        cone_radius=0.12,
        cylinder_height=length * 0.7,
        cone_height=length * 0.3,
    )
    arrow.paint_uniform_color([1.0, 0.55, 0.0])
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
        mc = o3d.geometry.TriangleMesh(scene._meshes[idx])
        mc.paint_uniform_color(list(rec.color))
        mc.compute_vertex_normals()
        meshes.append(mc)
        idx += 1
    return meshes


# ── Open3D keyboard helpers ───────────────────────────────────────────────────


def _setup_o3d_keyboard(vis, env, kb) -> dict[str, bool]:
    """Register GLFW key callbacks on *vis* (must be VisualizerWithKeyCallback).

    Returns a live key-state dict ``{"w": bool, "s": bool, "a": bool, "d": bool}``
    that is updated by the callbacks on every ``vis.poll_events()`` call.
    Motion keys use *register_key_action_callback* so held keys fire continuously.
    Command keys use *register_key_callback* (press-only).

    Returns an empty dict if the required API is absent.
    """
    if not hasattr(vis, "register_key_action_callback"):
        return {}

    state: dict[str, bool] = dict.fromkeys("wsad", False)

    # Motion keys — track press (action=1/repeat=2) vs release (action=0)
    for char, glfw_key in (("w", 87), ("s", 83), ("a", 65), ("d", 68)):

        def _motion_cb(_vis, action: int, _mods: int, c: str = char) -> bool:
            state[c] = action != 0  # True while held
            return False

        vis.register_key_action_callback(glfw_key, _motion_cb)

    # Command keys — press-only
    def _on_space(_vis) -> bool:
        if kb is not None and kb.env_ref is not None:
            kb._toggle_pause()
        return False

    def _on_r(_vis) -> bool:
        env.reset_flag = True
        return False

    def _on_x(_vis) -> bool:
        if kb is not None:
            kb._toggle_control_mode()
        return False

    def _on_esc(_vis) -> bool:
        env.quit_flag = True
        return False

    vis.register_key_callback(32, _on_space)  # GLFW_KEY_SPACE
    vis.register_key_callback(82, _on_r)  # R
    vis.register_key_callback(88, _on_x)  # X
    vis.register_key_callback(256, _on_esc)  # GLFW_KEY_ESCAPE

    return state


# ── sensor helpers ────────────────────────────────────────────────────────────


def _find_sensor(robot, sensor_type: str):
    for s in getattr(robot, "sensors", []):
        if getattr(s, "sensor_type", None) == sensor_type:
            return s
    return None


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="IR-SIM Open3D 3D LiDAR demo")
    ap.add_argument(
        "--headless", action="store_true", help="No Open3D window; run in terminal only"
    )
    ap.add_argument(
        "--profile",
        default="vlp16",
        choices=list(Scene3D.PROFILES.keys()),
        help="3D LiDAR profile (default: vlp16)",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Max simulation steps (default: 100000)",
    )
    ap.add_argument(
        "--range",
        type=float,
        default=20.0,
        help="3D LiDAR range_max in metres (default: 20)",
    )
    ap.add_argument(
        "--no-foxglove", action="store_true", help="Disable Foxglove bridge"
    )
    ap.add_argument(
        "--no-keyboard", action="store_true", help="Disable keyboard control"
    )
    ap.add_argument(
        "--fps", type=int, default=20, help="Foxglove publish rate Hz (default: 20)"
    )
    args = ap.parse_args()

    # ── Scene3D ───────────────────────────────────────────────────────────────
    print("Building Scene3D ...")
    scene = build_scene()
    n_ch, n_b, e_lo, e_hi = Scene3D.PROFILES[args.profile]
    print(f"  {args.profile}: {n_ch} ch x {n_b} beams, elev [{e_lo},{e_hi}] deg")

    # ── irsim ─────────────────────────────────────────────────────────────────
    env = irsim.make(os.path.join(HERE, "world.yaml"), headless=True)
    robot = env.robot_list[0]
    print("IR-SIM ready.")

    lidar2d = _find_sensor(robot, "lidar2d")
    imu_sensor = _find_sensor(robot, "imu")
    if lidar2d:
        print(
            f"  2D LiDAR: {lidar2d.number} beams, range [0.05, {lidar2d.range_max}] m, 360 deg"
        )
    if imu_sensor:
        print("  IMU: mpu6050 profile, noise=true")

    # ── keyboard ──────────────────────────────────────────────────────────────
    kb = None
    if not args.no_keyboard:
        if not _KEYBOARD_OK:
            print(
                "  WARNING: pynput not found; skip keyboard.  pip install ir-sim[keyboard]"
            )
        else:
            try:
                # env_ref=None so the pynput listener starts unconditionally
                # (headless env has display=False which would otherwise skip it).
                # global_hook=True disables MPL focus gating so keys are
                # captured on any OS window including the Open3D viewport.
                kb = KeyboardControl(
                    env_ref=None,
                    key_lv_max=2.0,
                    key_ang_max=1.5,
                    backend="pynput",
                    global_hook=True,
                )
                # Attach env_ref after construction for r/space/x/esc commands.
                kb.env_ref = env
                env.keyboard = kb
                env._world_param.control_mode = "keyboard"

                # Wrap _update_key_vel so every key-press prints to console.
                _orig_update = kb._update_key_vel

                def _debug_update() -> None:
                    _orig_update()
                    kv = kb.key_vel.ravel()
                    if any(kv != 0):
                        print(f"  [KB] key_vel=({kv[0]:.1f}, {kv[1]:.1f})")

                kb._update_key_vel = _debug_update  # type: ignore[method-assign]

                if kb.listener is None:
                    print(
                        "  WARNING: pynput not available — keyboard disabled."
                        "  Install with:  pip install ir-sim[keyboard]"
                    )
                else:
                    print(
                        "  Keyboard: w/s=fwd/back  a/d=turn  x=toggle-auto  "
                        "space=pause  r=reset  esc=quit"
                    )
            except Exception as exc:
                print(f"  WARNING: keyboard unavailable ({exc})")

    # ── Foxglove bridge ───────────────────────────────────────────────────────
    bridge = None
    if not args.no_foxglove:
        if not _FOXGLOVE_OK:
            print("  WARNING: foxglove_websocket not installed; bridge disabled.")
        else:
            bridge = FoxgloveBridge(
                port=FOXGLOVE_PORT, publish_hz=float(args.fps)
            ).start()
            print(
                f"  Foxglove: ws://0.0.0.0:{FOXGLOVE_PORT}  (connect Foxglove Studio)"
            )
            print("  Building occupancy grid ...", end=" ", flush=True)
            grid, origin_xy = build_occupancy_grid(env, resolution=0.2)
            bridge.update_map(grid, resolution=0.2, origin_xy=origin_xy)
            print(f"{grid.shape[1]}x{grid.shape[0]} cells at 0.2 m/cell")

            # /irsim/scene channel is temporarily disabled in FoxgloveBridge.
            # Obstacle scene markers will be published once it is re-enabled.

    # ── Open3D window ─────────────────────────────────────────────────────────
    vis = arrow_geom = robot_sphere = pcd = None
    prev_x = float(robot.state[0, 0])
    prev_y = float(robot.state[1, 0])

    if not args.headless:
        # VisualizerWithKeyCallback adds register_key_callback /
        # register_key_action_callback on top of the standard Visualizer API.
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="IR-SIM 3D LiDAR", width=1280, height=720)

        for m in _scene_meshes(scene):
            vis.add_geometry(m)

        goal_s = o3d.geometry.TriangleMesh.create_sphere(0.35)
        goal_s.paint_uniform_color([0.1, 0.85, 0.2])
        goal_s.compute_vertex_normals()
        goal_s.translate([7.0, 7.0, 0.35])
        vis.add_geometry(goal_s)

        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        robot_sphere = o3d.geometry.TriangleMesh.create_sphere(ROBOT_RADIUS)
        robot_sphere.paint_uniform_color([0.9, 0.2, 0.15])
        robot_sphere.compute_vertex_normals()
        robot_sphere.translate([prev_x, prev_y, ROBOT_RADIUS])
        vis.add_geometry(robot_sphere)

        vc = vis.get_view_control()
        vc.set_zoom(0.45)
        vc.set_front([0.3, -0.6, 0.75])
        vc.set_up([0, 0, 1])
        vc.set_lookat([0, 0, 0])

    # ── Open3D key callbacks (replaces pynput when visualizer is active) ──────
    # register_key_action_callback fires on press/repeat/release so held keys
    # update the state dict continuously — much more reliable on Windows than
    # pynput when an Open3D window is present.
    o3d_keys: dict[str, bool] = {}
    if vis is not None and kb is not None:
        o3d_keys = _setup_o3d_keyboard(vis, env, kb)
        if o3d_keys:
            # Stop pynput listener to avoid duplicate / racing updates.
            if getattr(kb, "listener", None) is not None:
                kb.listener.stop()
                kb.listener = None
            print(
                "  Keyboard (O3D): w/s=fwd/back  a/d=turn  "
                "space=pause  r=reset  x=toggle  esc=quit"
            )
            print("  Focus the 3D window to drive.")

    # ── simulation loop ───────────────────────────────────────────────────────
    print(f"Running (max {args.steps} steps) — Ctrl-C to stop ...")

    for step in range(args.steps):
        if getattr(env, "quit_flag", False):
            print("  Quit via keyboard.")
            break

        # Keyboard-triggered reset (r key)
        if getattr(env, "reset_flag", False):
            env.reset()
            env.reset_flag = False
            prev_x = float(robot.state[0, 0])
            prev_y = float(robot.state[1, 0])
            print(f"  [step {step}] manual reset")

        # Apply O3D key state to keyboard velocity before stepping.
        # Key callbacks fired during the previous vis.poll_events(); one-step
        # lag is imperceptible at normal simulation rates.
        if o3d_keys and kb is not None and env._world_param.control_mode == "keyboard":
            lv = (
                kb.key_lv_max
                if o3d_keys["w"]
                else -kb.key_lv_max
                if o3d_keys["s"]
                else 0.0
            )
            ang = (
                kb.key_ang_max
                if o3d_keys["a"]
                else -kb.key_ang_max
                if o3d_keys["d"]
                else 0.0
            )
            kb.key_vel = np.array([[lv], [ang], [0.0]])

        env.step()

        rx = float(robot.state[0, 0])
        ry = float(robot.state[1, 0])
        rth = float(robot.state[2, 0])

        # 3D LiDAR raycast
        pts = scene.cast_3d_lidar(
            [rx, ry, SENSOR_HEIGHT], profile=args.profile, range_max=args.range
        )

        if step % 20 == 0:
            mode = getattr(env._world_param, "control_mode", "?")
            kb_info = ""
            if kb is not None:
                kv = kb.key_vel.ravel()
                listener_ok = getattr(kb, "listener", None) is not None
                kb_info = f"  kb_vel=({kv[0]:.1f},{kv[1]:.1f})  listener={'on' if listener_ok else 'off(mpl)'}"
            print(
                f"  step {step:6d}  ({rx:6.2f},{ry:6.2f})  3d={len(pts):,}  mode={mode}{kb_info}"
            )

        # ── Foxglove publish ──────────────────────────────────────────────────
        if bridge is not None:
            bridge.update_pose(rx, ry, rth, robot_id=0, robot_name="robot_0")

            if lidar2d is not None:
                scan = lidar2d.get_scan()
                bridge.update_lidar2d(
                    np.asarray(scan["ranges"], dtype=np.float32),
                    [rx, ry, 0.2],
                    float(scan["angle_min"]),
                    float(scan["angle_max"]),
                    robot_id=0,
                    robot_name="robot_0",
                    sensor_name="lidar2d",
                )

            if imu_sensor is not None:
                meas = imu_sensor.get_measurement()
                bridge.update_imu(
                    meas["angular_velocity"],
                    meas["linear_acceleration"],
                    robot_id=0,
                    robot_name="robot_0",
                    sensor_name="imu",
                )

            try:
                enc = robot.encoder_readings
                if enc:
                    bridge.update_encoder(
                        enc, robot_id=0, robot_name="robot_0", sensor_name="encoder"
                    )
            except Exception:
                pass

            try:
                mot = robot.wheel_states
                if mot:
                    bridge.update_motor(
                        mot, robot_id=0, robot_name="robot_0", sensor_name="motor"
                    )
            except Exception:
                pass

            if len(pts) > 0:
                bridge.update_lidar3d(
                    pts[:, :3].astype(np.float32),
                    [rx, ry, SENSOR_HEIGHT],
                    robot_id=0,
                    robot_name="robot_0",
                    sensor_name="lidar3d",
                )

        if args.headless:
            continue

        # ── Open3D update ─────────────────────────────────────────────────────
        if len(pts) > 0:
            pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(_color_by_height(pts))
        else:
            pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        vis.update_geometry(pcd)

        robot_sphere.translate([rx - prev_x, ry - prev_y, 0.0])
        vis.update_geometry(robot_sphere)
        prev_x, prev_y = rx, ry

        if arrow_geom is not None:
            vis.remove_geometry(arrow_geom, reset_bounding_box=False)
        arrow_geom = _make_arrow_mesh(
            np.array([rx, ry, ROBOT_RADIUS * 2 + 0.1]),
            np.array([math.cos(rth), math.sin(rth), 0.0]),
        )
        vis.add_geometry(arrow_geom, reset_bounding_box=False)

        if not vis.poll_events():
            print("Window closed.")
            break
        vis.update_renderer()

    # ── cleanup ───────────────────────────────────────────────────────────────
    env.end()
    if vis is not None:
        vis.destroy_window()
    if bridge is not None:
        bridge.stop()
    print("Done.")


if __name__ == "__main__":
    main()
