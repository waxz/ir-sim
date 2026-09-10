"""
Foxglove Studio integration demo for IR-SIM.

Runs a real IR-SIM environment with one differential-drive robot and several
obstacles, streams all sensor and telemetry channels to Foxglove Studio, and
accepts remote-operation commands back from Studio.

Channels published (sim -> Studio)
------------------------------------
/irsim/pose        foxglove.PoseInFrame  - robot pose
/irsim/lidar2d     foxglove.LaserScan    - 2D horizontal scan
/irsim/imu         foxglove.Imu          - gyro + accelerometer
/irsim/encoder     irsim.Encoder         - per-wheel angle / ticks / speed
/irsim/motor       irsim.Motor           - per-wheel cmd / actual / motor shaft
/irsim/metrics     irsim.Metrics         - timing and CPU
/irsim/map         foxglove.Grid         - 2D occupancy map (static, sent once)
/irsim/scene       foxglove.SceneUpdate  - 3D bodies: robot, obstacles

Client channels (Studio -> sim)
------------------------------------
/irsim/cmd_vel     velocity override     {"linear":{"x":0.5},"angular":{"z":0.3}}
/irsim/cmd_pose    goal override         {"robot_id":0,"x":5.0,"y":3.0,"theta":0}

Services (Studio -> sim)
------------------------------------
/irsim/control     pause / resume / reset

Requirements::

    pip install ir-sim[foxglove]

Run::

    python usage/foxglove_demo.py

Then open Foxglove Studio -> Add connection -> Foxglove WebSocket -> ws://localhost:8765

Suggested Foxglove panel layout
---------------------------------
- 3D panel         : /irsim/scene (robot + obstacles), /irsim/lidar2d, /irsim/pose
- Image / Raw Msgs : /irsim/map
- Plot panel       : /irsim/metrics fields (fps, scan_2d_ms, cpu_pct)
- Publish panel    : topic=/irsim/cmd_vel  {"linear":{"x":1.0},"angular":{"z":0.5}}
- Service panel    : /irsim/control        {"command":"pause"}
"""

from __future__ import annotations

import math
import os
import tempfile
import time

import numpy as np

import irsim
from irsim.util.foxglove_bridge import FoxgloveBridge

# ---------------------------------------------------------------------------
# Inline YAML environment — one diff-drive robot + mixed obstacles
# ---------------------------------------------------------------------------

_WORLD_YAML = """
world:
  height: 20
  width: 20
  step_time: 0.05
  collision_mode: stop
  plot:
    no_axis: True

robot:
  - kinematics: {name: diff}
    shape: {name: circle, radius: 0.2}
    state: [-7.0, -7.0, 0.0]
    goal: [7.0, 7.0, 0.0]
    behavior: {name: dash}
    vel_min: [-1.5, -2.0]
    vel_max: [1.5, 2.0]
    sensors:
      - type: lidar2d
        range_min: 0.0
        range_max: 10.0
        angle_range: 3.14159
        number: 360
        noise: False
    plot:
      show_goal: True
      show_trajectory: True
      keep_traj_length: 200

obstacle:
  - shape: {name: circle, radius: 0.5}
    state: [0.0, 0.0, 0.0]

  - shape: {name: circle, radius: 0.4}
    state: [3.0, -3.0, 0.0]

  - shape: {name: rectangle, length: 1.5, width: 0.6}
    state: [-3.0, 2.0, 0.8]

  - shape: {name: rectangle, length: 1.0, width: 1.0}
    state: [4.0, 4.0, 0.5]

  - kinematics: {name: omni}
    shape: {name: circle, radius: 0.3}
    state: [5.0, -5.0, 0.0]
    goal: [-5.0, 5.0]
    behavior: {name: dash, loop: True}
    vel_min: [-0.8, -0.8]
    vel_max: [0.8, 0.8]

  - kinematics: {name: diff}
    shape: {name: circle, radius: 0.25}
    state: [-4.0, 4.0, 0.0]
    goal: [[4.0, -4.0], [-4.0, 4.0]]
    behavior: {name: dash, loop: True}
    vel_min: [-1.0, -2.0]
    vel_max: [1.0, 2.0]
"""

# ---------------------------------------------------------------------------
# Synthetic wheel and IMU models
# ---------------------------------------------------------------------------

_G = 9.80665
WHEEL_RADIUS = 0.033
TRACK = 0.16
ENCODER_CPR = 2048
_MOTOR_TAU = 0.030

_gyro_bias = np.zeros(3)
_accel_bias = np.zeros(3)
_prev_vel_world = np.zeros(2)
_theta_enc = {"left": 0.0, "right": 0.0}
_omega_act = {"left": 0.0, "right": 0.0}


def _synthetic_imu(theta, vx, vy, omega_z, dt):
    global _gyro_bias, _accel_bias, _prev_vel_world
    dv = np.array([vx, vy]) - _prev_vel_world
    accel_world = dv / dt
    c, s = math.cos(theta), math.sin(theta)
    ab = np.array(
        [
            c * accel_world[0] + s * accel_world[1],
            -s * accel_world[0] + c * accel_world[1],
            _G,
        ]
    )
    _gyro_bias += 1.75e-4 * math.sqrt(dt) * np.random.randn(3)
    _accel_bias += 1.96e-4 * math.sqrt(dt) * np.random.randn(3)
    omega_meas = (
        np.array([0.0, 0.0, omega_z])
        + _gyro_bias
        + 8.73e-5 / math.sqrt(dt) * np.random.randn(3)
    )
    accel_meas = ab + _accel_bias + 3.92e-3 / math.sqrt(dt) * np.random.randn(3)
    _prev_vel_world[:] = [vx, vy]
    return omega_meas, accel_meas


def _synthetic_wheels(v, omega_z, dt):
    omega_l = (v - omega_z * TRACK / 2) / WHEEL_RADIUS
    omega_r = (v + omega_z * TRACK / 2) / WHEEL_RADIUS
    decay = math.exp(-dt / _MOTOR_TAU)
    tpr = ENCODER_CPR / (2.0 * math.pi)
    for name, cmd in [("left", omega_l), ("right", omega_r)]:
        _omega_act[name] = cmd + (_omega_act[name] - cmd) * decay
        _theta_enc[name] += _omega_act[name] * dt
    encoder = {
        n: {
            "theta_enc": _theta_enc[n],
            "ticks": int(np.int32(round(_theta_enc[n] * tpr))),
            "omega_actual": _omega_act[n],
        }
        for n in ("left", "right")
    }
    motor = {
        "left": {
            "omega_cmd": omega_l,
            "omega_actual": _omega_act["left"],
            "motor_omega": _omega_act["left"] * 46.0,
        },
        "right": {
            "omega_cmd": omega_r,
            "omega_actual": _omega_act["right"],
            "motor_omega": _omega_act["right"] * 46.0,
        },
    }
    return encoder, motor


# ---------------------------------------------------------------------------
# Build a simple occupancy grid from the world YAML's obstacles
# ---------------------------------------------------------------------------


def _build_occupancy_grid(env, resolution=0.1):
    """Return (grid, origin_xy) from the env's static obstacle geometries."""
    world = env._world
    W = float(world.width)
    H = float(world.height)
    cols = int(W / resolution)
    rows = int(H / resolution)
    grid = np.zeros((rows, cols), dtype=np.float32)
    ox = -W / 2.0
    oy = -H / 2.0

    # Mark cells occupied by static obstacle boundaries
    for obj in env.obstacle_list:
        if not obj.static:
            continue
        geom = obj._geometry
        if geom is None:
            continue
        # Sample boundary points from shapely geometry
        try:
            coords = []
            gtype = geom.geom_type
            if gtype in ("Polygon", "MultiPolygon"):
                coords = list(
                    geom.exterior.coords
                    if gtype == "Polygon"
                    else [pt for g in geom.geoms for pt in g.exterior.coords]
                )
            elif gtype in ("LineString", "MultiLineString"):
                coords = list(
                    geom.coords
                    if gtype == "LineString"
                    else [pt for g in geom.geoms for pt in g.coords]
                )
            for gx, gy in coords:
                ci = int((gx - ox) / resolution)
                cj = int((gy - oy) / resolution)
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < cols and 0 <= nj < rows:
                            grid[nj, ni] = 100.0
        except Exception:
            pass

    # Add world boundary walls
    for j in range(rows):
        grid[j, 0] = 100.0
        grid[j, cols - 1] = 100.0
    for i in range(cols):
        grid[0, i] = 100.0
        grid[rows - 1, i] = 100.0

    return grid, [ox, oy]


# ---------------------------------------------------------------------------
# Scene helpers — colour palette
# ---------------------------------------------------------------------------

_ROBOT_COLOR = (0.2, 0.55, 1.0, 0.92)
_STATIC_OBS_COLOR = (0.75, 0.35, 0.1, 0.85)
_DYNAMIC_OBS_COLOR = (0.2, 0.85, 0.45, 0.85)


def _publish_scene(bridge, robot, obstacles):
    """Push robot + obstacle markers for Foxglove 3D panel."""
    sx, sy, stheta = (
        float(robot.state[0, 0]),
        float(robot.state[1, 0]),
        float(robot.state[2, 0]),
    )
    bridge.update_robot_marker(
        sx,
        sy,
        stheta,
        robot_id=robot._id,
        robot_name=robot.name,
        radius=float(robot.radius),
        height=0.5,
        color=_ROBOT_COLOR,
    )

    for obs in obstacles:
        ox_pos = float(obs.state[0, 0])
        oy_pos = float(obs.state[1, 0])
        oth = float(obs.state[2, 0])
        eid = f"obs_{obs._id}"
        color = _DYNAMIC_OBS_COLOR if not obs.static else _STATIC_OBS_COLOR

        if obs.shape == "circle":
            bridge.update_circle_marker(
                eid, ox_pos, oy_pos, radius=float(obs.radius), height=0.8, color=color
            )
        else:
            bridge.update_box_marker(
                eid,
                ox_pos,
                oy_pos,
                oth,
                length=float(obs.length),
                width=float(obs.width),
                height=0.8,
                color=color,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Starting Foxglove bridge on ws://localhost:8765 …")
    bridge = FoxgloveBridge(port=8765, publish_hz=10.0)
    bridge.start()
    print("Bridge ready — open Foxglove Studio and connect to ws://localhost:8765")

    # Write inline YAML to a temp file, then load the environment
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        tf.write(_WORLD_YAML)
        yaml_path = tf.name

    env = irsim.make(
        world_name=yaml_path,
        display=False,
        save_ani=False,
    )

    robot = env.robot_list[0]
    obstacles = env.obstacle_list
    DT = float(env._world.step_time)

    # Publish static occupancy map once (and refresh every 50 steps for dynamic map)
    grid, origin_xy = _build_occupancy_grid(env, resolution=0.1)
    bridge.update_map(grid, resolution=0.1, origin_xy=origin_xy)
    print(f"Map published: {grid.shape[1]}x{grid.shape[0]} cells @ 0.1 m/cell")

    step = 0
    print(f"Running simulation (step_time={DT:.3f} s, Ctrl-C to stop) …")

    try:
        while not env.done():
            t0 = time.perf_counter()

            # ── Handle remote pause ──────────────────────────────────────────
            if bridge.is_paused():
                time.sleep(DT)
                continue

            # ── Remote velocity override ─────────────────────────────────────
            cmd = bridge.pop_cmd_vel()
            if cmd is not None:
                vel = np.array([[cmd["linear"]], [cmd["angular"]]])
                robot.set_vel(vel)
                print(
                    f"  [step {step}] remote cmd_vel: v={cmd['linear']:.2f} w={cmd['angular']:.2f}"
                )

            # ── Remote goal override ─────────────────────────────────────────
            goal = bridge.pop_cmd_pose()
            if goal is not None:
                robot.set_goal(
                    [goal.get("x", 0.0), goal.get("y", 0.0), goal.get("theta", 0.0)]
                )
                print(
                    f"  [step {step}] remote goal: ({goal.get('x', 0):.1f}, {goal.get('y', 0):.1f})"
                )

            # ── Simulation step ──────────────────────────────────────────────
            env.step()

            # ── Extract robot state ──────────────────────────────────────────
            x = float(robot.state[0, 0])
            y = float(robot.state[1, 0])
            theta = float(robot.state[2, 0])
            vel_state = robot.velocity
            v = float(vel_state[0, 0]) if vel_state is not None else 0.0
            omega_z = float(vel_state[1, 0]) if vel_state is not None else 0.0
            vx = v * math.cos(theta)
            vy = v * math.sin(theta)

            # ── Pose ─────────────────────────────────────────────────────────
            bridge.update_pose(x, y, theta, robot_id=robot._id, robot_name=robot.name)

            # ── LiDAR 2D ────────────────────────────────────────────────────
            scan_t0 = time.perf_counter()
            scan = robot.get_lidar_scan()
            scan_2d_ms = (time.perf_counter() - scan_t0) * 1000
            if isinstance(scan, dict) and scan.get("ranges") is not None:
                ranges = np.asarray(scan["ranges"], dtype=float)
                a_min = float(scan.get("angle_min", -math.pi))
                a_max = float(scan.get("angle_max", math.pi))
                bridge.update_lidar2d(
                    ranges,
                    [x, y, 0.2],
                    angle_min=a_min,
                    angle_max=a_max,
                    robot_id=robot._id,
                    robot_name=robot.name,
                    sensor_name="lidar2d_0",
                )

            # ── IMU ──────────────────────────────────────────────────────────
            omega_meas, accel_meas = _synthetic_imu(theta, vx, vy, omega_z, DT)
            bridge.update_imu(
                omega_meas,
                accel_meas,
                robot_id=robot._id,
                robot_name=robot.name,
                sensor_name="imu_0",
            )

            # ── Encoder + Motor ──────────────────────────────────────────────
            encoder_data, motor_data = _synthetic_wheels(v, omega_z, DT)
            bridge.update_encoder(
                encoder_data,
                robot_id=robot._id,
                robot_name=robot.name,
                sensor_name="enc_0",
            )
            bridge.update_motor(
                motor_data,
                robot_id=robot._id,
                robot_name=robot.name,
                sensor_name="mot_0",
            )

            # ── Scene (3D markers) ───────────────────────────────────────────
            _publish_scene(bridge, robot, obstacles)

            # ── Metrics ──────────────────────────────────────────────────────
            cpu_est = scan_2d_ms / (DT * 1000) * 100
            bridge.update_metrics(scan_2d_ms=scan_2d_ms, cpu_pct=cpu_est, step=step)

            # ── Refresh map every 50 steps (dynamic obstacles may have moved) ─
            if step % 50 == 0 and step > 0:
                grid, origin_xy = _build_occupancy_grid(env, resolution=0.1)
                bridge.update_map(grid, resolution=0.1, origin_xy=origin_xy)

            step += 1

            # ── Real-time pacing ─────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            wait = DT - elapsed
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        bridge.stop()
        env.end()
        os.unlink(yaml_path)
        print(f"Done after {step} steps.")


if __name__ == "__main__":
    main()
