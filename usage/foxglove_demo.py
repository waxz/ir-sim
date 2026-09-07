"""
Foxglove Studio integration demo for IR-SIM.

Streams 2D/3D LiDAR, robot pose, IMU, wheel encoder, motor telemetry, and
performance metrics to Foxglove Studio at 10 Hz while the simulation runs
at full speed.

Channels
--------
/irsim/pose        foxglove.PoseInFrame  — robot pose
/irsim/lidar2d     foxglove.LaserScan    — 2D horizontal scan
/irsim/lidar3d     foxglove.PointCloud   — 3D spinning LiDAR (optional)
/irsim/imu         foxglove.Imu          — gyro + accelerometer
/irsim/encoder     irsim.Encoder         — per-wheel angle / ticks / speed
/irsim/motor       irsim.Motor           — per-wheel cmd / actual / motor shaft
/irsim/metrics     irsim.Metrics         — timing and CPU

Requirements::

    pip install ir-sim[foxglove]          # WebSocket bridge
    pip install ir-sim[lidar3d]           # 3D scene / LiDAR (optional)

Run::

    python usage/foxglove_demo.py

Then open Foxglove Studio → Add connection → Foxglove WebSocket → ws://localhost:8765
"""

from __future__ import annotations

import math
import time

import numpy as np

from irsim.util.foxglove_bridge import FoxgloveBridge

# ---------------------------------------------------------------------------
# Foxglove bridge — starts a background WebSocket server
# ---------------------------------------------------------------------------

bridge = FoxgloveBridge(port=8765, publish_hz=10.0)
bridge.start()
print("Foxglove bridge ready — connect Foxglove Studio to ws://localhost:8765")

# ---------------------------------------------------------------------------
# Optional: build a 3D scene (requires ir-sim[lidar3d])
# ---------------------------------------------------------------------------

try:
    from irsim.world.env3d import Scene3D

    scene = Scene3D()
    scene.add_ground(-15, 15, -15, 15, resolution=30, color=(0.3, 0.3, 0.3))
    scene.add_wall([-15, -15], [15, -15], height=2.5, thickness=0.2)
    scene.add_wall([-15, 15], [15, 15], height=2.5, thickness=0.2)
    scene.add_wall([-15, -15], [-15, 15], height=2.5, thickness=0.2)
    scene.add_wall([15, -15], [15, 15], height=2.5, thickness=0.2)
    scene.add_box([5.0, 3.0, 1.0], [2.0, 2.0, 2.0], label="box_a")
    scene.add_box([-4.0, -5.0, 0.75], [3.0, 1.0, 1.5], label="box_b")
    scene.add_car([-6.0, 6.0], yaw=math.radians(20), label="car_1")
    scene.build()
    HAS_3D = True
    print("3D scene built — 3D LiDAR enabled")
except ImportError:
    HAS_3D = False
    print("ir-sim[lidar3d] not installed — 3D LiDAR disabled")

# ---------------------------------------------------------------------------
# Synthetic IMU state (finite-difference acceleration from trajectory)
# ---------------------------------------------------------------------------

_G = 9.80665  # m/s²

# IMU bias random walk (MPU-6050 typical)
_gyro_bias = np.zeros(3)
_accel_bias = np.zeros(3)
_gyro_bias_walk = 1.75e-4  # rad/s/sqrt(s)
_accel_bias_walk = 1.96e-4  # m/s²/sqrt(s)
_gyro_noise = 8.73e-5  # rad/s/sqrt(Hz)
_accel_noise = 3.92e-3  # m/s²/sqrt(Hz)

_prev_vel_world = np.zeros(2)
_prev_theta = 0.0


def synthetic_imu(x, y, theta, vx, vy, omega_z, dt):
    """Derive noisy IMU measurement from ground-truth kinematics."""
    global _gyro_bias, _accel_bias, _prev_vel_world, _prev_theta

    # Ground-truth signals
    dv_world = np.array([vx, vy]) - _prev_vel_world
    accel_world = dv_world / dt

    # Rotate to body frame
    c, s = math.cos(theta), math.sin(theta)
    accel_body = np.array(
        [
            c * accel_world[0] + s * accel_world[1],
            -s * accel_world[0] + c * accel_world[1],
        ]
    )
    accel_true = np.array([accel_body[0], accel_body[1], _G])
    omega_true = np.array([0.0, 0.0, omega_z])

    # IEEE 517 noise model
    sigma_g = _gyro_noise / math.sqrt(dt)
    sigma_a = _accel_noise / math.sqrt(dt)
    _gyro_bias += _gyro_bias_walk * math.sqrt(dt) * np.random.randn(3)
    _accel_bias += _accel_bias_walk * math.sqrt(dt) * np.random.randn(3)

    omega_meas = omega_true + _gyro_bias + sigma_g * np.random.randn(3)
    accel_meas = accel_true + _accel_bias + sigma_a * np.random.randn(3)

    _prev_vel_world[:] = [vx, vy]
    _prev_theta = theta
    return omega_meas, accel_meas


# ---------------------------------------------------------------------------
# Synthetic wheel state (differential-drive robot)
# ---------------------------------------------------------------------------

WHEEL_RADIUS = 0.033  # m
TRACK = 0.16  # m
ENCODER_CPR = 2048  # counts per revolution

# Per-wheel accumulated encoder angle (rad)
_theta_enc = {"left": 0.0, "right": 0.0}

# Simple first-order motor lag: τ ≈ 30 ms (small_dc preset)
_MOTOR_TAU = 0.030
_omega_actual = {"left": 0.0, "right": 0.0}


def diff_inv_kin(v, omega):
    """Differential drive inverse kinematics → (omega_l, omega_r)."""
    omega_l = (v - omega * TRACK / 2) / WHEEL_RADIUS
    omega_r = (v + omega * TRACK / 2) / WHEEL_RADIUS
    return omega_l, omega_r


def synthetic_wheels(v, omega, dt):
    """Advance synthetic wheel state and return encoder + motor dicts."""
    omega_cmd_l, omega_cmd_r = diff_inv_kin(v, omega)

    # Motor first-order lag
    decay = math.exp(-dt / _MOTOR_TAU)
    for name, cmd in [("left", omega_cmd_l), ("right", omega_cmd_r)]:
        _omega_actual[name] = cmd + (_omega_actual[name] - cmd) * decay
        _theta_enc[name] += _omega_actual[name] * dt

    ticks_per_rad = ENCODER_CPR / (2.0 * math.pi)

    encoder = {
        name: {
            "theta_enc": _theta_enc[name],
            "ticks": int(np.int32(round(_theta_enc[name] * ticks_per_rad))),
            "omega_actual": _omega_actual[name],
        }
        for name in ("left", "right")
    }

    motor = {
        "left": {
            "omega_cmd": omega_cmd_l,
            "omega_actual": _omega_actual["left"],
            "motor_omega": _omega_actual["left"] * 46.0,  # 46:1 gearbox
        },
        "right": {
            "omega_cmd": omega_cmd_r,
            "omega_actual": _omega_actual["right"],
            "motor_omega": _omega_actual["right"] * 46.0,
        },
    }

    return encoder, motor


# ---------------------------------------------------------------------------
# Simulation loop — circular trajectory at 20 Hz
# ---------------------------------------------------------------------------

N_STEPS = 500
DT = 0.05  # 50 ms → 20 Hz sim rate
RADIUS = 8.0  # m
OMEGA_BODY = 0.3  # rad/s turn rate

print(f"Running {N_STEPS} steps ({N_STEPS * DT:.1f} s sim time) …")

for step in range(N_STEPS):
    t = step * DT
    theta = OMEGA_BODY * t

    # Robot pose
    x = RADIUS * math.cos(theta)
    y = RADIUS * math.sin(theta)
    heading = theta + math.pi / 2

    # Body-frame linear velocity (tangential) and angular velocity
    v_body = RADIUS * OMEGA_BODY
    vx = -v_body * math.sin(theta)
    vy = v_body * math.cos(theta)

    bridge.update_pose(x, y, heading, robot_id=0, robot_name="diff_bot")

    # ── IMU ───────────────────────────────────────────────────────────────
    omega_meas, accel_meas = synthetic_imu(x, y, heading, vx, vy, OMEGA_BODY, DT)
    bridge.update_imu(
        omega_meas, accel_meas, robot_id=0, robot_name="diff_bot", sensor_name="imu_0"
    )

    # ── Encoder + Motor ───────────────────────────────────────────────────
    encoder_data, motor_data = synthetic_wheels(v_body, OMEGA_BODY, DT)
    bridge.update_encoder(
        encoder_data, robot_id=0, robot_name="diff_bot", sensor_name="encoder_0"
    )
    bridge.update_motor(
        motor_data, robot_id=0, robot_name="diff_bot", sensor_name="motor_0"
    )

    # ── 2D LiDAR ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if HAS_3D:
        raw = scene.cast_2d_lidar([x, y], z_height=1.2, n_beams=720, range_max=20.0)
        ranges_2d = raw[:, 3].astype(float)
        scan_2d_ms = (time.perf_counter() - t0) * 1000
        bridge.update_lidar2d(
            ranges_2d,
            [x, y, 1.2],
            angle_min=-math.pi,
            angle_max=math.pi,
            robot_id=0,
            robot_name="diff_bot",
            sensor_name="lidar2d_0",
        )
    else:
        n = 720
        angles = np.linspace(-math.pi, math.pi, n, endpoint=False)
        ranges_2d = np.where(np.abs(angles) < 0.2, 0.0, 5.0 + np.random.randn(n) * 0.05)
        scan_2d_ms = (time.perf_counter() - t0) * 1000
        bridge.update_lidar2d(
            ranges_2d,
            [x, y, 1.2],
            angle_min=-math.pi,
            angle_max=math.pi,
            robot_id=0,
            robot_name="diff_bot",
            sensor_name="lidar2d_0",
        )

    # ── 3D LiDAR ──────────────────────────────────────────────────────────
    scan_3d_ms = 0.0
    if HAS_3D:
        t0 = time.perf_counter()
        pts3 = scene.cast_3d_lidar([x, y, 1.5], profile="vlp16", range_max=20.0)
        scan_3d_ms = (time.perf_counter() - t0) * 1000
        bridge.update_lidar3d(
            pts3[:, :3],
            [x, y, 1.5],
            robot_id=0,
            robot_name="diff_bot",
            sensor_name="vlp16_0",
        )

    # ── Metrics ───────────────────────────────────────────────────────────
    cpu_est = (scan_2d_ms + scan_3d_ms) / (DT * 1000) * 100
    bridge.update_metrics(
        scan_2d_ms=scan_2d_ms,
        scan_3d_ms=scan_3d_ms,
        cpu_pct=cpu_est,
        step=step,
    )

    # ── Remote operation (Studio → sim) ───────────────────────────────────
    # Pause/resume from Foxglove Studio "Service Call" panel:
    #   Service: /irsim/control   Body: {"command": "pause"}
    if bridge.is_paused():
        # In a real sim loop: skip env.step() while paused
        print(f"  [step {step}] paused — waiting for resume …", end="\r")
        time.sleep(DT)
        continue

    # Velocity override from Foxglove Studio "Publish" panel:
    #   Topic: /irsim/cmd_vel   Schema: irsim.CmdVel
    #   Body: {"linear": {"x": 1.0}, "angular": {"z": 0.5}}
    cmd = bridge.pop_cmd_vel()
    if cmd is not None:
        # In a real sim loop: pass cmd["linear"] / cmd["angular"] to the robot
        print(
            f"  [step {step}] remote cmd_vel:"
            f" v={cmd['linear']:.2f} m/s  w={cmd['angular']:.2f} rad/s"
        )

    # Goal pose from Foxglove Studio "Publish" panel:
    #   Topic: /irsim/cmd_pose  Schema: irsim.CmdPose
    #   Body: {"robot_id": 0, "x": 5.0, "y": 3.0, "theta": 1.57}
    goal = bridge.pop_cmd_pose()
    if goal is not None:
        print(
            f"  [step {step}] remote goal:"
            f" ({goal.get('x', 0):.1f}, {goal.get('y', 0):.1f})"
            f"  θ={goal.get('theta', 0):.2f}"
        )

    time.sleep(DT)

bridge.stop()
print("Done.")
