"""
Foxglove Studio integration demo for IR-SIM.

Streams 2D/3D LiDAR, robot pose, and performance metrics to Foxglove Studio
at 10 Hz while the simulation runs at full speed.

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
# Optional: build a 3D scene with Scene3D (requires ir-sim[lidar3d])
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
# Simple circular robot trajectory
# ---------------------------------------------------------------------------

N_STEPS = 500
DT = 0.05  # 50 ms → 20 Hz sim rate
RADIUS = 8.0  # m
OMEGA = 0.3  # rad/s

print(f"Running {N_STEPS} steps ({N_STEPS * DT:.1f} s sim time) …")

for step in range(N_STEPS):
    t = step * DT
    theta = OMEGA * t

    # Robot pose
    x = RADIUS * math.cos(theta)
    y = RADIUS * math.sin(theta)
    heading = theta + math.pi / 2
    bridge.update_pose(x, y, heading)

    # ── 2D LiDAR (synthetic) ──────────────────────────────────────────────
    t0 = time.perf_counter()
    if HAS_3D:
        raw = scene.cast_2d_lidar([x, y], z_height=1.2, n_beams=720, range_max=20.0)
        ranges_2d = raw[:, 3].astype(float)  # column 3 = distance
        scan_2d_ms = (time.perf_counter() - t0) * 1000
        bridge.update_lidar2d(
            ranges_2d, [x, y, 1.2], angle_min=-math.pi, angle_max=math.pi
        )
    else:
        # Synthetic ring with a gap
        n = 720
        angles = np.linspace(-math.pi, math.pi, n, endpoint=False)
        ranges_2d = np.where(np.abs(angles) < 0.2, 0.0, 5.0 + np.random.randn(n) * 0.05)
        scan_2d_ms = (time.perf_counter() - t0) * 1000
        bridge.update_lidar2d(
            ranges_2d, [x, y, 1.2], angle_min=-math.pi, angle_max=math.pi
        )

    # ── 3D LiDAR (VLP-16, only when Scene3D available) ───────────────────
    scan_3d_ms = 0.0
    if HAS_3D:
        t0 = time.perf_counter()
        pts3 = scene.cast_3d_lidar([x, y, 1.5], profile="vlp16", range_max=20.0)
        scan_3d_ms = (time.perf_counter() - t0) * 1000
        bridge.update_lidar3d(pts3[:, :3], [x, y, 1.5])

    # ── Metrics ───────────────────────────────────────────────────────────
    cpu_est = (scan_2d_ms + scan_3d_ms) / (DT * 1000) * 100
    bridge.update_metrics(
        scan_2d_ms=scan_2d_ms,
        scan_3d_ms=scan_3d_ms,
        cpu_pct=cpu_est,
        step=step,
    )

    time.sleep(DT)

bridge.stop()
print("Done.")
