"""
Foxglove Studio WebSocket bridge for IR-SIM.

Streams sensor frames and performance metrics to Foxglove Studio
at a configurable rate without blocking the simulation loop.

Channels published
------------------
/irsim/lidar2d     foxglove.LaserScan   — 2D horizontal scan (per robot x sensor)
/irsim/lidar3d     foxglove.PointCloud  — 3D spinning LiDAR  (per robot x sensor)
/irsim/pose        foxglove.PoseInFrame — robot pose (per robot)
/irsim/imu         foxglove.Imu         — IMU measurements    (per robot x sensor)
/irsim/encoder     irsim.Encoder        — wheel encoders      (per robot x sensor)
/irsim/motor       irsim.Motor          — motor telemetry     (per robot x sensor)
/irsim/metrics     irsim.Metrics        — timing and CPU

Multiple robots and multiple sensors per robot are supported on every channel.
The bridge keeps the latest frame for each (robot_id, sensor_name) pair and
publishes it at most ``publish_hz`` times per second.

Usage::

    from irsim.util.foxglove_bridge import FoxgloveBridge

    bridge = FoxgloveBridge(port=8765, publish_hz=10.0)
    bridge.start()

    # inside the sim loop — per-robot, per-sensor calls:
    bridge.update_lidar2d(ranges, origin_xyz,
                          robot_id=r._id, robot_name="robot_0",
                          sensor_name="lidar_front")
    bridge.update_lidar3d(points_xyz, origin_xyz,
                          robot_id=r._id, robot_name="robot_0",
                          sensor_name="vlp16_top")
    bridge.update_pose(x, y, theta,
                       robot_id=r._id, robot_name="robot_0")
    bridge.update_imu(omega_meas, accel_meas,
                      robot_id=r._id, robot_name="robot_0",
                      sensor_name="imu_0")
    bridge.update_encoder(readings,
                          robot_id=r._id, robot_name="robot_0",
                          sensor_name="encoder_0")
    bridge.update_motor(wheel_states,
                        robot_id=r._id, robot_name="robot_0",
                        sensor_name="motor_0")
    bridge.update_metrics(scan_2d_ms=1.2, scan_3d_ms=39.0, cpu_pct=5.0, step=100)

    # Open Foxglove Studio → Connect → ws://localhost:8765

Requires::

    pip install ir-sim[foxglove]
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import threading
import time
from typing import Any

import numpy as np

try:
    from foxglove_websocket.server import FoxgloveServer as _FoxgloveServer

    _HAS_FG = True
except ImportError:  # pragma: no cover
    _HAS_FG = False


# ---------------------------------------------------------------------------
# Foxglove JSON schemas
# ---------------------------------------------------------------------------

_TIME_DEF = {
    "type": "object",
    "properties": {
        "sec": {"type": "integer"},
        "nsec": {"type": "integer"},
    },
}
_VEC3_DEF = {
    "type": "object",
    "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
    },
}
_QUAT_DEF = {
    "type": "object",
    "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
        "w": {"type": "number"},
    },
}
_POSE_DEF = {
    "type": "object",
    "properties": {
        "position": _VEC3_DEF,
        "orientation": _QUAT_DEF,
    },
}
_DEFS = {
    "Time": _TIME_DEF,
    "Vector3": _VEC3_DEF,
    "Quaternion": _QUAT_DEF,
    "Pose": _POSE_DEF,
}

# Identity fields shared by every per-robot/per-sensor schema
_ID_PROPS = {
    "robot_id": {"type": "integer"},
    "robot_name": {"type": "string"},
    "sensor_name": {"type": "string"},
}

_SCHEMAS: dict[str, str] = {
    "/irsim/lidar2d": json.dumps(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "foxglove.LaserScan",
            "type": "object",
            "$defs": _DEFS,
            "properties": {
                "timestamp": {"$ref": "#/$defs/Time"},
                "frame_id": {"type": "string"},
                **_ID_PROPS,
                "pose": {"$ref": "#/$defs/Pose"},
                "start_angle": {"type": "number"},
                "end_angle": {"type": "number"},
                "ranges": {"type": "array", "items": {"type": "number"}},
                "intensities": {"type": "array", "items": {"type": "number"}},
            },
        }
    ),
    "/irsim/lidar3d": json.dumps(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "foxglove.PointCloud",
            "type": "object",
            "$defs": _DEFS,
            "properties": {
                "timestamp": {"$ref": "#/$defs/Time"},
                "frame_id": {"type": "string"},
                **_ID_PROPS,
                "pose": {"$ref": "#/$defs/Pose"},
                "point_stride": {"type": "integer"},
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "offset": {"type": "integer"},
                            "type": {"type": "integer"},
                        },
                    },
                },
                "data": {"type": "string", "contentEncoding": "base64"},
            },
        }
    ),
    "/irsim/pose": json.dumps(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "foxglove.PoseInFrame",
            "type": "object",
            "$defs": _DEFS,
            "properties": {
                "timestamp": {"$ref": "#/$defs/Time"},
                "frame_id": {"type": "string"},
                "robot_id": {"type": "integer"},
                "robot_name": {"type": "string"},
                "pose": {"$ref": "#/$defs/Pose"},
            },
        }
    ),
    "/irsim/metrics": json.dumps(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "irsim.Metrics",
            "type": "object",
            "$defs": {"Time": _TIME_DEF},
            "properties": {
                "timestamp": {"$ref": "#/$defs/Time"},
                "step": {"type": "integer"},
                "sim_time_s": {"type": "number"},
                "scan_2d_ms": {"type": "number"},
                "scan_3d_ms": {"type": "number"},
                "cpu_pct": {"type": "number"},
                "fps": {"type": "number"},
            },
        }
    ),
    "/irsim/imu": json.dumps(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "foxglove.Imu",
            "type": "object",
            "$defs": _DEFS,
            "properties": {
                "timestamp": {"$ref": "#/$defs/Time"},
                "frame_id": {"type": "string"},
                **_ID_PROPS,
                "linear_acceleration": {"$ref": "#/$defs/Vector3"},
                "angular_velocity": {"$ref": "#/$defs/Vector3"},
                "linear_acceleration_covariance": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "angular_velocity_covariance": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            },
        }
    ),
    "/irsim/encoder": json.dumps(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "irsim.Encoder",
            "type": "object",
            "$defs": {"Time": _TIME_DEF},
            "properties": {
                "timestamp": {"$ref": "#/$defs/Time"},
                **_ID_PROPS,
                "wheels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "theta_enc": {"type": "number"},
                            "ticks": {"type": "integer"},
                            "omega": {"type": "number"},
                        },
                    },
                },
            },
        }
    ),
    "/irsim/motor": json.dumps(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "irsim.Motor",
            "type": "object",
            "$defs": {"Time": _TIME_DEF},
            "properties": {
                "timestamp": {"$ref": "#/$defs/Time"},
                **_ID_PROPS,
                "wheels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "omega_cmd": {"type": "number"},
                            "omega_actual": {"type": "number"},
                            "motor_omega": {"type": "number"},
                            "delta_cmd": {"type": "number"},
                            "delta_actual": {"type": "number"},
                        },
                    },
                },
            },
        }
    ),
}

_SCHEMA_NAMES = {
    "/irsim/lidar2d": "foxglove.LaserScan",
    "/irsim/lidar3d": "foxglove.PointCloud",
    "/irsim/pose": "foxglove.PoseInFrame",
    "/irsim/metrics": "irsim.Metrics",
    "/irsim/imu": "foxglove.Imu",
    "/irsim/encoder": "irsim.Encoder",
    "/irsim/motor": "irsim.Motor",
}

# PackedElementField numeric type: 7 = FLOAT32
_PC_FIELDS = [
    {"name": "x", "offset": 0, "type": 7},
    {"name": "y", "offset": 4, "type": 7},
    {"name": "z", "offset": 8, "type": 7},
]
_PC_STRIDE = 12  # 3 x float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(ns: int) -> dict[str, int]:
    return {"sec": ns // 1_000_000_000, "nsec": ns % 1_000_000_000}


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> dict:
    c, s = math.cos(yaw / 2), math.sin(yaw / 2)
    return {
        "position": {"x": x, "y": y, "z": z},
        "orientation": {"x": 0.0, "y": 0.0, "z": s, "w": c},
    }


def _encode(msg: Any) -> bytes:
    return json.dumps(msg).encode()


# ---------------------------------------------------------------------------
# FoxgloveBridge
# ---------------------------------------------------------------------------


class FoxgloveBridge:
    """
    Periodic Foxglove Studio WebSocket bridge.

    Runs a background async server thread.  Call ``update_*`` from the sim
    loop at any rate; the bridge publishes at most ``publish_hz`` times per
    second per (robot, sensor) pair.

    Multiple robots and multiple sensors per robot are fully supported —
    each ``(robot_id, sensor_name)`` pair gets its own pending slot so no
    sensor overwrites another.

    Parameters
    ----------
    host : str
        Listen address (default ``"0.0.0.0"``).
    port : int
        WebSocket port (default ``8765``).
    publish_hz : float
        Maximum publish rate per channel slot (default ``10.0``).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        publish_hz: float = 10.0,
    ) -> None:
        if not _HAS_FG:
            raise ImportError(
                "foxglove-websocket is required.  "
                "Install it with:  pip install ir-sim[foxglove]"
            )
        self._host = host
        self._port = port
        self._period = 1.0 / max(publish_hz, 0.1)

        # Shared state: main thread writes, bg thread reads.
        # Key = (topic, sub_key) where sub_key = "{robot_id}:{sensor_name}"
        # so each (robot, sensor) pair gets its own slot.
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str], tuple[int, bytes]] = {}
        self._last_sent: dict[tuple[str, str], float] = {}

        # Performance counters (updated by update_metrics)
        self._step = 0
        self._sim_t0 = time.perf_counter()
        self._fps_t0 = time.perf_counter()
        self._fps_frames = 0
        self._fps = 0.0

        # Async runtime
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._server: _FoxgloveServer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> FoxgloveBridge:
        """Start the background WebSocket server.  Returns self for chaining."""
        self._thread = threading.Thread(
            target=self._bg, daemon=True, name="foxglove-bridge"
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)
        return self

    def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _bg(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        server = _FoxgloveServer(self._host, self._port, "IR-SIM")
        server.start()
        self._server = server

        chan_ids: dict[str, int] = {}
        for topic, schema in _SCHEMAS.items():
            cid = await server.add_channel(
                {
                    "topic": topic,
                    "encoding": "json",
                    "schemaName": _SCHEMA_NAMES[topic],
                    "schema": schema,
                }
            )
            chan_ids[topic] = cid

        self._ready.set()

        # Drain loop: flush the latest pending frame for each (topic, sub_key) slot
        while True:
            await asyncio.sleep(self._period)
            now = time.perf_counter()
            with self._lock:
                snapshot = dict(self._pending)
                self._pending.clear()

            for (topic, sub_key), (ts_ns, payload) in snapshot.items():
                last = self._last_sent.get((topic, sub_key), 0.0)
                if now - last >= self._period * 0.9 and topic in chan_ids:
                    with contextlib.suppress(Exception):
                        await server.send_message(chan_ids[topic], ts_ns, payload)
                    self._last_sent[(topic, sub_key)] = now

        await server.wait_closed()

    # ------------------------------------------------------------------
    # Queue helpers (called from main thread)
    # ------------------------------------------------------------------

    def _queue(self, topic: str, payload: bytes, sub_key: str = "") -> None:
        """Store latest payload for (topic, sub_key); overwrites previous unsent frame."""
        ts_ns = time.time_ns()
        with self._lock:
            self._pending[(topic, sub_key)] = (ts_ns, payload)

    @staticmethod
    def _sub(robot_id: int, sensor_name: str) -> str:
        return f"{robot_id}:{sensor_name}"

    # ------------------------------------------------------------------
    # Public update methods — call from the sim loop
    # ------------------------------------------------------------------

    def update_lidar2d(
        self,
        ranges: np.ndarray,
        origin_xyz: list | np.ndarray,
        angle_min: float = -math.pi,
        angle_max: float = math.pi,
        frame_id: str = "lidar2d",
        robot_id: int = 0,
        robot_name: str = "robot",
        sensor_name: str = "lidar2d_0",
    ) -> None:
        """
        Publish a 2D laser scan.

        Parameters
        ----------
        ranges : (N,) float array — measured ranges in metres (inf/nan → 0)
        origin_xyz : [x, y, z] sensor origin in world frame
        angle_min / angle_max : scan arc in radians
        frame_id : coordinate frame for Foxglove rendering (e.g. ``"map"``)
        robot_id : unique integer ID of the robot
        robot_name : human-readable robot label
        sensor_name : per-robot sensor identifier (e.g. ``"lidar_front"``)
        """
        ts_ns = time.time_ns()
        ox, oy, oz = (
            float(origin_xyz[0]),
            float(origin_xyz[1]),
            float(origin_xyz[2]) if len(origin_xyz) > 2 else 0.0,
        )
        msg = {
            "timestamp": _ts(ts_ns),
            "frame_id": frame_id,
            "robot_id": int(robot_id),
            "robot_name": str(robot_name),
            "sensor_name": str(sensor_name),
            "pose": _pose(ox, oy, oz),
            "start_angle": float(angle_min),
            "end_angle": float(angle_max),
            "ranges": [float(r) if np.isfinite(r) else 0.0 for r in ranges],
            "intensities": [],
        }
        self._queue("/irsim/lidar2d", _encode(msg), self._sub(robot_id, sensor_name))

    def update_lidar3d(
        self,
        points_xyz: np.ndarray,
        origin_xyz: list | np.ndarray,
        frame_id: str = "lidar3d",
        robot_id: int = 0,
        robot_name: str = "robot",
        sensor_name: str = "lidar3d_0",
    ) -> None:
        """
        Publish a 3D point cloud.

        Parameters
        ----------
        points_xyz : (N, 3) float32 array of hit points in world frame
        origin_xyz : [x, y, z] sensor origin
        frame_id : coordinate frame for Foxglove rendering
        robot_id : unique integer ID of the robot
        robot_name : human-readable robot label
        sensor_name : per-robot sensor identifier (e.g. ``"vlp16_top"``)
        """
        ts_ns = time.time_ns()
        pts = np.asarray(points_xyz, dtype=np.float32)
        ox, oy, oz = (
            float(origin_xyz[0]),
            float(origin_xyz[1]),
            float(origin_xyz[2]) if len(origin_xyz) > 2 else 0.0,
        )

        # Pack as contiguous float32 binary, base64-encode for JSON transport
        data_b64 = base64.b64encode(pts[:, :3].tobytes()).decode()

        msg = {
            "timestamp": _ts(ts_ns),
            "frame_id": frame_id,
            "robot_id": int(robot_id),
            "robot_name": str(robot_name),
            "sensor_name": str(sensor_name),
            "pose": _pose(ox, oy, oz),
            "point_stride": _PC_STRIDE,
            "fields": _PC_FIELDS,
            "data": data_b64,
        }
        self._queue("/irsim/lidar3d", _encode(msg), self._sub(robot_id, sensor_name))

    def update_pose(
        self,
        x: float,
        y: float,
        theta: float,
        z: float = 0.0,
        frame_id: str = "map",
        robot_id: int = 0,
        robot_name: str = "robot",
    ) -> None:
        """
        Publish robot pose (x, y, theta in radians, optional z).

        Parameters
        ----------
        robot_id : unique integer ID of the robot
        robot_name : human-readable robot label
        """
        ts_ns = time.time_ns()
        msg = {
            "timestamp": _ts(ts_ns),
            "frame_id": frame_id,
            "robot_id": int(robot_id),
            "robot_name": str(robot_name),
            "pose": _pose(x, y, z, theta),
        }
        self._queue("/irsim/pose", _encode(msg), str(robot_id))

    def update_metrics(
        self,
        scan_2d_ms: float = 0.0,
        scan_3d_ms: float = 0.0,
        cpu_pct: float = 0.0,
        step: int | None = None,
    ) -> None:
        """
        Publish performance metrics.

        Parameters
        ----------
        scan_2d_ms : 2D LiDAR cast time in milliseconds
        scan_3d_ms : 3D LiDAR cast time in milliseconds
        cpu_pct : estimated CPU usage percentage (scan_ms x hz / 10)
        step : simulation step counter (auto-increments if None)
        """
        ts_ns = time.time_ns()
        if step is not None:
            self._step = step
        else:
            self._step += 1

        self._fps_frames += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 1.0:
            self._fps = self._fps_frames / (now - self._fps_t0)
            self._fps_frames = 0
            self._fps_t0 = now

        msg = {
            "timestamp": _ts(ts_ns),
            "step": self._step,
            "sim_time_s": round(now - self._sim_t0, 3),
            "scan_2d_ms": round(scan_2d_ms, 3),
            "scan_3d_ms": round(scan_3d_ms, 3),
            "cpu_pct": round(cpu_pct, 2),
            "fps": round(self._fps, 1),
        }
        self._queue("/irsim/metrics", _encode(msg))

    def update_imu(
        self,
        angular_velocity: np.ndarray,
        linear_acceleration: np.ndarray,
        frame_id: str = "imu",
        robot_id: int = 0,
        robot_name: str = "robot",
        sensor_name: str = "imu_0",
    ) -> None:
        """
        Publish IMU measurements.

        Parameters
        ----------
        angular_velocity : array-like, shape (3,) — [wx, wy, wz] in rad/s
        linear_acceleration : array-like, shape (3,) — [ax, ay, az] in m/s²
            (az includes static gravity ~+9.807 m/s² for a level robot)
        frame_id : coordinate frame of the sensor
        robot_id : unique integer ID of the robot
        robot_name : human-readable robot label
        sensor_name : per-robot IMU identifier (e.g. ``"imu_front"``)
        """
        ts_ns = time.time_ns()
        av = np.asarray(angular_velocity).ravel()
        la = np.asarray(linear_acceleration).ravel()
        msg = {
            "timestamp": _ts(ts_ns),
            "frame_id": frame_id,
            "robot_id": int(robot_id),
            "robot_name": str(robot_name),
            "sensor_name": str(sensor_name),
            "angular_velocity": {
                "x": float(av[0]),
                "y": float(av[1]),
                "z": float(av[2]),
            },
            "linear_acceleration": {
                "x": float(la[0]),
                "y": float(la[1]),
                "z": float(la[2]),
            },
            "angular_velocity_covariance": [],
            "linear_acceleration_covariance": [],
        }
        self._queue("/irsim/imu", _encode(msg), self._sub(robot_id, sensor_name))

    def update_encoder(
        self,
        readings: dict,
        robot_id: int = 0,
        robot_name: str = "robot",
        sensor_name: str = "encoder_0",
    ) -> None:
        """
        Publish wheel encoder readings.

        Parameters
        ----------
        readings : dict
            Keyed by wheel name.  Each value is a dict with any subset of:
            ``{"theta_enc": float, "ticks": int, "omega_actual": float}``.
            Compatible with ``ObjectBase.encoder_readings`` and
            ``WheelLayout.get_encoder_readings()``.
        robot_id : unique integer ID of the robot (matches ``ObjectBase._id``)
        robot_name : human-readable label (e.g. ``"robot_0"``)
        sensor_name : per-robot encoder unit identifier (e.g. ``"enc_left_axle"``)

        Example
        -------
        ::

            bridge.update_encoder(robot.encoder_readings,
                                  robot_id=robot._id, robot_name="diff_bot",
                                  sensor_name="encoder_0")
        """
        ts_ns = time.time_ns()
        wheels = [
            {
                "name": name,
                "theta_enc": float(r.get("theta_enc", 0.0)),
                "ticks": int(r.get("ticks", 0)),
                "omega": float(r.get("omega_actual", r.get("omega", 0.0))),
            }
            for name, r in readings.items()
        ]
        msg = {
            "timestamp": _ts(ts_ns),
            "robot_id": int(robot_id),
            "robot_name": str(robot_name),
            "sensor_name": str(sensor_name),
            "wheels": wheels,
        }
        self._queue("/irsim/encoder", _encode(msg), self._sub(robot_id, sensor_name))

    def update_motor(
        self,
        wheel_states: dict,
        robot_id: int = 0,
        robot_name: str = "robot",
        sensor_name: str = "motor_0",
    ) -> None:
        """
        Publish motor telemetry.

        Parameters
        ----------
        wheel_states : dict
            Keyed by wheel name.  Each value may be a ``WheelState`` dataclass
            instance or a plain dict with any subset of:
            ``{"omega_cmd", "omega_actual", "motor_omega",
               "delta_cmd", "delta_actual"}``.
            Compatible with ``WheelLayout.get_wheel_states()`` and
            ``ObjectBase.wheel_states``.
        robot_id : unique integer ID of the robot (matches ``ObjectBase._id``)
        robot_name : human-readable label (e.g. ``"robot_0"``)
        sensor_name : per-robot motor controller identifier (e.g. ``"motor_ctrl_0"``)

        Example
        -------
        ::

            bridge.update_motor(robot.wheel_states,
                                robot_id=robot._id, robot_name="diff_bot",
                                sensor_name="motor_0")
        """
        ts_ns = time.time_ns()

        def _get(w: Any, key: str, default: float = 0.0) -> float:
            if isinstance(w, dict):
                return float(w.get(key, default))
            return float(getattr(w, key, default))

        wheels = [
            {
                "name": name,
                "omega_cmd": _get(w, "omega_cmd"),
                "omega_actual": _get(w, "omega_actual"),
                "motor_omega": _get(w, "motor_omega"),
                "delta_cmd": _get(w, "delta_cmd"),
                "delta_actual": _get(w, "delta_actual"),
            }
            for name, w in wheel_states.items()
        ]
        msg = {
            "timestamp": _ts(ts_ns),
            "robot_id": int(robot_id),
            "robot_name": str(robot_name),
            "sensor_name": str(sensor_name),
            "wheels": wheels,
        }
        self._queue("/irsim/motor", _encode(msg), self._sub(robot_id, sensor_name))
