"""
Foxglove Studio WebSocket bridge for IR-SIM.

Bidirectional: streams sensor telemetry to Foxglove Studio and receives
remote-operation commands back from it, all over a single WebSocket server.

Channels published (sim → Studio)
----------------------------------
/irsim/lidar2d     foxglove.LaserScan   — 2D horizontal scan (per robot x sensor)
/irsim/lidar3d     foxglove.PointCloud  — 3D spinning LiDAR  (per robot x sensor)
/irsim/pose        foxglove.PoseInFrame — robot pose (per robot)
/irsim/imu         foxglove.Imu         — IMU measurements    (per robot x sensor)
/irsim/encoder     irsim.Encoder        — wheel encoders      (per robot x sensor)
/irsim/motor       irsim.Motor          — motor telemetry     (per robot x sensor)
/irsim/metrics     irsim.Metrics        — timing and CPU

Client channels received (Studio → sim)
-----------------------------------------
/irsim/cmd_vel     irsim.CmdVel  — velocity command  {"linear": float, "angular": float}
/irsim/cmd_pose    irsim.CmdPose — goal pose         {"x", "y", "theta", "robot_id"}
<any topic>        any schema    — get_latest(topic) returns raw decoded dict

Services (Studio → sim, request/response)
------------------------------------------
/irsim/control     irsim.Control — {"command": "pause"|"resume"|"reset"}
                                   response: {"ok": bool, "message": str}

Multiple robots and multiple sensors per robot are supported on every channel.
The bridge keeps the latest frame for each (robot_id, sensor_name) pair and
publishes it at most ``publish_hz`` times per second.

Usage::

    from irsim.util.foxglove_bridge import FoxgloveBridge

    bridge = FoxgloveBridge(port=8765, publish_hz=10.0)
    bridge.start()

    # Publish telemetry (sim → Studio)
    bridge.update_pose(x, y, theta, robot_id=r._id, robot_name="robot_0")
    bridge.update_lidar2d(ranges, origin_xyz, robot_id=r._id, sensor_name="lidar_front")

    # Receive remote-operation commands (Studio → sim) — non-blocking poll
    cmd = bridge.pop_cmd_vel()           # {"linear": float, "angular": float} or None
    goal = bridge.pop_cmd_pose()         # {"x", "y", "theta", "robot_id"} or None
    msg = bridge.get_latest("/my/topic") # raw dict or None

    if bridge.is_paused():               # True while Studio sent "pause"
        continue

    # Open Foxglove Studio → Connect → ws://localhost:8765
    # Publish panel: topic=/irsim/cmd_vel  {"linear":{"x":0.5},"angular":{"z":0.3}}
    # Service Call panel: /irsim/control   {"command":"pause"}

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
    from foxglove_websocket.server import (
        FoxgloveServerListener as _FoxgloveServerListener,
    )
    from foxglove_websocket.types import ClientChannelId as _ClientChannelId

    _HAS_FG = True
except ImportError:  # pragma: no cover
    _HAS_FG = False
    _FoxgloveServerListener = object  # type: ignore[assignment,misc]
    _ClientChannelId = int  # type: ignore[assignment,misc]


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
# Service schemas (Studio → sim)
# ---------------------------------------------------------------------------

_SVC_CONTROL_REQ = json.dumps(
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "irsim.ControlRequest",
        "type": "object",
        "properties": {
            "command": {"type": "string", "enum": ["pause", "resume", "reset"]},
        },
        "required": ["command"],
    }
)
_SVC_CONTROL_RESP = json.dumps(
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "irsim.ControlResponse",
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "message": {"type": "string"},
        },
    }
)

# Client-channel schemas (documented here; the client declares them)
_CLIENT_SCHEMAS = {
    "/irsim/cmd_vel": json.dumps(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "irsim.CmdVel",
            "type": "object",
            "description": "Velocity command from Foxglove Studio to IR-SIM.",
            "properties": {
                "robot_id": {"type": "integer", "default": 0},
                "linear": {
                    "oneOf": [
                        {"type": "number"},
                        {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                        },
                    ]
                },
                "angular": {
                    "oneOf": [
                        {"type": "number"},
                        {
                            "type": "object",
                            "properties": {"z": {"type": "number"}},
                        },
                    ]
                },
            },
        }
    ),
    "/irsim/cmd_pose": json.dumps(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "irsim.CmdPose",
            "type": "object",
            "description": "Goal pose command from Foxglove Studio to IR-SIM.",
            "properties": {
                "robot_id": {"type": "integer", "default": 0},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "theta": {"type": "number"},
            },
        }
    ),
}


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
# Listener — receives messages and service calls from Foxglove Studio
# ---------------------------------------------------------------------------


class _BridgeListener(_FoxgloveServerListener):
    """Routes incoming client messages and service calls into the bridge RX buffer."""

    def __init__(self, bridge: FoxgloveBridge) -> None:
        self._bridge = bridge
        # Map assigned channel_id → topic string (filled in on_client_advertise)
        self._ch_map: dict[int, str] = {}

    async def on_client_advertise(self, server: Any, channel: Any) -> None:
        self._ch_map[channel["id"]] = channel["topic"]

    async def on_client_unadvertise(self, server: Any, channel_id: int) -> None:
        self._ch_map.pop(channel_id, None)

    async def on_client_message(
        self, server: Any, channel_id: int, payload: bytes
    ) -> None:
        topic = self._ch_map.get(channel_id)
        if topic is None:
            return
        with contextlib.suppress(Exception):
            msg = json.loads(payload)
            with self._bridge._rx_lock:
                self._bridge._rx_buf[topic] = (time.time(), msg)

    async def on_service_request(
        self,
        server: Any,
        service_id: int,
        call_id: str,
        encoding: str,
        payload: bytes,
    ) -> bytes:
        svc_name = self._bridge._svc_map.get(service_id, "")
        if svc_name == "/irsim/control":
            with contextlib.suppress(Exception):
                req = json.loads(payload)
                cmd = str(req.get("command", "")).lower()
                if cmd == "pause":
                    self._bridge._paused = True
                    return _encode({"ok": True, "message": "paused"})
                if cmd == "resume":
                    self._bridge._paused = False
                    return _encode({"ok": True, "message": "resumed"})
                if cmd == "reset":
                    self._bridge._paused = False
                    with self._bridge._rx_lock:
                        self._bridge._rx_buf.clear()
                    return _encode({"ok": True, "message": "reset"})
                return _encode({"ok": False, "message": f"unknown command: {cmd}"})
        return _encode({"ok": False, "message": "unknown service"})


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

        # TX: main thread writes, bg thread reads.
        # Key = (topic, sub_key) where sub_key = "{robot_id}:{sensor_name}"
        # so each (robot, sensor) pair gets its own slot.
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str], tuple[int, bytes]] = {}
        self._last_sent: dict[tuple[str, str], float] = {}

        # RX: bg thread writes (from Studio), main thread reads (sim loop polls).
        self._rx_lock = threading.Lock()
        self._rx_buf: dict[str, tuple[float, Any]] = {}  # topic → (ts, msg)
        self._paused = False  # set by /irsim/control service
        self._svc_map: dict[int, str] = {}  # service_id → service name

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

        # Attach listener before start() so no events are missed
        server.set_listener(_BridgeListener(self))
        server.start()
        self._server = server

        # Register TX channels (sim → Studio)
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

        # Register /irsim/control service (Studio → sim)
        svc_id = await server.add_service(
            {
                "name": "/irsim/control",
                "type": "irsim.Control",
                "request": {
                    "encoding": "json",
                    "schemaName": "irsim.ControlRequest",
                    "schemaEncoding": "jsonschema",
                    "schema": _SVC_CONTROL_REQ,
                },
                "response": {
                    "encoding": "json",
                    "schemaName": "irsim.ControlResponse",
                    "schemaEncoding": "jsonschema",
                    "schema": _SVC_CONTROL_RESP,
                },
            }
        )
        self._svc_map[svc_id] = "/irsim/control"

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

    # ------------------------------------------------------------------
    # Remote-operation receive methods — poll from the sim loop
    # ------------------------------------------------------------------

    def get_latest(self, topic: str) -> dict | None:
        """
        Return and consume the latest message received on *topic*, or ``None``.

        Thread-safe; non-blocking.  The message is removed from the buffer so
        the next call returns ``None`` until a new message arrives.

        Parameters
        ----------
        topic : str
            Any topic the Foxglove Studio "Publish" panel has sent to this
            bridge, e.g. ``"/irsim/cmd_vel"`` or ``"/irsim/cmd_pose"``.
        """
        with self._rx_lock:
            entry = self._rx_buf.pop(topic, None)
        return None if entry is None else entry[1]

    def pop_cmd_vel(self, robot_id: int = 0) -> dict | None:
        """
        Return the latest velocity command for *robot_id* and clear it.

        The Foxglove Studio "Publish" panel should publish on ``/irsim/cmd_vel``
        with schema ``irsim.CmdVel``::

            {"robot_id": 0, "linear": {"x": 0.5}, "angular": {"z": 0.3}}

        Returns a normalised dict ``{"linear": float, "angular": float, "robot_id": int}``
        or ``None`` if no command has arrived since the last call.
        """
        msg = self.get_latest("/irsim/cmd_vel")
        if msg is None:
            return None
        # Accept both flat scalars and nested {"x":…} / {"z":…} objects
        lin = msg.get("linear", 0.0)
        ang = msg.get("angular", 0.0)
        return {
            "robot_id": int(msg.get("robot_id", robot_id)),
            "linear": float(lin["x"] if isinstance(lin, dict) else lin),
            "angular": float(ang["z"] if isinstance(ang, dict) else ang),
        }

    def pop_cmd_pose(self) -> dict | None:
        """
        Return the latest goal-pose command and clear it.

        The Foxglove Studio "Publish" panel should publish on ``/irsim/cmd_pose``
        with schema ``irsim.CmdPose``::

            {"robot_id": 0, "x": 5.0, "y": 3.0, "theta": 1.57}

        Returns the raw decoded dict or ``None`` if no command has arrived.
        """
        return self.get_latest("/irsim/cmd_pose")

    def is_paused(self) -> bool:
        """
        Return ``True`` while Foxglove Studio has paused the simulation.

        Set by the ``/irsim/control`` service (``{"command": "pause"}``).
        Cleared by ``{"command": "resume"}`` or ``{"command": "reset"}``.
        Can also be toggled from the sim side via :meth:`set_paused`.
        """
        return self._paused

    def set_paused(self, paused: bool) -> None:
        """Override the pause state from the simulation side."""
        self._paused = paused

    @property
    def client_schemas(self) -> dict[str, str]:
        """
        JSON schemas for the client channels this bridge expects to receive.

        Keys are topic names; values are JSON Schema strings.  Useful for
        documentation or for generating a Foxglove panel config.
        """
        return dict(_CLIENT_SCHEMAS)
