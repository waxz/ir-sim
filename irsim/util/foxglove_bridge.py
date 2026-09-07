"""
Foxglove Studio WebSocket bridge for IR-SIM.

Streams sensor frames and performance metrics to Foxglove Studio
at a configurable rate without blocking the simulation loop.

Channels published
------------------
/irsim/lidar2d     foxglove.LaserScan  — 2D horizontal scan
/irsim/lidar3d     foxglove.PointCloud — 3D spinning LiDAR
/irsim/pose        foxglove.PoseInFrame — robot pose (x, y, theta)
/irsim/metrics     irsim.Metrics       — timing and CPU

Usage::

    from irsim.util.foxglove_bridge import FoxgloveBridge

    bridge = FoxgloveBridge(port=8765, publish_hz=10.0)
    bridge.start()

    # inside the sim loop — call as often as you like, bridge rate-limits:
    bridge.update_lidar2d(ranges, origin_xyz, angle_min=-pi, angle_max=pi)
    bridge.update_lidar3d(points_xyz, origin_xyz)   # (N,3) ndarray
    bridge.update_pose(x, y, theta)
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
}

_SCHEMA_NAMES = {
    "/irsim/lidar2d": "foxglove.LaserScan",
    "/irsim/lidar3d": "foxglove.PointCloud",
    "/irsim/pose": "foxglove.PoseInFrame",
    "/irsim/metrics": "irsim.Metrics",
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
    second.

    Parameters
    ----------
    host : str
        Listen address (default ``"0.0.0.0"``).
    port : int
        WebSocket port (default ``8765``).
    publish_hz : float
        Maximum publish rate for each channel (default ``10.0``).
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

        # Shared state: main thread writes, bg thread reads
        self._lock = threading.Lock()
        self._pending: dict[str, tuple[int, bytes]] = {}  # topic → (ts_ns, payload)
        self._last_sent: dict[str, float] = {}

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

        # Drain loop: flush the latest pending message for each channel
        while True:
            await asyncio.sleep(self._period)
            now = time.perf_counter()
            with self._lock:
                snapshot = dict(self._pending)
                self._pending.clear()

            for topic, (ts_ns, payload) in snapshot.items():
                last = self._last_sent.get(topic, 0.0)
                if now - last >= self._period * 0.9 and topic in chan_ids:
                    with contextlib.suppress(Exception):
                        await server.send_message(chan_ids[topic], ts_ns, payload)
                    self._last_sent[topic] = now

        await server.wait_closed()

    # ------------------------------------------------------------------
    # Queue helpers (called from main thread)
    # ------------------------------------------------------------------

    def _queue(self, topic: str, payload: bytes) -> None:
        """Store latest payload; overwrites previous unsent frame."""
        ts_ns = time.time_ns()
        with self._lock:
            self._pending[topic] = (ts_ns, payload)

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
    ) -> None:
        """
        Publish a 2D laser scan.

        Parameters
        ----------
        ranges : (N,) float array — measured ranges in metres (inf / nan for no return)
        origin_xyz : [x, y, z] sensor origin in world frame
        angle_min / angle_max : scan arc in radians
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
            "pose": _pose(ox, oy, oz),
            "start_angle": float(angle_min),
            "end_angle": float(angle_max),
            "ranges": [float(r) if np.isfinite(r) else 0.0 for r in ranges],
            "intensities": [],
        }
        self._queue("/irsim/lidar2d", _encode(msg))

    def update_lidar3d(
        self,
        points_xyz: np.ndarray,
        origin_xyz: list | np.ndarray,
        frame_id: str = "lidar3d",
    ) -> None:
        """
        Publish a 3D point cloud.

        Parameters
        ----------
        points_xyz : (N, 3) float32 array of hit points in world frame
        origin_xyz : [x, y, z] sensor origin
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
            "pose": _pose(ox, oy, oz),
            "point_stride": _PC_STRIDE,
            "fields": _PC_FIELDS,
            "data": data_b64,
        }
        self._queue("/irsim/lidar3d", _encode(msg))

    def update_pose(
        self,
        x: float,
        y: float,
        theta: float,
        z: float = 0.0,
        frame_id: str = "map",
    ) -> None:
        """Publish robot pose (x, y, theta in radians, optional z)."""
        ts_ns = time.time_ns()
        msg = {
            "timestamp": _ts(ts_ns),
            "frame_id": frame_id,
            "pose": _pose(x, y, z, theta),
        }
        self._queue("/irsim/pose", _encode(msg))

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
