"""
Headless WebSocket verification client for FoxgloveBridge.

Starts the bridge, pumps synthetic telemetry, connects as a raw WebSocket
client (using the ``websockets`` library), and verifies every layer of the
Foxglove WebSocket v1 protocol:

  1. serverInfo     — capabilities, supportedEncodings
  2. Channel adverts — all 7 topics present with correct schemaNames
  3. Service advert  — /irsim/control registered
  4. Data reception  — subscribe to all channels, assert a frame arrives on each
  5. Service call    — pause / resume / reset round-trip over binary framing
  6. Client publish  — cmd_vel and cmd_pose messages received by bridge

Wire-format reference (foxglove.websocket.v1)
----------------------------------------------
Server → client binary (MESSAGE_DATA)  : [op=1:u8][sub_id:u32][ts_ns:u64][payload]
Client → server binary (publish)       : [op=1:u8][chan_id:u32][payload]
Client → server binary (service call)  : [op=2:u8][svc_id:u32][call_id:u32][enc_len:u32][enc][payload]
Server → client binary (svc response)  : [op=3:u8][svc_id:u32][call_id:u32][enc_len:u32][enc][response]

All struct formats use little-endian.

Exits 0 on all-pass, 1 on any failure.

Usage::

    python usage/verify_bridge.py
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Test bridge on a non-default port to avoid clashing with a running server
# ---------------------------------------------------------------------------

_PORT = 18765
_PUBLISH_HZ = 20.0  # fast drain loop so tests finish quickly

# ---------------------------------------------------------------------------
# Binary frame helpers
# ---------------------------------------------------------------------------

_MSG_HDR = struct.Struct("<BIQ")  # SERVER→client: op, sub_id, ts_ns
_PUB_HDR = struct.Struct("<BI")  # client→SERVER: op=1, chan_id
_SVC_HDR = struct.Struct("<BIII")  # both directions: op, svc/call/enc_len

_OP_MSG = 1  # BinaryOpcode.MESSAGE_DATA
_OP_SVC_RESP = 3  # BinaryOpcode.SERVICE_CALL_RESPONSE
_CL_OP_PUB = 1  # ClientBinaryOpcode.MESSAGE_DATA
_CL_OP_SVC = 2  # ClientBinaryOpcode.SERVICE_CALL_REQUEST


def _svc_request(svc_id: int, call_id: int, payload: bytes) -> bytes:
    enc = b"json"
    hdr = _SVC_HDR.pack(_CL_OP_SVC, svc_id, call_id, len(enc))
    return hdr + enc + payload


def _client_publish(chan_id: int, payload: bytes) -> bytes:
    return _PUB_HDR.pack(_CL_OP_PUB, chan_id) + payload


# ---------------------------------------------------------------------------
# Minimal result collector
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _check(label: str, cond: bool, detail: str = "") -> None:
    _results.append((label, cond, detail))
    tag = "PASS" if cond else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {label}{suffix}")


# ---------------------------------------------------------------------------
# Async verification coroutine
# ---------------------------------------------------------------------------

_EXPECTED = {
    "/irsim/pose": "foxglove.PoseInFrame",
    "/irsim/lidar2d": "foxglove.LaserScan",
    "/irsim/lidar3d": "foxglove.PointCloud",
    "/irsim/imu": "foxglove.Imu",
    "/irsim/encoder": "irsim.Encoder",
    "/irsim/motor": "irsim.Motor",
    "/irsim/metrics": "irsim.Metrics",
}


async def _verify(bridge) -> None:
    import websockets

    uri = f"ws://localhost:{_PORT}"

    async with websockets.connect(
        uri, subprotocols=["foxglove.websocket.v1"], open_timeout=5
    ) as ws:
        # ------------------------------------------------------------------
        # 1. serverInfo
        # ------------------------------------------------------------------
        print("\n--- 1. serverInfo ---")
        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        msg = json.loads(raw)
        _check("op == serverInfo", msg.get("op") == "serverInfo")
        caps = msg.get("capabilities", [])
        _check("capability: clientPublish", "clientPublish" in caps, str(caps))
        _check("capability: services", "services" in caps, str(caps))
        encs = msg.get("supportedEncodings") or []
        _check("supportedEncodings: json", "json" in encs, str(encs))

        # ------------------------------------------------------------------
        # 2. Channel advertisements
        # ------------------------------------------------------------------
        print("\n--- 2. Channel advertisements ---")
        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        msg = json.loads(raw)
        _check("op == advertise", msg.get("op") == "advertise")
        channels: dict[str, dict] = {ch["topic"]: ch for ch in msg.get("channels", [])}
        for topic, schema_name in _EXPECTED.items():
            got = channels.get(topic, {}).get("schemaName", "MISSING")
            _check(
                f"channel {topic}",
                got == schema_name,
                f"got schemaName={got!r}",
            )

        # ------------------------------------------------------------------
        # 3. Service advertisements
        # ------------------------------------------------------------------
        print("\n--- 3. Service advertisements ---")
        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        msg = json.loads(raw)
        _check("op == advertiseServices", msg.get("op") == "advertiseServices")
        services: dict[str, dict] = {
            svc["name"]: svc for svc in msg.get("services", [])
        }
        _check("/irsim/control advertised", "/irsim/control" in services)

        # ------------------------------------------------------------------
        # 4. Data reception — subscribe, pump, collect frames
        # ------------------------------------------------------------------
        print("\n--- 4. Data reception ---")
        subs = [
            {"id": i + 1, "channelId": ch["id"]}
            for i, ch in enumerate(channels.values())
        ]
        sub_id_to_topic = {i + 1: topic for i, topic in enumerate(channels.keys())}
        await ws.send(json.dumps({"op": "subscribe", "subscriptions": subs}))

        # Pump one frame of telemetry so the drain loop has something to send
        _pump_telemetry(bridge)

        # Collect for two drain periods
        drain_period = 1.0 / _PUBLISH_HZ
        wait_s = drain_period * 3.0
        received: dict[str, bytes] = {}
        deadline = asyncio.get_event_loop().time() + wait_s
        while asyncio.get_event_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=drain_period)
            except asyncio.TimeoutError:
                continue
            if not isinstance(raw, bytes):
                continue  # text status messages
            if len(raw) < _MSG_HDR.size:
                continue
            op = raw[0]
            if op != _OP_MSG:
                continue
            _, sub_id, _ = _MSG_HDR.unpack_from(raw)
            topic = sub_id_to_topic.get(sub_id)
            if topic and topic not in received:
                payload = raw[_MSG_HDR.size :]
                received[topic] = payload

        for topic in _EXPECTED:
            have = topic in received
            _check(
                f"data on {topic}", have, f"payload {len(received.get(topic, b''))} B"
            )

        # Spot-check one decoded pose message
        if "/irsim/pose" in received:
            try:
                pose_msg = json.loads(received["/irsim/pose"])
                has_pose = "pose" in pose_msg
                _check(
                    "pose payload decodable",
                    has_pose,
                    list(pose_msg.keys())[:5],
                )
            except Exception as exc:
                _check("pose payload decodable", False, str(exc))

        # ------------------------------------------------------------------
        # 5a. Service call: pause
        # ------------------------------------------------------------------
        print("\n--- 5. Service calls ---")
        svc_id = services["/irsim/control"]["id"]
        call_id = 7

        await ws.send(_svc_request(svc_id, call_id, b'{"command":"pause"}'))
        resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
        op, _r_svc, r_call, enc_len = _SVC_HDR.unpack_from(resp)
        r_payload = resp[_SVC_HDR.size + enc_len :]
        r_msg = json.loads(r_payload)
        _check("pause: op == SERVICE_CALL_RESPONSE", op == _OP_SVC_RESP)
        _check("pause: call_id echoed", r_call == call_id)
        _check("pause: ok=True", r_msg.get("ok") is True, str(r_msg))
        _check("pause: bridge.is_paused()", bridge.is_paused())

        # 5b. resume
        call_id += 1
        await ws.send(_svc_request(svc_id, call_id, b'{"command":"resume"}'))
        resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
        _, _, r_call, enc_len = _SVC_HDR.unpack_from(resp)
        r_msg = json.loads(resp[_SVC_HDR.size + enc_len :])
        _check("resume: ok=True", r_msg.get("ok") is True, str(r_msg))
        _check("resume: bridge not paused", not bridge.is_paused())

        # 5c. reset
        call_id += 1
        await ws.send(_svc_request(svc_id, call_id, b'{"command":"reset"}'))
        resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
        _, _, r_call, enc_len = _SVC_HDR.unpack_from(resp)
        r_msg = json.loads(resp[_SVC_HDR.size + enc_len :])
        _check("reset: ok=True", r_msg.get("ok") is True, str(r_msg))

        # 5d. unknown command → ok=False
        call_id += 1
        await ws.send(_svc_request(svc_id, call_id, b'{"command":"explode"}'))
        resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
        _, _, _, enc_len = _SVC_HDR.unpack_from(resp)
        r_msg = json.loads(resp[_SVC_HDR.size + enc_len :])
        _check("bad command: ok=False", r_msg.get("ok") is False, str(r_msg))

        # ------------------------------------------------------------------
        # 6. Client publish (Studio → sim)
        # ------------------------------------------------------------------
        print("\n--- 6. Client publish ---")
        client_chan_cmd_vel = 1
        client_chan_cmd_pose = 2

        await ws.send(
            json.dumps(
                {
                    "op": "advertise",
                    "channels": [
                        {
                            "id": client_chan_cmd_vel,
                            "topic": "/irsim/cmd_vel",
                            "encoding": "json",
                            "schemaName": "irsim.CmdVel",
                        },
                        {
                            "id": client_chan_cmd_pose,
                            "topic": "/irsim/cmd_pose",
                            "encoding": "json",
                            "schemaName": "irsim.CmdPose",
                        },
                    ],
                }
            )
        )
        await asyncio.sleep(0.05)  # let listener process the advertise

        # Publish cmd_vel
        cmd_vel_payload = json.dumps(
            {"linear": {"x": 1.5}, "angular": {"z": 0.3}}
        ).encode()
        await ws.send(_client_publish(client_chan_cmd_vel, cmd_vel_payload))
        await asyncio.sleep(0.05)

        cmd = bridge.pop_cmd_vel()
        _check("cmd_vel received", cmd is not None, str(cmd))
        if cmd is not None:
            _check(
                "cmd_vel linear=1.5",
                abs(cmd["linear"] - 1.5) < 1e-6,
                f"got {cmd['linear']}",
            )
            _check(
                "cmd_vel angular=0.3",
                abs(cmd["angular"] - 0.3) < 1e-6,
                f"got {cmd['angular']}",
            )

        # Publish cmd_pose
        cmd_pose_payload = json.dumps(
            {"robot_id": 0, "x": 5.0, "y": -3.0, "theta": 1.57}
        ).encode()
        await ws.send(_client_publish(client_chan_cmd_pose, cmd_pose_payload))
        await asyncio.sleep(0.05)

        goal = bridge.pop_cmd_pose()
        _check("cmd_pose received", goal is not None, str(goal))
        if goal is not None:
            _check(
                "cmd_pose x=5.0",
                abs(goal.get("x", 0) - 5.0) < 1e-6,
                f"got {goal.get('x')}",
            )


# ---------------------------------------------------------------------------
# Synthetic telemetry pump
# ---------------------------------------------------------------------------


def _pump_telemetry(bridge) -> None:
    """Push one frame of synthetic data into the bridge's pending queue."""
    x, y, theta = 1.0, 2.0, math.pi / 4
    bridge.update_pose(x, y, theta, robot_id=0, robot_name="verify_bot")

    omega = np.array([0.0, 0.0, 0.1])
    accel = np.array([0.01, 0.0, 9.81])
    bridge.update_imu(
        omega, accel, robot_id=0, robot_name="verify_bot", sensor_name="imu_0"
    )

    ranges = np.full(360, 5.0)
    bridge.update_lidar2d(
        ranges,
        [x, y, 1.2],
        angle_min=-math.pi,
        angle_max=math.pi,
        robot_id=0,
        robot_name="verify_bot",
        sensor_name="lidar2d_0",
    )

    # Minimal fake 3D point cloud (10 points)
    pts3d = np.random.rand(10, 3).astype(np.float32)
    bridge.update_lidar3d(
        pts3d, [x, y, 1.5], robot_id=0, robot_name="verify_bot", sensor_name="vlp16_0"
    )

    encoder_data = {
        "left": {"theta_enc": 1.0, "ticks": 32, "omega_actual": 10.0},
        "right": {"theta_enc": 1.0, "ticks": 32, "omega_actual": 10.0},
    }
    bridge.update_encoder(
        encoder_data, robot_id=0, robot_name="verify_bot", sensor_name="enc_0"
    )

    motor_data = {
        "left": {"omega_cmd": 10.0, "omega_actual": 9.9, "motor_omega": 456.0},
        "right": {"omega_cmd": 10.0, "omega_actual": 9.9, "motor_omega": 456.0},
    }
    bridge.update_motor(
        motor_data, robot_id=0, robot_name="verify_bot", sensor_name="mot_0"
    )

    bridge.update_metrics(scan_2d_ms=1.2, scan_3d_ms=4.5, cpu_pct=3.0, step=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"Starting FoxgloveBridge on port {_PORT} at {_PUBLISH_HZ} Hz …")

    try:
        from irsim.util.foxglove_bridge import FoxgloveBridge
    except ImportError as exc:
        print(f"ERROR: could not import FoxgloveBridge — {exc}")
        return 1

    bridge = FoxgloveBridge(port=_PORT, publish_hz=_PUBLISH_HZ)
    bridge.start()
    time.sleep(0.3)  # wait for asyncio event loop and WebSocket server to open

    print("Running verification …")
    try:
        asyncio.run(_verify(bridge))
    except Exception as exc:
        print(f"\nFATAL: unhandled exception in test coroutine: {exc}")
        bridge.stop()
        return 1
    finally:
        bridge.stop()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    failed = [(lbl, detail) for lbl, ok, detail in _results if not ok]

    print(f"\n{'=' * 55}")
    print(f"Result: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({len(failed)} FAILED)")
        for lbl, detail in failed:
            print(f"  - {lbl}" + (f": {detail}" if detail else ""))
    else:
        print("  — all OK")
    print("=" * 55)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
