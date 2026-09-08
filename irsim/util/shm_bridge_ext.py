"""
shm_bridge_ext.py -- Extended POSIX shm bridge with IMU, encoder, and
large point-cloud channels.

Segment layout (1 049 344 bytes total):
  offset      0 : IrsimExtHeader    128 B  ready flag + channel metadata
  offset    128 : IrsimStateSlot    128 B  robot state  (from base bridge)
  offset    256 : IrsimCmdSlot      128 B  velocity cmd (from base bridge)
  offset    384 : IrsimImuSlot      128 B  6-axis IMU @ up to 2 kHz
  offset    512 : IrsimEncoderSlot  128 B  4-wheel encoder @ up to 2 kHz
  offset    640 : IrsimPcSlot       128 B  point-cloud seqlock header
  offset    768 : point-cloud data  65 536 x 16 B = 1 048 576 B (1 MiB)

All fixed-size channels use the same seqlock protocol as the base bridge:
  writer: seq++ (even→odd), write, seq++ (odd→even), seq2 = seq
  reader: s1=seq; read; s2=seq2; valid iff s1==s2 and !(s1&1)

The point-cloud channel extends the seqlock over a raw numpy array that
maps the data region starting at offset 768.
"""

from __future__ import annotations

import ctypes
import mmap
import os

import numpy as np

from irsim.util.shm_bridge import (
    _O_CREAT,
    _O_EXCL,
    _O_RDWR,
    _IrsimCmdSlot,
    _IrsimStateSlot,
    _libc,
)

# ── Layout constants ──────────────────────────────────────────────────────────

EXT_SHM_NAME = "/irsim_bridge_ext_v1"
N_MAX_POINTS = 65_536  # maximum points per point-cloud frame
POINT_STRIDE = 16  # bytes per point: x, y, z, intensity (4 x float32)
PC_DATA_BYTES = N_MAX_POINTS * POINT_STRIDE  # 1 048 576 B = 1 MiB

EXT_PC_DATA_OFFSET = 768  # byte offset where raw point data begins
EXT_SHM_SIZE = EXT_PC_DATA_OFFSET + PC_DATA_BYTES  # 1 049 344 B

_O_RDONLY = 0o0

# ── ctypes structs ────────────────────────────────────────────────────────────


class _IrsimImu(ctypes.Structure):
    """40 bytes — 6-axis IMU sample (accelerometer + gyroscope + temperature)."""

    _fields_ = [
        ("timestamp", ctypes.c_double),  # offset  0   seconds
        ("accel_x", ctypes.c_float),  # offset  8   m/s²
        ("accel_y", ctypes.c_float),  # offset 12
        ("accel_z", ctypes.c_float),  # offset 16
        ("gyro_x", ctypes.c_float),  # offset 20   rad/s
        ("gyro_y", ctypes.c_float),  # offset 24
        ("gyro_z", ctypes.c_float),  # offset 28
        ("temp", ctypes.c_float),  # offset 32   °C
        ("_pad", ctypes.c_uint8 * 4),  # offset 36
    ]


assert ctypes.sizeof(_IrsimImu) == 40, ctypes.sizeof(_IrsimImu)


class _IrsimImuSlot(ctypes.Structure):
    """128 bytes — seqlock wrapper for IMU data."""

    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("imu", _IrsimImu),  # 40 bytes at +8
        ("seq2", ctypes.c_uint64),
        ("_fill", ctypes.c_uint8 * 72),
    ]


assert ctypes.sizeof(_IrsimImuSlot) == 128


class _IrsimEncoder(ctypes.Structure):
    """40 bytes — 4-wheel encoder state (position + velocity)."""

    _fields_ = [
        ("timestamp", ctypes.c_double),  # offset  0   seconds
        ("position", ctypes.c_float * 4),  # offset  8   rad (4 wheels)
        ("velocity", ctypes.c_float * 4),  # offset 24   rad/s
    ]


assert ctypes.sizeof(_IrsimEncoder) == 40


class _IrsimEncoderSlot(ctypes.Structure):
    """128 bytes — seqlock wrapper for encoder data."""

    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("enc", _IrsimEncoder),  # 40 bytes at +8
        ("seq2", ctypes.c_uint64),
        ("_fill", ctypes.c_uint8 * 72),
    ]


assert ctypes.sizeof(_IrsimEncoderSlot) == 128


class _IrsimPcHdr(ctypes.Structure):
    """64 bytes — seqlock header guarding the raw point-cloud data area."""

    _fields_ = [
        ("seq", ctypes.c_uint64),  # offset  0
        ("timestamp", ctypes.c_double),  # offset  8   seconds
        ("num_points", ctypes.c_uint32),  # offset 16
        ("frame_id", ctypes.c_uint32),  # offset 20   monotone frame counter
        ("seq2", ctypes.c_uint64),  # offset 24
        ("_fill", ctypes.c_uint8 * 32),  # offset 32
    ]


assert ctypes.sizeof(_IrsimPcHdr) == 64


class _IrsimPcSlot(ctypes.Structure):
    """128 bytes — full cache-line slot containing the point-cloud header."""

    _fields_ = [
        ("hdr", _IrsimPcHdr),  # 64 bytes
        ("_fill", ctypes.c_uint8 * 64),
    ]


assert ctypes.sizeof(_IrsimPcSlot) == 128


class _IrsimExtHeader(ctypes.Structure):
    """128 bytes — extended block header with channel metadata."""

    _fields_ = [
        ("ready", ctypes.c_uint64),  # offset  0
        ("imu_rate_hz", ctypes.c_float),  # offset  8
        ("enc_rate_hz", ctypes.c_float),  # offset 12
        ("pc_rate_hz", ctypes.c_float),  # offset 16
        ("n_max_points", ctypes.c_uint32),  # offset 20
        ("_fill", ctypes.c_uint8 * 100),  # offset 24
    ]


assert ctypes.sizeof(_IrsimExtHeader) == 128


class _IrsimExtBlock(ctypes.Structure):
    """768 bytes — all fixed-size slots (header + 5 data slots)."""

    _fields_ = [
        ("header", _IrsimExtHeader),  # offset   0
        ("state", _IrsimStateSlot),  # offset 128
        ("cmd", _IrsimCmdSlot),  # offset 256
        ("imu", _IrsimImuSlot),  # offset 384
        ("enc", _IrsimEncoderSlot),  # offset 512
        ("pc_slot", _IrsimPcSlot),  # offset 640
    ]


assert ctypes.sizeof(_IrsimExtBlock) == 768


# ── ExtShmBridge ──────────────────────────────────────────────────────────────


class ExtShmBridge:
    """
    Extended shared-memory bridge.

    Adds 6-axis IMU, 4-wheel encoder, and large point-cloud channels on top
    of the base robot-state / velocity-cmd slots.

    Typical lifecycle::

        bridge = ExtShmBridge()
        bridge.open()          # or use as context manager
        bridge.write_imu(...)
        bridge.write_encoder(...)
        bridge.write_pointcloud(points, t)
        bridge.close()

    Use ``attach()`` in a second process to read without creating the segment.
    """

    def __init__(
        self,
        shm_name: str = EXT_SHM_NAME,
        shm_size: int = EXT_SHM_SIZE,
    ) -> None:
        self._name = shm_name.encode()
        self._size = shm_size
        self._mm: mmap.mmap | None = None
        self._blk: _IrsimExtBlock | None = None
        self._is_owner = False
        # seqlock counters (writer-side)
        self._imu_seq: int = 0
        self._enc_seq: int = 0
        self._pc_seq: int = 0
        self._pc_frame_id: int = 0
        # cached slot references (set in open/attach, cleared in close)
        self._imu_slot: _IrsimImuSlot | None = None
        self._enc_slot: _IrsimEncoderSlot | None = None
        self._pc_hdr: _IrsimPcHdr | None = None
        self._pc_data: np.ndarray | None = None  # (N_MAX_POINTS*4,) float32 view

    # ── internal helper ──────────────────────────────────────────────────────

    def _map_and_cache(self, fd: int) -> None:
        self._mm = mmap.mmap(
            fd, self._size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )
        os.close(fd)
        self._blk = _IrsimExtBlock.from_buffer(self._mm)
        self._imu_slot = self._blk.imu
        self._enc_slot = self._blk.enc
        self._pc_hdr = self._blk.pc_slot.hdr
        self._pc_data = np.frombuffer(
            self._mm,
            dtype=np.float32,
            count=N_MAX_POINTS * 4,
            offset=EXT_PC_DATA_OFFSET,
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Create and own the POSIX shm segment."""
        _libc.shm_unlink(self._name)
        fd = _libc.shm_open(self._name, _O_CREAT | _O_RDWR | _O_EXCL, 0o666)
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), self._name.decode())
        if _libc.ftruncate(fd, self._size) != 0:
            err = ctypes.get_errno()
            os.close(fd)
            raise OSError(err, os.strerror(err))
        self._map_and_cache(fd)
        self._mm.write(b"\x00" * self._size)  # type: ignore[union-attr]
        self._mm.seek(0)  # type: ignore[union-attr]
        # re-bind block after zero-init (from_buffer is still valid)
        self._blk.header.ready = 1
        self._blk.header.n_max_points = N_MAX_POINTS
        self._is_owner = True

    def attach(self) -> None:
        """Attach to an existing segment (no create, no unlink on close)."""
        fd = _libc.shm_open(self._name, _O_RDWR, 0o666)
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), self._name.decode())
        self._map_and_cache(fd)
        self._is_owner = False

    def close(self) -> None:
        """Release all ctypes / numpy references, then unmap and optionally unlink."""
        self._pc_data = None
        self._pc_hdr = None
        self._enc_slot = None
        self._imu_slot = None
        if self._blk is not None:
            self._blk = None
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._is_owner:
            _libc.shm_unlink(self._name)

    def __enter__(self) -> ExtShmBridge:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── IMU ──────────────────────────────────────────────────────────────────

    def write_imu(
        self,
        timestamp: float,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
        temp: float = 25.0,
    ) -> None:
        assert self._imu_slot is not None, "call open() first"
        slot = self._imu_slot
        d = slot.imu
        self._imu_seq += 1
        slot.seq = self._imu_seq
        d.timestamp = timestamp
        d.accel_x = accel_x
        d.accel_y = accel_y
        d.accel_z = accel_z
        d.gyro_x = gyro_x
        d.gyro_y = gyro_y
        d.gyro_z = gyro_z
        d.temp = temp
        self._imu_seq += 1
        slot.seq = self._imu_seq
        slot.seq2 = self._imu_seq

    def read_imu(self) -> dict | None:
        """Non-blocking seqlock read. Returns dict or None if mid-write / no data."""
        assert self._imu_slot is not None, "call open() or attach() first"
        slot = self._imu_slot
        s1 = slot.seq
        d = slot.imu
        ts = d.timestamp
        ax, ay, az = d.accel_x, d.accel_y, d.accel_z
        gx, gy, gz = d.gyro_x, d.gyro_y, d.gyro_z
        tmp = d.temp
        s2 = slot.seq2
        if s1 != s2 or (s1 & 1) or s1 == 0:
            return None
        return {
            "timestamp": ts,
            "accel": (float(ax), float(ay), float(az)),
            "gyro": (float(gx), float(gy), float(gz)),
            "temp": float(tmp),
        }

    # ── Encoder ──────────────────────────────────────────────────────────────

    def write_encoder(
        self,
        timestamp: float,
        position: tuple[float, float, float, float],
        velocity: tuple[float, float, float, float],
    ) -> None:
        assert self._enc_slot is not None, "call open() first"
        slot = self._enc_slot
        d = slot.enc
        self._enc_seq += 1
        slot.seq = self._enc_seq
        d.timestamp = timestamp
        for i, v in enumerate(position):
            d.position[i] = v
        for i, v in enumerate(velocity):
            d.velocity[i] = v
        self._enc_seq += 1
        slot.seq = self._enc_seq
        slot.seq2 = self._enc_seq

    def read_encoder(self) -> dict | None:
        """Non-blocking seqlock read. Returns dict or None."""
        assert self._enc_slot is not None, "call open() or attach() first"
        slot = self._enc_slot
        s1 = slot.seq
        d = slot.enc
        ts = d.timestamp
        pos = tuple(float(d.position[i]) for i in range(4))
        vel = tuple(float(d.velocity[i]) for i in range(4))
        s2 = slot.seq2
        if s1 != s2 or (s1 & 1) or s1 == 0:
            return None
        return {"timestamp": ts, "position": pos, "velocity": vel}

    # ── Point Cloud ───────────────────────────────────────────────────────────

    def write_pointcloud(
        self,
        points: np.ndarray,
        timestamp: float,
    ) -> None:
        """
        Write a point cloud to shared memory.

        Parameters
        ----------
        points :
            (N, 4) float32 array  — columns: x, y, z, intensity.
            N must be <= N_MAX_POINTS (65 536).
        timestamp :
            Acquisition timestamp in seconds.
        """
        assert self._pc_hdr is not None, "call open() first"
        assert points.ndim == 2
        assert points.shape[1] == 4
        n = points.shape[0]
        assert n <= N_MAX_POINTS, f"too many points: {n} > {N_MAX_POINTS}"
        hdr = self._pc_hdr
        self._pc_seq += 1
        hdr.seq = self._pc_seq
        hdr.timestamp = timestamp
        hdr.num_points = n
        hdr.frame_id = self._pc_frame_id
        self._pc_frame_id += 1
        flat = (
            points.ravel()
            if points.flags["C_CONTIGUOUS"]
            else np.ascontiguousarray(points).ravel()
        )
        self._pc_data[: n * 4] = flat
        self._pc_seq += 1
        hdr.seq = self._pc_seq
        hdr.seq2 = self._pc_seq

    def read_pointcloud_header(self) -> dict | None:
        """Read the point-cloud header only (no data copy). Non-blocking."""
        assert self._pc_hdr is not None, "call open() or attach() first"
        hdr = self._pc_hdr
        s1 = hdr.seq
        ts = hdr.timestamp
        n_pts = hdr.num_points
        fid = hdr.frame_id
        s2 = hdr.seq2
        if s1 != s2 or (s1 & 1) or s1 == 0:
            return None
        return {"timestamp": ts, "num_points": n_pts, "frame_id": fid, "seq": s1}

    def read_pointcloud(self) -> tuple[dict, np.ndarray] | None:
        """
        Read header + copy all point data (seqlock-guarded).

        Returns ``(header_dict, points_array)`` or ``None`` if mid-write / no data.
        The returned array is a fresh copy (not a view) safe to hold across
        subsequent writes.
        """
        assert self._pc_hdr is not None, "call open() or attach() first"
        hdr = self._pc_hdr
        s1 = hdr.seq
        ts = hdr.timestamp
        n_pts = hdr.num_points
        fid = hdr.frame_id
        if n_pts > 0:
            data = self._pc_data[: n_pts * 4].copy().reshape(-1, 4)
        else:
            data = np.empty((0, 4), dtype=np.float32)
        s2 = hdr.seq2
        if s1 != s2 or (s1 & 1) or s1 == 0:
            return None
        meta = {"timestamp": ts, "num_points": n_pts, "frame_id": fid, "seq": s1}
        return meta, data
