"""
ctypes struct definitions for shmbridge schema v2.

Layout (per robot, 256 bytes; N robots = 128 + 256*N total):

  IrsimHeader   (128 B)  offset   0  - magic, version, ready, n_robots
  IrsimStateSlot(128 B)  offset 128  - robot state written by the sim
  IrsimCmdSlot  (128 B)  offset 256  - velocity cmd written by C++

Wire format is identical to schema v1 for the first robot slot; v1 C++
controllers can read a v2 segment without modification.  New fields
(writer_ts_ns, magic, schema_version, n_robots) occupy bytes that were
padding in v1.
"""

from __future__ import annotations

import ctypes

MAGIC: int = 0x53484D42  # ASCII "SHMB"
SCHEMA_VERSION: int = 2
SHM_NAME_DEFAULT = "/irsim_bridge_v2"
EXT_SHM_NAME_DEFAULT = "/irsim_bridge_ext_v2"


def _shm_size(n_robots: int = 1, n_consumers: int = 1) -> int:
    """Segment byte size for *n_robots* robots and *n_consumers* cmd writers, page-aligned."""
    raw = 128 + 128 * n_robots + 128 * n_robots * n_consumers
    page = 4096
    return (raw + page - 1) & ~(page - 1)


# ── Core payload structs (unchanged from v1; C++ ABI stable) ──────────────


class _IrsimState(ctypes.Structure):
    """72 bytes - robot pose/velocity snapshot written by the simulator."""

    _fields_ = [
        ("x", ctypes.c_double),  # offset  0   world-frame x (m)
        ("y", ctypes.c_double),  # offset  8   world-frame y (m)
        ("heading", ctypes.c_double),  # offset 16   orientation (rad)
        ("vx", ctypes.c_float),  # offset 24   velocity x (m/s)
        ("vy", ctypes.c_float),  # offset 28   velocity y (m/s)
        ("omega", ctypes.c_float),  # offset 32   angular velocity (rad/s)
        ("goal_x", ctypes.c_float),  # offset 36   goal position (m)
        ("goal_y", ctypes.c_float),  # offset 40
        ("goal_dist", ctypes.c_float),  # offset 44   distance to goal (m)
        ("step", ctypes.c_uint64),  # offset 48   sim step counter
        ("sim_time", ctypes.c_double),  # offset 56   simulated time (s)
        ("reached", ctypes.c_uint8),  # offset 64   1 = goal reached
        ("collision", ctypes.c_uint8),  # offset 65   1 = in collision
        ("_pad", ctypes.c_uint8 * 6),
    ]


assert ctypes.sizeof(_IrsimState) == 72, ctypes.sizeof(_IrsimState)


class _IrsimCmd(ctypes.Structure):
    """16 bytes - velocity command written by the external controller."""

    _fields_ = [
        ("linear", ctypes.c_float),  # forward velocity (m/s)
        ("angular", ctypes.c_float),  # angular velocity (rad/s, CCW+)
        ("seq", ctypes.c_uint32),  # command counter (monotone)
        ("valid", ctypes.c_uint32),  # nonzero = command is fresh
    ]


assert ctypes.sizeof(_IrsimCmd) == 16, ctypes.sizeof(_IrsimCmd)


# ── Seqlock slot wrappers (128 bytes each, cache-line aligned) ────────────


class _IrsimStateSlot(ctypes.Structure):
    """
    128 bytes.  v2 adds writer_ts_ns at offset 88 (was _fill in v1).
    Backward-compatible: a v1 C++ reader ignores bytes 88-95 silently.
    """

    _fields_ = [
        ("seq", ctypes.c_uint64),  # offset  0   odd while writing
        ("state", _IrsimState),  # offset  8   72 bytes
        ("seq2", ctypes.c_uint64),  # offset 80   mirrors seq
        ("writer_ts_ns", ctypes.c_uint64),  # offset 88   monotonic ns (NEW)
        ("_fill", ctypes.c_uint8 * 32),  # offset 96   pad to 128
    ]


assert ctypes.sizeof(_IrsimStateSlot) == 128, ctypes.sizeof(_IrsimStateSlot)


class _IrsimCmdSlot(ctypes.Structure):
    """
    128 bytes.  v2 adds writer_ts_ns at offset 40 (was _fill in v1).
    """

    _fields_ = [
        ("seq", ctypes.c_uint64),  # offset  0
        ("cmd", _IrsimCmd),  # offset  8  16 bytes
        ("seq2", ctypes.c_uint64),  # offset 24
        ("writer_ts_ns", ctypes.c_uint64),  # offset 32  (NEW)
        ("_fill", ctypes.c_uint8 * 88),  # offset 40  pad to 128
    ]


assert ctypes.sizeof(_IrsimCmdSlot) == 128, ctypes.sizeof(_IrsimCmdSlot)


class _IrsimHeader(ctypes.Structure):
    """
    128 bytes.  v2 layout (backward-compatible: ready is still at offset 0).

      offset  0  ready          (uint64) - 1 = segment initialized
      offset  8  magic          (uint32) - 0x53484D42 "SHMB"
      offset 12  schema_version (uint32) - 2
      offset 16  n_robots       (uint8)  - number of robot slots
      offset 17  _fill[111]
    """

    _fields_ = [
        ("ready", ctypes.c_uint64),  # offset  0  UNCHANGED from v1
        ("magic", ctypes.c_uint32),  # offset  8  NEW
        ("schema_version", ctypes.c_uint32),  # offset 12  NEW
        ("n_robots", ctypes.c_uint8),  # offset 16  NEW
        ("n_consumers", ctypes.c_uint8),  # offset 17  NEW - cmd writers per robot
        ("_fill", ctypes.c_uint8 * 110),  # offset 18  pad to 128
    ]


assert ctypes.sizeof(_IrsimHeader) == 128, ctypes.sizeof(_IrsimHeader)


def make_block_type(n_robots: int = 1, n_consumers: int = 1) -> type[ctypes.Structure]:
    """
    Build a ctypes Structure for *n_robots* robots and *n_consumers* cmd writers.

    Layout:
      IrsimHeader (128)
      IrsimStateSlot[n_robots]           (n_robots * 128)
      IrsimCmdSlot[n_robots * n_consumers] (n_robots * n_consumers * 128)

    Cmd slot index for (robot r, consumer c): r * n_consumers + c.
    When n_consumers == 1 the layout is identical to schema v2.
    """
    n_cmd = n_robots * n_consumers

    class _IrsimBlock(ctypes.Structure):
        _fields_ = [
            ("header", _IrsimHeader),
            ("states", _IrsimStateSlot * n_robots),
            ("cmds", _IrsimCmdSlot * n_cmd),
        ]

    expected = 128 + 128 * n_robots + 128 * n_cmd
    assert ctypes.sizeof(_IrsimBlock) == expected, (
        f"Block size {ctypes.sizeof(_IrsimBlock)} != {expected}"
    )
    return _IrsimBlock


# ── Extended segment (IMU / encoder / point cloud) ────────────────────────


class _IrsimImu(ctypes.Structure):
    """40 bytes - accelerometer + gyroscope + magnetometer."""

    _fields_ = [
        ("ax", ctypes.c_float),
        ("ay", ctypes.c_float),
        ("az", ctypes.c_float),
        ("gx", ctypes.c_float),
        ("gy", ctypes.c_float),
        ("gz", ctypes.c_float),
        ("mx", ctypes.c_float),
        ("my", ctypes.c_float),
        ("mz", ctypes.c_float),
        ("ts", ctypes.c_float),
    ]


assert ctypes.sizeof(_IrsimImu) == 40, ctypes.sizeof(_IrsimImu)


class _IrsimImuSlot(ctypes.Structure):
    """128 bytes.  8+40+8+8+64 = 128."""

    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("imu", _IrsimImu),
        ("seq2", ctypes.c_uint64),
        ("writer_ts_ns", ctypes.c_uint64),
        ("_fill", ctypes.c_uint8 * 64),
    ]


assert ctypes.sizeof(_IrsimImuSlot) == 128, ctypes.sizeof(_IrsimImuSlot)


class _IrsimEncoder(ctypes.Structure):
    """40 bytes - wheel encoders (up to 4 wheels).  4*4+4*4+4+4 = 40."""

    _fields_ = [
        ("ticks", ctypes.c_int32 * 4),
        ("speed", ctypes.c_float * 4),
        ("ts", ctypes.c_float),
        ("_pad", ctypes.c_uint8 * 4),
    ]


assert ctypes.sizeof(_IrsimEncoder) == 40, ctypes.sizeof(_IrsimEncoder)


class _IrsimEncoderSlot(ctypes.Structure):
    """128 bytes.  8+40+8+8+64 = 128."""

    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("encoder", _IrsimEncoder),
        ("seq2", ctypes.c_uint64),
        ("writer_ts_ns", ctypes.c_uint64),
        ("_fill", ctypes.c_uint8 * 64),
    ]


assert ctypes.sizeof(_IrsimEncoderSlot) == 128, ctypes.sizeof(_IrsimEncoderSlot)


class _IrsimPcHdr(ctypes.Structure):
    """64 bytes - point cloud metadata header.  8+4+4+8+40 = 64."""

    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("n_points", ctypes.c_uint32),
        ("max_pts", ctypes.c_uint32),
        ("ts", ctypes.c_double),
        ("_fill", ctypes.c_uint8 * 40),
    ]


assert ctypes.sizeof(_IrsimPcHdr) == 64, ctypes.sizeof(_IrsimPcHdr)


class _IrsimPcSlot(ctypes.Structure):
    """128 bytes.  8+64+8+8+40 = 128."""

    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("hdr", _IrsimPcHdr),
        ("seq2", ctypes.c_uint64),
        ("writer_ts_ns", ctypes.c_uint64),
        ("_fill", ctypes.c_uint8 * 40),
    ]


assert ctypes.sizeof(_IrsimPcSlot) == 128, ctypes.sizeof(_IrsimPcSlot)

# Maximum points in a point-cloud payload (each point: x,y,z,intensity as float32)
PC_MAX_POINTS = 65_536
PC_POINT_BYTES = 16  # 4 * float32
PC_DATA_BYTES = PC_MAX_POINTS * PC_POINT_BYTES  # 1 MiB

EXT_SHM_SIZE = (
    768 + PC_DATA_BYTES
)  # header(128)+state(128)+cmd(128)+imu(128)+enc(128)+pc_slot(128)+data


class _IrsimExtBlock(ctypes.Structure):
    """Top 768 bytes of the extended segment; raw PC data follows."""

    _fields_ = [
        ("header", _IrsimHeader),
        ("state", _IrsimStateSlot),
        ("cmd", _IrsimCmdSlot),
        ("imu", _IrsimImuSlot),
        ("encoder", _IrsimEncoderSlot),
        ("pc_slot", _IrsimPcSlot),
    ]


assert ctypes.sizeof(_IrsimExtBlock) == 768, ctypes.sizeof(_IrsimExtBlock)
