"""
shm_bridge.py -- IR-SIM shared-memory bridge for C++/Rust robot stacks.

Creates a POSIX shm segment whose layout exactly matches
cpp_bridge/shm_types.h.  The C++ controller mmap-s the same segment,
reads robot state via a seqlock, and writes velocity commands back.

Layout (384 bytes in a 512-byte segment):
  offset   0  IrsimHeader   (128 B)  ready flag
  offset 128  IrsimStateSlot(128 B)  robot state (IR-SIM → C++)
  offset 256  IrsimCmdSlot  (128 B)  velocity cmd (C++ → IR-SIM)

Seqlock protocol (single writer, wait-free reader):
  writer: seq++ (even→odd), write data, seq++ (odd→even), seq2 = seq
  reader: s1 = seq; read; s2 = seq2; valid iff s1==s2 and !(s1&1)

On x86-64 (TSO), store ordering between ctypes field writes is guaranteed
by the hardware.  On AArch64 (Jetson, Pi 5), wrap this module in a short
C extension that adds dmb barriers, or use mmap.flush() after each write.

Typical use in a step loop::

    bridge = ShmBridge()
    bridge.open()
    while not env.done():
        env.step(bridge.read_cmd())          # apply last cmd
        robot = env.robot_list[0]
        bridge.write_state(robot, step, t)   # publish new state
    bridge.close()
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os

# ── layout constants (must match shm_types.h) ─────────────────────────────

SHM_NAME = "/irsim_bridge_v1"
SHM_SIZE = 512

# ── ctypes structs ─────────────────────────────────────────────────────────


class _IrsimState(ctypes.Structure):
    """72 bytes -- matches IrsimState in shm_types.h."""

    _fields_ = [
        ("x", ctypes.c_double),  # offset  0
        ("y", ctypes.c_double),  # offset  8
        ("heading", ctypes.c_double),  # offset 16
        ("vx", ctypes.c_float),  # offset 24
        ("vy", ctypes.c_float),  # offset 28
        ("omega", ctypes.c_float),  # offset 32
        ("goal_x", ctypes.c_float),  # offset 36
        ("goal_y", ctypes.c_float),  # offset 40
        ("goal_dist", ctypes.c_float),  # offset 44
        ("step", ctypes.c_uint64),  # offset 48
        ("sim_time", ctypes.c_double),  # offset 56
        ("reached", ctypes.c_uint8),  # offset 64
        ("collision", ctypes.c_uint8),  # offset 65
        ("_pad", ctypes.c_uint8 * 6),
    ]


assert ctypes.sizeof(_IrsimState) == 72, (
    f"_IrsimState size mismatch: {ctypes.sizeof(_IrsimState)} (expected 72). "
    "Check shm_types.h field alignment."
)


class _IrsimCmd(ctypes.Structure):
    """16 bytes -- matches IrsimCmd in shm_types.h."""

    _fields_ = [
        ("linear", ctypes.c_float),  # m/s forward
        ("angular", ctypes.c_float),  # rad/s CCW+
        ("seq", ctypes.c_uint32),  # command counter
        ("valid", ctypes.c_uint32),  # nonzero = fresh
    ]


assert ctypes.sizeof(_IrsimCmd) == 16


class _IrsimStateSlot(ctypes.Structure):
    """128 bytes -- matches IrsimStateSlot in shm_types.h."""

    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("state", _IrsimState),
        ("seq2", ctypes.c_uint64),
        ("_fill", ctypes.c_uint8 * 40),
    ]


assert ctypes.sizeof(_IrsimStateSlot) == 128


class _IrsimCmdSlot(ctypes.Structure):
    """128 bytes -- matches IrsimCmdSlot in shm_types.h."""

    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("cmd", _IrsimCmd),
        ("seq2", ctypes.c_uint64),
        ("_fill", ctypes.c_uint8 * 96),
    ]


assert ctypes.sizeof(_IrsimCmdSlot) == 128


class _IrsimBlock(ctypes.Structure):
    """384 bytes -- matches IrsimBlock in shm_types.h."""

    _fields_ = [
        ("ready", ctypes.c_uint64),
        ("_fill", ctypes.c_uint8 * 120),
        ("state", _IrsimStateSlot),  # offset 128
        ("cmd", _IrsimCmdSlot),  # offset 256
    ]


assert ctypes.sizeof(_IrsimBlock) == 384


# ── POSIX shm wrappers (avoid subprocess for shm_open) ────────────────────

_libc = ctypes.CDLL(None, use_errno=True)  # libc is already loaded

_libc.shm_open.restype = ctypes.c_int
_libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
_libc.shm_unlink.restype = ctypes.c_int
_libc.shm_unlink.argtypes = [ctypes.c_char_p]
_libc.ftruncate.restype = ctypes.c_int
_libc.ftruncate.argtypes = [ctypes.c_int, ctypes.c_long]

_O_CREAT = 0o100  # Linux
_O_RDWR = 0o2
_O_EXCL = 0o200


# ── ShmBridge ─────────────────────────────────────────────────────────────


class ShmBridge:
    """
    Shared-memory bridge between IR-SIM and a C++ robot controller.

    The Python side creates and owns the segment; the C++ side attaches
    read-write.  Call ``open()`` before the simulation loop and
    ``close()`` (or use as a context manager) at the end.

    Hot-path slot references (``_state_slot``, ``_cmd_slot``,
    ``_state_inner``) are cached in ``open()`` to eliminate repeated
    ctypes attribute traversal on every write/read call.
    """

    def __init__(
        self,
        shm_name: str = SHM_NAME,
        shm_size: int = SHM_SIZE,
    ) -> None:
        self._name = shm_name.encode()
        self._size = shm_size
        self._mm: mmap.mmap | None = None
        self._blk: _IrsimBlock | None = None
        # seqlock counter: even = valid, odd = writing
        self._state_seq: int = 0
        # cached slot references (set in open(), cleared in close())
        self._state_slot: _IrsimStateSlot | None = None
        self._cmd_slot: _IrsimCmdSlot | None = None
        self._state_inner: _IrsimState | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def open(self) -> None:
        """Create the POSIX shm segment and zero-initialise it."""
        _libc.shm_unlink(self._name)  # remove any stale segment

        fd = _libc.shm_open(self._name, _O_CREAT | _O_RDWR | _O_EXCL, 0o666)
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), self._name.decode())

        if _libc.ftruncate(fd, self._size) != 0:
            err = ctypes.get_errno()
            os.close(fd)
            raise OSError(err, os.strerror(err))

        self._mm = mmap.mmap(
            fd, self._size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )
        os.close(fd)

        self._mm.write(b"\x00" * self._size)
        self._mm.seek(0)
        self._blk = _IrsimBlock.from_buffer(self._mm)
        self._blk.ready = 1  # signal C++ side that segment is ready

        # cache slot references to avoid per-call ctypes attribute traversal
        self._state_slot = self._blk.state
        self._cmd_slot = self._blk.cmd
        self._state_inner = self._state_slot.state

    def close(self) -> None:
        """Unmap and delete the shm segment."""
        self._state_inner = None
        self._state_slot = None
        self._cmd_slot = None
        if self._blk is not None:
            self._blk = None
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        _libc.shm_unlink(self._name)

    def __enter__(self) -> ShmBridge:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── write state (call after env.step()) ───────────────────────────────

    def write_state_from_robot(
        self,
        robot: object,
        step: int,
        sim_time: float,
    ) -> None:
        """
        Convenience wrapper: extract position, velocity, and goal from an
        IR-SIM ObjectBase robot and publish to shm.

        Parameters
        ----------
        robot :
            An ``irsim.world.object_base.ObjectBase`` instance.
        step :
            Current simulation step count.
        sim_time :
            Current simulated time in seconds.
        """
        st = robot.state  # np.ndarray [x, y, heading, ...]
        vel = robot.velocity  # np.ndarray [linear, angular] for diff

        # .item() extracts a Python scalar directly from the numpy buffer
        # without creating an intermediate array object.
        x = st.item(0)
        y = st.item(1)
        heading = st.item(2)
        vx = vel.item(0) if vel.size > 0 else 0.0
        vy = vel.item(1) if vel.size > 1 else 0.0
        omega = vel.item(2) if vel.size > 2 else 0.0

        goal = robot.goal  # column vector or None
        if goal is not None:
            goal_x = goal.item(0)
            goal_y = goal.item(1)
            dx, dy = goal_x - x, goal_y - y
            goal_dist = math.sqrt(dx * dx + dy * dy)
        else:
            goal_x = goal_y = goal_dist = 0.0

        self.write_state(
            x=x,
            y=y,
            heading=heading,
            vx=vx,
            vy=vy,
            omega=omega,
            goal_x=goal_x,
            goal_y=goal_y,
            goal_dist=goal_dist,
            step=step,
            sim_time=sim_time,
            reached=bool(robot.arrive),
            collision=bool(robot.collision),
        )

    def write_state(
        self,
        x: float,
        y: float,
        heading: float,
        vx: float,
        vy: float,
        omega: float,
        goal_x: float,
        goal_y: float,
        goal_dist: float,
        step: int,
        sim_time: float,
        reached: bool = False,
        collision: bool = False,
    ) -> None:
        """Write raw state values directly (no robot object needed)."""
        assert self._state_slot is not None, "call open() first"
        slot = self._state_slot
        s = self._state_inner

        # seqlock begin-write: even → odd
        self._state_seq += 1
        slot.seq = self._state_seq

        s.x = x
        s.y = y
        s.heading = heading
        s.vx = vx
        s.vy = vy
        s.omega = omega
        s.goal_x = goal_x
        s.goal_y = goal_y
        s.goal_dist = goal_dist
        s.step = step
        s.sim_time = sim_time
        s.reached = int(reached)
        s.collision = int(collision)

        # seqlock end-write: odd → even
        self._state_seq += 1
        slot.seq = self._state_seq
        slot.seq2 = self._state_seq

    # ── read command (call before env.step()) ─────────────────────────────

    def read_cmd(self) -> tuple[float, float] | None:
        """
        Non-blocking seqlock read of the latest velocity command.

        Returns ``(linear, angular)`` if a valid, fresh command is
        available, or ``None`` if the slot is mid-write or no command
        has been written yet.
        """
        assert self._cmd_slot is not None, "call open() first"
        slot = self._cmd_slot
        s1 = slot.seq
        linear = slot.cmd.linear
        angular = slot.cmd.angular
        valid = slot.cmd.valid
        s2 = slot.seq2
        if s1 != s2 or (s1 & 1) or not valid:
            return None
        return float(linear), float(angular)
