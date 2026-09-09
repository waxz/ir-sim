"""
bridge.py - ShmBridge: Python ↔ C++ shared-memory robot control bridge.

Schema v2 improvements over irsim/util/shm_bridge.py v1:

  • magic + schema_version in header  → ABI mismatch detected on attach
  • writer_ts_ns in every slot        → liveness/heartbeat detection
  • multi-robot (n_robots parameter)  → one segment for N robots
  • Linux + macOS O_* constants       → portable open()
  • read_cmd_blocking(timeout_ms)     → deadline-based cmd wait
  • is_writer_alive(max_age_ms)       → detect stale sim (C++ side)
  • is_controller_alive(max_age_ms)   → detect stale controller (Py side)
  • RobotState / RobotCmd dataclasses → typed API, no raw tuples
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
import struct
import time
from dataclasses import dataclass

from ._libc import _libc, _monotonic_ns
from ._platform import O_CREAT, O_EXCL, O_RDWR
from ._types import (
    MAGIC,
    SCHEMA_VERSION,
    SHM_NAME_DEFAULT,
    _IrsimCmdSlot,
    _IrsimStateSlot,
    _shm_size,
    make_block_type,
)

# struct format for the 72-byte _IrsimState payload (matches _types.py field layout)
# ddd: x, y, heading | ffffff: vx, vy, omega, goal_x, goal_y, goal_dist
# Q: step | d: sim_time | BB: reached, collision | 6x: pad
_STATE_FMT = struct.Struct("<dddffffffQdBB6x")

# ── Public data types ──────────────────────────────────────────────────────


@dataclass(slots=True)
class RobotState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    goal_x: float = 0.0
    goal_y: float = 0.0
    goal_dist: float = 0.0
    step: int = 0
    sim_time: float = 0.0
    reached: bool = False
    collision: bool = False


@dataclass(slots=True)
class RobotCmd:
    linear: float = 0.0
    angular: float = 0.0
    seq: int = 0


# ── ShmBridge ─────────────────────────────────────────────────────────────


class ShmBridge:
    """
    Shared-memory bridge for one or more robots.

    The Python sim creates and owns the segment; the C++ controller attaches
    read-write.  Write robot state after each env.step(); read back the
    velocity command the controller posted.

    Parameters
    ----------
    shm_name : str
        POSIX shm name (must start with '/').
    n_robots : int
        Number of robot slots in the segment (default 1).
    n_consumers : int
        Number of independent C++ cmd writers per robot.
        n_consumers == 1 (default) is byte-for-byte identical to schema v2.
    heartbeat_every : int
        Update writer_ts_ns once every N state writes.
        Higher values reduce clock_gettime overhead; 1 = every write (default).
    """

    def __init__(
        self,
        shm_name: str = SHM_NAME_DEFAULT,
        n_robots: int = 1,
        n_consumers: int = 1,
        heartbeat_every: int = 1,
    ) -> None:
        if n_robots < 1:
            raise ValueError("n_robots must be >= 1")
        if n_consumers < 1:
            raise ValueError("n_consumers must be >= 1")
        if heartbeat_every < 1:
            raise ValueError("heartbeat_every must be >= 1")
        self._name = shm_name.encode()
        self._n = n_robots
        self._nc = n_consumers
        self._heartbeat_every = heartbeat_every
        self._size = _shm_size(n_robots, n_consumers)
        self._mm: mmap.mmap | None = None
        self._blk_type = make_block_type(n_robots, n_consumers)
        self._blk: ctypes.Structure | None = None
        # per-robot seqlock counters and write-count for heartbeat batching
        self._state_seqs: list[int] = [0] * n_robots
        self._write_counts: list[int] = [0] * n_robots
        # _state_slots[r]; _cmd_slots[r][c]
        self._state_slots: list[_IrsimStateSlot] | None = None
        self._cmd_slots: list[list[_IrsimCmdSlot]] | None = None
        # byte offset of state[r].state payload inside the mmap (fast-path)
        self._state_offsets: list[int] = []

    # ── lifecycle ─────────────────────────────────────────────────────────

    def open(self, mlock: bool = False) -> None:
        """
        Create and zero-init the POSIX shm segment.

        Parameters
        ----------
        mlock : bool
            If True, call mlock() to pin the segment in RAM.  Eliminates
            minor page-fault latency on first-access writes (~10-50 µs per page).
            Requires sufficient RLIMIT_MEMLOCK (or CAP_IPC_LOCK).
        """
        _libc.shm_unlink(self._name)
        fd = _libc.shm_open(self._name, O_CREAT | O_RDWR | O_EXCL, 0o666)
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
        self._blk = self._blk_type.from_buffer(self._mm)
        if mlock:
            addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))
            if _libc.mlock(addr, self._size) != 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err), "mlock")
        # write header
        hdr = self._blk.header
        hdr.magic = MAGIC
        hdr.schema_version = SCHEMA_VERSION
        hdr.n_robots = self._n
        hdr.n_consumers = self._nc
        hdr.ready = 1  # signal C++ side: segment is ready
        self._cache_slots()

    def attach(self) -> None:
        """Attach to an existing segment as a secondary reader/writer."""
        fd = _libc.shm_open(self._name, O_RDWR, 0o666)
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), self._name.decode())
        self._mm = mmap.mmap(
            fd, self._size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )
        os.close(fd)
        self._blk = self._blk_type.from_buffer(self._mm)
        hdr = self._blk.header
        if hdr.magic and hdr.magic != MAGIC:
            self.close()
            raise ValueError(
                f"Schema mismatch: magic=0x{hdr.magic:08X} (expected 0x{MAGIC:08X})"
            )
        if hdr.schema_version and hdr.schema_version != SCHEMA_VERSION:
            self.close()
            raise ValueError(
                f"Schema version mismatch: got {hdr.schema_version}, "
                f"expected {SCHEMA_VERSION}"
            )
        # honour n_consumers from the existing segment
        if hdr.n_consumers and hdr.n_consumers != self._nc:
            self._nc = hdr.n_consumers
            self._blk_type = make_block_type(self._n, self._nc)
            self._blk = self._blk_type.from_buffer(self._mm)
        self._cache_slots()

    def _cache_slots(self) -> None:
        """Build fast-access slot caches after open() or attach()."""
        n, nc = self._n, self._nc
        self._state_slots = [self._blk.states[r] for r in range(n)]
        self._cmd_slots = [
            [self._blk.cmds[r * nc + c] for c in range(nc)] for r in range(n)
        ]
        # Byte offset of state[r].state (payload) inside the mmap:
        # header(128) + r * StateSlot(128) + seq(8)
        self._state_offsets = [128 + r * 128 + 8 for r in range(n)]

    def close(self) -> None:
        """Unmap and (if owner) delete the segment."""
        # Release ctypes sub-object references in order: slots → block → mmap.
        # ctypes objects backed by an mmap hold an internal buffer export; the
        # mmap cannot be closed while any exported buffer is alive.
        self._state_slots = None
        self._cmd_slots = None
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
        robot_idx: int = 0,
    ) -> None:
        assert self._state_slots is not None, "call open() first"
        slot = self._state_slots[robot_idx]
        s = slot.state

        self._state_seqs[robot_idx] += 1
        slot.seq = self._state_seqs[robot_idx]  # even → odd

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

        self._state_seqs[robot_idx] += 1
        seq = self._state_seqs[robot_idx]
        slot.seq = seq  # odd → even
        slot.seq2 = seq
        # heartbeat: update timestamp every heartbeat_every writes
        self._write_counts[robot_idx] += 1
        if self._write_counts[robot_idx] % self._heartbeat_every == 0:
            slot.writer_ts_ns = _monotonic_ns()

    def write_state_fast(
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
        robot_idx: int = 0,
    ) -> None:
        """
        Low-latency write path: one struct.pack_into instead of 11 ctypes field sets.

        Replaces ~11 ctypes descriptor calls (~550 ns) with a single memcpy-based
        pack (~80 ns on x86-64).  The seqlock protocol is still honoured: the ctypes
        seq/seq2 writes bracket the struct.pack_into, which acts as a store fence on
        x86 TSO.  Use this on hard-RT paths once you have verified correctness with
        the standard write_state() first.

        On AArch64 (weaker memory model) add a CPU memory barrier between the seq
        write and the pack_into call for full correctness; see the C++ header's
        _SB_FENCE() macro for the pattern.
        """
        assert self._state_slots is not None, "call open() first"
        slot = self._state_slots[robot_idx]

        self._state_seqs[robot_idx] += 1
        slot.seq = self._state_seqs[robot_idx]  # even → odd

        # Pack all 72 state bytes in one call; avoids ctypes attribute overhead
        _STATE_FMT.pack_into(
            self._mm,
            self._state_offsets[robot_idx],
            x,
            y,
            heading,
            vx,
            vy,
            omega,
            goal_x,
            goal_y,
            goal_dist,
            step,
            sim_time,
            int(reached),
            int(collision),
        )

        self._state_seqs[robot_idx] += 1
        seq = self._state_seqs[robot_idx]
        slot.seq = seq  # odd → even
        slot.seq2 = seq
        self._write_counts[robot_idx] += 1
        if self._write_counts[robot_idx] % self._heartbeat_every == 0:
            slot.writer_ts_ns = _monotonic_ns()

    def write_state_from_robot(
        self,
        robot: object,
        step: int,
        sim_time: float,
        robot_idx: int = 0,
    ) -> None:
        """Extract pose/velocity/goal from an ir-sim ObjectBase and publish."""
        st = robot.state  # np.ndarray [x, y, heading, ...]
        vel = robot.velocity  # np.ndarray [linear, angular, ...]

        x = st.item(0)
        y = st.item(1)
        heading = st.item(2)
        vx = vel.item(0) if vel.size > 0 else 0.0
        vy = vel.item(1) if vel.size > 1 else 0.0
        omega = vel.item(2) if vel.size > 2 else 0.0

        goal = robot.goal
        if goal is not None:
            goal_x, goal_y = goal.item(0), goal.item(1)
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
            robot_idx=robot_idx,
        )

    def write_state_obj(
        self,
        state: RobotState,
        step: int,
        sim_time: float,
        robot_idx: int = 0,
    ) -> None:
        """Write from a RobotState dataclass."""
        self.write_state(
            x=state.x,
            y=state.y,
            heading=state.heading,
            vx=state.vx,
            vy=state.vy,
            omega=state.omega,
            goal_x=state.goal_x,
            goal_y=state.goal_y,
            goal_dist=state.goal_dist,
            step=step,
            sim_time=sim_time,
            reached=state.reached,
            collision=state.collision,
            robot_idx=robot_idx,
        )

    # ── read command ──────────────────────────────────────────────────────

    def read_cmd(self, robot_idx: int = 0, consumer_idx: int = 0) -> RobotCmd | None:
        """
        Non-blocking seqlock read of the velocity command from one consumer slot.

        Parameters
        ----------
        consumer_idx : int
            Which consumer's slot to read (0..n_consumers-1).

        Returns ``RobotCmd`` if a valid command is available,
        or ``None`` if the slot is mid-write or no command written yet.
        """
        assert self._cmd_slots is not None, "call open() first"
        slot = self._cmd_slots[robot_idx][consumer_idx]
        s1 = slot.seq
        linear = slot.cmd.linear
        angular = slot.cmd.angular
        seq = slot.cmd.seq
        valid = slot.cmd.valid
        s2 = slot.seq2
        if s1 != s2 or (s1 & 1) or not valid:
            return None
        return RobotCmd(linear=float(linear), angular=float(angular), seq=seq)

    def read_best_cmd(self, robot_idx: int = 0) -> RobotCmd | None:
        """
        Read across all consumer slots for *robot_idx* and return the one
        with the highest cmd.seq (most recently written valid command).

        This is the recommended arbiter for multi-consumer deployments.
        Returns ``None`` if no consumer has posted a valid command yet.
        """
        assert self._cmd_slots is not None, "call open() first"
        best: RobotCmd | None = None
        for slot in self._cmd_slots[robot_idx]:
            s1 = slot.seq
            linear = slot.cmd.linear
            angular = slot.cmd.angular
            seq = slot.cmd.seq
            valid = slot.cmd.valid
            s2 = slot.seq2
            if s1 != s2 or (s1 & 1) or not valid:
                continue
            cmd = RobotCmd(linear=float(linear), angular=float(angular), seq=seq)
            if best is None or cmd.seq > best.seq:
                best = cmd
        return best

    def read_cmd_blocking(
        self,
        timeout_ms: float = 10.0,
        robot_idx: int = 0,
        consumer_idx: int = 0,
    ) -> RobotCmd | None:
        """
        Block until a valid command arrives or *timeout_ms* elapses.

        Busy-polls the seqlock; suitable for hard RT loops.
        Returns ``None`` on timeout.
        """
        deadline = time.monotonic() + timeout_ms * 1e-3
        while time.monotonic() < deadline:
            cmd = self.read_cmd(robot_idx, consumer_idx)
            if cmd is not None:
                return cmd
        return None

    # ── liveness ──────────────────────────────────────────────────────────

    def is_writer_alive(
        self,
        max_age_ms: float = 100.0,
        robot_idx: int = 0,
    ) -> bool:
        """
        Return True if the sim has written a state within *max_age_ms* ms.

        Uses the ``writer_ts_ns`` heartbeat field in the state slot.
        Always returns True if no write has ever been posted (ts == 0).
        """
        assert self._state_slots is not None, "call open() or attach() first"
        ts = self._state_slots[robot_idx].writer_ts_ns
        if ts == 0:
            return True  # no write yet; not stale
        age_ms = (_monotonic_ns() - ts) / 1_000_000.0
        return age_ms < max_age_ms

    def is_controller_alive(
        self,
        max_age_ms: float = 100.0,
        robot_idx: int = 0,
    ) -> bool:
        """
        Return True if the C++ controller posted a cmd within *max_age_ms*.

        Uses the ``writer_ts_ns`` heartbeat field in the cmd slot.
        Always returns True if no cmd has ever been posted (ts == 0).
        """
        assert self._cmd_slots is not None, "call open() or attach() first"
        # Use consumer 0 as the representative liveness signal
        ts = self._cmd_slots[robot_idx][0].writer_ts_ns
        if ts == 0:
            return True
        age_ms = (_monotonic_ns() - ts) / 1_000_000.0
        return age_ms < max_age_ms
