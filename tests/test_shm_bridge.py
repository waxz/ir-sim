"""
Unit tests for irsim.util.shm_bridge.ShmBridge.

Covers:
- Struct size / offset contract (matches shm_types.h)
- Lifecycle: open, close, context manager, double-close safety
- write_state / read_cmd seqlock round-trip
- write_state_from_robot with a mock ObjectBase
- Edge cases: closed bridge, no command written, mid-write detection
- Segment cleanup on close and __exit__
"""

from __future__ import annotations

import ctypes
import os
import time

import numpy as np
import pytest

# ── availability probe ────────────────────────────────────────────────────────

try:
    _probe_libc = ctypes.CDLL(None, use_errno=True)
    _probe_fd = _probe_libc.shm_open(b"/irsim_probe_xyz", 0o100 | 0o2 | 0o200, 0o666)
    if _probe_fd >= 0:
        os.close(_probe_fd)
        _probe_libc.shm_unlink(b"/irsim_probe_xyz")
    _SHM_AVAILABLE = _probe_fd >= 0
except Exception:
    _SHM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SHM_AVAILABLE, reason="POSIX shm_open not available on this platform"
)

from irsim.util.shm_bridge import (  # noqa: E402
    ShmBridge,
    _IrsimBlock,
    _IrsimCmd,
    _IrsimCmdSlot,
    _IrsimState,
    _IrsimStateSlot,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _unique_name() -> str:
    return f"/irsim_ut_{os.getpid()}_{int(time.monotonic_ns() % 10**9)}"


def _fresh_bridge() -> ShmBridge:
    """Return an opened bridge using a unique segment name."""
    b = ShmBridge(shm_name=_unique_name())
    b.open()
    return b


def _mock_robot(
    x: float = 1.0,
    y: float = 2.0,
    heading: float = 0.5,
    vx: float = 0.3,
    vy: float = 0.0,
    omega: float = 0.1,
    goal: tuple[float, float] | None = (5.0, 5.0),
    arrive: bool = False,
    collision: bool = False,
):
    """Minimal stand-in for an IR-SIM ObjectBase robot."""

    class _Robot:
        pass

    r = _Robot()
    r.state = np.array([[x], [y], [heading]], dtype=float)
    r.velocity = np.array([[vx], [vy], [omega]], dtype=float)
    r.goal = None if goal is None else np.array([[goal[0]], [goal[1]]], dtype=float)
    r.arrive = arrive
    r.collision = collision
    return r


# ── struct contract tests ─────────────────────────────────────────────────────


class TestStructLayout:
    def test_irsim_state_size(self):
        assert ctypes.sizeof(_IrsimState) == 72

    def test_irsim_cmd_size(self):
        assert ctypes.sizeof(_IrsimCmd) == 16

    def test_irsim_state_slot_size(self):
        assert ctypes.sizeof(_IrsimStateSlot) == 128

    def test_irsim_cmd_slot_size(self):
        assert ctypes.sizeof(_IrsimCmdSlot) == 128

    def test_irsim_block_size(self):
        assert ctypes.sizeof(_IrsimBlock) == 384

    def test_state_field_offsets(self):
        s = _IrsimState()
        offsets = {
            "x": 0,
            "y": 8,
            "heading": 16,
            "vx": 24,
            "vy": 28,
            "omega": 32,
            "goal_x": 36,
            "goal_y": 40,
            "goal_dist": 44,
            "step": 48,
            "sim_time": 56,
            "reached": 64,
            "collision": 65,
        }
        for field, expected in offsets.items():
            actual = getattr(type(s), field).offset
            assert actual == expected, f"{field}: offset {actual} != {expected}"

    def test_block_slot_offsets(self):
        s = _IrsimBlock()
        assert type(s).state.offset == 128
        assert type(s).cmd.offset == 256


# ── lifecycle tests ───────────────────────────────────────────────────────────


class TestLifecycle:
    def test_open_creates_segment(self):
        b = ShmBridge(shm_name=_unique_name())
        b.open()
        assert b._mm is not None
        assert b._blk is not None
        b.close()

    def test_ready_flag_set_after_open(self):
        b = ShmBridge(shm_name=_unique_name())
        b.open()
        assert b._blk.ready == 1
        b.close()

    def test_close_clears_internal_refs(self):
        b = ShmBridge(shm_name=_unique_name())
        b.open()
        b.close()
        assert b._mm is None
        assert b._blk is None

    def test_double_close_is_safe(self):
        b = ShmBridge(shm_name=_unique_name())
        b.open()
        b.close()
        b.close()  # must not raise

    def test_context_manager_opens_and_closes(self):
        name = _unique_name()
        with ShmBridge(shm_name=name) as b:
            assert b._mm is not None
        assert b._mm is None

    def test_context_manager_closes_on_exception(self):
        name = _unique_name()
        b_obj = ShmBridge(shm_name=name)

        def _run() -> None:
            with b_obj:
                raise RuntimeError("deliberate")

        with pytest.raises(RuntimeError):
            _run()
        assert b_obj._mm is None

    def test_stale_segment_is_cleaned_on_open(self):
        """open() unlinks a stale segment name before creating a fresh one."""
        name = _unique_name()
        # Create a segment but do NOT call close() so the name persists.
        b1 = ShmBridge(shm_name=name)
        b1.open()
        # Simulate process crash: discard references without calling close.
        # The segment name still exists in /dev/shm at this point.
        b1._state_inner = None
        b1._state_slot = None
        b1._cmd_slot = None
        b1._blk = None
        b1._mm = None  # leak the mmap -- segment name still lives

        # A second bridge with the same name should succeed (open() unlinks first).
        b2 = ShmBridge(shm_name=name)
        b2.open()
        b2.close()


# ── write_state / read_cmd seqlock tests ─────────────────────────────────────


class TestSeqlock:
    def test_read_cmd_returns_none_before_any_write(self):
        b = _fresh_bridge()
        try:
            assert b.read_cmd() is None
        finally:
            b.close()

    def test_write_and_read_cmd_round_trip(self):
        # Use context manager and access slots through bridge attrs (released in close())
        result = None
        with ShmBridge(shm_name=_unique_name()) as b:
            # Simulate C++ irsim_write_cmd: seq must be even after write.
            b._cmd_slot.seq = 2
            b._cmd_slot.cmd.linear = 0.5
            b._cmd_slot.cmd.angular = 0.2
            b._cmd_slot.cmd.valid = 1
            b._cmd_slot.seq2 = 2
            result = b.read_cmd()
        # assert outside the with-block so all ctypes refs are gone
        assert result is not None
        lin, ang = result
        assert lin == pytest.approx(0.5, abs=1e-5)
        assert ang == pytest.approx(0.2, abs=1e-5)

    def test_read_cmd_returns_none_when_mid_write(self):
        """seq odd → writer is mid-update → return None."""
        result = "sentinel"
        with ShmBridge(shm_name=_unique_name()) as b:
            b._cmd_slot.seq = 3  # odd: writer started but not done
            b._cmd_slot.cmd.linear = 1.0
            b._cmd_slot.cmd.valid = 1
            b._cmd_slot.seq2 = 3  # matches but both odd → invalid
            result = b.read_cmd()
        assert result is None

    def test_read_cmd_returns_none_when_seq_mismatch(self):
        """seq != seq2 → torn read → return None."""
        result = "sentinel"
        with ShmBridge(shm_name=_unique_name()) as b:
            b._cmd_slot.seq = 4  # even
            b._cmd_slot.cmd.linear = 1.0
            b._cmd_slot.cmd.valid = 1
            b._cmd_slot.seq2 = 2  # mismatch → mid-write
            result = b.read_cmd()
        assert result is None

    def test_read_cmd_returns_none_when_valid_zero(self):
        result = "sentinel"
        with ShmBridge(shm_name=_unique_name()) as b:
            b._cmd_slot.seq = 2
            b._cmd_slot.cmd.linear = 0.5
            b._cmd_slot.cmd.valid = 0
            b._cmd_slot.seq2 = 2
            result = b.read_cmd()
        assert result is None

    def test_write_state_increments_seq(self):
        """Each write increments _state_seq by 2 (even→odd→even)."""
        b = _fresh_bridge()
        try:
            b.write_state(1.0, 2.0, 0.5, 0.3, 0.0, 0.1, 5.0, 5.0, 4.24, 10, 0.5)
            assert b._state_seq == 2  # 0+1+1 = 2 (even after write)
            assert b._blk.state.seq == 1  # seq set to odd (begin-write value)
            assert b._blk.state.seq2 == 2  # seq2 set to even (end-write value)

            b.write_state(1.1, 2.1, 0.6, 0.3, 0.0, 0.1, 5.0, 5.0, 4.10, 11, 0.55)
            assert b._state_seq == 4
            assert b._blk.state.seq == 3
            assert b._blk.state.seq2 == 4
        finally:
            b.close()

    def test_write_state_values_stored_correctly(self):
        # Extract Python scalars from b._state_inner inside the with-block;
        # they're released automatically and mmap.close() will succeed.
        vals: dict = {}
        with ShmBridge(shm_name=_unique_name()) as b:
            b.write_state(
                x=1.5,
                y=2.5,
                heading=0.7,
                vx=0.4,
                vy=0.1,
                omega=0.05,
                goal_x=8.0,
                goal_y=8.0,
                goal_dist=7.77,
                step=42,
                sim_time=2.1,
                reached=True,
                collision=False,
            )
            si = b._state_inner  # same object as b._state_inner; released in close()
            vals = {
                "x": si.x,
                "y": si.y,
                "heading": si.heading,
                "vx": si.vx,
                "vy": si.vy,
                "omega": si.omega,
                "goal_x": si.goal_x,
                "goal_y": si.goal_y,
                "goal_dist": si.goal_dist,
                "step": si.step,
                "sim_time": si.sim_time,
                "reached": si.reached,
                "collision": si.collision,
            }
            del si  # release ctypes ref before close()
        assert vals["x"] == pytest.approx(1.5)
        assert vals["y"] == pytest.approx(2.5)
        assert vals["heading"] == pytest.approx(0.7)
        assert vals["vx"] == pytest.approx(0.4, abs=1e-5)
        assert vals["vy"] == pytest.approx(0.1, abs=1e-5)
        assert vals["omega"] == pytest.approx(0.05, abs=1e-5)
        assert vals["goal_x"] == pytest.approx(8.0, abs=1e-4)
        assert vals["goal_y"] == pytest.approx(8.0, abs=1e-4)
        assert vals["goal_dist"] == pytest.approx(7.77, abs=1e-3)
        assert vals["step"] == 42
        assert vals["sim_time"] == pytest.approx(2.1)
        assert vals["reached"] == 1
        assert vals["collision"] == 0

    def test_write_state_requires_open(self):
        b = ShmBridge(shm_name=_unique_name())
        with pytest.raises(AssertionError, match="call open"):
            b.write_state(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)

    def test_read_cmd_requires_open(self):
        b = ShmBridge(shm_name=_unique_name())
        with pytest.raises(AssertionError, match="call open"):
            b.read_cmd()


# ── write_state_from_robot tests ─────────────────────────────────────────────


def _read_state_inner(b: ShmBridge) -> dict:
    """Extract all _IrsimState fields as Python scalars (no lingering ctypes ref)."""
    si = b._state_inner
    d = {
        "x": si.x,
        "y": si.y,
        "heading": si.heading,
        "vx": si.vx,
        "vy": si.vy,
        "omega": si.omega,
        "goal_x": si.goal_x,
        "goal_y": si.goal_y,
        "goal_dist": si.goal_dist,
        "step": si.step,
        "sim_time": si.sim_time,
        "reached": si.reached,
        "collision": si.collision,
    }
    del si
    return d


class TestWriteStateFromRobot:
    def test_basic_robot_extracts_position(self):
        with ShmBridge(shm_name=_unique_name()) as b:
            robot = _mock_robot(x=3.0, y=4.0, heading=1.0)
            b.write_state_from_robot(robot, step=5, sim_time=0.25)
            v = _read_state_inner(b)
        assert v["x"] == pytest.approx(3.0)
        assert v["y"] == pytest.approx(4.0)
        assert v["heading"] == pytest.approx(1.0)

    def test_goal_dist_computed_correctly(self):
        with ShmBridge(shm_name=_unique_name()) as b:
            robot = _mock_robot(x=0.0, y=0.0, goal=(3.0, 4.0))
            b.write_state_from_robot(robot, step=1, sim_time=0.05)
            v = _read_state_inner(b)
        assert v["goal_dist"] == pytest.approx(5.0, abs=1e-4)

    def test_goal_none_uses_zeros(self):
        with ShmBridge(shm_name=_unique_name()) as b:
            robot = _mock_robot(x=1.0, y=1.0, goal=None)
            b.write_state_from_robot(robot, step=1, sim_time=0.05)
            v = _read_state_inner(b)
        assert v["goal_x"] == pytest.approx(0.0, abs=1e-5)
        assert v["goal_y"] == pytest.approx(0.0, abs=1e-5)
        assert v["goal_dist"] == pytest.approx(0.0, abs=1e-5)

    def test_arrive_collision_flags(self):
        with ShmBridge(shm_name=_unique_name()) as b:
            b.write_state_from_robot(_mock_robot(arrive=True, collision=False), 1, 0.05)
            v1 = _read_state_inner(b)
            b.write_state_from_robot(_mock_robot(arrive=False, collision=True), 2, 0.10)
            v2 = _read_state_inner(b)
        assert v1["reached"] == 1
        assert v1["collision"] == 0
        assert v2["reached"] == 0
        assert v2["collision"] == 1

    def test_velocity_extracted_correctly(self):
        with ShmBridge(shm_name=_unique_name()) as b:
            b.write_state_from_robot(_mock_robot(vx=0.8, vy=0.0, omega=0.3), 1, 0.05)
            v = _read_state_inner(b)
        assert v["vx"] == pytest.approx(0.8, abs=1e-5)
        assert v["omega"] == pytest.approx(0.3, abs=1e-5)

    def test_velocity_shorter_than_3(self):
        """Robot with only [linear, angular] (len 2) fills omega=0."""
        with ShmBridge(shm_name=_unique_name()) as b:
            robot = _mock_robot()
            robot.velocity = np.array([[0.5], [0.2]], dtype=float)
            b.write_state_from_robot(robot, step=1, sim_time=0.05)
            v = _read_state_inner(b)
        assert v["vx"] == pytest.approx(0.5, abs=1e-5)
        assert v["vy"] == pytest.approx(0.2, abs=1e-5)
        assert v["omega"] == pytest.approx(0.0, abs=1e-5)

    def test_step_and_sim_time_stored(self):
        with ShmBridge(shm_name=_unique_name()) as b:
            b.write_state_from_robot(_mock_robot(), step=99, sim_time=4.95)
            v = _read_state_inner(b)
        assert v["step"] == 99
        assert v["sim_time"] == pytest.approx(4.95)


# ── multiple writes integrity ─────────────────────────────────────────────────


class TestMultipleWrites:
    def test_seq_monotonically_increases(self):
        """_state_seq increases by 2 per write; seq2 (end-write) is always even."""
        b = _fresh_bridge()
        try:
            for i in range(1, 11):
                b.write_state(
                    float(i),
                    float(i),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    i,
                    float(i) * 0.05,
                )
                # seq (begin-write) is odd, seq2 (end-write) is even
                assert b._blk.state.seq == (2 * i - 1)
                assert b._blk.state.seq2 == (2 * i)
        finally:
            b.close()

    def test_seqlock_seq2_always_even_after_write(self):
        """After any write, seq2 must be even (valid for C++ reader)."""
        b = _fresh_bridge()
        try:
            for _ in range(20):
                b.write_state(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0)
            assert (b._blk.state.seq2 & 1) == 0
        finally:
            b.close()
