"""
test_shmbridge_irsim.py — integration test: IR-SIM + shmbridge round-trip.

Tests the full publish-subscribe cycle without spawning external processes:
  1. ShmPublisher opened by the "sim" side (mimics sim.py).
  2. ShmSubscriber attached by the "controller" side (mimics controller_py.py
     or the C++ controller running over the same shmbridge ABI).
  3. State written, read back, command written, read back — all verified.

A second test drives the actual irsim environment for 200 steps with a
proportional-heading controller piped through shared memory and confirms
the robot moves toward its goal.

Run:
    pytest usage/27shmbridge_control/test_shmbridge_irsim.py -v

Backend note
-----------
The C++ extension (`_BACKEND == "cpp"`) exposes:
  pub.write_state(robot_idx, RobotState)          — takes an object, not kwargs
  sub.read_state_spin(robot_idx, max_retries=64)  — spin count, not timeout
  sub.write_cmd(robot_idx, consumer_idx, linear, angular)  — no seq arg

The Python fallback (`_BACKEND == "python"`) has compatible subscriber
signatures but publisher write_state accepts individual keyword arguments.
The helper _write_state() below adapts for both so the test body is uniform.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_sb_src = os.path.join(os.path.dirname(__file__), "..", "..", "shmbridge", "src")
if os.path.isdir(_sb_src):
    sys.path.insert(0, os.path.abspath(_sb_src))

import shmbridge  # noqa: E402
from shmbridge import RobotState, ShmPublisher, ShmSubscriber  # noqa: E402

SHM_NAME = "/irsim_test_integration"


# ── backend-compatibility helpers ─────────────────────────────────────────

def _make_state(**kwargs) -> RobotState:
    """Create a RobotState regardless of backend (dataclass vs C++ class)."""
    s = RobotState()
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def _write_state(pub, robot_idx: int, state: RobotState) -> None:
    """Write robot state — adapts for C++ (takes object) and Python (kwargs)."""
    if shmbridge._BACKEND == "cpp":
        pub.write_state(robot_idx, state)
    else:
        pub.write_state(
            x=state.x, y=state.y, heading=state.heading,
            vx=state.vx, vy=state.vy, omega=state.omega,
            goal_x=state.goal_x, goal_y=state.goal_y,
            goal_dist=state.goal_dist,
            step=state.step, sim_time=state.sim_time,
            reached=state.reached, collision=state.collision,
            robot_idx=robot_idx,
        )


def _read_state_spin(sub, robot_idx: int = 0, max_retries: int = 1000):
    """Read state spin — both backends use max_retries."""
    return sub.read_state_spin(robot_idx, max_retries)


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def pub_sub():
    """Open a publisher, attach a subscriber; yield both; clean up after."""
    pub = ShmPublisher(SHM_NAME, n_robots=1)
    pub.open()
    sub = ShmSubscriber(SHM_NAME, n_robots=1)
    sub.attach(timeout_ms=2_000)
    yield pub, sub
    pub.close()


# ── unit tests ────────────────────────────────────────────────────────────

def test_backend_available():
    """shmbridge must load (C++ or Python fallback)."""
    assert shmbridge._BACKEND in ("cpp", "python")


def test_state_round_trip(pub_sub):
    """State written by publisher is faithfully read by subscriber."""
    pub, sub = pub_sub

    s = _make_state(
        x=1.5, y=2.5, heading=math.pi / 4,
        vx=0.3, vy=0.0, omega=0.1,
        goal_x=9.0, goal_y=9.0, goal_dist=math.sqrt(2) * 7.5,
        step=42, sim_time=2.1,
        reached=False, collision=False,
    )
    _write_state(pub, 0, s)

    state = _read_state_spin(sub, 0)
    assert state is not None, "read_state_spin returned None"

    assert abs(state.x - 1.5) < 1e-9
    assert abs(state.y - 2.5) < 1e-9
    assert abs(state.heading - math.pi / 4) < 1e-9
    assert abs(state.vx - 0.3) < 1e-6
    assert abs(state.omega - 0.1) < 1e-6
    assert abs(state.goal_x - 9.0) < 1e-9
    assert state.step == 42
    assert abs(state.sim_time - 2.1) < 1e-9
    assert not state.reached
    assert not state.collision


def test_cmd_round_trip(pub_sub):
    """Command written by subscriber is read back by publisher."""
    pub, sub = pub_sub

    # Write an initial state so the segment is live.
    _write_state(pub, 0, _make_state(
        x=0, y=0, heading=0, vx=0, vy=0, omega=0,
        goal_x=5, goal_y=5, goal_dist=7.07,
        step=0, sim_time=0.0,
    ))

    # Controller writes a command.
    sub.write_cmd(0, 0, 0.8, -0.3)

    cmd = pub.read_best_cmd(robot_idx=0)
    assert cmd is not None, "read_best_cmd returned None"
    assert abs(cmd.linear - 0.8) < 1e-6
    assert abs(cmd.angular - (-0.3)) < 1e-6


def test_multiple_state_writes(pub_sub):
    """Latest state is returned after several writes."""
    pub, sub = pub_sub

    for step in range(10):
        _write_state(pub, 0, _make_state(
            x=float(step) * 0.1, y=0.0, heading=0.0,
            vx=0.5, vy=0.0, omega=0.0,
            goal_x=9.0, goal_y=0.0, goal_dist=9.0 - step * 0.1,
            step=step, sim_time=step * 0.05,
        ))

    state = _read_state_spin(sub, 0)
    assert state is not None
    assert state.step == 9
    assert abs(state.x - 0.9) < 1e-9


# ── integration test with irsim ───────────────────────────────────────────

def _robot_to_state(robot, step: int, sim_time: float) -> RobotState:
    """Extract pose/velocity/goal from an ir-sim robot into a RobotState."""
    import math as _math

    st = robot.state
    vel = robot.velocity

    s = RobotState()
    s.x = float(st.item(0))
    s.y = float(st.item(1))
    s.heading = float(st.item(2))
    s.vx = float(vel.item(0)) if vel.size > 0 else 0.0
    s.vy = float(vel.item(1)) if vel.size > 1 else 0.0
    s.omega = float(vel.item(2)) if vel.size > 2 else 0.0
    goal = robot.goal
    if goal is not None:
        s.goal_x = float(goal.item(0))
        s.goal_y = float(goal.item(1))
        dx, dy = s.goal_x - s.x, s.goal_y - s.y
        s.goal_dist = _math.sqrt(dx * dx + dy * dy)
    s.step = step
    s.sim_time = sim_time
    s.reached = bool(robot.arrive)
    s.collision = bool(robot.collision)
    return s


def _bearing_error(state: RobotState) -> float:
    dx = state.goal_x - state.x
    dy = state.goal_y - state.y
    tgt = math.atan2(dy, dx)
    err = tgt - state.heading
    while err > math.pi:
        err -= 2 * math.pi
    while err < -math.pi:
        err += 2 * math.pi
    return err


def _proportional_cmd(state: RobotState) -> tuple[float, float]:
    V_MAX, OMEGA_MAX, K_ANG, GOAL_TOL = 1.0, 1.5, 2.0, 0.3
    if state.reached or state.goal_dist < GOAL_TOL:
        return 0.0, 0.0
    err = _bearing_error(state)
    angular = max(-OMEGA_MAX, min(OMEGA_MAX, K_ANG * err))
    linear = V_MAX * max(0.0, math.cos(err))
    return linear, angular


def test_irsim_shm_control_loop():
    """
    Run IR-SIM for up to 200 steps with a P-controller over shmbridge and
    confirm the robot reduces its distance to goal by at least 30%.
    """
    import irsim

    yaml_path = os.path.join(os.path.dirname(__file__), "world.yaml")
    env = irsim.make(yaml_path, headless=True)

    shm = "/irsim_test_control_loop"
    pub = ShmPublisher(shm, n_robots=1)
    pub.open()

    sub = ShmSubscriber(shm, n_robots=1)
    sub.attach(timeout_ms=2_000)

    robot = env.robot_list[0]
    initial_dist: float | None = None

    try:
        for step in range(200):
            # 1. Build and publish robot state.
            rs = _robot_to_state(robot, step=step, sim_time=step * 0.05)
            _write_state(pub, 0, rs)

            if initial_dist is None:
                initial_dist = rs.goal_dist

            # 2. Controller reads state and writes command.
            observed = _read_state_spin(sub, 0)
            assert observed is not None, f"state timeout at step {step}"

            linear, angular = _proportional_cmd(observed)
            sub.write_cmd(0, 0, linear, angular)

            # 3. Sim reads best command and steps.
            best = pub.read_best_cmd(robot_idx=0)
            if best is not None:
                env.step([best.linear, best.angular])
            else:
                env.step()

            if env.done():
                break

    finally:
        pub.close()
        env.end()

    assert initial_dist is not None
    final_dist = float(np.linalg.norm(
        env.robot_list[0].state[:2].flatten() - np.array([9.0, 9.0])
    ))
    reduction = (initial_dist - final_dist) / initial_dist
    assert reduction > 0.30, (
        f"Robot should reduce goal distance by ≥30%, "
        f"got {reduction:.1%} (initial={initial_dist:.2f} final={final_dist:.2f})"
    )
