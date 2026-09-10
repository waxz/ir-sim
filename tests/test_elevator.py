"""Test: differential-drive robot navigates from floor 1 to floor 2 via elevator.

IR-SIM kinematics are 2-D (x, y, θ).  A multi-floor building is modelled by
augmenting the robot's pose with an explicit z coordinate managed by a
:class:`FloorManager`.  The elevator is a circular zone in the shared xy plane:
when the robot enters it and calls ``ride_elevator()``, the floor index and z
height update atomically.

Building layout
---------------
  Floor 0 (ground): z = 0.0 m — start at (1, 1), goal (1, 1) on this floor
  Floor 1 (upper):  z = 3.0 m — goal at (9, 9)
  Elevator zone:    circle centred at (5, 5) with radius 0.8 m
                    (accessible from both floors at the same xy position)

Navigation uses the ``differential_kinematics`` function with a simple
proportional heading controller to steer toward each waypoint.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.lib.algorithm.kinematics import differential_kinematics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLOOR_HEIGHT = 3.0  # vertical separation between floors (m)
ELEVATOR_CENTER = np.array([5.0, 5.0])
ELEVATOR_RADIUS = 0.8  # robot must be within this radius to call elevator
DT = 0.05  # simulation timestep (s)
V_MAX = 0.8  # maximum linear speed (m/s)
K_ANG = 2.0  # proportional gain for heading controller
GOAL_TOL = 0.15  # distance tolerance to declare waypoint reached (m)
MAX_STEPS = 3_000  # safety cap for each navigation phase


# ---------------------------------------------------------------------------
# FloorManager — elevator state machine
# ---------------------------------------------------------------------------


class FloorManager:
    """Tracks the robot's current floor and enforces elevator-zone entry.

    Args:
        n_floors (int): Total number of floors in the building.
        floor_height (float): Vertical separation between consecutive floors (m).
    """

    def __init__(self, n_floors: int = 2, floor_height: float = FLOOR_HEIGHT) -> None:
        self.n_floors = n_floors
        self.floor_height = floor_height
        self.current_floor: int = 0

    @property
    def z(self) -> float:
        """Current z elevation of the robot (m)."""
        return self.current_floor * self.floor_height

    def in_elevator_zone(self, pos: np.ndarray) -> bool:
        """Return True when *pos* (x, y) is inside the elevator footprint."""
        return float(np.linalg.norm(pos - ELEVATOR_CENTER)) < ELEVATOR_RADIUS

    def ride_elevator(self, pos: np.ndarray, target_floor: int) -> None:
        """Transition to *target_floor*.

        The robot must be inside the elevator zone; raises ``RuntimeError``
        otherwise to prevent silent teleportation bugs.

        Args:
            pos: Current (x, y) position.
            target_floor: Destination floor index (0-based).

        Raises:
            RuntimeError: Robot is not in the elevator zone.
            ValueError: *target_floor* is out of range.
        """
        if not self.in_elevator_zone(pos):
            raise RuntimeError(
                f"Robot at {pos} is not within the elevator zone "
                f"(centre={ELEVATOR_CENTER}, r={ELEVATOR_RADIUS})."
            )
        if target_floor < 0 or target_floor >= self.n_floors:
            raise ValueError(
                f"target_floor={target_floor} is out of range [0, {self.n_floors})."
            )
        self.current_floor = target_floor


# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------


def navigate_to(
    state: np.ndarray,
    goal_xy: np.ndarray,
    *,
    floor_manager: FloorManager,
    max_steps: int = MAX_STEPS,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Drive the robot toward *goal_xy* using a proportional heading controller.

    Args:
        state: Current robot state column vector [x, y, θ]ᵀ, shape (3, 1).
        goal_xy: Target (x, y) position.
        floor_manager: Active :class:`FloorManager` instance.
        max_steps: Maximum number of DT steps before giving up.

    Returns:
        ``(final_state, trajectory)`` where *trajectory* is a list of
        ``[x, y, z]`` vectors recorded at every step.
    """
    trajectory: list[np.ndarray] = []
    for _ in range(max_steps):
        x, y, theta = float(state[0, 0]), float(state[1, 0]), float(state[2, 0])
        trajectory.append(np.array([x, y, floor_manager.z]))

        diff = goal_xy - np.array([x, y])
        dist = float(np.linalg.norm(diff))
        if dist < GOAL_TOL:
            break

        desired_heading = math.atan2(float(diff[1]), float(diff[0]))
        heading_error = math.atan2(
            math.sin(desired_heading - theta),
            math.cos(desired_heading - theta),
        )
        v = min(V_MAX, dist)
        omega = K_ANG * heading_error
        velocity = np.array([[v], [omega]])
        state = differential_kinematics(state, velocity, DT)

    return state, trajectory


# ---------------------------------------------------------------------------
# 1. Unit tests for FloorManager
# ---------------------------------------------------------------------------


class TestFloorManager:
    def test_initial_floor_is_zero(self):
        fm = FloorManager()
        assert fm.current_floor == 0
        assert fm.z == pytest.approx(0.0)

    def test_z_matches_floor_index(self):
        fm = FloorManager(n_floors=3, floor_height=4.0)
        fm.current_floor = 2
        assert fm.z == pytest.approx(8.0)

    def test_in_elevator_zone_true(self):
        fm = FloorManager()
        assert fm.in_elevator_zone(ELEVATOR_CENTER)

    def test_in_elevator_zone_just_inside(self):
        fm = FloorManager()
        pos = ELEVATOR_CENTER + np.array([ELEVATOR_RADIUS - 0.01, 0.0])
        assert fm.in_elevator_zone(pos)

    def test_in_elevator_zone_just_outside(self):
        fm = FloorManager()
        pos = ELEVATOR_CENTER + np.array([ELEVATOR_RADIUS + 0.01, 0.0])
        assert not fm.in_elevator_zone(pos)

    def test_in_elevator_zone_at_start(self):
        """Robot start position (1, 1) is not in the elevator zone."""
        fm = FloorManager()
        assert not fm.in_elevator_zone(np.array([1.0, 1.0]))

    def test_ride_elevator_succeeds_in_zone(self):
        fm = FloorManager()
        fm.ride_elevator(ELEVATOR_CENTER, target_floor=1)
        assert fm.current_floor == 1
        assert fm.z == pytest.approx(FLOOR_HEIGHT)

    def test_ride_elevator_raises_outside_zone(self):
        fm = FloorManager()
        with pytest.raises(RuntimeError, match="elevator zone"):
            fm.ride_elevator(np.array([1.0, 1.0]), target_floor=1)

    def test_ride_elevator_raises_bad_floor(self):
        fm = FloorManager()
        with pytest.raises(ValueError, match="out of range"):
            fm.ride_elevator(ELEVATOR_CENTER, target_floor=5)

    def test_ride_elevator_same_floor_is_noop(self):
        fm = FloorManager()
        fm.ride_elevator(ELEVATOR_CENTER, target_floor=0)
        assert fm.current_floor == 0


# ---------------------------------------------------------------------------
# 2. Navigation unit tests
# ---------------------------------------------------------------------------


class TestNavigation:
    def test_navigate_reaches_goal(self):
        state = np.array([[1.0], [1.0], [0.0]])
        goal = np.array([3.0, 3.0])
        fm = FloorManager()
        final, _traj = navigate_to(state, goal, floor_manager=fm)
        pos = np.array([float(final[0]), float(final[1])])
        assert np.linalg.norm(pos - goal) < GOAL_TOL

    def test_trajectory_records_z_at_every_step(self):
        state = np.array([[1.0], [1.0], [0.0]])
        goal = np.array([2.0, 2.0])
        fm = FloorManager()
        _, traj = navigate_to(state, goal, floor_manager=fm)
        assert all(len(pt) == 3 for pt in traj)
        # All on floor 0 — z should be 0 throughout
        assert all(pt[2] == pytest.approx(0.0) for pt in traj)

    def test_navigate_already_at_goal(self):
        goal = np.array([1.0, 1.0])
        state = np.array([[1.0], [1.0], [0.0]])
        fm = FloorManager()
        final, traj = navigate_to(state, goal, floor_manager=fm)
        pos = np.array([float(final[0]), float(final[1])])
        assert np.linalg.norm(pos - goal) < GOAL_TOL
        assert len(traj) >= 1


# ---------------------------------------------------------------------------
# 3. End-to-end elevator scenario
# ---------------------------------------------------------------------------


class TestElevatorScenario:
    """Integration test: floor-0 start → elevator → floor-1 goal."""

    def test_robot_reaches_elevator_zone(self):
        """Phase 1: robot navigates from (1,1) to the elevator on floor 0."""
        state = np.array([[1.0], [1.0], [0.0]])
        fm = FloorManager()

        final, _ = navigate_to(state, ELEVATOR_CENTER, floor_manager=fm)
        pos = np.array([float(final[0]), float(final[1])])

        assert fm.in_elevator_zone(pos), (
            f"Robot at {pos} should be inside elevator zone after navigation"
        )
        assert fm.current_floor == 0, "Floor should not change during phase 1"

    def test_floor_changes_after_elevator(self):
        """Phase 2: robot uses elevator to ascend from floor 0 to floor 1."""
        fm = FloorManager()
        pos = ELEVATOR_CENTER.copy()
        assert fm.z == pytest.approx(0.0)

        fm.ride_elevator(pos, target_floor=1)

        assert fm.current_floor == 1
        assert fm.z == pytest.approx(FLOOR_HEIGHT)

    def test_robot_reaches_goal_on_floor_1(self):
        """Phase 3: robot navigates from elevator exit to (9, 9) on floor 1."""
        state = np.array([[ELEVATOR_CENTER[0]], [ELEVATOR_CENTER[1]], [0.0]])
        fm = FloorManager()
        fm.current_floor = 1  # already ascended

        goal = np.array([9.0, 9.0])
        final, traj = navigate_to(state, goal, floor_manager=fm)
        pos = np.array([float(final[0]), float(final[1])])

        assert np.linalg.norm(pos - goal) < GOAL_TOL, (
            f"Robot at {pos} did not reach floor-1 goal {goal}"
        )
        assert fm.current_floor == 1
        # All trajectory z values should be FLOOR_HEIGHT
        assert all(pt[2] == pytest.approx(FLOOR_HEIGHT) for pt in traj)

    def test_full_floor0_to_floor1_trajectory(self):
        """Complete end-to-end scenario: floor 0 → elevator → floor 1.

        Asserts each phase transition in order:
        1. Robot starts at (1, 1, z=0) on floor 0.
        2. Navigates to elevator zone and enters (still z=0).
        3. Rides elevator; z becomes FLOOR_HEIGHT, floor index becomes 1.
        4. Navigates to final goal (9, 9, z=FLOOR_HEIGHT) on floor 1.
        """
        state = np.array([[1.0], [1.0], [0.0]])
        fm = FloorManager()

        # ── Phase 1: approach elevator ─────────────────────────────────────
        assert fm.current_floor == 0
        assert fm.z == pytest.approx(0.0)

        state, traj_p1 = navigate_to(state, ELEVATOR_CENTER, floor_manager=fm)
        pos = np.array([float(state[0]), float(state[1])])

        assert fm.in_elevator_zone(pos), "Robot must enter elevator zone"
        assert fm.current_floor == 0, "Floor unchanged during approach"

        # z stays at 0 throughout phase 1
        assert all(pt[2] == pytest.approx(0.0) for pt in traj_p1)

        # ── Phase 2: ride elevator to floor 1 ─────────────────────────────
        z_before = fm.z
        fm.ride_elevator(pos, target_floor=1)
        z_after = fm.z

        assert z_after > z_before, "Elevation must increase"
        assert fm.current_floor == 1
        assert z_after == pytest.approx(FLOOR_HEIGHT)

        # ── Phase 3: navigate to goal on floor 1 ──────────────────────────
        goal = np.array([9.0, 9.0])
        state, traj_p3 = navigate_to(state, goal, floor_manager=fm)
        pos = np.array([float(state[0]), float(state[1])])

        assert np.linalg.norm(pos - goal) < GOAL_TOL, (
            f"Floor-1 goal not reached; final pos={pos}"
        )
        assert fm.current_floor == 1
        assert fm.z == pytest.approx(FLOOR_HEIGHT)

        # z stays at FLOOR_HEIGHT throughout phase 3
        assert all(pt[2] == pytest.approx(FLOOR_HEIGHT) for pt in traj_p3)

        # ── Sanity: combined trajectory spans both floors ──────────────────
        all_z = [pt[2] for pt in traj_p1 + traj_p3]
        assert min(all_z) == pytest.approx(0.0)
        assert max(all_z) == pytest.approx(FLOOR_HEIGHT)

    def test_3d_state_vector_at_each_phase(self):
        """[x, y, z] state has correct z on each floor."""
        fm = FloorManager()

        # Floor 0
        state_3d = np.array([1.0, 1.0, fm.z])
        assert state_3d[2] == pytest.approx(0.0)

        # After elevator
        fm.ride_elevator(ELEVATOR_CENTER, target_floor=1)
        state_3d = np.array(
            [float(ELEVATOR_CENTER[0]), float(ELEVATOR_CENTER[1]), fm.z]
        )
        assert state_3d[2] == pytest.approx(FLOOR_HEIGHT)

    def test_cannot_skip_elevator_zone(self):
        """Robot cannot teleport to floor 1 without entering elevator zone."""
        fm = FloorManager()
        with pytest.raises(RuntimeError, match="elevator zone"):
            fm.ride_elevator(np.array([9.0, 9.0]), target_floor=1)

    def test_world3d_z_range_contains_both_floors(self):
        """World3D z_range covers both floor z-heights."""
        from irsim.world.world3d import World3D

        # Building: ground=0 m, upper=3 m — choose depth > FLOOR_HEIGHT
        depth = FLOOR_HEIGHT + 1.0
        world = World3D(name="test_building", depth=depth, offset=[0, 0, 0])

        z_lo, z_hi = world.z_range
        assert z_lo <= 0.0
        assert z_hi >= FLOOR_HEIGHT, (
            f"World z_range [{z_lo}, {z_hi}] must cover floor 1 at z={FLOOR_HEIGHT}"
        )
