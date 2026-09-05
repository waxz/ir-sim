"""Strapdown IMU pose estimator for 2-D dead-reckoning.

Integrates noisy gyroscope and accelerometer measurements from :class:`IMU`
to produce a running pose estimate (x, y, θ).  Uses the midpoint-angle
(trapezoidal) method for heading integration and the corresponding rotated
acceleration for reduced linearisation error.

Typical usage::

    from irsim.world.sensors.imu import IMU
    from irsim.lib.algorithm.imu_pose_estimator import IMUPoseEstimator

    imu   = IMU(initial_state, step_time=dt)
    est   = IMUPoseEstimator(initial_state, dt=dt)

    for state in trajectory:
        imu.step(state)
        pose = est.update(imu.angular_velocity, imu.linear_acceleration)
"""

from __future__ import annotations

import numpy as np


class IMUPoseEstimator:
    """2-D strapdown IMU integration (dead-reckoning pose estimator).

    At each step the estimator:

    1. Integrates angular velocity with the midpoint angle to update heading.
    2. Rotates body-frame acceleration into the world frame using that midpoint
       angle (reduces linearisation error by ~50 % vs. start-of-step angle).
    3. Integrates acceleration → velocity → position (Euler forward).

    Args:
        initial_state (array-like): ``[x, y, theta]`` starting pose in metres
            and radians.
        dt (float): Fixed integration timestep in seconds.
        initial_velocity (array-like | None): World-frame initial velocity
            ``[vx, vy]``.  Defaults to zero.

    Attr:
        pos (np.ndarray): Current position ``[x, y]`` (m).
        vel (np.ndarray): Current velocity ``[vx, vy]`` (m/s).
        theta (float): Current heading (rad).
        history_pos (list[np.ndarray]): Position at every update call.
        history_theta (list[float]): Heading at every update call.
    """

    def __init__(
        self,
        initial_state: np.ndarray | list,
        dt: float,
        initial_velocity: np.ndarray | list | None = None,
    ) -> None:
        s = np.asarray(initial_state, dtype=float).ravel()
        self.pos: np.ndarray = s[:2].copy()
        self.theta: float = float(s[2]) if len(s) > 2 else 0.0
        self.vel: np.ndarray = (
            np.zeros(2)
            if initial_velocity is None
            else np.asarray(initial_velocity, dtype=float).ravel()[:2]
        )
        self.dt = float(dt)

        self.history_pos: list[np.ndarray] = [self.pos.copy()]
        self.history_theta: list[float] = [self.theta]

    # ------------------------------------------------------------------
    def update(self, omega: float, accel_body: np.ndarray) -> np.ndarray:
        """Integrate one IMU measurement.

        Args:
            omega (float): Gyroscope measurement (rad/s), positive = CCW.
            accel_body (np.ndarray): Accelerometer measurement in body frame,
                shape (2,), units m/s².

        Returns:
            np.ndarray: Updated pose ``[x, y, theta]``.
        """
        dt = self.dt
        accel_body = np.asarray(accel_body, dtype=float).ravel()[:2]

        # Midpoint heading avoids large-step linearisation bias
        theta_mid = self.theta + 0.5 * omega * dt

        # Update heading
        self.theta += omega * dt

        # Rotate body-frame accel to world frame at midpoint angle
        c, s = np.cos(theta_mid), np.sin(theta_mid)
        R = np.array([[c, -s], [s, c]])
        accel_world = R @ accel_body

        # Euler integrate velocity then position
        self.vel += accel_world * dt
        self.pos += self.vel * dt

        self.history_pos.append(self.pos.copy())
        self.history_theta.append(self.theta)

        return np.array([self.pos[0], self.pos[1], self.theta])

    # ------------------------------------------------------------------
    def reset(self, initial_state: np.ndarray | list) -> None:
        """Reset the estimator to a new starting pose with zero velocity."""
        s = np.asarray(initial_state, dtype=float).ravel()
        self.pos = s[:2].copy()
        self.theta = float(s[2]) if len(s) > 2 else 0.0
        self.vel = np.zeros(2)
        self.history_pos = [self.pos.copy()]
        self.history_theta = [self.theta]

    # ------------------------------------------------------------------
    def get_pose(self) -> np.ndarray:
        """Return the current pose as ``[x, y, theta]``."""
        return np.array([self.pos[0], self.pos[1], self.theta])

    # ------------------------------------------------------------------
    @staticmethod
    def position_rmse(
        estimated: list[np.ndarray], ground_truth: list[np.ndarray]
    ) -> float:
        """RMS Euclidean position error over a trajectory.

        Args:
            estimated: Sequence of ``[x, y]`` estimated positions.
            ground_truth: Sequence of ``[x, y]`` true positions (same length).

        Returns:
            float: RMSE in metres.
        """
        n = min(len(estimated), len(ground_truth))
        errors = [
            np.linalg.norm(np.asarray(estimated[i]) - np.asarray(ground_truth[i]))
            for i in range(n)
        ]
        return float(np.sqrt(np.mean(np.square(errors))))

    @staticmethod
    def heading_rmse(
        estimated: list[float], ground_truth: list[float]
    ) -> float:
        """RMS heading error, wrapped to [-π, π].

        Args:
            estimated: Sequence of estimated headings (rad).
            ground_truth: Sequence of true headings (rad).

        Returns:
            float: Heading RMSE in radians.
        """
        n = min(len(estimated), len(ground_truth))
        diffs = [
            np.arctan2(
                np.sin(estimated[i] - ground_truth[i]),
                np.cos(estimated[i] - ground_truth[i]),
            )
            for i in range(n)
        ]
        return float(np.sqrt(np.mean(np.square(diffs))))
