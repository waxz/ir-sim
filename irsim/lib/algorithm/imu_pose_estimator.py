"""Strapdown IMU pose estimators for 2-D dead-reckoning.

Provides four integration algorithms of increasing order/accuracy:

* :class:`EulerIntegrator`     — 1st-order forward Euler
* :class:`MidpointIntegrator`  — 2nd-order midpoint / trapezoidal (default)
* :class:`RK4Integrator`       — 4th-order Runge-Kutta
* :class:`StrapdownIntegrator` — Midpoint + sculling velocity correction

All share the same interface as :class:`IMUPoseEstimatorBase` and can be
used interchangeably::

    from irsim.lib.algorithm.imu_pose_estimator import RK4Integrator
    est = RK4Integrator(initial_state, dt=0.001)
    for state in trajectory:
        imu.step(state)
        pose = est.update(imu.angular_velocity, imu.linear_acceleration)

A convenience alias ``IMUPoseEstimator`` points to :class:`MidpointIntegrator`
for backward compatibility.
"""

from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────


class IMUPoseEstimatorBase:
    """Common interface and shared helpers for all 2-D IMU integrators.

    Args:
        initial_state (array-like): ``[x, y, theta]`` starting pose.
        dt (float): Fixed integration timestep (s).
        initial_velocity (array-like | None): World-frame ``[vx, vy]``.
    """

    name: str = "base"

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
            else np.asarray(initial_velocity, dtype=float).ravel()[:2].copy()
        )
        self.dt = float(dt)
        self.history_pos: list[np.ndarray] = [self.pos.copy()]
        self.history_theta: list[float] = [self.theta]

    def update(self, omega: float, accel_body: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self, initial_state: np.ndarray | list) -> None:
        """Reset to a new starting pose with zero velocity."""
        s = np.asarray(initial_state, dtype=float).ravel()
        self.pos = s[:2].copy()
        self.theta = float(s[2]) if len(s) > 2 else 0.0
        self.vel = np.zeros(2)
        self.history_pos = [self.pos.copy()]
        self.history_theta = [self.theta]

    def get_pose(self) -> np.ndarray:
        """Return current pose ``[x, y, theta]``."""
        return np.array([self.pos[0], self.pos[1], self.theta])

    @staticmethod
    def _rot2d(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def position_rmse(
        estimated: list[np.ndarray], ground_truth: list[np.ndarray]
    ) -> float:
        """RMS Euclidean position error (m)."""
        n = min(len(estimated), len(ground_truth))
        errs = [
            np.linalg.norm(np.asarray(estimated[i]) - np.asarray(ground_truth[i]))
            for i in range(n)
        ]
        return float(np.sqrt(np.mean(np.square(errs))))

    @staticmethod
    def heading_rmse(estimated: list[float], ground_truth: list[float]) -> float:
        """RMS heading error wrapped to [-π, π] (rad)."""
        n = min(len(estimated), len(ground_truth))
        diffs = [
            np.arctan2(
                np.sin(estimated[i] - ground_truth[i]),
                np.cos(estimated[i] - ground_truth[i]),
            )
            for i in range(n)
        ]
        return float(np.sqrt(np.mean(np.square(diffs))))


# ──────────────────────────────────────────────────────────────────────────────
# 1. Euler (1st order)
# ──────────────────────────────────────────────────────────────────────────────


class EulerIntegrator(IMUPoseEstimatorBase):
    """First-order forward Euler integration.

    Uses the heading and velocity at the **start** of each step to compute
    the increment.  Fast but accumulates O(dt) heading error, leading to
    O(dt²) position error per step — visible on curved trajectories.

    Update equations::

        θ[k+1] = θ[k] + ω·dt
        v[k+1] = v[k] + R(θ[k]) @ a_body · dt
        p[k+1] = p[k] + v[k] · dt
    """

    name = "Euler"

    def update(self, omega: float, accel_body: np.ndarray) -> np.ndarray:
        dt = self.dt
        ab = np.asarray(accel_body, dtype=float).ravel()[:2]

        # Rotate at start-of-step heading
        accel_world = self._rot2d(self.theta) @ ab

        # Euler position update (uses start-of-step velocity)
        self.pos += self.vel * dt

        # Integrate velocity and heading
        self.vel += accel_world * dt
        self.theta += omega * dt

        self.history_pos.append(self.pos.copy())
        self.history_theta.append(self.theta)
        return np.array([self.pos[0], self.pos[1], self.theta])


# ──────────────────────────────────────────────────────────────────────────────
# 2. Midpoint / trapezoidal (2nd order)
# ──────────────────────────────────────────────────────────────────────────────


class MidpointIntegrator(IMUPoseEstimatorBase):
    """Second-order midpoint integration (trapezoidal heading).

    Rotates body-frame acceleration at the **midpoint heading**
    θ + ½·ω·dt, halving the linearisation error of pure Euler on
    curved trajectories.  O(dt²) per step in heading, O(dt³) in position.

    Update equations::

        θ_mid  = θ[k] + 0.5·ω·dt
        θ[k+1] = θ[k] + ω·dt
        v[k+1] = v[k] + R(θ_mid) @ a_body · dt
        p[k+1] = p[k] + v[k+1] · dt
    """

    name = "Midpoint"

    def update(self, omega: float, accel_body: np.ndarray) -> np.ndarray:
        dt = self.dt
        ab = np.asarray(accel_body, dtype=float).ravel()[:2]

        theta_mid = self.theta + 0.5 * omega * dt
        self.theta += omega * dt
        accel_world = self._rot2d(theta_mid) @ ab
        self.vel += accel_world * dt
        self.pos += self.vel * dt

        self.history_pos.append(self.pos.copy())
        self.history_theta.append(self.theta)
        return np.array([self.pos[0], self.pos[1], self.theta])


# ──────────────────────────────────────────────────────────────────────────────
# 3. Runge-Kutta 4th order
# ──────────────────────────────────────────────────────────────────────────────


class RK4Integrator(IMUPoseEstimatorBase):
    """Fourth-order Runge-Kutta integration.

    Treats the strapdown ODE as a 5-D state  ``[px, py, vx, vy, θ]``  with
    a constant control input ``(ω, a_body)`` over the step.  Four function
    evaluations per step give O(dt⁴) local truncation error.

    This is the highest-accuracy scheme here; at dt = 0.001 s its numerical
    error is negligible compared to IMU noise.

    Update equations::

        x = [px, py, vx, vy, θ]
        f(x, ω, a) = [vx, vy, R(θ)@a, ω]   (2-D strapdown ODE)

        k1 = f(x[k],           ω, a)
        k2 = f(x[k]+dt/2·k1,  ω, a)
        k3 = f(x[k]+dt/2·k2,  ω, a)
        k4 = f(x[k]+dt·k3,    ω, a)
        x[k+1] = x[k] + dt/6·(k1 + 2k2 + 2k3 + k4)
    """

    name = "RK4"

    def update(self, omega: float, accel_body: np.ndarray) -> np.ndarray:
        dt = self.dt
        ab = np.asarray(accel_body, dtype=float).ravel()[:2]

        x = np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1], self.theta])

        def f(state: np.ndarray) -> np.ndarray:
            vx, vy, th = state[2], state[3], state[4]
            aw = self._rot2d(th) @ ab
            return np.array([vx, vy, aw[0], aw[1], omega])

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)

        xn = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.pos = xn[:2].copy()
        self.vel = xn[2:4].copy()
        self.theta = float(xn[4])

        self.history_pos.append(self.pos.copy())
        self.history_theta.append(self.theta)
        return np.array([self.pos[0], self.pos[1], self.theta])


# ──────────────────────────────────────────────────────────────────────────────
# 4. Strapdown + sculling correction (2nd order + sculling)
# ──────────────────────────────────────────────────────────────────────────────


class StrapdownIntegrator(IMUPoseEstimatorBase):
    """Midpoint heading + sculling velocity correction.

    Sculling (a.k.a. coning-sculling) accounts for the cross-coupling error
    that occurs when the body simultaneously rotates and accelerates.  In the
    2-D case the sculling correction adds a velocity increment perpendicular
    to the current acceleration, proportional to the rotation rate x previous
    velocity increment::

        a[k]   = a_body * dt          (body-frame velocity increment)
        phi[k] = omega * dt           (rotation increment)
        dv_scull = 0.5 * (phi[k-1] x a[k] + phi[k] x a[k-1])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          2-D cross product: (p,0)x(q,r) = (p*r, -p*q)

    The sculling correction matters most when dt is large (> 10 ms) and the
    trajectory has simultaneous rotation and acceleration.  At 1 kHz (dt = 1 ms)
    it is numerically tiny but still reduces the systematic bias.
    """

    name = "Strapdown+Sculling"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Previous step's body-frame velocity increment and rotation increment
        self._prev_alpha: np.ndarray = np.zeros(2)
        self._prev_phi: float = 0.0

    def reset(self, initial_state: np.ndarray | list) -> None:
        super().reset(initial_state)
        self._prev_alpha = np.zeros(2)
        self._prev_phi = 0.0

    @staticmethod
    def _cross2d(phi: float, alpha: np.ndarray) -> np.ndarray:
        """2-D 'cross product': scalar phi x [ax, ay] = phi*[-ay, ax]."""
        return phi * np.array([-alpha[1], alpha[0]])

    def update(self, omega: float, accel_body: np.ndarray) -> np.ndarray:
        dt = self.dt
        ab = np.asarray(accel_body, dtype=float).ravel()[:2]

        alpha = ab * dt  # body-frame velocity increment this step
        phi = omega * dt  # rotation increment this step

        # Sculling velocity correction (body frame)
        delta_v_scull = 0.5 * (
            self._cross2d(self._prev_phi, alpha) + self._cross2d(phi, self._prev_alpha)
        )

        # Total velocity increment in body frame
        delta_v_body = alpha + delta_v_scull

        # Midpoint heading for rotation to world frame
        theta_mid = self.theta + 0.5 * phi
        self.theta += phi
        accel_world = self._rot2d(theta_mid) @ (delta_v_body / dt)

        self.vel += accel_world * dt
        self.pos += self.vel * dt

        # Advance sculling history
        self._prev_alpha = alpha.copy()
        self._prev_phi = phi

        self.history_pos.append(self.pos.copy())
        self.history_theta.append(self.theta)
        return np.array([self.pos[0], self.pos[1], self.theta])


# ── Backward-compatible alias ──────────────────────────────────────────────────
IMUPoseEstimator = MidpointIntegrator
