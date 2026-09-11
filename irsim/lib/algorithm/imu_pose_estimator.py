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

    @staticmethod
    def _extract_omega(omega) -> float:
        """Extract scalar ωz from a scalar or 3-D angular-velocity vector."""
        arr = np.asarray(omega, dtype=float).ravel()
        return float(arr[-1]) if len(arr) > 1 else float(arr[0])

    @staticmethod
    def _extract_accel(accel_body: np.ndarray) -> np.ndarray:
        """Extract 2-D body-frame [ax, ay] from a 2-D or 3-D array."""
        return np.asarray(accel_body, dtype=float).ravel()[:2]

    def update(self, omega, accel_body: np.ndarray) -> np.ndarray:
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

    def update(self, omega, accel_body: np.ndarray) -> np.ndarray:
        dt = self.dt
        omega = self._extract_omega(omega)
        ab = self._extract_accel(accel_body)

        # Rotate at start-of-step heading
        accel_world = self._rot2d(self.theta) @ ab

        # Euler position update (uses start-of-step velocity)
        self.pos += self.vel * dt

        # Integrate velocity and heading
        self.vel += accel_world * dt
        self.theta += omega * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

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

    def update(self, omega, accel_body: np.ndarray) -> np.ndarray:
        dt = self.dt
        omega = self._extract_omega(omega)
        ab = self._extract_accel(accel_body)

        theta_mid = self.theta + 0.5 * omega * dt
        self.theta += omega * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
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

    def update(self, omega, accel_body: np.ndarray) -> np.ndarray:
        dt = self.dt
        omega = self._extract_omega(omega)
        ab = self._extract_accel(accel_body)

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
        self.theta = (float(xn[4]) + np.pi) % (2 * np.pi) - np.pi

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

    def update(self, omega, accel_body: np.ndarray) -> np.ndarray:
        dt = self.dt
        omega = self._extract_omega(omega)
        ab = self._extract_accel(accel_body)

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
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        accel_world = self._rot2d(theta_mid) @ (delta_v_body / dt)

        self.vel += accel_world * dt
        self.pos += self.vel * dt

        # Advance sculling history
        self._prev_alpha = alpha.copy()
        self._prev_phi = phi

        self.history_pos.append(self.pos.copy())
        self.history_theta.append(self.theta)
        return np.array([self.pos[0], self.pos[1], self.theta])


# ──────────────────────────────────────────────────────────────────────────────
# 5. C-extension accelerated integrator (sub-step aware)
# ──────────────────────────────────────────────────────────────────────────────

_ALGORITHMS = ("euler", "midpoint", "rk4", "strapdown")


class CExtIntegrator(IMUPoseEstimatorBase):
    """Pose estimator backed by the C+OpenMP batch integration library.

    Accepts both a single IMU measurement (shape ``(3,)``) and a full batch
    from an ``IMU`` running at a higher rate than the simulator (shape
    ``(n_sub, 3)``).  When the batch interface is used (``imu_rate > 0`` on
    the ``IMU`` sensor), all sub-steps are dispatched to C in a single call,
    making the combined Python + C overhead effectively constant regardless
    of ``n_sub``.

    Falls back to :class:`MidpointIntegrator` transparently when the C
    extension is not available or when the incremental entry-points are
    absent from an older compiled library.

    Args:
        initial_state: ``[x, y, theta]`` starting pose.
        dt: IMU sub-step timestep in seconds (e.g. ``1/imu_rate``).
        initial_velocity: World-frame ``[vx, vy]`` (optional).
        algorithm: One of ``"euler"``, ``"midpoint"`` (default), ``"rk4"``,
            ``"strapdown"``.
    """

    name = "CExtMidpoint"

    def __init__(
        self,
        initial_state: np.ndarray | list,
        dt: float,
        initial_velocity: np.ndarray | list | None = None,
        algorithm: str = "midpoint",
    ) -> None:
        if algorithm not in _ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {_ALGORITHMS}, got '{algorithm}'"
            )
        super().__init__(initial_state, dt, initial_velocity)
        self.name = f"CExt-{algorithm}"
        self._algorithm = algorithm

        # Probe C extension availability
        self._c_ok = False
        try:
            from irsim.lib.algorithm.imu_c_integrators import (
                _has_step_fn,
                ensure_built,
            )

            fn_name = f"imu_{algorithm if algorithm != 'strapdown' else 'strap'}_step"
            self._c_ok = ensure_built() and _has_step_fn(fn_name)
        except Exception:
            pass

        # Internal 5-D (or 8-D for strapdown) C state vector
        self._c_state: np.ndarray = self._make_c_state()

        # Pure-Python fallback (shares no state with self)
        if not self._c_ok:
            self._fallback: IMUPoseEstimatorBase = MidpointIntegrator(
                initial_state, dt, initial_velocity
            )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_c_state(self) -> np.ndarray:
        """Build the flat C state vector from current Python state."""
        s5 = np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1], self.theta])
        if self._algorithm == "strapdown":
            return np.append(s5, [0.0, 0.0, 0.0])
        return s5

    def _sync_from_c_state(self) -> None:
        """Copy C state vector back into Python attributes."""
        self.pos[0] = self._c_state[0]
        self.pos[1] = self._c_state[1]
        self.vel[0] = self._c_state[2]
        self.vel[1] = self._c_state[3]
        self.theta = (float(self._c_state[4]) + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _to_1d(_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (omega_z, ax, ay) 1-D arrays from (N,3) or (3,) inputs."""
        raise NotImplementedError  # handled inline below

    # ── main interface ─────────────────────────────────────────────────────────

    def update(self, omega, accel_body: np.ndarray) -> np.ndarray:
        """Integrate one or more IMU measurements and return the new pose.

        Args:
            omega: Angular velocity — scalar, ``(3,)``, or ``(n_sub, 3)``.
            accel_body: Body-frame acceleration — ``(3,)`` or ``(n_sub, 3)``.

        Returns:
            ``[x, y, theta]`` after integrating all sub-steps.
        """
        omega_arr = np.asarray(omega, dtype=float)
        accel_arr = np.asarray(accel_body, dtype=float)

        # Normalise to 2-D: rows = sub-steps
        if omega_arr.ndim == 1:
            omega_arr = omega_arr[np.newaxis, :]
        if accel_arr.ndim == 1:
            accel_arr = accel_arr[np.newaxis, :]

        omega_z = np.ascontiguousarray(omega_arr[:, -1])
        ax = np.ascontiguousarray(accel_arr[:, 0])
        ay = np.ascontiguousarray(accel_arr[:, 1])

        if self._c_ok:
            from irsim.lib.algorithm.imu_c_integrators import (
                step_euler,
                step_midpoint,
                step_rk4,
                step_strapdown,
            )

            _step_fns = {
                "euler": step_euler,
                "midpoint": step_midpoint,
                "rk4": step_rk4,
                "strapdown": step_strapdown,
            }
            self._c_state = _step_fns[self._algorithm](
                self._c_state, self.dt, omega_z, ax, ay
            )
            self._sync_from_c_state()
        else:
            # Python fallback — process sub-steps one at a time
            for k in range(len(omega_z)):
                self._fallback.update(
                    np.array([0.0, 0.0, omega_z[k]]),
                    np.array([ax[k], ay[k], 0.0]),
                )
            self.pos = self._fallback.pos.copy()
            self.vel = self._fallback.vel.copy()
            self.theta = self._fallback.theta

        pose = np.array([self.pos[0], self.pos[1], self.theta])
        self.history_pos.append(self.pos.copy())
        self.history_theta.append(self.theta)
        return pose

    def reset(self, initial_state: np.ndarray | list) -> None:
        super().reset(initial_state)
        self._c_state = self._make_c_state()
        if not self._c_ok:
            self._fallback.reset(initial_state)


# ── Backward-compatible alias ──────────────────────────────────────────────────
IMUPoseEstimator = MidpointIntegrator
