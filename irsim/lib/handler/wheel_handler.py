"""Wheel-level actuator, encoder, and layout models for IR-SIM robots.

Each ``WheelLayout`` subclass mirrors the physical wheel arrangement of one
robot type.  On every simulation step it:

1. Converts the body-velocity command to per-wheel commands via inverse
   kinematics (``wheel_kinematics.py``).
2. Runs each wheel through a realistic DC-motor or servo actuator model.
3. Updates the optical/magnetic encoder state for each drive wheel.

The original ``KinematicsHandler.step()`` ground-truth pose update is never
modified; the wheel layer only derives *additional* per-wheel state from the
same velocity command that flows into the kinematics step.

Motor models
------------
``DCMotorActuator`` — reduced electrical model (electrical TC ≪ mechanical):

    J · dω/dt = K_motor · (ω_cmd - ω) - K_back · ω
              = K_motor · ω_cmd - (K_motor + K_back) · ω

where ``K_motor = Kt·Kp_ctrl / R`` (speed-controller forcing gain) and
``K_back = Kt·Kb / R + B`` (back-EMF + viscous friction term).
Steady-state: ω → ω_cmd.  Time constant: τ = J / (K_motor + K_back).

``ServoActuator`` — 2nd-order PD position controller:

    J_s · δ̈ = K_p · (δ_cmd - δ) - (K_d + B_s) · δ̇

Integrated with Euler; angle and rate clamped to physical limits.

Named presets
-------------
Drive motors: ``"small_dc"``, ``"agv_hub_motor"``, ``"forklift_drive"``
Steer servos: ``"light_servo"``, ``"agv_servo"``, ``"forklift_steer"``,
              ``"swerve_module"``

Usage::

    from irsim.lib.handler.wheel_handler import DiffWheelLayout, MOTOR_PRESETS

    layout = DiffWheelLayout(wheel_radius=0.033, track=0.16,
                             motor="small_dc", encoder_cpr=512)
    layout.step(np.array([[1.0], [0.3]]), dt=0.05)
    print(layout.get_encoder_readings())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from irsim.lib.algorithm.wheel_kinematics import (
    acker_inv_kin,
    diff_inv_kin,
    forklift_inv_kin,
    mecanum_inv_kin,
    swerve_inv_kin,
)

# ---------------------------------------------------------------------------
# Motor parameter dataclasses + presets
# ---------------------------------------------------------------------------


@dataclass
class DCMotorParams:
    """Wheel-shaft-level DC motor parameters (after gearbox).

    The simplified equation of motion (electrical TC assumed ≪ mechanical)::

        J · dω/dt = K_motor · (ω_cmd - ω) - K_back · ω

    Attributes:
        J: Effective rotational inertia at wheel shaft (kg·m²).
        K_motor: Speed-controller forcing gain K_t·K_p_ctrl/R (N·m·s/rad).
        K_back: Back-EMF + friction term K_t·K_b/R + B (N·m·s/rad).
        omega_max: Maximum wheel angular velocity (rad/s).
    """

    J: float = 2e-4
    K_motor: float = 2.5
    K_back: float = 1.5
    omega_max: float = 20.0

    @property
    def tau(self) -> float:
        """Open-loop mechanical time constant (s)."""
        return self.J / (self.K_motor + self.K_back)


@dataclass
class ServoParams:
    """Steering-servo parameters for a 2nd-order PD position controller::

        J_s · δ̈ = K_p · (δ_cmd - δ) - (K_d + B_s) · δ̇

    Attributes:
        J_s: Steering assembly inertia (kg·m²).
        B_s: Viscous friction coefficient (N·m·s/rad).
        K_p: Proportional position gain (N·m/rad).
        K_d: Derivative gain (N·m·s/rad).
        delta_min: Minimum steering angle (rad).
        delta_max: Maximum steering angle (rad).
        omega_max: Maximum steering rate (rad/s).
    """

    J_s: float = 0.05
    B_s: float = 0.02
    K_p: float = 4.0
    K_d: float = 0.8
    delta_min: float = -np.pi / 2
    delta_max: float = np.pi / 2
    omega_max: float = 5.0

    @property
    def omega_n(self) -> float:
        """Undamped natural frequency (rad/s)."""
        return float(np.sqrt(self.K_p / self.J_s))

    @property
    def zeta(self) -> float:
        """Damping ratio."""
        return (self.K_d + self.B_s) / (2.0 * np.sqrt(self.K_p * self.J_s))


# ── Named motor presets ──────────────────────────────────────────────────────
# Parameters chosen to give realistic closed-loop time constants:
#   small_dc     → τ ≈ 0.05 s  (lightweight diff robot, e.g. TurtleBot class)
#   agv_hub_motor→ τ ≈ 0.15 s  (AMR/AGV hub motor, 500 W class)
#   forklift_drive→ τ ≈ 0.30 s (heavy AGV/forklift, high-inertia drivetrain)

MOTOR_PRESETS: dict[str, DCMotorParams] = {
    "small_dc": DCMotorParams(J=2e-4, K_motor=2.5, K_back=1.5, omega_max=20.0),
    "agv_hub_motor": DCMotorParams(J=1e-2, K_motor=40.0, K_back=27.0, omega_max=8.0),
    "forklift_drive": DCMotorParams(J=5e-2, K_motor=100.0, K_back=67.0, omega_max=4.0),
}

# ── Named servo presets ──────────────────────────────────────────────────────
# ωn and ζ approximate real steering assemblies:
#   light_servo   → ωn ≈ 8.9 rad/s, ζ ≈ 0.92  (small robot steering)
#   agv_servo     → ωn ≈ 6.3 rad/s, ζ ≈ 0.83  (AGV, full ±π rotation)
#   forklift_steer→ ωn ≈ 3.5 rad/s, ζ ≈ 1.24  (heavy, overdamped)
#   swerve_module → ωn ≈ 10  rad/s, ζ ≈ 0.68  (fast FRC-style swerve)

SERVO_PRESETS: dict[str, ServoParams] = {
    "light_servo": ServoParams(
        J_s=0.05,
        B_s=0.02,
        K_p=4.0,
        K_d=0.8,
        delta_min=-np.pi / 2,
        delta_max=np.pi / 2,
        omega_max=5.0,
    ),
    "agv_servo": ServoParams(
        J_s=0.20,
        B_s=0.10,
        K_p=8.0,
        K_d=2.0,
        delta_min=-np.pi,
        delta_max=np.pi,
        omega_max=3.0,
    ),
    "forklift_steer": ServoParams(
        J_s=0.50,
        B_s=0.30,
        K_p=6.0,
        K_d=4.0,
        delta_min=-0.75 * np.pi,
        delta_max=0.75 * np.pi,
        omega_max=1.5,
    ),
    "swerve_module": ServoParams(
        J_s=0.03,
        B_s=0.01,
        K_p=3.0,
        K_d=0.4,
        delta_min=-np.pi,
        delta_max=np.pi,
        omega_max=8.0,
    ),
}


# ---------------------------------------------------------------------------
# Wheel state
# ---------------------------------------------------------------------------


@dataclass
class WheelState:
    """Mutable per-wheel state updated each simulation step.

    Attributes:
        name: Unique label (e.g. ``"left"``, ``"FL"``, ``"rear_center"``).
        role: ``"drive"`` | ``"steer"`` | ``"drive_steer"``.
        omega_cmd: Commanded angular velocity (rad/s) from IK.
        omega_actual: Actual angular velocity after motor lag (rad/s).
        theta_enc: Cumulative encoder angle (rad), always monotonically changes.
        delta_cmd: Commanded steering angle (rad); steer/drive_steer only.
        delta_actual: Actual steering angle after servo lag (rad).
        delta_dot: Steering angular rate (rad/s); internal servo state.
        ticks: Quantised encoder tick count (0 when cpr=0).
    """

    name: str
    role: str
    omega_cmd: float = 0.0
    omega_actual: float = 0.0
    theta_enc: float = 0.0
    delta_cmd: float = 0.0
    delta_actual: float = 0.0
    delta_dot: float = 0.0
    ticks: int = 0


# ---------------------------------------------------------------------------
# Actuators
# ---------------------------------------------------------------------------


class DCMotorActuator:
    """Drive-wheel actuator using a simplified DC motor model.

    The governing ODE::

        J · dω/dt = K_motor · (ω_cmd - ω) - K_back · ω

    is integrated with a first-order Euler step, giving time constant
    ``τ = J / (K_motor + K_back)``.

    Args:
        params: A :class:`DCMotorParams` instance or a preset name string.
    """

    def __init__(self, params: DCMotorParams | str = "small_dc") -> None:
        if isinstance(params, str):
            if params not in MOTOR_PRESETS:
                raise ValueError(
                    f"Unknown motor preset {params!r}. Available: {list(MOTOR_PRESETS)}"
                )
            self.params = MOTOR_PRESETS[params]
        else:
            self.params = params

    def step(self, wheel: WheelState, omega_cmd: float, dt: float) -> None:
        """Advance motor state by one timestep.

        Args:
            wheel: Mutable :class:`WheelState` to update in-place.
            omega_cmd: Desired wheel angular velocity (rad/s).
            dt: Simulation timestep (s).
        """
        p = self.params
        wheel.omega_cmd = float(omega_cmd)
        # Exact solution for J·dω/dt = (K_motor+K_back)·(ω_cmd - ω):
        #   ω(t+dt) = ω_cmd + (ω - ω_cmd)·exp(-dt/τ)
        tau = p.J / (p.K_motor + p.K_back)
        decay = float(np.exp(-dt / tau))
        new_omega = omega_cmd + (wheel.omega_actual - omega_cmd) * decay
        wheel.omega_actual = float(np.clip(new_omega, -p.omega_max, p.omega_max))


class ServoActuator:
    """Steering-wheel actuator using a 2nd-order PD position controller.

    The governing ODE::

        J_s · δ̈ = K_p · (δ_cmd - δ) - (K_d + B_s) · δ̇

    is integrated with Euler; angle and rate are clamped to their limits.

    Args:
        params: A :class:`ServoParams` instance or a preset name string.
    """

    def __init__(self, params: ServoParams | str = "light_servo") -> None:
        if isinstance(params, str):
            if params not in SERVO_PRESETS:
                raise ValueError(
                    f"Unknown servo preset {params!r}. Available: {list(SERVO_PRESETS)}"
                )
            self.params = SERVO_PRESETS[params]
        else:
            self.params = params

    def step(self, wheel: WheelState, delta_cmd: float, dt: float) -> None:
        """Advance servo state by one timestep.

        Args:
            wheel: Mutable :class:`WheelState` to update in-place.
            delta_cmd: Desired steering angle (rad).
            dt: Simulation timestep (s).
        """
        p = self.params
        delta_cmd = float(np.clip(delta_cmd, p.delta_min, p.delta_max))
        wheel.delta_cmd = delta_cmd
        tau = (
            p.K_p * (delta_cmd - wheel.delta_actual) - (p.K_d + p.B_s) * wheel.delta_dot
        )
        delta_ddot = tau / p.J_s
        new_dot = float(
            np.clip(wheel.delta_dot + delta_ddot * dt, -p.omega_max, p.omega_max)
        )
        new_delta = float(
            np.clip(wheel.delta_actual + new_dot * dt, p.delta_min, p.delta_max)
        )
        wheel.delta_dot = new_dot
        wheel.delta_actual = new_delta


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class WheelEncoder:
    """Optical/magnetic encoder attached to a drive wheel.

    Integrates ``theta_enc`` from the wheel's ``omega_actual`` and optionally
    quantises to integer ticks (counts per revolution).

    Args:
        cpr: Counts per revolution (CPR). ``0`` means continuous radian
            output with no quantisation.
    """

    def __init__(self, cpr: int = 0) -> None:
        self.cpr = int(cpr)
        self._ticks_per_rad = cpr / (2.0 * np.pi) if cpr > 0 else 0.0

    def step(self, wheel: WheelState, dt: float) -> None:
        """Accumulate encoder angle and update tick count.

        Args:
            wheel: :class:`WheelState` to update in-place.
            dt: Simulation timestep (s).
        """
        wheel.theta_enc += wheel.omega_actual * dt
        if self.cpr > 0:
            wheel.ticks = round(wheel.theta_enc * self._ticks_per_rad)


# ---------------------------------------------------------------------------
# WheelLayout — abstract base
# ---------------------------------------------------------------------------


class WheelLayout(ABC):
    """Abstract base class for a robot's complete wheel arrangement.

    Subclasses define the wheel count and geometry, then implement
    :meth:`_inv_kinematics` to convert a body-velocity command to per-wheel
    ``(omega_cmd, delta_cmd)`` pairs.

    Args:
        motor: DC motor params or preset name for all drive wheels.
        servo: Servo params or preset name for all steer wheels.
        encoder_cpr: Encoder counts per revolution (0 = continuous rad).
    """

    def __init__(
        self,
        motor: DCMotorParams | str = "small_dc",
        servo: ServoParams | str = "light_servo",
        encoder_cpr: int = 0,
    ) -> None:
        self.wheels: list[WheelState] = []
        self._motor_act: dict[str, DCMotorActuator] = {}
        self._servo_act: dict[str, ServoActuator] = {}
        self._encoders: dict[str, WheelEncoder] = {}
        self._motor_params = motor
        self._servo_params = servo
        self._encoder_cpr = encoder_cpr

    def _add_drive_wheel(self, name: str) -> None:
        ws = WheelState(name=name, role="drive")
        self.wheels.append(ws)
        self._motor_act[name] = DCMotorActuator(self._motor_params)
        self._encoders[name] = WheelEncoder(self._encoder_cpr)

    def _add_steer_wheel(self, name: str) -> None:
        ws = WheelState(name=name, role="steer")
        self.wheels.append(ws)
        self._servo_act[name] = ServoActuator(self._servo_params)

    def _add_drive_steer_wheel(self, name: str) -> None:
        ws = WheelState(name=name, role="drive_steer")
        self.wheels.append(ws)
        self._motor_act[name] = DCMotorActuator(self._motor_params)
        self._servo_act[name] = ServoActuator(self._servo_params)
        self._encoders[name] = WheelEncoder(self._encoder_cpr)

    @abstractmethod
    def _inv_kinematics(self, body_vel: np.ndarray) -> dict[str, dict[str, float]]:
        """Return per-wheel commands as ``{name: {"omega": ..., "delta": ...}}``.

        Only ``"omega"`` is required for drive wheels; only ``"delta"`` for
        steer wheels; both for drive_steer wheels.
        """

    def step(self, body_vel: np.ndarray, dt: float) -> None:
        """Advance all wheel actuators and encoders one step.

        Args:
            body_vel: Body-frame velocity command (same column vector that
                ``KinematicsHandler.step()`` received), shape ``(2,1)`` or
                ``(3,1)`` depending on the kinematics model.
            dt: Simulation timestep (s).
        """
        cmds = self._inv_kinematics(body_vel)
        wheel_map: dict[str, WheelState] = {w.name: w for w in self.wheels}
        for name, cmd in cmds.items():
            w = wheel_map[name]
            if w.role in ("drive", "drive_steer"):
                self._motor_act[name].step(w, cmd.get("omega", 0.0), dt)
                self._encoders[name].step(w, dt)
            if w.role in ("steer", "drive_steer"):
                self._servo_act[name].step(w, cmd.get("delta", 0.0), dt)

    def get_wheel_states(self) -> dict[str, WheelState]:
        """Return current wheel states keyed by wheel name."""
        return {w.name: w for w in self.wheels}

    def get_encoder_readings(self) -> dict[str, dict[str, Any]]:
        """Return encoder readings for all drive wheels.

        Returns:
            Dict keyed by wheel name, each containing::

                {
                    "theta_enc":    float,  # cumulative angle (rad)
                    "ticks":        int,    # quantised tick count
                    "omega_actual": float,  # actual wheel speed (rad/s)
                }
        """
        return {
            w.name: {
                "theta_enc": w.theta_enc,
                "ticks": w.ticks,
                "omega_actual": w.omega_actual,
            }
            for w in self.wheels
            if w.role in ("drive", "drive_steer")
        }

    def reset(self) -> None:
        """Reset all wheel states to zero."""
        for w in self.wheels:
            w.omega_cmd = 0.0
            w.omega_actual = 0.0
            w.theta_enc = 0.0
            w.delta_cmd = 0.0
            w.delta_actual = 0.0
            w.delta_dot = 0.0
            w.ticks = 0


# ---------------------------------------------------------------------------
# Differential drive layout
# ---------------------------------------------------------------------------


class DiffWheelLayout(WheelLayout):
    """Two drive wheels (left, right) + one unmodelled passive caster.

    Body velocity format: ``[v, omega]`` shape ``(2, 1)``.

    Args:
        wheel_radius: Wheel radius r (m).
        track: Distance between left and right wheel centres (m).
        motor: DC motor preset name or :class:`DCMotorParams`.
        encoder_cpr: Encoder CPR per drive wheel (0 = continuous).
    """

    def __init__(
        self,
        wheel_radius: float = 0.033,
        track: float = 0.16,
        motor: DCMotorParams | str = "small_dc",
        encoder_cpr: int = 0,
    ) -> None:
        super().__init__(motor=motor, encoder_cpr=encoder_cpr)
        self.wheel_radius = float(wheel_radius)
        self.track = float(track)
        self._add_drive_wheel("left")
        self._add_drive_wheel("right")

    def _inv_kinematics(self, body_vel: np.ndarray) -> dict[str, dict[str, float]]:
        v = float(body_vel[0, 0])
        omega = float(body_vel[1, 0]) if body_vel.shape[0] > 1 else 0.0
        omega_l, omega_r = diff_inv_kin(v, omega, self.wheel_radius, self.track)
        return {"left": {"omega": omega_l}, "right": {"omega": omega_r}}


# ---------------------------------------------------------------------------
# Mecanum (4-wheel omni) layout
# ---------------------------------------------------------------------------


class MecanumWheelLayout(WheelLayout):
    """Four mecanum drive wheels at ±45° roller orientation.

    Body velocity format: ``[vx, vy]`` (omni) or ``[vx, vy, omega_z]``
    (omni_angular), shape ``(2,1)`` or ``(3,1)``.

    Args:
        wheel_radius: Wheel radius r (m).
        half_length: Half-length from body centre to front/rear axle (m).
        half_width: Half-width from body centre to left/right wheels (m).
        motor: DC motor preset or params.
        encoder_cpr: Encoder CPR per wheel.
    """

    def __init__(
        self,
        wheel_radius: float = 0.05,
        half_length: float = 0.15,
        half_width: float = 0.15,
        motor: DCMotorParams | str = "agv_hub_motor",
        encoder_cpr: int = 0,
    ) -> None:
        super().__init__(motor=motor, encoder_cpr=encoder_cpr)
        self.wheel_radius = float(wheel_radius)
        self.half_length = float(half_length)
        self.half_width = float(half_width)
        for name in ("FL", "FR", "RL", "RR"):
            self._add_drive_wheel(name)

    def _inv_kinematics(self, body_vel: np.ndarray) -> dict[str, dict[str, float]]:
        vx = float(body_vel[0, 0])
        vy = float(body_vel[1, 0]) if body_vel.shape[0] > 1 else 0.0
        omega_z = float(body_vel[2, 0]) if body_vel.shape[0] > 2 else 0.0
        fl, fr, rl, rr = mecanum_inv_kin(
            vx, vy, omega_z, self.wheel_radius, self.half_length, self.half_width
        )
        return {
            "FL": {"omega": fl},
            "FR": {"omega": fr},
            "RL": {"omega": rl},
            "RR": {"omega": rr},
        }


# ---------------------------------------------------------------------------
# Ackermann layout
# ---------------------------------------------------------------------------


class AckerWheelLayout(WheelLayout):
    """Two rear drive wheels + two front steering wheels (Ackermann bicycle).

    Body velocity format: ``[v, psi]`` (steer mode) shape ``(2, 1)``.
    ``psi`` is the bicycle-model steering angle, identical to the Ackermann
    kinematics handler's ``state[3]`` at the previous step.

    Args:
        wheel_radius: Wheel radius r (m).
        wheelbase: Axle-to-axle distance L (m).
        track: Track width T (m).
        motor: DC motor preset for rear drive wheels.
        servo: Servo preset for front steer wheels.
        encoder_cpr: Encoder CPR for rear wheels.
    """

    def __init__(
        self,
        wheel_radius: float = 0.15,
        wheelbase: float = 1.0,
        track: float = 0.5,
        motor: DCMotorParams | str = "agv_hub_motor",
        servo: ServoParams | str = "light_servo",
        encoder_cpr: int = 0,
    ) -> None:
        super().__init__(motor=motor, servo=servo, encoder_cpr=encoder_cpr)
        self.wheel_radius = float(wheel_radius)
        self.wheelbase = float(wheelbase)
        self.track = float(track)
        self._add_drive_wheel("RL")
        self._add_drive_wheel("RR")
        self._add_steer_wheel("FL")
        self._add_steer_wheel("FR")

    def _inv_kinematics(self, body_vel: np.ndarray) -> dict[str, dict[str, float]]:
        v = float(body_vel[0, 0])
        psi = float(body_vel[1, 0]) if body_vel.shape[0] > 1 else 0.0
        cmds = acker_inv_kin(v, psi, self.wheel_radius, self.wheelbase, self.track)
        return {
            "RL": {"omega": cmds["omega_rl"]},
            "RR": {"omega": cmds["omega_rr"]},
            "FL": {"delta": cmds["delta_fl"]},
            "FR": {"delta": cmds["delta_fr"]},
        }


# ---------------------------------------------------------------------------
# Forklift layout (2 front drive + 1 rear steer+drive)
# ---------------------------------------------------------------------------


class ForkiftWheelLayout(WheelLayout):
    """Three-wheel counterbalance forklift layout.

    Wheel arrangement::

        FL ──── FR       ← front drive axle (fork side)
             ┆
            RC           ← single rear steer+drive wheel (operator side)

    Body velocity format: ``[v, omega]`` (same as differential-drive model),
    shape ``(2, 1)``.

    Args:
        wheel_radius: Wheel radius for all wheels (m).
        half_wheelbase: Distance from body centre to each axle (m).
        track: Front-axle track width (m).
        motor: DC motor preset or params.
        servo: Servo preset or params for rear steer wheel.
        encoder_cpr: Encoder CPR for drive wheels.
    """

    def __init__(
        self,
        wheel_radius: float = 0.15,
        half_wheelbase: float = 0.6,
        track: float = 0.5,
        motor: DCMotorParams | str = "forklift_drive",
        servo: ServoParams | str = "forklift_steer",
        encoder_cpr: int = 0,
    ) -> None:
        super().__init__(motor=motor, servo=servo, encoder_cpr=encoder_cpr)
        self.wheel_radius = float(wheel_radius)
        self.half_wheelbase = float(half_wheelbase)
        self.track = float(track)
        self._add_drive_wheel("FL")
        self._add_drive_wheel("FR")
        self._add_drive_steer_wheel("RC")

    def _inv_kinematics(self, body_vel: np.ndarray) -> dict[str, dict[str, float]]:
        v = float(body_vel[0, 0])
        omega = float(body_vel[1, 0]) if body_vel.shape[0] > 1 else 0.0
        cmds = forklift_inv_kin(
            v, omega, self.wheel_radius, self.half_wheelbase, self.track
        )
        return {
            "FL": {"omega": cmds["omega_fl"]},
            "FR": {"omega": cmds["omega_fr"]},
            "RC": {"omega": cmds["omega_rc"], "delta": cmds["delta_rc"]},
        }


# ---------------------------------------------------------------------------
# Dual steering-drive layout (AGV tandem, 2 active + 2 passive)
# ---------------------------------------------------------------------------


class DualSteerWheelLayout(WheelLayout):
    """Tandem dual-steering AGV: front and rear active steer+drive wheels.

    Each active wheel has an independent drive motor and steering servo,
    enabling full holonomic motion (vx, vy, omega_z).  Two passive casters
    at the sides are not modelled.

    Wheel positions in body frame (x = forward)::

        ── FC ──         x = +half_wheelbase, y = 0
           ┆
        ── RC ──         x = -half_wheelbase, y = 0

    Body velocity format: ``[vx, vy, omega_z]`` shape ``(3, 1)``
    (same as ``omni_angular`` kinematics).

    Args:
        wheel_radius: Wheel radius r (m).
        half_wheelbase: Distance from body centre to each active wheel (m).
        motor: Drive motor preset.
        servo: Steering servo preset.
        encoder_cpr: Encoder CPR per active wheel.
    """

    def __init__(
        self,
        wheel_radius: float = 0.10,
        half_wheelbase: float = 0.4,
        motor: DCMotorParams | str = "agv_hub_motor",
        servo: ServoParams | str = "agv_servo",
        encoder_cpr: int = 0,
    ) -> None:
        super().__init__(motor=motor, servo=servo, encoder_cpr=encoder_cpr)
        self.wheel_radius = float(wheel_radius)
        self.half_wheelbase = float(half_wheelbase)
        # body-frame positions [x, y]
        self._positions = np.array([[half_wheelbase, 0.0], [-half_wheelbase, 0.0]])
        self._add_drive_steer_wheel("FC")
        self._add_drive_steer_wheel("RC")

    def _inv_kinematics(self, body_vel: np.ndarray) -> dict[str, dict[str, float]]:
        vx = float(body_vel[0, 0])
        vy = float(body_vel[1, 0]) if body_vel.shape[0] > 1 else 0.0
        omega_z = float(body_vel[2, 0]) if body_vel.shape[0] > 2 else 0.0
        cmds = swerve_inv_kin(vx, vy, omega_z, self.wheel_radius, self._positions)
        # cmds shape (2, 2): [[delta_FC, omega_FC], [delta_RC, omega_RC]]
        return {
            "FC": {"delta": float(cmds[0, 0]), "omega": float(cmds[0, 1])},
            "RC": {"delta": float(cmds[1, 0]), "omega": float(cmds[1, 1])},
        }


# ---------------------------------------------------------------------------
# Quadruple steering-drive layout (4-wheel swerve)
# ---------------------------------------------------------------------------


class QuadSteerWheelLayout(WheelLayout):
    """Four-wheel independent steer+drive (swerve drive).

    Every wheel has its own drive motor and steering servo, giving full
    holonomic control plus redundancy.  Wheel positions::

        FL(+L,+W) ──── FR(+L,-W)
            |                |
        RL(-L,+W) ──── RR(-L,-W)

    Body velocity format: ``[vx, vy, omega_z]`` shape ``(3, 1)``
    (same as ``omni_angular`` kinematics).

    Args:
        wheel_radius: Wheel radius r (m).
        half_length: Distance from body centre to front/rear axle (m).
        half_width: Distance from body centre to left/right wheels (m).
        motor: Drive motor preset.
        servo: Steering servo preset.
        encoder_cpr: Encoder CPR per wheel.
    """

    def __init__(
        self,
        wheel_radius: float = 0.05,
        half_length: float = 0.15,
        half_width: float = 0.15,
        motor: DCMotorParams | str = "agv_hub_motor",
        servo: ServoParams | str = "swerve_module",
        encoder_cpr: int = 0,
    ) -> None:
        super().__init__(motor=motor, servo=servo, encoder_cpr=encoder_cpr)
        self.wheel_radius = float(wheel_radius)
        self.half_length = float(half_length)
        self.half_width = float(half_width)
        L, W = half_length, half_width
        self._positions = np.array([[L, W], [L, -W], [-L, W], [-L, -W]])
        for name in ("FL", "FR", "RL", "RR"):
            self._add_drive_steer_wheel(name)

    def _inv_kinematics(self, body_vel: np.ndarray) -> dict[str, dict[str, float]]:
        vx = float(body_vel[0, 0])
        vy = float(body_vel[1, 0]) if body_vel.shape[0] > 1 else 0.0
        omega_z = float(body_vel[2, 0]) if body_vel.shape[0] > 2 else 0.0
        cmds = swerve_inv_kin(vx, vy, omega_z, self.wheel_radius, self._positions)
        names = ("FL", "FR", "RL", "RR")
        return {
            n: {"delta": float(cmds[i, 0]), "omega": float(cmds[i, 1])}
            for i, n in enumerate(names)
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LAYOUT_REGISTRY: dict[str, type[WheelLayout]] = {
    "diff": DiffWheelLayout,
    "mecanum": MecanumWheelLayout,
    "acker": AckerWheelLayout,
    "forklift": ForkiftWheelLayout,
    "dual_steer": DualSteerWheelLayout,
    "quad_steer": QuadSteerWheelLayout,
}


class WheelLayoutFactory:
    """Create a :class:`WheelLayout` from a layout-name string and kwargs.

    Args:
        layout: One of ``"diff"``, ``"mecanum"``, ``"acker"``,
            ``"forklift"``, ``"dual_steer"``, ``"quad_steer"``.
        **kwargs: Forwarded to the layout class ``__init__``.

    Returns:
        WheelLayout: Configured layout instance.

    Raises:
        ValueError: If ``layout`` is not a registered name.
    """

    @staticmethod
    def create(layout: str, **kwargs: Any) -> WheelLayout:
        """Instantiate a layout by name."""
        key = layout.lower()
        cls = _LAYOUT_REGISTRY.get(key)
        if cls is None:
            raise ValueError(
                f"Unknown wheel layout {layout!r}. Available: {list(_LAYOUT_REGISTRY)}"
            )
        return cls(**kwargs)
