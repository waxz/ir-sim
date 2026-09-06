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

Built-in controller module
--------------------------
Real motor drives (Dynamixel, EPOS4, Elmo Gold, CANopen DS402) expose PID
gains as tunable registers separate from the motor physics.  The optional
``MotorController`` class models this layer:

- ``ControlMode``: ``VELOCITY``, ``POSITION``, or ``TORQUE`` operating mode.
- ``ControllerParams``: PID gains (Kp, Ki, Kd) plus feedforward gains (Kff,
  Kff_acc) and anti-windup / output limits — equivalent to the gain registers
  in a real drive.
- ``MotorController``: stateful PID + feedforward.  Passed as ``controller=``
  to either actuator to replace the implicit P/PD with an explicit loop.

Feedforward reduces tracking error at its source:

- ``Kff = K_back``: cancels back-EMF; zero steady-state droop with P-only.
- ``Kff_acc = J``: pre-injects inertial torque during ramps; tracking lag
  drops from ``tau * alpha`` to near zero (alpha = d(omega)/dt of profile).

Gearbox model
-------------
:class:`GearboxParams` captures the three gearbox properties that matter for
command-profile design: gear ratio (speed scaling), efficiency (torque loss),
and backlash (angular dead zone).  Attach one to :class:`DCMotorParams` via
the ``gearbox=`` field.  The actuator then:

- exposes ``motor_omega_max`` / ``motor_rpm`` to size the motor-shaft profile
- reports ``WheelState.motor_omega = omega_actual * ratio`` each step
- applies the backlash dead zone so velocity drops to zero at direction
  reversals until the full backlash play is consumed

Named presets
-------------
Drive motors: ``"small_dc"``, ``"agv_hub_motor"``, ``"forklift_drive"``
Commercial motors: ``"dynamixel_xl430"``, ``"pololu_37d_50"``, ``"maxon_ec45_43"``
Steer servos: ``"light_servo"``, ``"agv_servo"``, ``"forklift_steer"``,
              ``"swerve_module"``
Velocity controllers: ``VELOCITY_CONTROLLER_PRESETS`` (key = motor name)
Position controllers: ``POSITION_CONTROLLER_PRESETS`` (key = servo name)

Usage::

    from irsim.lib.handler.wheel_handler import (
        DiffWheelLayout, MOTOR_PRESETS,
        ControllerParams, MotorController, ControlMode,
    )

    # Default (exact first-order solution, backward-compatible):
    layout = DiffWheelLayout(wheel_radius=0.033, track=0.16,
                             motor="small_dc", encoder_cpr=4096)
    layout.step(np.array([[1.0], [0.3]]), dt=0.05)

    # With explicit PI velocity controller:
    from irsim.lib.handler.wheel_handler import VELOCITY_CONTROLLER_PRESETS
    ctrl_params = VELOCITY_CONTROLLER_PRESETS["small_dc"]   # Kp=0.065, Ki=0.010
    actuator = DCMotorActuator("small_dc", controller=ctrl_params)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
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
class GearboxParams:
    """Gearbox between motor shaft and wheel shaft.

    Encodes the three properties that change a velocity profile's shape at the
    motor shaft vs. the wheel shaft:

    - **Ratio**: the motor spins ``ratio`` times faster than the wheel; commands
      and velocities scale by this factor between the two shafts.
    - **Efficiency**: only ``efficiency`` fraction of motor power reaches the
      wheel, so the effective back-EMF torque seen by the controller is
      reduced by η at rated load.
    - **Backlash**: a mechanical dead zone of ``backlash`` radians (at the
      output shaft) where the wheel does not move even though the motor is
      turning.  This causes velocity ripple at direction reversals and a
      position tracking floor of backlash/2.

    Typical values by gearbox type:

    ============= ========= ============= =====================
    Gearbox type  ratio     efficiency    backlash (output, rad)
    ============= ========= ============= =====================
    Precision planetary (Maxon GP)  3-111   0.80-0.90   0.009 (0.5°)
    Standard planetary (Dynamixel)  40-512  0.65-0.80   0.003 (0.18°)
    Spur / helical                  5-50    0.85-0.95   0.017 (1°)
    Chain / sprocket                5-30    0.75-0.85   0.026 (1.5°)
    Direct drive (hub motor)        1       0.90-0.97   0
    ============= ========= ============= =====================

    Attributes:
        ratio:      Gear reduction ratio N (motor shaft turns N times per
                    output shaft turn).  Must be >= 1.0.
        efficiency: Power transmission efficiency eta (0-1).
        backlash:   Total angular play at the output shaft (rad).
    """

    ratio: float = 1.0
    efficiency: float = 1.0
    backlash: float = 0.0


@dataclass
class DCMotorParams:
    """Wheel-shaft-level DC motor parameters (after gearbox reflection).

    The simplified equation of motion (electrical TC assumed ≪ mechanical)::

        J · dω/dt = K_motor · (ω_cmd - ω) - K_back · ω

    All quantities are expressed at the **wheel shaft** (after gearbox).
    Attach a :class:`GearboxParams` to expose motor-shaft properties
    (``motor_omega_max``, ``motor_rpm``) and enable the backlash simulation.

    Attributes:
        J: Effective rotational inertia at wheel shaft (kg·m²).
        K_motor: Speed-controller forcing gain K_t·K_p_ctrl/R (N·m·s/rad).
        K_back: Back-EMF + friction term K_t·K_b/R + B (N·m·s/rad).
        omega_max: Maximum wheel angular velocity (rad/s).
        gearbox: Optional :class:`GearboxParams`.  ``None`` means parameters
            are already reflected to the wheel shaft and no backlash is
            applied (backward-compatible default).
    """

    J: float = 2e-4
    K_motor: float = 2.5
    K_back: float = 1.5
    omega_max: float = 20.0
    gearbox: GearboxParams | None = None

    @property
    def tau(self) -> float:
        """Open-loop mechanical time constant (s)."""
        return self.J / (self.K_motor + self.K_back)

    @property
    def gear_ratio(self) -> float:
        """Gear reduction ratio (1.0 if no gearbox attached)."""
        return self.gearbox.ratio if self.gearbox is not None else 1.0

    @property
    def motor_omega_max(self) -> float:
        """Maximum motor shaft angular velocity (rad/s)."""
        return self.omega_max * self.gear_ratio

    @property
    def motor_rpm(self) -> float:
        """No-load motor shaft speed (RPM)."""
        return self.motor_omega_max * 60.0 / (2.0 * np.pi)


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
# All parameters are at the wheel shaft after gearbox reflection.
# Time constant τ = J / (K_motor + K_back).  Convergence time (2% band) = 4τ.
#
#   small_dc      → τ = 30 ms  (46:1 planetary gearmotor; J ≈ J_motor x 46²)
#     J_motor of typical brushed servo ≈ 1.5 µN·m·s² → J_wheel ≈ 3.2e-3 kg·m².
#     Encoder: 48-64 CPR on motor shaft x 46 = 2200-2950 CPR at wheel.
#     Recommend: encoder_cpr=2048.
#
#   agv_hub_motor → τ = 26 ms  (BLDC hub motor with FOC, direct drive)
#     Wheel + rotor inertia ~7e-3 kg·m² (1-1.5 kg hub at 10 cm radius).
#     FOC current bandwidth ~2 kHz; velocity bandwidth ~200 Hz (τ_vel ≈ 1 ms);
#     effective closed-loop τ ≈ 26 ms with load and discretisation included.
#     Encoder: 512-4096 pulse/rev magnetic at wheel shaft.
#     Recommend: encoder_cpr=1024.
#
#   forklift_drive→ τ = 200 ms (heavy brush motor, 20:1 chain/spur reduction)
#     Large reflected inertia; chain compliance reduces effective bandwidth.
#     Encoder: 200-500 CPR at wheel shaft (industrial resolver or disk encoder).
#     Recommend: encoder_cpr=500.
#
# Real-world commercial motor presets (wheel-shaft quantities from datasheets):
#
#   dynamixel_xl430 → ROBOTIS XL430-W250-T, 46.13:1 planetary, 12 V
#     No-load 61 RPM out = 6.4 rad/s; stall 1.5 N·m.
#     Built-in 12-bit absolute encoder: 4096 CPR at output shaft.
#     τ = 9e-4 / 0.09 = 10 ms. Velocity PI bandwidth ~100-150 Hz.
#     Recommend: encoder_cpr=4096.
#
#   pololu_37d_50 → Pololu 37D metal gearmotor 50:1, 12 V
#     No-load 120 RPM out = 12.6 rad/s; stall 0.54 N·m.
#     Motor-shaft encoder: 64 CPR x 50:1 = 3200 CPR at wheel.
#     τ = 2.3e-3 / 0.066 = 35 ms. No built-in controller (requires MCU PID).
#     Recommend: encoder_cpr=3200.
#
#   maxon_ec45_43 → Maxon EC 45 flat + GP 42 C 43:1, 24 V BLDC
#     No-load 87 RPM out = 9.1 rad/s; peak torque 4.0 N·m.
#     Motor-shaft encoder: 2048 CPR; EPOS4 velocity bandwidth ~500 Hz.
#     τ = 5.4e-3 / 1.35 = 4 ms.
#     Recommend: encoder_cpr=4096.

MOTOR_PRESETS: dict[str, DCMotorParams] = {
    "small_dc": DCMotorParams(
        J=3e-3,  # J_motor x 46² ≈ 3.2e-3 kg·m²; τ = 3e-3/0.10 = 30 ms
        K_motor=0.065,
        K_back=0.035,
        omega_max=20.0,
        gearbox=GearboxParams(ratio=46.0, efficiency=0.72, backlash=np.radians(0.18)),
    ),
    "agv_hub_motor": DCMotorParams(
        J=7e-3,  # wheel+rotor ≈ 7e-3 kg·m²; τ = 7e-3/0.27 = 26 ms
        K_motor=0.20,  # increased for BLDC-FOC bandwidth (~200 Hz velocity loop)
        K_back=0.07,
        omega_max=8.0,
        gearbox=GearboxParams(ratio=1.0, efficiency=0.92, backlash=0.0),
    ),
    "forklift_drive": DCMotorParams(
        J=3.3e-1,  # heavy vehicle reflected inertia; τ = 3.3e-1/1.67 = 198 ms
        K_motor=1.10,
        K_back=0.57,
        omega_max=4.0,
        gearbox=GearboxParams(ratio=20.0, efficiency=0.80, backlash=np.radians(1.5)),
    ),
    # ── Commercial motor presets ──────────────────────────────────────────────
    "dynamixel_xl430": DCMotorParams(
        J=9e-4,  # J_motor x 46.13² ≈ 8.5e-4 kg·m²; τ = 9e-4/0.09 = 10 ms
        K_motor=0.060,
        K_back=0.030,
        omega_max=6.4,
        gearbox=GearboxParams(ratio=46.13, efficiency=0.72, backlash=np.radians(0.18)),
    ),
    "pololu_37d_50": DCMotorParams(
        J=2.3e-3,  # J_motor x 50² ≈ 2.5e-3 kg·m²; τ = 2.3e-3/0.066 = 35 ms
        K_motor=0.044,
        K_back=0.022,
        omega_max=12.6,
        gearbox=GearboxParams(ratio=50.0, efficiency=0.67, backlash=np.radians(1.0)),
    ),
    "maxon_ec45_43": DCMotorParams(
        J=5.4e-3,  # EC45 J_motor ≈ 2.9e-6 x 43² ≈ 5.4e-3 kg·m²; τ = 4 ms
        K_motor=1.20,
        K_back=0.15,
        omega_max=9.1,
        gearbox=GearboxParams(ratio=43.0, efficiency=0.82, backlash=np.radians(0.5)),
    ),
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
# Controller module  (ControlMode / ControllerParams / MotorController)
# ---------------------------------------------------------------------------


class ControlMode(str, Enum):
    """Motor drive operating mode.

    Matches the profile modes common to servo-drive standards (CANopen DS402,
    Dynamixel protocol 2.0, Modbus RTU drives):

    - ``VELOCITY``: PI/PID speed loop; typical for drive wheels.
    - ``POSITION``: PID position loop, often cascaded over an inner speed loop.
    - ``TORQUE``:   Direct torque/current reference; no feedback in this layer.
    """

    VELOCITY = "velocity"
    POSITION = "position"
    TORQUE = "torque"


@dataclass
class ControllerParams:
    """Built-in PID controller gains — the tunable registers of a servo drive.

    Real examples of what these map to:

    - **Dynamixel XL430**: Position_P_Gain (addr 84), Position_I_Gain (86),
      Position_D_Gain (88), Velocity_P_Gain (78), Velocity_I_Gain (76).
    - **Maxon EPOS4**: Velocity PI (Kp, Ki in the Motion Controller tab),
      Position PID with velocity feedforward.
    - **Elmo Gold**: CL[1]/CL[2] current-loop PI; VL[1]/VL[2] velocity PI.

    Attributes:
        Kp: Proportional gain.
        Ki: Integral gain (0 = P/PD only; add to eliminate steady-state error
            under constant load or gravity).
        Kd: Derivative gain.
        Kff: Velocity feedforward gain (N·m·s/rad).  Set to the motor's
            ``K_back`` value to cancel back-EMF at the setpoint, eliminating
            steady-state droop with P-only control (EPOS4 "Velocity FF Gain";
            Dynamixel "Feedforward 2nd Gain").
        Kff_acc: Acceleration feedforward gain (N·m·s²/rad ≈ effective inertia
            J).  Pre-injects the inertial torque needed to follow a ramp
            command, reducing ramp tracking lag to near zero.  Matches the
            "Feedforward 1st Gain" register on Dynamixel XL430.
        i_limit: Anti-windup clamp on the integrator state (same units as
            ``Kp * error``).  ``inf`` means unlimited.
        output_limit: Hard clamp on total controller output magnitude.
    """

    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    Kff: float = 0.0
    Kff_acc: float = 0.0
    i_limit: float = float("inf")
    output_limit: float = float("inf")


class MotorController:
    """Stateful PID controller modelling a motor drive's built-in control loop.

    Maintains integrator and previous-error state across :meth:`step` calls.
    Call :meth:`reset` after abrupt setpoint changes to avoid integrator kick.

    Derivative is computed on the **error** (not measurement), which is the
    standard for velocity loops.  For position loops the same formula holds
    when the setpoint is constant (d(error)/dt = -δ̇), matching the existing
    ``ServoActuator`` PD equation.

    Args:
        params: :class:`ControllerParams` with gains and limits.
        mode:   :class:`ControlMode` — ``VELOCITY``, ``POSITION``, or
                ``TORQUE``.
    """

    def __init__(
        self,
        params: ControllerParams,
        mode: ControlMode = ControlMode.VELOCITY,
    ) -> None:
        self.params = params
        self.mode = mode
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._initialized: bool = False

    def step(
        self,
        setpoint: float,
        measurement: float,
        dt: float,
        setpoint_rate: float = 0.0,
    ) -> float:
        """Compute one PID + feedforward step.

        The feedforward path pre-computes the torque the plant needs and leaves
        the PID loop to correct only model mismatch and disturbances::

            u = Kff * setpoint + Kff_acc * setpoint_rate
              + Kp * e + Ki * integral(e) + Kd * de/dt

        With ``Kff = K_back`` the back-EMF droop is cancelled at steady state,
        giving zero error even with ``Ki = 0``.  Adding ``Kff_acc = J`` also
        pre-injects the inertial torque during ramps, reducing ramp lag to
        near zero.

        Args:
            setpoint:      Target value (rad/s for velocity; rad for position).
            measurement:   Current measured value.
            dt:            Timestep (s).
            setpoint_rate: Rate of change of the setpoint (rad/s² for velocity
                           mode).  Used by the acceleration feedforward term.
                           Pass 0.0 (default) if unknown or for step inputs.

        Returns:
            Controller output.  For velocity/torque mode this is a torque
            command (N·m); for position mode it is the generalized force
            driving the plant's angular acceleration (N·m, divided by J_s
            inside the actuator).
        """
        p = self.params
        error = setpoint - measurement
        # Model-inversion feedforward (velocity + acceleration)
        ff_term = p.Kff * setpoint + p.Kff_acc * setpoint_rate
        # Integrator with symmetric anti-windup clamp
        self._integral = float(
            np.clip(self._integral + error * dt, -p.i_limit, p.i_limit)
        )
        # Derivative skipped on the very first call to avoid a spike
        if not self._initialized or dt <= 0.0:
            d_term = 0.0
            self._initialized = True
        else:
            d_term = p.Kd * (error - self._prev_error) / dt
        self._prev_error = error
        output = ff_term + p.Kp * error + p.Ki * self._integral + d_term
        return float(np.clip(output, -p.output_limit, p.output_limit))

    def reset(self) -> None:
        """Clear integrator and derivative history."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False

    @property
    def integral(self) -> float:
        """Current integrator state (read-only)."""
        return self._integral


# ── Default velocity controller presets ──────────────────────────────────────
# Kp matches the motor's K_motor so the P-only step response is identical to
# the exact first-order solution.  Ki adds a slow integral to eliminate the
# small steady-state error that appears under constant load (e.g. incline).
# output_limit is set to K_motor·ω_max + K_back·ω_max = (K_motor+K_back)·ω_max
# (max torque the drive can command at full speed).

VELOCITY_CONTROLLER_PRESETS: dict[str, ControllerParams] = {
    # Feedforward gains: Kff = K_back (back-EMF cancellation, eliminates
    # steady-state droop with P-only), Kff_acc = J (inertia pre-compensation,
    # reduces ramp tracking lag to near zero).
    "small_dc": ControllerParams(
        Kp=0.065,
        Ki=0.010,
        Kff=0.035,  # K_back of small_dc
        Kff_acc=3e-3,  # J of small_dc
        i_limit=0.5,
        output_limit=2.0,
        # small_dc: (0.065+0.035)*20 = 2.0 N*m ceiling
    ),
    "agv_hub_motor": ControllerParams(
        Kp=0.200,  # matches K_motor (updated for BLDC-FOC bandwidth)
        Ki=0.015,
        Kff=0.07,  # K_back of agv_hub_motor
        Kff_acc=7e-3,  # J of agv_hub_motor
        i_limit=1.0,
        output_limit=2.2,
        # agv_hub: (0.20+0.07)*8 = 2.16 N*m
    ),
    "forklift_drive": ControllerParams(
        Kp=1.100,
        Ki=0.050,
        Kff=0.57,  # K_back of forklift_drive
        Kff_acc=3.3e-1,  # J of forklift_drive
        i_limit=4.0,
        output_limit=6.7,
        # forklift: (1.10+0.57)*4 = 6.68 N*m
    ),
    # ── Commercial motor controller presets ───────────────────────────────────
    # Kff = K_back (back-EMF cancel), Kff_acc = J (inertia pre-compensation).
    # output_limit = (Kp + Kback) * omega_max (max torque at full speed).
    "dynamixel_xl430": ControllerParams(
        Kp=0.060,
        Ki=0.012,
        Kff=0.030,  # K_back
        Kff_acc=9e-4,  # J
        i_limit=0.30,
        output_limit=0.576,
        # (0.06+0.03)*6.4 = 0.576 N*m
    ),
    "pololu_37d_50": ControllerParams(
        Kp=0.044,
        Ki=0.008,
        Kff=0.022,  # K_back
        Kff_acc=2.3e-3,  # J
        i_limit=0.40,
        output_limit=0.83,
        # (0.044+0.022)*12.6 = 0.83 N*m
    ),
    "maxon_ec45_43": ControllerParams(
        Kp=1.200,
        Ki=0.200,
        Kff=0.150,  # K_back
        Kff_acc=5.4e-3,  # J
        i_limit=2.0,
        output_limit=12.3,
        # (1.20+0.15)*9.1 = 12.3 N*m
    ),
}

# ── Default position controller presets ──────────────────────────────────────
# Kp and Kd match the K_p / K_d values already in SERVO_PRESETS so the
# default step response is unchanged.  Ki=0 by default; set a small value
# (e.g. 0.1-0.5) to eliminate steady-state position error under external load.

POSITION_CONTROLLER_PRESETS: dict[str, ControllerParams] = {
    "light_servo": ControllerParams(
        Kp=4.0, Ki=0.0, Kd=0.8, i_limit=1.0, output_limit=float("inf")
    ),
    "agv_servo": ControllerParams(
        Kp=8.0, Ki=0.0, Kd=2.0, i_limit=2.0, output_limit=float("inf")
    ),
    "forklift_steer": ControllerParams(
        Kp=6.0, Ki=0.0, Kd=4.0, i_limit=2.0, output_limit=float("inf")
    ),
    "swerve_module": ControllerParams(
        Kp=3.0, Ki=0.0, Kd=0.4, i_limit=2.0, output_limit=float("inf")
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
        motor_omega: Motor shaft angular velocity (rad/s) = omega_actual * gear_ratio.
            Equal to omega_actual when no gearbox is attached.
        theta_enc: Cumulative encoder angle (rad), always monotonically changes.
        delta_cmd: Commanded steering angle (rad); steer/drive_steer only.
        delta_actual: Actual steering angle after servo lag (rad).
        delta_dot: Steering angular rate (rad/s); internal servo state.
        ticks: Quantised encoder tick count, wrapped to int32 range to simulate
            a 32-bit MCU hardware counter (0 when cpr=0).
    """

    name: str
    role: str
    omega_cmd: float = 0.0
    omega_actual: float = 0.0
    motor_omega: float = 0.0
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

    **Default (no controller)** — exact first-order solution::

        J · dω/dt = K_motor · (ω_cmd - ω) - K_back · ω
        ω(t+dt)  = ω_cmd + (ω - ω_cmd) · exp(-dt/τ),   τ = J/(K_motor+K_back)

    **With explicit controller** — the motor drive's built-in velocity PI/PID
    loop drives the plant equation::

        torque   = controller.step(ω_cmd, ω_actual, dt)
        dω/dt    = (torque - K_back · ω) / J          [Euler integration]

    The explicit controller path lets you tune Kp, Ki, Kd independently of
    the mechanical plant, matching real drive electronics (Dynamixel velocity
    PI registers, EPOS4 velocity loop, Elmo VL[1]/VL[2]).

    Args:
        params:     :class:`DCMotorParams` instance or preset name string.
        controller: Optional :class:`ControllerParams` for the velocity PI/PID
                    loop.  ``None`` (default) uses the exact analytical solution
                    — backward-compatible with all existing code.
    """

    def __init__(
        self,
        params: DCMotorParams | str = "small_dc",
        controller: ControllerParams | None = None,
    ) -> None:
        if isinstance(params, str):
            if params not in MOTOR_PRESETS:
                raise ValueError(
                    f"Unknown motor preset {params!r}. Available: {list(MOTOR_PRESETS)}"
                )
            self.params = MOTOR_PRESETS[params]
        else:
            self.params = params

        if controller is not None:
            self._controller: MotorController | None = MotorController(
                controller, mode=ControlMode.VELOCITY
            )
        else:
            self._controller = None
        # Tracks previous command for computing dω_cmd/dt (acceleration FF)
        self._prev_omega_cmd: float | None = None
        # Backlash simulation state
        self._theta_ideal: float = 0.0  # ideal output shaft angle (no backlash)
        self._theta_out: float = 0.0  # actual output shaft angle (after backlash)
        self._backlash_contact: float = 0.0  # edge of the backlash dead zone (rad)

    @property
    def control_mode(self) -> ControlMode:
        """Active control mode (always ``VELOCITY`` for a drive actuator)."""
        return ControlMode.VELOCITY

    def reset_controller(self) -> None:
        """Clear controller integrator, derivative, FF, and backlash state."""
        if self._controller is not None:
            self._controller.reset()
        self._prev_omega_cmd = None
        self._theta_ideal = 0.0
        self._theta_out = 0.0
        self._backlash_contact = 0.0

    def step(self, wheel: WheelState, omega_cmd: float, dt: float) -> None:
        """Advance motor state by one timestep.

        Args:
            wheel:     Mutable :class:`WheelState` to update in-place.
            omega_cmd: Desired wheel angular velocity (rad/s).
            dt:        Simulation timestep (s).
        """
        p = self.params
        wheel.omega_cmd = float(omega_cmd)

        if self._controller is not None:
            # Compute dω_cmd/dt for the acceleration feedforward term.
            if self._prev_omega_cmd is not None and dt > 0.0:
                cmd_rate = (omega_cmd - self._prev_omega_cmd) / dt
            else:
                cmd_rate = 0.0
            self._prev_omega_cmd = float(omega_cmd)
            # Explicit velocity PID + FF path: controller output is motor torque.
            # Plant: J·dω/dt = torque - K_back·ω
            torque = self._controller.step(omega_cmd, wheel.omega_actual, dt, cmd_rate)
            dw_dt = (torque - p.K_back * wheel.omega_actual) / p.J
            new_omega = wheel.omega_actual + dw_dt * dt
        else:
            # Exact first-order solution (backward-compatible default):
            #   ω(t+dt) = ω_cmd + (ω - ω_cmd)·exp(-dt/τ)
            tau = p.J / (p.K_motor + p.K_back)
            decay = float(np.exp(-dt / tau))
            new_omega = omega_cmd + (wheel.omega_actual - omega_cmd) * decay

        # Apply gearbox backlash dead zone.
        # _theta_ideal accumulates the frictionless output position.  The real
        # output (_theta_out) only moves when _theta_ideal escapes the backlash
        # window; inside the window the wheel stands still even though the motor
        # is turning (energy absorbed by the gear play).
        gb = p.gearbox
        if gb is not None and gb.backlash > 0.0 and dt > 0.0:
            bl = gb.backlash
            self._theta_ideal += new_omega * dt
            prev_out = self._theta_out
            if self._theta_ideal > self._backlash_contact + bl / 2:
                self._backlash_contact = self._theta_ideal - bl / 2
                self._theta_out = self._backlash_contact
            elif self._theta_ideal < self._backlash_contact - bl / 2:
                self._backlash_contact = self._theta_ideal + bl / 2
                self._theta_out = self._backlash_contact
            # else: inside dead zone — _theta_out unchanged
            effective_omega = (self._theta_out - prev_out) / dt
            wheel.omega_actual = float(
                np.clip(effective_omega, -p.omega_max, p.omega_max)
            )
        else:
            wheel.omega_actual = float(np.clip(new_omega, -p.omega_max, p.omega_max))

        # Report motor-shaft speed (raw motor before gearbox).
        ratio = gb.ratio if gb is not None else 1.0
        wheel.motor_omega = float(wheel.omega_actual * ratio)


class ServoActuator:
    """Steering-wheel actuator using a 2nd-order position controller.

    **Default (no controller)** — PD control via :class:`ServoParams`::

        J_s · δ̈ = K_p · (δ_cmd - δ) - (K_d + B_s) · δ̇

    is integrated with Euler; angle and rate are clamped to their limits.

    **With explicit controller** — the servo drive's built-in PID loop
    (e.g., Dynamixel position PID registers) drives the plant::

        force   = controller.step(δ_cmd, δ_actual, dt)   [N·m, PID output]
        δ̈       = (force - B_s · δ̇) / J_s               [Euler integration]

    Adding ``Ki > 0`` in :class:`ControllerParams` eliminates the small
    steady-state angular error that appears when the servo holds against
    an external moment (gravity on a tilted arm, for example).

    Args:
        params:     :class:`ServoParams` instance or preset name string.
        controller: Optional :class:`ControllerParams` for the position PID
                    loop.  ``None`` (default) uses ``K_p``/``K_d`` from
                    ``params`` directly — identical to the previous behaviour.
    """

    def __init__(
        self,
        params: ServoParams | str = "light_servo",
        controller: ControllerParams | None = None,
    ) -> None:
        if isinstance(params, str):
            if params not in SERVO_PRESETS:
                raise ValueError(
                    f"Unknown servo preset {params!r}. Available: {list(SERVO_PRESETS)}"
                )
            self.params = SERVO_PRESETS[params]
        else:
            self.params = params

        if controller is not None:
            self._controller: MotorController | None = MotorController(
                controller, mode=ControlMode.POSITION
            )
        else:
            self._controller = None

    @property
    def control_mode(self) -> ControlMode:
        """Active control mode (always ``POSITION`` for a steering actuator)."""
        return ControlMode.POSITION

    def reset_controller(self) -> None:
        """Clear the built-in controller's integrator and derivative history."""
        if self._controller is not None:
            self._controller.reset()

    def step(self, wheel: WheelState, delta_cmd: float, dt: float) -> None:
        """Advance servo state by one timestep.

        Args:
            wheel:     Mutable :class:`WheelState` to update in-place.
            delta_cmd: Desired steering angle (rad).
            dt:        Simulation timestep (s).
        """
        p = self.params
        delta_cmd = float(np.clip(delta_cmd, p.delta_min, p.delta_max))
        wheel.delta_cmd = delta_cmd

        if self._controller is not None:
            # Explicit position PID: controller drives the plant J_s·δ̈ = force - B_s·δ̇
            force = self._controller.step(delta_cmd, wheel.delta_actual, dt)
            delta_ddot = (force - p.B_s * wheel.delta_dot) / p.J_s
        else:
            # Default PD from ServoParams (backward-compatible):
            #   J_s·δ̈ = K_p·(e) - (K_d + B_s)·δ̇
            force = (
                p.K_p * (delta_cmd - wheel.delta_actual)
                - (p.K_d + p.B_s) * wheel.delta_dot
            )
            delta_ddot = force / p.J_s

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
            # Wrap to int32 range to simulate 32-bit hardware counter overflow.
            wheel.ticks = int(np.int32(round(wheel.theta_enc * self._ticks_per_rad)))


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
        motor_controller: ControllerParams | None = None,
        servo_controller: ControllerParams | None = None,
    ) -> None:
        self.wheels: list[WheelState] = []
        self._motor_act: dict[str, DCMotorActuator] = {}
        self._servo_act: dict[str, ServoActuator] = {}
        self._encoders: dict[str, WheelEncoder] = {}
        self._motor_params = motor
        self._servo_params = servo
        self._encoder_cpr = encoder_cpr
        self._motor_controller = motor_controller
        self._servo_controller = servo_controller

    def _add_drive_wheel(self, name: str) -> None:
        ws = WheelState(name=name, role="drive")
        self.wheels.append(ws)
        self._motor_act[name] = DCMotorActuator(
            self._motor_params, self._motor_controller
        )
        self._encoders[name] = WheelEncoder(self._encoder_cpr)

    def _add_steer_wheel(self, name: str) -> None:
        ws = WheelState(name=name, role="steer")
        self.wheels.append(ws)
        self._servo_act[name] = ServoActuator(
            self._servo_params, self._servo_controller
        )

    def _add_drive_steer_wheel(self, name: str) -> None:
        ws = WheelState(name=name, role="drive_steer")
        self.wheels.append(ws)
        self._motor_act[name] = DCMotorActuator(
            self._motor_params, self._motor_controller
        )
        self._servo_act[name] = ServoActuator(
            self._servo_params, self._servo_controller
        )
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

    def reset_controllers(self) -> None:
        """Clear integrator and derivative history of all built-in controllers."""
        for act in self._motor_act.values():
            act.reset_controller()
        for act in self._servo_act.values():
            act.reset_controller()

    def reset(self) -> None:
        """Reset all wheel states and controller history to zero."""
        for w in self.wheels:
            w.omega_cmd = 0.0
            w.omega_actual = 0.0
            w.theta_enc = 0.0
            w.delta_cmd = 0.0
            w.delta_actual = 0.0
            w.delta_dot = 0.0
            w.ticks = 0
        self.reset_controllers()


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
        motor_controller: ControllerParams | None = None,
    ) -> None:
        super().__init__(
            motor=motor, encoder_cpr=encoder_cpr, motor_controller=motor_controller
        )
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
        motor_controller: ControllerParams | None = None,
    ) -> None:
        super().__init__(
            motor=motor, encoder_cpr=encoder_cpr, motor_controller=motor_controller
        )
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
        motor_controller: ControllerParams | None = None,
        servo_controller: ControllerParams | None = None,
    ) -> None:
        super().__init__(
            motor=motor,
            servo=servo,
            encoder_cpr=encoder_cpr,
            motor_controller=motor_controller,
            servo_controller=servo_controller,
        )
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
        motor_controller: ControllerParams | None = None,
        servo_controller: ControllerParams | None = None,
    ) -> None:
        super().__init__(
            motor=motor,
            servo=servo,
            encoder_cpr=encoder_cpr,
            motor_controller=motor_controller,
            servo_controller=servo_controller,
        )
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
        motor_controller: ControllerParams | None = None,
        servo_controller: ControllerParams | None = None,
    ) -> None:
        super().__init__(
            motor=motor,
            servo=servo,
            encoder_cpr=encoder_cpr,
            motor_controller=motor_controller,
            servo_controller=servo_controller,
        )
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
        motor_controller: ControllerParams | None = None,
        servo_controller: ControllerParams | None = None,
    ) -> None:
        super().__init__(
            motor=motor,
            servo=servo,
            encoder_cpr=encoder_cpr,
            motor_controller=motor_controller,
            servo_controller=servo_controller,
        )
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
