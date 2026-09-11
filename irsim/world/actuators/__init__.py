"""Actuator models for IR-SIM.

This package provides motor-level actuation models that operate below the
kinematic abstraction layer.  They are **standalone** — no dependency on
ObjectBase or KinematicsFactory — and can be used independently or composed
into higher-level chassis models.

Classes
-------
Motor
    Simulated DC motor with built-in PID controller and quadrature encoder.
MotorDiffChassis
    Differential drive chassis driven by two independent :class:`Motor`
    instances; integrates motor dynamics, encoder state, and 2-D pose.
"""

from irsim.world.actuators.motor import Motor
from irsim.world.actuators.motor_diff_chassis import MotorDiffChassis

__all__ = ["Motor", "MotorDiffChassis"]
