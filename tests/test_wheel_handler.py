"""Tests for the per-wheel actuator/encoder/layout system.

Covers:
- Pure IK/FK round-trips in wheel_kinematics.py
- DC motor and servo actuator dynamics
- Encoder accumulation and CPR quantisation
- All six WheelLayout concrete classes
- WheelLayoutFactory registry
- reset() functionality
- ObjectBase.wheel_states / encoder_readings properties
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.lib.algorithm.wheel_kinematics import (
    acker_inv_kin,
    acker_steer_angles,
    diff_fwd_kin,
    diff_inv_kin,
    forklift_inv_kin,
    mecanum_fwd_kin,
    mecanum_inv_kin,
    swerve_inv_kin,
    swerve_wheel_cmd,
)
from irsim.lib.handler.wheel_handler import (
    MOTOR_PRESETS,
    SERVO_PRESETS,
    AckerWheelLayout,
    DCMotorActuator,
    DCMotorParams,
    DiffWheelLayout,
    DualSteerWheelLayout,
    ForkiftWheelLayout,
    MecanumWheelLayout,
    QuadSteerWheelLayout,
    ServoActuator,
    ServoParams,
    WheelEncoder,
    WheelLayoutFactory,
    WheelState,
)

# ---------------------------------------------------------------------------
# IK / FK helpers
# ---------------------------------------------------------------------------


def test_diff_ik_fk_round_trip():
    """diff_fwd_kin(diff_inv_kin(v, omega)) reproduces (v, omega)."""
    v, omega = 0.5, 0.3
    r, track = 0.033, 0.16
    ol, or_ = diff_inv_kin(v, omega, r, track)
    v2, omega2 = diff_fwd_kin(ol, or_, r, track)
    assert abs(v2 - v) < 1e-12
    assert abs(omega2 - omega) < 1e-12


def test_diff_ik_zero():
    """Zero v and omega → zero wheel speeds."""
    ol, or_ = diff_inv_kin(0.0, 0.0, 0.033, 0.16)
    assert ol == pytest.approx(0.0)
    assert or_ == pytest.approx(0.0)


def test_diff_ik_straight():
    """Straight motion → left and right wheels equal."""
    ol, or_ = diff_inv_kin(1.0, 0.0, 0.1, 0.3)
    assert ol == pytest.approx(or_)


def test_mecanum_ik_fk_round_trip_pure_vx():
    """Pure vx → mecanum IK → FK recovers vx with vy=0, omega=0."""
    vx, vy, omega_z = 1.0, 0.0, 0.0
    r, L, W = 0.05, 0.15, 0.15
    fl, fr, rl, rr = mecanum_inv_kin(vx, vy, omega_z, r, L, W)
    vx2, vy2, omega_z2 = mecanum_fwd_kin(fl, fr, rl, rr, r, L, W)
    assert vx2 == pytest.approx(vx, abs=1e-12)
    assert vy2 == pytest.approx(vy, abs=1e-12)
    assert omega_z2 == pytest.approx(omega_z, abs=1e-12)


def test_mecanum_ik_pure_vy():
    """Pure lateral motion → all wheel speeds equal magnitude."""
    vy = 0.5
    fl, fr, rl, rr = mecanum_inv_kin(0.0, vy, 0.0, 0.05, 0.15, 0.15)
    inv_r = 1.0 / 0.05
    assert fl == pytest.approx(-inv_r * vy, abs=1e-12)
    assert fr == pytest.approx(inv_r * vy, abs=1e-12)
    assert rl == pytest.approx(inv_r * vy, abs=1e-12)
    assert rr == pytest.approx(-inv_r * vy, abs=1e-12)


def test_acker_steer_angles_zero():
    """Zero steering angle → both wheel angles zero."""
    dl, dr = acker_steer_angles(0.0, 1.0, 0.5)
    assert dl == pytest.approx(0.0)
    assert dr == pytest.approx(0.0)


def test_acker_steer_angles_symmetry():
    """Ackermann correction: inner wheel has larger angle magnitude."""
    psi = (
        0.3  # right turn (positive psi = left in convention, but let's check magnitude)
    )
    dl, dr = acker_steer_angles(psi, 1.0, 0.5)
    # One angle should be larger than the other (they differ due to track width)
    assert abs(dl) != pytest.approx(abs(dr), abs=1e-6)


def test_acker_inv_kin_straight():
    """Straight ahead (psi=0): both rear wheels same speed, zero steer."""
    result = acker_inv_kin(1.0, 0.0, 0.1, 1.0, 0.5)
    assert result["omega_rl"] == pytest.approx(result["omega_rr"])
    assert result["delta_fl"] == pytest.approx(0.0)
    assert result["delta_fr"] == pytest.approx(0.0)


def test_forklift_inv_kin_straight():
    """Straight motion (omega=0): rear delta_rc = 0."""
    result = forklift_inv_kin(1.0, 0.0, 0.15, 0.6, 0.5)
    assert result["delta_rc"] == pytest.approx(0.0)
    assert result["omega_fl"] == pytest.approx(result["omega_fr"])


def test_forklift_inv_kin_turn():
    """Non-zero yaw rate: rear wheel steers in expected direction."""
    result = forklift_inv_kin(0.0, 0.5, 0.15, 0.6, 0.5)
    # v_rc_y = -omega * half_wheelbase = -0.5 * 0.6 = -0.3 → delta_rc negative
    assert result["delta_rc"] < 0.0


def test_swerve_single_module_forward():
    """Pure forward motion: swerve module steer = 0."""
    delta, omega_w = swerve_wheel_cmd(1.0, 0.0, 0.0, 0.15, 0.15, 0.05)
    assert delta == pytest.approx(0.0, abs=1e-12)
    assert omega_w == pytest.approx(1.0 / 0.05, abs=1e-12)


def test_swerve_inv_kin_shape():
    """swerve_inv_kin returns (N,2) for N wheels."""
    positions = np.array([[0.15, 0.15], [0.15, -0.15], [-0.15, 0.15], [-0.15, -0.15]])
    cmds = swerve_inv_kin(1.0, 0.0, 0.0, 0.05, positions)
    assert cmds.shape == (4, 2)


# ---------------------------------------------------------------------------
# DCMotorActuator
# ---------------------------------------------------------------------------


def test_dc_motor_converges_to_command():
    """DC motor converges to commanded speed within several time constants."""
    params = DCMotorParams()
    act = DCMotorActuator(params)
    wheel = WheelState(name="test", role="drive")
    dt = 0.01
    omega_cmd = 10.0
    for _ in range(200):
        act.step(wheel, omega_cmd, dt)
    assert wheel.omega_actual == pytest.approx(omega_cmd, rel=0.01)


def test_dc_motor_respects_omega_max():
    """DC motor cannot exceed omega_max."""
    params = DCMotorParams(omega_max=5.0)
    act = DCMotorActuator(params)
    wheel = WheelState(name="test", role="drive")
    for _ in range(500):
        act.step(wheel, 1000.0, 0.01)
    assert abs(wheel.omega_actual) <= params.omega_max + 1e-9


def test_dc_motor_preset_names():
    """All MOTOR_PRESETS entries are valid DCMotorParams."""
    for name, p in MOTOR_PRESETS.items():
        assert isinstance(p, DCMotorParams), f"{name} is not DCMotorParams"
        assert p.tau > 0


def test_dc_motor_unknown_preset_raises():
    """DCMotorActuator raises ValueError for unknown preset name."""
    with pytest.raises(ValueError, match="motor preset"):
        DCMotorActuator("nonexistent_motor_xyz")


# ---------------------------------------------------------------------------
# ServoActuator
# ---------------------------------------------------------------------------


def test_servo_converges_to_command():
    """Servo reaches commanded angle within its limits."""
    params = ServoParams(K_p=8.0, K_d=2.0, J_s=0.1, B_s=0.05)
    act = ServoActuator(params)
    wheel = WheelState(name="test", role="steer")
    dt = 0.01
    delta_cmd = 0.4
    for _ in range(300):
        act.step(wheel, delta_cmd, dt)
    assert wheel.delta_actual == pytest.approx(delta_cmd, abs=0.01)


def test_servo_respects_limits():
    """Servo clamps to [delta_min, delta_max]."""
    params = ServoParams(delta_min=-0.5, delta_max=0.5)
    act = ServoActuator(params)
    wheel = WheelState(name="test", role="steer")
    for _ in range(500):
        act.step(wheel, 10.0, 0.01)
    assert wheel.delta_actual <= params.delta_max + 1e-9


def test_servo_preset_names():
    """All SERVO_PRESETS entries are valid ServoParams."""
    for name, p in SERVO_PRESETS.items():
        assert isinstance(p, ServoParams), f"{name} is not ServoParams"


# ---------------------------------------------------------------------------
# WheelEncoder
# ---------------------------------------------------------------------------


def test_encoder_accumulates():
    """Encoder theta_enc accumulates omega_actual * dt."""
    enc = WheelEncoder(cpr=0)
    wheel = WheelState(name="test", role="drive")
    wheel.omega_actual = 5.0
    enc.step(wheel, 0.1)
    assert wheel.theta_enc == pytest.approx(0.5, abs=1e-12)


def test_encoder_cpr_quantises():
    """With CPR > 0, ticks are integer multiples of 1."""
    cpr = 512
    enc = WheelEncoder(cpr=cpr)
    wheel = WheelState(name="test", role="drive")
    wheel.omega_actual = 2 * math.pi  # 1 full revolution per second
    enc.step(wheel, 1.0)  # dt=1s → theta_enc = 2π → cpr ticks
    assert wheel.ticks == cpr


# ---------------------------------------------------------------------------
# WheelLayout — DiffWheelLayout
# ---------------------------------------------------------------------------


def test_diff_layout_step_updates_wheels():
    """DiffWheelLayout.step() updates omega_actual for both wheels."""
    layout = DiffWheelLayout()
    vel = np.array([[0.5], [0.3]])
    for _ in range(100):
        layout.step(vel, 0.01)
    states = layout.get_wheel_states()
    assert "left" in states
    assert "right" in states
    assert abs(states["left"].omega_actual) > 0


def test_diff_layout_encoder_readings():
    """get_encoder_readings returns keys theta_enc and ticks."""
    layout = DiffWheelLayout(encoder_cpr=0)
    vel = np.array([[0.5], [0.0]])
    for _ in range(10):
        layout.step(vel, 0.01)
    readings = layout.get_encoder_readings()
    for _wheel_name, data in readings.items():
        assert "theta_enc" in data
        assert "ticks" in data


def test_diff_layout_reset():
    """reset() zeros all wheel states."""
    layout = DiffWheelLayout()
    vel = np.array([[1.0], [0.0]])
    for _ in range(50):
        layout.step(vel, 0.01)
    layout.reset()
    for ws in layout.get_wheel_states().values():
        assert ws.omega_actual == pytest.approx(0.0)
        assert ws.theta_enc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# WheelLayout — AckerWheelLayout
# ---------------------------------------------------------------------------


def test_acker_layout_straight_no_steer():
    """Straight motion: front wheel steer angles stay near zero."""
    layout = AckerWheelLayout()
    vel = np.array([[1.0], [0.0]])
    for _ in range(300):
        layout.step(vel, 0.01)
    states = layout.get_wheel_states()
    assert abs(states["FL"].delta_actual) < 0.05
    assert abs(states["FR"].delta_actual) < 0.05


# ---------------------------------------------------------------------------
# WheelLayout — ForkiftWheelLayout
# ---------------------------------------------------------------------------


def test_forklift_layout_straight_delta_zero():
    """Forklift: straight motion → rear-center wheel delta converges to 0."""
    layout = ForkiftWheelLayout()
    vel = np.array([[1.0], [0.0]])
    for _ in range(300):
        layout.step(vel, 0.01)
    states = layout.get_wheel_states()
    assert abs(states["RC"].delta_actual) < 0.1


# ---------------------------------------------------------------------------
# WheelLayout — DualSteerWheelLayout
# ---------------------------------------------------------------------------


def test_dual_steer_layout_has_fc_rc():
    """DualSteerWheelLayout has exactly 'FC' and 'RC' wheels."""
    layout = DualSteerWheelLayout()
    states = layout.get_wheel_states()
    assert set(states.keys()) == {"FC", "RC"}


# ---------------------------------------------------------------------------
# WheelLayout — QuadSteerWheelLayout
# ---------------------------------------------------------------------------


def test_quad_steer_layout_has_four_wheels():
    """QuadSteerWheelLayout has exactly FL, FR, RL, RR."""
    layout = QuadSteerWheelLayout()
    states = layout.get_wheel_states()
    assert set(states.keys()) == {"FL", "FR", "RL", "RR"}


def test_quad_steer_pure_forward():
    """QuadSteer pure forward: all steer angles converge near 0."""
    layout = QuadSteerWheelLayout()
    vel = np.array([[1.0], [0.0], [0.0]])
    for _ in range(300):
        layout.step(vel, 0.01)
    states = layout.get_wheel_states()
    for wname, ws in states.items():
        assert abs(ws.delta_actual) < 0.1, f"wheel {wname} steer not near 0"


# ---------------------------------------------------------------------------
# WheelLayoutFactory
# ---------------------------------------------------------------------------


def test_factory_creates_all_layouts():
    """Factory creates each registered layout type without error."""
    for name in ("diff", "mecanum", "acker", "forklift", "dual_steer", "quad_steer"):
        layout = WheelLayoutFactory.create(name)
        assert layout is not None


def test_factory_invalid_name_raises():
    """Factory raises ValueError for unregistered layout name."""
    with pytest.raises(ValueError, match=r"[Uu]nknown wheel layout"):
        WheelLayoutFactory.create("nonexistent_layout_xyz")


def test_factory_custom_kwargs_passed():
    """Factory passes kwargs to the layout constructor."""
    layout = WheelLayoutFactory.create("diff", wheel_radius=0.05, track=0.20)
    assert layout.wheel_radius == pytest.approx(0.05)
    assert layout.track == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# MecanumWheelLayout
# ---------------------------------------------------------------------------


def test_mecanum_layout_has_four_wheels():
    """MecanumWheelLayout has FL, FR, RL, RR."""
    layout = MecanumWheelLayout()
    states = layout.get_wheel_states()
    assert set(states.keys()) == {"FL", "FR", "RL", "RR"}
