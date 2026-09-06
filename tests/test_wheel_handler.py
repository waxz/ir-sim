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
    POSITION_CONTROLLER_PRESETS,
    SERVO_PRESETS,
    VELOCITY_CONTROLLER_PRESETS,
    AckerWheelLayout,
    ControllerParams,
    ControlMode,
    DCMotorActuator,
    DCMotorParams,
    DiffWheelLayout,
    DualSteerWheelLayout,
    ForkiftWheelLayout,
    MecanumWheelLayout,
    MotorController,
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


# ---------------------------------------------------------------------------
# ControlMode / ControllerParams / MotorController
# ---------------------------------------------------------------------------


def test_control_mode_values():
    """ControlMode has the three standard string values."""
    assert ControlMode.VELOCITY == "velocity"
    assert ControlMode.POSITION == "position"
    assert ControlMode.TORQUE == "torque"


def test_controller_params_defaults():
    """ControllerParams default Kp=1, Ki=Kd=Kff=Kff_acc=0, unlimited limits."""
    p = ControllerParams()
    assert p.Kp == 1.0
    assert p.Ki == 0.0
    assert p.Kd == 0.0
    assert p.Kff == 0.0
    assert p.Kff_acc == 0.0
    assert p.i_limit == float("inf")
    assert p.output_limit == float("inf")


def test_motor_controller_p_only():
    """P-only MotorController produces Kp * error output."""
    ctrl = MotorController(ControllerParams(Kp=2.0), mode=ControlMode.VELOCITY)
    out = ctrl.step(setpoint=5.0, measurement=3.0, dt=0.01)
    assert out == pytest.approx(4.0)  # 2.0 * (5-3)


def test_motor_controller_integral_accumulates():
    """Integral term accumulates over multiple steps."""
    ctrl = MotorController(ControllerParams(Kp=0.0, Ki=1.0), mode=ControlMode.VELOCITY)
    ctrl.step(1.0, 0.0, 0.1)  # integral += 1.0 * 0.1 = 0.1
    ctrl.step(1.0, 0.0, 0.1)  # integral += 1.0 * 0.1 = 0.2
    assert ctrl.integral == pytest.approx(0.2, abs=1e-10)


def test_motor_controller_anti_windup():
    """Integrator clamps to i_limit."""
    ctrl = MotorController(
        ControllerParams(Kp=0.0, Ki=10.0, i_limit=0.5), mode=ControlMode.VELOCITY
    )
    for _ in range(100):
        ctrl.step(1.0, 0.0, 0.1)
    assert abs(ctrl.integral) <= 0.5 + 1e-9


def test_motor_controller_output_clamp():
    """Output is clamped to output_limit."""
    ctrl = MotorController(
        ControllerParams(Kp=100.0, output_limit=3.0), mode=ControlMode.VELOCITY
    )
    out = ctrl.step(1.0, 0.0, 0.01)
    assert abs(out) <= 3.0 + 1e-9


def test_motor_controller_reset():
    """reset() clears integrator and derivative state."""
    ctrl = MotorController(ControllerParams(Kp=0.0, Ki=1.0), mode=ControlMode.VELOCITY)
    ctrl.step(1.0, 0.0, 0.1)
    ctrl.reset()
    assert ctrl.integral == 0.0


def test_motor_controller_derivative_zero_first_step():
    """Derivative term is zero on the first call (no spike)."""
    ctrl = MotorController(ControllerParams(Kp=0.0, Kd=100.0), mode=ControlMode.VELOCITY)
    out = ctrl.step(1.0, 0.0, 0.01)
    assert out == pytest.approx(0.0)


def test_motor_controller_velocity_feedforward():
    """Kff * setpoint is added to the output (velocity feedforward)."""
    ctrl = MotorController(ControllerParams(Kp=0.0, Kff=2.0), mode=ControlMode.VELOCITY)
    out = ctrl.step(setpoint=5.0, measurement=5.0, dt=0.01)  # error=0, FF=2*5=10
    assert out == pytest.approx(10.0)


def test_motor_controller_acceleration_feedforward():
    """Kff_acc * setpoint_rate is added to the output (acceleration feedforward)."""
    ctrl = MotorController(ControllerParams(Kp=0.0, Kff_acc=0.5), mode=ControlMode.VELOCITY)
    out = ctrl.step(setpoint=0.0, measurement=0.0, dt=0.01, setpoint_rate=4.0)
    assert out == pytest.approx(2.0)  # 0.5 * 4.0


def test_motor_controller_ff_cancels_back_emf():
    """Kff = K_back achieves zero steady-state error with P-only (no integral needed)."""
    J, K_back = 5e-3, 0.035
    params = DCMotorParams(J=J, K_motor=0.065, K_back=K_back, omega_max=20.0)
    # P-only with back-EMF feedforward — ω_ss should approach ω_cmd
    ctrl_ff = ControllerParams(Kp=0.065, Ki=0.0, Kff=K_back, output_limit=5.0)
    act = DCMotorActuator(params, controller=ctrl_ff)
    wheel = WheelState(name="test", role="drive")
    for _ in range(2000):
        act.step(wheel, 20.0, 0.001)
    # With feedforward, error should be near zero (< 1% of cmd)
    assert abs(wheel.omega_actual - 20.0) < 0.2


def test_motor_controller_ff_reduces_ramp_lag():
    """Acceleration feedforward reduces ramp-tracking error vs. P-only with no FF."""
    J, K_back = 3e-2, 0.07
    params = DCMotorParams(J=J, K_motor=0.13, K_back=K_back, omega_max=8.0)
    tau = J / (params.K_motor + K_back)
    ramp_rate = 8.0 / (3 * tau)  # same ramp as the report (omega_max over 3*tau)

    ctrl_no_ff = ControllerParams(Kp=0.13, Ki=0.0, Kff=0.0, Kff_acc=0.0, output_limit=3.0)
    ctrl_ff = ControllerParams(Kp=0.13, Ki=0.0, Kff=K_back, Kff_acc=J, output_limit=3.0)

    act_no_ff = DCMotorActuator(params, controller=ctrl_no_ff)
    act_ff = DCMotorActuator(params, controller=ctrl_ff)
    w_no_ff = WheelState(name="a", role="drive")
    w_ff = WheelState(name="b", role="drive")

    dt = 0.001
    t_ramp = 3 * tau
    errors_no_ff, errors_ff = [], []
    t = 0.0
    while t < t_ramp:
        cmd = min(ramp_rate * t, 8.0)
        act_no_ff.step(w_no_ff, cmd, dt)
        act_ff.step(w_ff, cmd, dt)
        errors_no_ff.append(abs(cmd - w_no_ff.omega_actual))
        errors_ff.append(abs(cmd - w_ff.omega_actual))
        t += dt

    rms_no_ff = (sum(e**2 for e in errors_no_ff) / len(errors_no_ff)) ** 0.5
    rms_ff = (sum(e**2 for e in errors_ff) / len(errors_ff)) ** 0.5
    assert rms_ff < rms_no_ff * 0.5, (
        f"FF ramp RMS {rms_ff:.3f} should be <50% of no-FF {rms_no_ff:.3f}"
    )


def test_velocity_controller_presets_keys():
    """VELOCITY_CONTROLLER_PRESETS has an entry for every MOTOR_PRESETS key."""
    for key in MOTOR_PRESETS:
        assert key in VELOCITY_CONTROLLER_PRESETS, f"Missing velocity preset for {key!r}"


def test_velocity_controller_presets_have_feedforward():
    """Every velocity controller preset has nonzero Kff and Kff_acc."""
    for key, params in VELOCITY_CONTROLLER_PRESETS.items():
        assert params.Kff > 0, f"{key}: Kff should be nonzero"
        assert params.Kff_acc > 0, f"{key}: Kff_acc should be nonzero"


def test_position_controller_presets_keys():
    """POSITION_CONTROLLER_PRESETS has an entry for every SERVO_PRESETS key."""
    for key in SERVO_PRESETS:
        assert key in POSITION_CONTROLLER_PRESETS, f"Missing position preset for {key!r}"


# ---------------------------------------------------------------------------
# DCMotorActuator with explicit controller
# ---------------------------------------------------------------------------


def test_dc_motor_actuator_explicit_controller_p_steady_state():
    """P-only controller reaches ω_ss = Kp*cmd/(Kp+K_back) (back-EMF limits steady state)."""
    params = DCMotorParams(J=5e-3, K_motor=0.065, K_back=0.035, omega_max=20.0)
    ctrl_p = ControllerParams(Kp=0.065, Ki=0.0, output_limit=5.0)
    act = DCMotorActuator(params, controller=ctrl_p)
    wheel = WheelState(name="test", role="drive")
    # Run well past 5τ (τ = J/(Kp+K_back) = 5e-3/0.1 = 50 ms → 500 steps)
    for _ in range(1000):
        act.step(wheel, 20.0, 0.001)
    # Expected steady state: ω_ss = Kp*cmd/(Kp+K_back) = 0.065*20/0.1 = 13 rad/s
    omega_ss_expected = 0.065 * 20.0 / (0.065 + 0.035)
    assert wheel.omega_actual == pytest.approx(omega_ss_expected, abs=0.1)


def test_dc_motor_actuator_pi_controller_eliminates_offset():
    """PI controller drives ω_actual → ω_cmd (integral eliminates back-EMF offset)."""
    params = DCMotorParams(J=5e-3, K_motor=0.065, K_back=0.035, omega_max=20.0)
    # P-only — has steady-state error proportional to K_back
    p_ctrl = ControllerParams(Kp=0.065, Ki=0.0, output_limit=3.0)
    act_p = DCMotorActuator(params, controller=p_ctrl)
    wheel_p = WheelState(name="p", role="drive")
    # PI — integral eliminates the back-EMF steady-state error
    pi_ctrl = ControllerParams(Kp=0.065, Ki=0.5, i_limit=2.0, output_limit=3.0)
    act_pi = DCMotorActuator(params, controller=pi_ctrl)
    wheel_pi = WheelState(name="pi", role="drive")
    # Run 2 seconds at 1 ms step
    for _ in range(2000):
        act_p.step(wheel_p, 10.0, 0.001)
        act_pi.step(wheel_pi, 10.0, 0.001)
    err_p = abs(wheel_p.omega_actual - 10.0)
    err_pi = abs(wheel_pi.omega_actual - 10.0)
    assert err_pi < err_p, f"PI error {err_pi:.4f} should be less than P error {err_p:.4f}"


def test_dc_motor_actuator_control_mode_property():
    """control_mode is always VELOCITY for DCMotorActuator."""
    act = DCMotorActuator("small_dc")
    assert act.control_mode == ControlMode.VELOCITY


def test_dc_motor_actuator_reset_controller_noop_when_none():
    """reset_controller() does not raise when no explicit controller set."""
    act = DCMotorActuator("small_dc")
    act.reset_controller()  # should not raise


# ---------------------------------------------------------------------------
# ServoActuator with explicit controller
# ---------------------------------------------------------------------------


def test_servo_actuator_explicit_controller_converges():
    """ServoActuator with explicit PD controller reaches target angle."""
    ctrl_pd = ControllerParams(Kp=4.0, Kd=0.8, output_limit=50.0)
    act = ServoActuator("light_servo", controller=ctrl_pd)
    wheel = WheelState(name="test", role="steer")
    for _ in range(2000):
        act.step(wheel, 0.5, 0.001)
    assert wheel.delta_actual == pytest.approx(0.5, abs=0.05)


def test_servo_actuator_control_mode_property():
    """control_mode is always POSITION for ServoActuator."""
    act = ServoActuator("light_servo")
    assert act.control_mode == ControlMode.POSITION


def test_servo_actuator_pid_integral_state():
    """ServoActuator with Ki > 0 accumulates integral state."""
    ctrl_pid = ControllerParams(Kp=4.0, Ki=1.0, Kd=0.8, i_limit=5.0)
    act = ServoActuator("light_servo", controller=ctrl_pid)
    assert act._controller is not None
    wheel = WheelState(name="test", role="steer")
    for _ in range(100):
        act.step(wheel, 0.3, 0.01)
    # Integral should be non-zero after tracking a non-zero setpoint
    assert act._controller.integral != 0.0


# ---------------------------------------------------------------------------
# WheelLayout with motor_controller / servo_controller
# ---------------------------------------------------------------------------


def test_diff_layout_with_motor_controller():
    """DiffWheelLayout accepts motor_controller and passes it to actuators."""
    ctrl = ControllerParams(Kp=0.065, Ki=0.01, i_limit=0.5, output_limit=2.0)
    layout = DiffWheelLayout(motor_controller=ctrl)
    vel = np.array([[0.5], [0.0]])
    for _ in range(50):
        layout.step(vel, 0.01)
    # Both actuators should have a MotorController attached
    for act in layout._motor_act.values():
        assert act._controller is not None


def test_layout_reset_clears_controller_state():
    """layout.reset() clears controller integrator state."""
    ctrl = ControllerParams(Kp=0.065, Ki=1.0, i_limit=10.0, output_limit=5.0)
    layout = DiffWheelLayout(motor_controller=ctrl)
    vel = np.array([[1.0], [0.0]])
    for _ in range(200):
        layout.step(vel, 0.01)
    # Integrators should be non-zero
    for act in layout._motor_act.values():
        assert act._controller is not None
    layout.reset()
    for act in layout._motor_act.values():
        assert act._controller is not None
        assert act._controller.integral == 0.0
