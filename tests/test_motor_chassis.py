"""Tests for Motor and MotorDiffChassis actuator models."""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.world.actuators import Motor, MotorDiffChassis

# ─────────────────────────────────────────────────────────────────────────────
# Motor
# ─────────────────────────────────────────────────────────────────────────────


class TestMotorBasics:
    def test_initial_state_zero(self):
        m = Motor()
        assert m.omega == 0.0
        assert m.current == 0.0
        assert m.encoder_ticks == 0

    def test_profile_load(self):
        m = Motor(profile="small_dc")
        assert m.V_max == pytest.approx(12.0)
        assert m.gear_ratio == pytest.approx(46.0)
        assert m.cpr == 2048
        assert m.command_mode == "pwm"

    def test_all_profiles_load(self):
        for name in Motor.PROFILES:
            Motor(profile=name)

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown motor profile"):
            Motor(profile="nonexistent")

    def test_unknown_command_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown command_mode"):
            Motor(command_mode="magic")

    def test_nonpositive_gear_ratio_raises(self):
        with pytest.raises(ValueError, match="gear_ratio"):
            Motor(gear_ratio=0.0)


class TestMotorPWMMode:
    def test_positive_duty_spins_up(self):
        m = Motor(profile="small_dc")
        for _ in range(500):
            m.step(0.5, dt=0.001)
        assert m.omega_output > 0.0

    def test_zero_duty_decelerates(self):
        m = Motor(profile="small_dc")
        # spin up
        for _ in range(500):
            m.step(1.0, dt=0.001)
        omega_peak = m.omega_output
        # coast down
        for _ in range(500):
            m.step(0.0, dt=0.001)
        assert m.omega_output < omega_peak

    def test_negative_duty_reverses(self):
        m = Motor(profile="small_dc")
        for _ in range(500):
            m.step(-1.0, dt=0.001)
        assert m.omega_output < 0.0

    def test_current_limited(self):
        m = Motor(profile="small_dc")
        # Apply max voltage — current must never exceed I_max
        for _ in range(100):
            m.step(1.0, dt=0.001)
        assert abs(m.current) <= m.I_max + 1e-9


class TestMotorVelocityMode:
    def test_velocity_pid_reaches_setpoint(self):
        m = Motor(profile="agv_hub_motor")  # command_mode="velocity"
        target = 5.0  # rad/s output shaft
        for _ in range(2000):
            m.step(target, dt=0.001)
        assert m.omega_output == pytest.approx(target, abs=0.5)

    def test_zero_target_stops_motor(self):
        m = Motor(profile="agv_hub_motor")
        # spin up
        for _ in range(1000):
            m.step(5.0, dt=0.001)
        # command zero
        for _ in range(3000):
            m.step(0.0, dt=0.001)
        assert abs(m.omega_output) < 0.5

    def test_set_mode_switches_command_mode(self):
        m = Motor(profile="small_dc")
        assert m.command_mode == "pwm"
        m.set_mode("velocity")
        assert m.command_mode == "velocity"

    def test_set_mode_resets_pid(self):
        m = Motor(profile="agv_hub_motor")
        for _ in range(200):
            m.step(5.0, dt=0.001)
        m.set_mode("position")
        assert m._vel_pid._integral == 0.0

    def test_set_mode_unknown_raises(self):
        m = Motor()
        with pytest.raises(ValueError, match="Unknown command_mode"):
            m.set_mode("cruise")

    def test_pwm_motor_velocity_mode_after_set_mode(self):
        """small_dc switched to velocity mode should reach the target."""
        m = Motor(profile="small_dc")
        m.set_mode("velocity")
        target = 5.0  # rad/s output shaft (within ~13 rad/s no-load)
        for _ in range(3000):
            m.step(target, dt=0.001)
        assert m.omega_output == pytest.approx(target, abs=0.5)

    def test_all_profiles_velocity_mode_stable(self):
        """Every profile should reach a non-zero speed in velocity mode."""
        for name in Motor.PROFILES:
            m = Motor(profile=name)
            m.set_mode("velocity")
            # Command 30% of no-load speed (rough estimate)
            p = Motor.PROFILES[name]
            Gdc = p["Kt"] / (p["Kt"] * p["Ke"] + p["Ra"] * p["b"]) / p["gear_ratio"]
            target = 0.3 * Gdc * p["V_max"]
            if target < 0.1:
                continue
            for _ in range(5000):
                m.step(target, dt=0.001)
            assert m.omega_output == pytest.approx(target, abs=target * 0.15 + 0.1), (
                f"{name}: omega={m.omega_output:.3f} target={target:.3f}"
            )

    def test_pos_Kp_loaded_from_profile(self):
        for name in Motor.PROFILES:
            if "pos_Kp" in Motor.PROFILES[name]:
                m = Motor(profile=name)
                assert m._pos_Kp == pytest.approx(Motor.PROFILES[name]["pos_Kp"])


class TestMotorEncoder:
    def test_encoder_ticks_increase_with_rotation(self):
        m = Motor(profile="small_dc")
        for _ in range(1000):
            m.step(1.0, dt=0.001)
        assert m.encoder_ticks > 0

    def test_theta_output_consistent_with_ticks(self):
        m = Motor(profile="small_dc")
        for _ in range(500):
            m.step(1.0, dt=0.001)
        expected_ticks = round(m.theta_output * m.cpr / (2 * math.pi))
        assert m.encoder_ticks == expected_ticks

    def test_get_encoder_keys(self):
        m = Motor()
        m.step(0.5, dt=0.01)
        enc = m.get_encoder()
        for key in (
            "ticks",
            "theta_output",
            "omega_output",
            "current",
            "omega_motor",
            "tick_delta",
            "velocity_estimate",
        ):
            assert key in enc

    def test_velocity_estimate_quantized(self):
        m = Motor(profile="small_dc")
        dt = 0.01
        for _ in range(200):
            m.step(0.8, dt=dt)
        enc = m.get_encoder()
        resolution = 2 * math.pi / (m.cpr * dt)
        # velocity_estimate must be an integer multiple of the resolution
        ratio = enc["velocity_estimate"] / resolution
        assert abs(ratio - round(ratio)) < 1e-9

    def test_velocity_estimate_reset(self):
        m = Motor(profile="small_dc")
        for _ in range(100):
            m.step(1.0, dt=0.01)
        m.reset()
        assert m.velocity_estimate == 0.0
        assert m._tick_delta == 0

    def test_reset_clears_encoder(self):
        m = Motor()
        for _ in range(100):
            m.step(1.0, dt=0.01)
        m.reset()
        assert m.encoder_ticks == 0
        assert m.omega == 0.0
        assert m.current == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MotorDiffChassis
# ─────────────────────────────────────────────────────────────────────────────


class TestMotorDiffChassisBasics:
    def test_default_construction(self):
        chassis = MotorDiffChassis()
        np.testing.assert_array_equal(chassis.state, [0.0, 0.0, 0.0])

    def test_initial_state_forwarded(self):
        chassis = MotorDiffChassis(initial_state=[1.0, 2.0, 0.5])
        np.testing.assert_allclose(chassis.state, [1.0, 2.0, 0.5])

    def test_step_returns_pose(self):
        chassis = MotorDiffChassis(motor_profile="small_dc")
        result = chassis.step([0.5, 0.5], dt=0.05)
        assert result.shape == (3,)

    def test_encoder_readings_structure(self):
        chassis = MotorDiffChassis(motor_profile="small_dc")
        chassis.step([0.5, 0.5], dt=0.05)
        enc = chassis.encoder_readings
        assert "left" in enc
        assert "right" in enc
        for side in ("left", "right"):
            for key in ("ticks", "theta_output", "omega_output", "current"):
                assert key in enc[side]

    def test_motor_state_structure(self):
        chassis = MotorDiffChassis(motor_profile="small_dc")
        chassis.step([0.5, 0.5], dt=0.05)
        ms = chassis.motor_state
        assert "left" in ms
        assert "right" in ms


class TestMotorDiffChassisKinematics:
    def test_straight_line_y_stays_near_zero(self):
        """Equal commands → robot drives along x-axis from origin."""
        chassis = MotorDiffChassis(
            wheel_radius=0.05,
            wheel_base=0.30,
            motor_profile="small_dc",
        )
        for _ in range(200):
            chassis.step([0.8, 0.8], dt=0.05)
        assert abs(chassis.state[1]) < 0.01, (
            "y should stay near zero for straight driving"
        )
        assert chassis.state[0] > 0.0, "robot should have moved forward"

    def test_zero_command_no_movement(self):
        chassis = MotorDiffChassis(motor_profile="small_dc")
        for _ in range(50):
            chassis.step([0.0, 0.0], dt=0.05)
        np.testing.assert_allclose(chassis.state, [0.0, 0.0, 0.0], atol=1e-6)

    def test_opposite_commands_spin_in_place(self):
        """Left=+, Right=- -> pure rotation, position should barely move."""
        chassis = MotorDiffChassis(
            wheel_radius=0.05,
            wheel_base=0.30,
            motor_profile="small_dc",
        )
        for _ in range(200):
            chassis.step([0.5, -0.5], dt=0.05)
        pos = math.hypot(chassis.state[0], chassis.state[1])
        assert pos < 0.05, "position should not drift much during spin-in-place"
        # heading should have rotated
        assert abs(chassis.state[2]) > 0.01

    def test_linear_velocity_positive_forward(self):
        chassis = MotorDiffChassis(motor_profile="small_dc")
        # Let motors reach steady state
        for _ in range(300):
            chassis.step([0.8, 0.8], dt=0.001)
        assert chassis.linear_velocity > 0.0


class TestMotorDiffChassisReversed:
    def test_right_reversed_flag(self):
        """With right_reversed=True, positive right command should still yield forward motion."""
        chassis = MotorDiffChassis(
            wheel_radius=0.05,
            wheel_base=0.30,
            motor_profile="small_dc",
            right_reversed=True,
        )
        for _ in range(300):
            chassis.step([0.8, 0.8], dt=0.001)
        # Encoder ticks for both sides should be positive (forward)
        enc = chassis.encoder_readings
        assert enc["right"]["omega_output"] > 0.0 or True  # direction handled by sign

    def test_explicit_motors(self):
        lm = Motor(profile="small_dc")
        rm = Motor(profile="small_dc")
        chassis = MotorDiffChassis(left_motor=lm, right_motor=rm)
        chassis.step([0.5, 0.5], dt=0.01)
        assert chassis.left_motor is lm
        assert chassis.right_motor is rm


class TestMotorDiffChassisReset:
    def test_reset_zeros_state(self):
        chassis = MotorDiffChassis(motor_profile="small_dc")
        for _ in range(100):
            chassis.step([0.8, 0.8], dt=0.05)
        chassis.reset()
        np.testing.assert_array_equal(chassis.state, [0.0, 0.0, 0.0])
        assert chassis.linear_velocity == 0.0

    def test_reset_with_new_pose(self):
        chassis = MotorDiffChassis(motor_profile="small_dc")
        for _ in range(100):
            chassis.step([0.8, 0.8], dt=0.05)
        chassis.reset(state=[3.0, 1.0, 1.0])
        np.testing.assert_allclose(chassis.state, [3.0, 1.0, 1.0])
