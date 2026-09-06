"""Correctness tests for per-wheel actuator / encoder / kinematics layer.

Each test:
  1. Drives a layout through a straight phase (0-2 s) then a turning phase (2-5 s).
  2. Compares ground-truth pose (from the original kinematics functions) with the
     encoder-reconstructed pose (FK from actual wheel speeds).
  3. Checks that motor/servo responses track their commands within tolerance.
  4. Verifies encoder accumulation matches integrated omega_actual.

Motor time-constants are all < 1 ms, so at dt=0.01 s motors are effectively
ideal; the tolerance for trajectory convergence reflects only floating-point
accumulation.  Servos have settling times of 0.5-1.5 s and introduce real
trajectory lag during the turning phase.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.lib.algorithm.kinematics import (
    ackermann_kinematics,
    differential_kinematics,
    omni_angular_kinematics,
)
from irsim.lib.algorithm.wheel_kinematics import diff_fwd_kin, mecanum_fwd_kin
from irsim.lib.handler.wheel_handler import (
    AckerWheelLayout,
    DiffWheelLayout,
    DualSteerWheelLayout,
    ForkiftWheelLayout,
    MecanumWheelLayout,
    QuadSteerWheelLayout,
)

DT = 0.01
N_STRAIGHT = 200  # 2 s
N_TURN = 300  # 3 s
N_TOTAL = N_STRAIGHT + N_TURN


def _make_vel(*vals: float) -> np.ndarray:
    return np.array(vals, dtype=float).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Differential Drive
# ---------------------------------------------------------------------------


class TestDiffCorrectness:
    """Diff-drive: v=0.5, omega=0 then v=0.3, omega=0.5."""

    VEL_STRAIGHT = _make_vel(0.5, 0.0)
    VEL_TURN = _make_vel(0.3, 0.5)
    R = 0.033
    TRACK = 0.16

    def _simulate(self):
        layout = DiffWheelLayout(
            wheel_radius=self.R, track=self.TRACK, encoder_cpr=1024
        )
        gt_state = np.zeros((3, 1))
        fk_state = np.zeros((3, 1))
        data = {
            "gt": [],
            "fk": [],
            "omega_cmd": [],
            "omega_act_l": [],
            "omega_act_r": [],
            "theta_enc_l": [],
            "theta_enc_r": [],
        }

        for i in range(N_TOTAL):
            vel = self.VEL_STRAIGHT if i < N_STRAIGHT else self.VEL_TURN
            gt_state = differential_kinematics(gt_state, vel, DT)

            layout.step(vel, DT)
            ws = layout.get_wheel_states()
            enc = layout.get_encoder_readings()
            lw, rw = ws["left"], ws["right"]

            v_fk, om_fk = diff_fwd_kin(
                lw.omega_actual, rw.omega_actual, self.R, self.TRACK
            )
            fk_state = differential_kinematics(fk_state, _make_vel(v_fk, om_fk), DT)

            data["gt"].append(gt_state.copy())
            data["fk"].append(fk_state.copy())
            data["omega_cmd"].append(float(vel[0, 0]) / self.R)
            data["omega_act_l"].append(lw.omega_actual)
            data["omega_act_r"].append(rw.omega_actual)
            data["theta_enc_l"].append(enc["left"]["theta_enc"])
            data["theta_enc_r"].append(enc["right"]["theta_enc"])

        return data

    def test_motor_tracks_command_straight(self):
        """After straight phase, motor speed matches command to < 1%."""
        data = self._simulate()
        # At i=N_STRAIGHT-1, motor has been running at constant speed for 2 s
        # tau << dt, so convergence is essentially perfect
        cmd = self.VEL_STRAIGHT[0, 0] / self.R
        act_l = data["omega_act_l"][N_STRAIGHT - 1]
        act_r = data["omega_act_r"][N_STRAIGHT - 1]
        assert abs(act_l - cmd) / (abs(cmd) + 1e-9) < 0.01, (
            f"Left motor lag: cmd={cmd:.4f}, actual={act_l:.4f}"
        )
        assert abs(act_r - cmd) / (abs(cmd) + 1e-9) < 0.01, (
            f"Right motor lag: cmd={cmd:.4f}, actual={act_r:.4f}"
        )

    def test_encoder_accumulates_correctly(self):
        """theta_enc matches direct integration of omega_actual * dt."""
        layout = DiffWheelLayout(wheel_radius=self.R, track=self.TRACK, encoder_cpr=0)
        manual_theta = 0.0
        for _ in range(50):
            layout.step(self.VEL_STRAIGHT, DT)
            ws = layout.get_wheel_states()
            manual_theta += ws["left"].omega_actual * DT

        enc = layout.get_encoder_readings()
        assert enc["left"]["theta_enc"] == pytest.approx(manual_theta, rel=1e-9)

    def test_fk_trajectory_matches_gt_straight(self):
        """FK-reconstructed trajectory matches ground truth during straight phase."""
        data = self._simulate()
        gt = data["gt"][N_STRAIGHT - 1]
        fk = data["fk"][N_STRAIGHT - 1]
        # Negligible error (motors ideal at this dt)
        assert abs(float(fk[0, 0]) - float(gt[0, 0])) < 0.001, "x mismatch"
        assert abs(float(fk[1, 0]) - float(gt[1, 0])) < 0.001, "y mismatch"

    def test_fk_trajectory_reasonable_during_turn(self):
        """FK trajectory stays within 5 cm of ground truth at end of turn phase."""
        data = self._simulate()
        gt = data["gt"][-1]
        fk = data["fk"][-1]
        err = math.sqrt(
            (float(gt[0, 0]) - float(fk[0, 0])) ** 2
            + (float(gt[1, 0]) - float(fk[1, 0])) ** 2
        )
        assert err < 0.05, f"Trajectory error {err:.4f} m > 5 cm during turn phase"

    def test_encoder_cpr_ticks(self):
        """1 full revolution ≙ exactly cpr ticks."""
        cpr = 100
        layout = DiffWheelLayout(wheel_radius=1.0, track=1.0, encoder_cpr=cpr)
        # Force omega_actual = 2π → one rev per second
        ws = layout.get_wheel_states()
        ws["left"].omega_actual = 2 * math.pi
        ws["right"].omega_actual = 2 * math.pi
        # Manually call encoder via layout step with artificial omega
        # Use a single step at dt=1s to get exactly one revolution
        layout._encoders["left"].step(ws["left"], 1.0)
        assert ws["left"].ticks == cpr


# ---------------------------------------------------------------------------
# Mecanum Drive
# ---------------------------------------------------------------------------


class TestMecanumCorrectness:
    """Mecanum: vx=0.3 straight (within motor limit) then vx=0.2, vy=0.2, omega=0.3.

    agv_hub_motor omega_max=8 rad/s → vx_max=8*0.05=0.4 m/s; use 0.3 to avoid saturation.
    """

    VEL_STRAIGHT = _make_vel(0.3, 0.0, 0.0)
    VEL_TURN = _make_vel(0.2, 0.2, 0.3)
    R, HL, HW = 0.05, 0.15, 0.15

    def _simulate(self):
        layout = MecanumWheelLayout(
            wheel_radius=self.R,
            half_length=self.HL,
            half_width=self.HW,
            encoder_cpr=512,
        )
        gt_state = np.zeros((3, 1))
        fk_state = np.zeros((3, 1))
        data = {
            "gt": [],
            "fk": [],
            "omega_act": {n: [] for n in ["FL", "FR", "RL", "RR"]},
        }

        for i in range(N_TOTAL):
            vel = self.VEL_STRAIGHT if i < N_STRAIGHT else self.VEL_TURN
            gt_state = omni_angular_kinematics(gt_state, vel, DT)
            layout.step(vel, DT)
            ws = layout.get_wheel_states()

            vx_fk, vy_fk, om_fk = mecanum_fwd_kin(
                ws["FL"].omega_actual,
                ws["FR"].omega_actual,
                ws["RL"].omega_actual,
                ws["RR"].omega_actual,
                self.R,
                self.HL,
                self.HW,
            )
            fk_state = omni_angular_kinematics(
                fk_state, _make_vel(vx_fk, vy_fk, om_fk), DT
            )

            data["gt"].append(gt_state.copy())
            data["fk"].append(fk_state.copy())
            for nm in ["FL", "FR", "RL", "RR"]:
                data["omega_act"][nm].append(ws[nm].omega_actual)

        return data

    def test_pure_vx_symmetric_wheels(self):
        """Pure vx motion: FL==RL and FR==RR in magnitude (mecanum symmetry)."""
        layout = MecanumWheelLayout(
            wheel_radius=self.R, half_length=self.HL, half_width=self.HW
        )
        for _ in range(200):
            layout.step(self.VEL_STRAIGHT, DT)
        ws = layout.get_wheel_states()
        # For pure forward motion, FL=RL and FR=RR (standard mecanum)
        assert abs(ws["FL"].omega_actual - ws["RL"].omega_actual) < 0.01
        assert abs(ws["FR"].omega_actual - ws["RR"].omega_actual) < 0.01

    def test_fk_trajectory_matches_gt_straight(self):
        """FK trajectory matches ground truth during straight phase < 1 mm."""
        data = self._simulate()
        gt = data["gt"][N_STRAIGHT - 1]
        fk = data["fk"][N_STRAIGHT - 1]
        err = math.sqrt(
            (float(gt[0, 0]) - float(fk[0, 0])) ** 2
            + (float(gt[1, 0]) - float(fk[1, 0])) ** 2
        )
        assert err < 0.001, f"Trajectory error {err:.4f} m in straight phase"

    def test_encoder_readings_present(self):
        """All four wheels provide theta_enc in encoder readings."""
        layout = MecanumWheelLayout(
            wheel_radius=self.R, half_length=self.HL, half_width=self.HW, encoder_cpr=0
        )
        for _ in range(10):
            layout.step(self.VEL_STRAIGHT, DT)
        enc = layout.get_encoder_readings()
        for nm in ["FL", "FR", "RL", "RR"]:
            assert nm in enc
            assert enc[nm]["theta_enc"] > 0, f"Wheel {nm} encoder not accumulating"

    def test_omega_correct_sign_lateral(self):
        """Pure lateral (vy>0): FL and RR negative, FR and RL positive (mecanum convention)."""
        layout = MecanumWheelLayout(
            wheel_radius=self.R, half_length=self.HL, half_width=self.HW
        )
        vel = _make_vel(0.0, 0.5, 0.0)
        for _ in range(200):
            layout.step(vel, DT)
        ws = layout.get_wheel_states()
        assert ws["FL"].omega_actual < 0
        assert ws["RR"].omega_actual < 0
        assert ws["FR"].omega_actual > 0
        assert ws["RL"].omega_actual > 0


# ---------------------------------------------------------------------------
# Ackermann Drive
# ---------------------------------------------------------------------------


class TestAckerCorrectness:
    """Ackermann: v=1.0 straight (within motor limit) then v=1.0, psi=0.3.

    agv_hub_motor omega_max=8 rad/s → v_max=8*0.15=1.2 m/s; use 1.0 to avoid saturation.
    """

    VEL_STRAIGHT = _make_vel(1.0, 0.0)
    VEL_TURN = _make_vel(1.0, 0.3)
    R, WB, TRACK = 0.15, 1.0, 0.5

    def _simulate(self):
        layout = AckerWheelLayout(
            wheel_radius=self.R, wheelbase=self.WB, track=self.TRACK, encoder_cpr=512
        )
        gt_state = np.zeros((4, 1))
        data = {
            "gt": [],
            "omega_rl": [],
            "omega_rr": [],
            "delta_fl": [],
            "delta_fr": [],
        }

        for i in range(N_TOTAL):
            vel = self.VEL_STRAIGHT if i < N_STRAIGHT else self.VEL_TURN
            gt_state = ackermann_kinematics(gt_state, vel, DT)
            layout.step(vel, DT)
            ws = layout.get_wheel_states()

            data["gt"].append(gt_state.copy())
            data["omega_rl"].append(ws["RL"].omega_actual)
            data["omega_rr"].append(ws["RR"].omega_actual)
            data["delta_fl"].append(ws["FL"].delta_actual)
            data["delta_fr"].append(ws["FR"].delta_actual)

        return data

    def test_straight_rear_wheels_equal(self):
        """Straight motion: RL and RR have equal omega (symmetric)."""
        data = self._simulate()
        # After 2s straight, steady state
        assert (
            abs(data["omega_rl"][N_STRAIGHT - 1] - data["omega_rr"][N_STRAIGHT - 1])
            < 0.01
        )

    def test_straight_steer_near_zero(self):
        """Straight motion: front wheel steer angles converge to near 0."""
        data = self._simulate()
        # Servo convergence: light_servo settles in ~0.5s, so at N_STRAIGHT-1 (t=2s) it's done
        assert abs(data["delta_fl"][N_STRAIGHT - 1]) < 0.02
        assert abs(data["delta_fr"][N_STRAIGHT - 1]) < 0.02

    def test_turn_steer_nonzero(self):
        """During turn phase, steer angles converge to nonzero commanded value."""
        data = self._simulate()
        # At the end of 5s simulation, steer should have converged
        assert abs(data["delta_fl"][-1]) > 0.15, (
            "Front left steer did not converge during turn"
        )

    def test_drive_speed_tracks_command(self):
        """Drive wheel omega matches v/R command to < 2% (v=1.0 < v_max=1.2)."""
        data = self._simulate()
        cmd_omega = self.VEL_STRAIGHT[0, 0] / self.R
        act = data["omega_rl"][N_STRAIGHT - 1]
        err = abs(act - cmd_omega) / cmd_omega
        assert err < 0.02, f"Drive wheel lag: cmd={cmd_omega:.3f}, actual={act:.3f}"

    def test_motor_saturates_at_omega_max(self):
        """When commanded speed exceeds omega_max, motor clamps to omega_max."""
        layout = AckerWheelLayout(
            wheel_radius=self.R, wheelbase=self.WB, track=self.TRACK
        )
        # v=2.0 → omega_cmd=13.3 > omega_max=8
        vel_fast = _make_vel(2.0, 0.0)
        for _ in range(100):
            layout.step(vel_fast, DT)
        ws = layout.get_wheel_states()
        from irsim.lib.handler.wheel_handler import MOTOR_PRESETS

        omega_max = MOTOR_PRESETS["agv_hub_motor"].omega_max
        assert ws["RL"].omega_actual <= omega_max + 1e-9
        assert ws["RR"].omega_actual <= omega_max + 1e-9

    def test_encoder_accumulates(self):
        """Encoder theta_enc grows monotonically during forward motion."""
        layout = AckerWheelLayout(
            wheel_radius=self.R, wheelbase=self.WB, track=self.TRACK, encoder_cpr=0
        )
        for _ in range(50):
            layout.step(self.VEL_STRAIGHT, DT)
        enc = layout.get_encoder_readings()
        assert enc["RL"]["theta_enc"] > 0
        assert enc["RR"]["theta_enc"] > 0


# ---------------------------------------------------------------------------
# Forklift
# ---------------------------------------------------------------------------


class TestForkiftCorrectness:
    """Forklift: v=0.4 straight (within motor limit) then v=0.4, omega=0.4.

    forklift_drive omega_max=4 rad/s → v_max=4*0.15=0.6 m/s; use 0.4 to avoid saturation.
    """

    VEL_STRAIGHT = _make_vel(0.4, 0.0)
    VEL_TURN = _make_vel(0.4, 0.4)
    R, HWB, TRACK = 0.15, 0.6, 0.5

    def _simulate(self):
        layout = ForkiftWheelLayout(
            wheel_radius=self.R,
            half_wheelbase=self.HWB,
            track=self.TRACK,
            encoder_cpr=512,
        )
        gt_state = np.zeros((3, 1))
        data = {"gt": [], "omega_fl": [], "omega_fr": [], "delta_rc": []}

        for i in range(N_TOTAL):
            vel = self.VEL_STRAIGHT if i < N_STRAIGHT else self.VEL_TURN
            gt_state = differential_kinematics(gt_state, vel, DT)
            layout.step(vel, DT)
            ws = layout.get_wheel_states()

            data["gt"].append(gt_state.copy())
            data["omega_fl"].append(ws["FL"].omega_actual)
            data["omega_fr"].append(ws["FR"].omega_actual)
            data["delta_rc"].append(ws["RC"].delta_actual)

        return data

    def test_straight_rear_steer_zero(self):
        """Straight motion: rear-center steer angle converges to 0."""
        data = self._simulate()
        assert abs(data["delta_rc"][N_STRAIGHT - 1]) < 0.05

    def test_turn_rear_steer_nonzero(self):
        """Turn phase: rear-center steer angle becomes nonzero."""
        data = self._simulate()
        assert abs(data["delta_rc"][-1]) > 0.05, (
            "Rear steer did not respond to turn command"
        )

    def test_front_wheels_equal_straight(self):
        """Straight motion: FL and FR have equal omega."""
        data = self._simulate()
        assert (
            abs(data["omega_fl"][N_STRAIGHT - 1] - data["omega_fr"][N_STRAIGHT - 1])
            < 0.01
        )

    def test_drive_speed_tracks_straight(self):
        """Drive wheel omega ≈ v/R during straight phase (v=0.4 < v_max=0.6)."""
        data = self._simulate()
        cmd_omega = self.VEL_STRAIGHT[0, 0] / self.R
        act = data["omega_fl"][N_STRAIGHT - 1]
        err = abs(act - cmd_omega) / cmd_omega
        assert err < 0.02, f"Drive speed lag: cmd={cmd_omega:.3f}, actual={act:.3f}"

    def test_motor_saturates_at_omega_max(self):
        """When commanded speed exceeds forklift_drive omega_max, motor clamps."""
        layout = ForkiftWheelLayout(
            wheel_radius=self.R, half_wheelbase=self.HWB, track=self.TRACK
        )
        vel_fast = _make_vel(1.5, 0.0)  # omega_cmd=10 >> omega_max=4
        for _ in range(100):
            layout.step(vel_fast, DT)
        ws = layout.get_wheel_states()
        from irsim.lib.handler.wheel_handler import MOTOR_PRESETS

        omega_max = MOTOR_PRESETS["forklift_drive"].omega_max
        assert ws["FL"].omega_actual <= omega_max + 1e-9
        assert ws["FR"].omega_actual <= omega_max + 1e-9


# ---------------------------------------------------------------------------
# Dual Steer
# ---------------------------------------------------------------------------


class TestDualSteerCorrectness:
    """Dual steer: v=1.0 straight then v=0.8, omega=0.6."""

    VEL_STRAIGHT = _make_vel(1.0, 0.0)
    VEL_TURN = _make_vel(0.8, 0.6)
    R, HWB = 0.10, 0.4

    def _simulate(self):
        layout = DualSteerWheelLayout(
            wheel_radius=self.R, half_wheelbase=self.HWB, encoder_cpr=512
        )
        data = {"omega_fc": [], "omega_rc": [], "delta_fc": [], "delta_rc": []}

        for i in range(N_TOTAL):
            vel = self.VEL_STRAIGHT if i < N_STRAIGHT else self.VEL_TURN
            layout.step(vel, DT)
            ws = layout.get_wheel_states()
            data["omega_fc"].append(ws["FC"].omega_actual)
            data["omega_rc"].append(ws["RC"].omega_actual)
            data["delta_fc"].append(ws["FC"].delta_actual)
            data["delta_rc"].append(ws["RC"].delta_actual)

        return data

    def test_has_fc_rc_wheels(self):
        """DualSteerWheelLayout provides FC and RC wheel states."""
        layout = DualSteerWheelLayout()
        assert set(layout.get_wheel_states().keys()) == {"FC", "RC"}

    def test_straight_steer_zero(self):
        """Straight motion: both steer angles converge near 0."""
        data = self._simulate()
        assert abs(data["delta_fc"][N_STRAIGHT - 1]) < 0.05
        assert abs(data["delta_rc"][N_STRAIGHT - 1]) < 0.05

    def test_turn_steer_nonzero(self):
        """Turn phase: both steer angles become nonzero."""
        data = self._simulate()
        # At end of simulation steer should have converged
        assert abs(data["delta_fc"][-1]) > 0.05, "FC steer did not respond"
        assert abs(data["delta_rc"][-1]) > 0.05, "RC steer did not respond"

    def test_equal_drive_straight(self):
        """Straight motion: FC and RC wheels have equal omega."""
        data = self._simulate()
        assert (
            abs(data["omega_fc"][N_STRAIGHT - 1] - data["omega_rc"][N_STRAIGHT - 1])
            < 0.05
        )

    def test_encoder_readings(self):
        """FC and RC encoder theta_enc accumulate during motion."""
        layout = DualSteerWheelLayout(wheel_radius=self.R, half_wheelbase=self.HWB)
        for _ in range(50):
            layout.step(self.VEL_STRAIGHT, DT)
        enc = layout.get_encoder_readings()
        assert enc["FC"]["theta_enc"] > 0
        assert enc["RC"]["theta_enc"] > 0


# ---------------------------------------------------------------------------
# Quad Steer (Swerve)
# ---------------------------------------------------------------------------


class TestQuadSteerCorrectness:
    """Quad steer (swerve drive): vx=0.5 straight then vx=0.3, vy=0.2, omega=0.4."""

    VEL_STRAIGHT = _make_vel(0.5, 0.0, 0.0)
    VEL_COMPLEX = _make_vel(0.3, 0.2, 0.4)
    R, HL, HW = 0.05, 0.15, 0.15

    def _simulate(self):
        layout = QuadSteerWheelLayout(
            wheel_radius=self.R,
            half_length=self.HL,
            half_width=self.HW,
            encoder_cpr=512,
        )
        data = {
            "omega": {n: [] for n in ["FL", "FR", "RL", "RR"]},
            "delta": {n: [] for n in ["FL", "FR", "RL", "RR"]},
        }

        for i in range(N_TOTAL):
            vel = self.VEL_STRAIGHT if i < N_STRAIGHT else self.VEL_COMPLEX
            layout.step(vel, DT)
            ws = layout.get_wheel_states()
            for nm in ["FL", "FR", "RL", "RR"]:
                data["omega"][nm].append(ws[nm].omega_actual)
                data["delta"][nm].append(ws[nm].delta_actual)

        return data

    def test_has_four_wheels(self):
        """QuadSteerWheelLayout provides FL, FR, RL, RR."""
        layout = QuadSteerWheelLayout()
        assert set(layout.get_wheel_states().keys()) == {"FL", "FR", "RL", "RR"}

    def test_forward_steer_near_zero(self):
        """Pure forward motion: all steer angles converge near 0."""
        data = self._simulate()
        for nm in ["FL", "FR", "RL", "RR"]:
            assert abs(data["delta"][nm][N_STRAIGHT - 1]) < 0.15, (
                f"Wheel {nm} steer not converging: {data['delta'][nm][N_STRAIGHT - 1]:.3f}"
            )

    def test_forward_all_drive_positive(self):
        """Pure forward: all four drive wheels positive omega."""
        data = self._simulate()
        for nm in ["FL", "FR", "RL", "RR"]:
            assert data["omega"][nm][N_STRAIGHT - 1] > 0, (
                f"Wheel {nm} not spinning forward"
            )

    def test_complex_steer_responds(self):
        """Complex motion: steer angles move away from 0."""
        data = self._simulate()
        # At least one wheel should have a non-trivial steer angle after turning phase
        max_delta = max(abs(data["delta"][nm][-1]) for nm in ["FL", "FR", "RL", "RR"])
        assert max_delta > 0.1, (
            f"No steer response to complex cmd (max={max_delta:.3f})"
        )

    def test_encoder_all_wheels(self):
        """All four wheels have non-zero encoder readings after forward motion."""
        layout = QuadSteerWheelLayout(
            wheel_radius=self.R, half_length=self.HL, half_width=self.HW
        )
        for _ in range(50):
            layout.step(self.VEL_STRAIGHT, DT)
        enc = layout.get_encoder_readings()
        for nm in ["FL", "FR", "RL", "RR"]:
            assert nm in enc
            assert enc[nm]["theta_enc"] > 0, f"Wheel {nm} encoder not accumulating"

    def test_reset_clears_encoder(self):
        """After reset, all wheel states return to zero."""
        layout = QuadSteerWheelLayout(
            wheel_radius=self.R, half_length=self.HL, half_width=self.HW
        )
        for _ in range(50):
            layout.step(self.VEL_STRAIGHT, DT)
        layout.reset()
        for ws in layout.get_wheel_states().values():
            assert ws.theta_enc == pytest.approx(0.0)
            assert ws.omega_actual == pytest.approx(0.0)
            assert ws.delta_actual == pytest.approx(0.0)
