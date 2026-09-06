"""Verify FK correctness for all 6 IR-SIM wheel layout models.

Design
------
Ground Truth (GT) = commanded trajectory: integrate the body velocity that was
commanded at each step, representing the ideal reference motion.

FK Reconstructed trajectory: estimated body velocity derived from quantised
encoder tick differences  (omega_est = delta_ticks * 2pi/CPR / dt), then
propagated through the FK formula and integrated.

Sources of FK error (both present in a real system):
  1. Encoder quantisation  -- omega_est has +-1-tick uncertainty per step.
  2. Hold-constant approximation -- omega is sampled once per dt interval;
     any intra-interval change creates a small prediction error.
  3. DC motor lag -- τ=50 ms (diff), 150 ms (mecanum/acker/swerve), 300 ms (forklift).
  4. Servo phase lag (steer models) -- actual steer angle lags commanded angle.

Command tracking error (separate metric): distance between the commanded
trajectory (GT) and the "true physical" trajectory integrated from omega_actual,
showing combined motor + servo lag.

Dual Steer / Quad Steer simplified mode: all wheels receive the same forward
speed v and steer angle psi (no holonomic swerve), so the robot behaves like
a single-steering-wheel vehicle. This makes servo lag clearly visible.

Usage::

    python simulate_wheel_fk.py            # run all 6 models, print summary
    python simulate_wheel_fk.py --save     # also write sim_data.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

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

DT = 0.01  # 10 ms step — motor control loop & encoder topic rate (100 Hz, ROS-typical)
# Note: raw encoder hardware runs at kHz; motor FOC at 1 kHz; this is the FK update rate.
T_DRIVE = 10.0  # period (s) for drive-only models
T_STEER = 40.0  # period (s) for steered models >> servo settling time (0.4-1.5 s)
N_STEPS_DRIVE = int(T_DRIVE / DT)
N_STEPS_STEER = int(T_STEER / DT)

# 4096 CPR -- Dynamixel XL430-W250 / AMT102 class (12-bit magnetic/optical encoder).
# The 32-bit MCU counter wraps at ±2^31 ticks (~524 k revolutions at 4096 CPR).
RAD_PER_TICK = 2.0 * math.pi / 4096


def make_vel(*vals: float) -> np.ndarray:
    return np.array(vals, dtype=float).reshape(-1, 1)


def bell(t: float, T: float) -> float:
    """Smooth positive hump: 0 to 1 to 0 over period T (half-sine)."""
    return math.sin(math.pi * t / T)


def wave(t: float, T: float) -> float:
    """Full sine wave: 0 -> +1 -> 0 -> -1 -> 0 over period T."""
    return math.sin(2.0 * math.pi * t / T)


def omega_from_ticks(ticks_now: int, ticks_prev: int, rad_per_tick: float) -> float:
    """Estimate omega from integer encoder tick difference over DT."""
    return (ticks_now - ticks_prev) * rad_per_tick / DT


# ---------------------------------------------------------------------------
# Differential Drive
# ---------------------------------------------------------------------------


def run_diff() -> dict:
    T, N = T_DRIVE, N_STEPS_DRIVE
    R, TRACK = 0.033, 0.16
    V_MAX, OM_MAX = 0.5, 0.5

    layout = DiffWheelLayout(
        wheel_radius=R, track=TRACK, motor="small_dc", encoder_cpr=4096
    )
    gt_state = np.zeros((3, 1))  # commanded trajectory
    fk_state = np.zeros((3, 1))  # FK from quantised encoder
    act_state = np.zeros((3, 1))  # true physical trajectory (from omega_actual)
    gt_traj = [gt_state.flatten().tolist()]
    fk_traj = [fk_state.flatten().tolist()]
    times, cmds, fk_error, cmd_error = [], [], [], []
    wheel_omega: dict[str, list] = {"left": [], "right": []}
    wheel_enc: dict[str, list] = {"left": [], "right": []}
    prev_ticks: dict[str, int] = {"left": 0, "right": 0}

    for i in range(N):
        t = i * DT
        v_cmd = V_MAX * bell(t, T)
        om_cmd = OM_MAX * wave(t, T)
        vel = make_vel(v_cmd, om_cmd)

        layout.step(vel, DT)
        ws = layout.get_wheel_states()
        enc = layout.get_encoder_readings()

        for nm in ["left", "right"]:
            wheel_omega[nm].append(ws[nm].omega_actual)
            wheel_enc[nm].append(enc[nm]["theta_enc"])
        times.append(t)
        cmds.append([v_cmd, om_cmd])

        # GT: commanded trajectory
        gt_state = differential_kinematics(gt_state, vel, DT)
        gt_traj.append(gt_state.flatten().tolist())

        # FK from quantised encoder ticks
        om_l = omega_from_ticks(enc["left"]["ticks"], prev_ticks["left"], RAD_PER_TICK)
        om_r = omega_from_ticks(
            enc["right"]["ticks"], prev_ticks["right"], RAD_PER_TICK
        )
        v_fk, omz_fk = diff_fwd_kin(om_l, om_r, R, TRACK)
        fk_state = differential_kinematics(fk_state, make_vel(v_fk, omz_fk), DT)
        fk_traj.append(fk_state.flatten().tolist())

        # True physical trajectory (omega_actual, no quantisation)
        v_act, omz_act = diff_fwd_kin(
            ws["left"].omega_actual, ws["right"].omega_actual, R, TRACK
        )
        act_state = differential_kinematics(act_state, make_vel(v_act, omz_act), DT)

        fk_error.append(
            math.hypot(fk_state[0, 0] - gt_state[0, 0], fk_state[1, 0] - gt_state[1, 0])
        )
        cmd_error.append(
            math.hypot(
                act_state[0, 0] - gt_state[0, 0], act_state[1, 0] - gt_state[1, 0]
            )
        )

        prev_ticks["left"] = enc["left"]["ticks"]
        prev_ticks["right"] = enc["right"]["ticks"]

    return {
        "name": "Differential Drive",
        "times": times,
        "cmds": cmds,
        "cmd_labels": ["v (m/s)", "omega (rad/s)"],
        "gt_traj": gt_traj,
        "enc_traj": fk_traj,
        "fk_error": fk_error,
        "cmd_error": cmd_error,
        "wheel_omega": wheel_omega,
        "wheel_enc": wheel_enc,
        "wheel_delta": {},
    }


# ---------------------------------------------------------------------------
# Mecanum Drive
# ---------------------------------------------------------------------------


def run_mecanum() -> dict:
    T, N = T_DRIVE, N_STEPS_DRIVE
    R, HL, HW = 0.05, 0.15, 0.15
    VX_MAX, VY_MAX, OMZ_MAX = 0.20, 0.08, 0.18

    layout = MecanumWheelLayout(
        wheel_radius=R,
        half_length=HL,
        half_width=HW,
        motor="agv_hub_motor",
        encoder_cpr=4096,
    )
    gt_state = np.zeros((3, 1))
    fk_state = np.zeros((3, 1))
    act_state = np.zeros((3, 1))
    gt_traj = [gt_state.flatten().tolist()]
    fk_traj = [fk_state.flatten().tolist()]
    times, cmds, fk_error, cmd_error = [], [], [], []
    names4 = ["FL", "FR", "RL", "RR"]
    wheel_omega: dict[str, list] = {n: [] for n in names4}
    wheel_enc: dict[str, list] = {n: [] for n in names4}
    prev_ticks: dict[str, int] = dict.fromkeys(names4, 0)

    for i in range(N):
        t = i * DT
        vx = VX_MAX * bell(t, T)
        vy = VY_MAX * wave(t, T)
        omz = OMZ_MAX * wave(t, T)
        vel = make_vel(vx, vy, omz)

        layout.step(vel, DT)
        ws = layout.get_wheel_states()
        enc = layout.get_encoder_readings()

        for nm in names4:
            wheel_omega[nm].append(ws[nm].omega_actual)
            wheel_enc[nm].append(enc[nm]["theta_enc"])
        times.append(t)
        cmds.append([vx, vy, omz])

        gt_state = omni_angular_kinematics(gt_state, vel, DT)
        gt_traj.append(gt_state.flatten().tolist())

        om = {
            nm: omega_from_ticks(enc[nm]["ticks"], prev_ticks[nm], RAD_PER_TICK)
            for nm in names4
        }
        vx_fk, vy_fk, omz_fk = mecanum_fwd_kin(
            om["FL"], om["FR"], om["RL"], om["RR"], R, HL, HW
        )
        fk_state = omni_angular_kinematics(fk_state, make_vel(vx_fk, vy_fk, omz_fk), DT)
        fk_traj.append(fk_state.flatten().tolist())

        vx_a, vy_a, omz_a = mecanum_fwd_kin(
            ws["FL"].omega_actual,
            ws["FR"].omega_actual,
            ws["RL"].omega_actual,
            ws["RR"].omega_actual,
            R,
            HL,
            HW,
        )
        act_state = omni_angular_kinematics(act_state, make_vel(vx_a, vy_a, omz_a), DT)

        fk_error.append(
            math.hypot(fk_state[0, 0] - gt_state[0, 0], fk_state[1, 0] - gt_state[1, 0])
        )
        cmd_error.append(
            math.hypot(
                act_state[0, 0] - gt_state[0, 0], act_state[1, 0] - gt_state[1, 0]
            )
        )

        for nm in names4:
            prev_ticks[nm] = enc[nm]["ticks"]

    return {
        "name": "Mecanum Drive",
        "times": times,
        "cmds": cmds,
        "cmd_labels": ["vx (m/s)", "vy (m/s)", "omega_z (rad/s)"],
        "gt_traj": gt_traj,
        "enc_traj": fk_traj,
        "fk_error": fk_error,
        "cmd_error": cmd_error,
        "wheel_omega": wheel_omega,
        "wheel_enc": wheel_enc,
        "wheel_delta": {},
    }


# ---------------------------------------------------------------------------
# Ackermann (Car-like)
# ---------------------------------------------------------------------------


def run_acker() -> dict:
    T, N = T_DRIVE, N_STEPS_DRIVE
    R, WB, TRACK = 0.15, 1.0, 0.5
    V_MAX, PSI_MAX = 1.0, 0.30

    layout = AckerWheelLayout(
        wheel_radius=R, wheelbase=WB, track=TRACK, motor="agv_hub_motor", encoder_cpr=4096
    )
    gt_state = np.zeros((4, 1))
    fk_state = np.zeros((4, 1))
    act_state = np.zeros((4, 1))
    gt_traj = [gt_state[:3].flatten().tolist()]
    fk_traj = [fk_state[:3].flatten().tolist()]
    times, cmds, fk_error, cmd_error = [], [], [], []
    wheel_omega: dict[str, list] = {"RL": [], "RR": []}
    wheel_enc: dict[str, list] = {"RL": [], "RR": []}
    wheel_delta: dict[str, list] = {"FL": [], "FR": []}
    prev_ticks: dict[str, int] = {"RL": 0, "RR": 0}

    for i in range(N):
        t = i * DT
        v_cmd = V_MAX * bell(t, T)
        psi_cmd = PSI_MAX * wave(t, T)
        vel = make_vel(v_cmd, psi_cmd)

        layout.step(vel, DT)
        ws = layout.get_wheel_states()
        enc = layout.get_encoder_readings()

        wheel_omega["RL"].append(ws["RL"].omega_actual)
        wheel_omega["RR"].append(ws["RR"].omega_actual)
        wheel_enc["RL"].append(enc["RL"]["theta_enc"])
        wheel_enc["RR"].append(enc["RR"]["theta_enc"])
        wheel_delta["FL"].append(ws["FL"].delta_actual)
        wheel_delta["FR"].append(ws["FR"].delta_actual)
        times.append(t)
        cmds.append([v_cmd, psi_cmd])

        gt_state = ackermann_kinematics(gt_state, vel, DT)
        gt_traj.append(gt_state[:3].flatten().tolist())

        # FK from ticks: v from avg rear speed, psi from speed ratio
        om_rl = omega_from_ticks(enc["RL"]["ticks"], prev_ticks["RL"], RAD_PER_TICK)
        om_rr = omega_from_ticks(enc["RR"]["ticks"], prev_ticks["RR"], RAD_PER_TICK)
        v_fk = R * (om_rl + om_rr) / 2.0
        diff_spd = om_rr - om_rl
        if abs(diff_spd) < 1e-9:
            psi_fk = 0.0
        else:
            R_turn = TRACK * (om_rl + om_rr) / (2.0 * diff_spd)
            psi_fk = float(np.arctan2(WB, R_turn))
        fk_state = ackermann_kinematics(fk_state, make_vel(v_fk, psi_fk), DT)
        fk_traj.append(fk_state[:3].flatten().tolist())

        v_act = R * (ws["RL"].omega_actual + ws["RR"].omega_actual) / 2.0
        d_act = ws["RR"].omega_actual - ws["RL"].omega_actual
        psi_act = (
            0.0
            if abs(d_act) < 1e-9
            else float(
                np.arctan2(
                    WB,
                    TRACK
                    * (ws["RL"].omega_actual + ws["RR"].omega_actual)
                    / (2.0 * d_act),
                )
            )
        )
        act_state = ackermann_kinematics(act_state, make_vel(v_act, psi_act), DT)

        fk_error.append(
            math.hypot(fk_state[0, 0] - gt_state[0, 0], fk_state[1, 0] - gt_state[1, 0])
        )
        cmd_error.append(
            math.hypot(
                act_state[0, 0] - gt_state[0, 0], act_state[1, 0] - gt_state[1, 0]
            )
        )

        prev_ticks["RL"] = enc["RL"]["ticks"]
        prev_ticks["RR"] = enc["RR"]["ticks"]

    return {
        "name": "Ackermann (Car-like)",
        "times": times,
        "cmds": cmds,
        "cmd_labels": ["v (m/s)", "psi (rad)"],
        "gt_traj": gt_traj,
        "enc_traj": fk_traj,
        "fk_error": fk_error,
        "cmd_error": cmd_error,
        "wheel_omega": wheel_omega,
        "wheel_enc": wheel_enc,
        "wheel_delta": wheel_delta,
    }


# ---------------------------------------------------------------------------
# Forklift (2 front drive + 1 rear steer+drive)
# ---------------------------------------------------------------------------


def run_forklift() -> dict:
    T, N = T_DRIVE, N_STEPS_DRIVE
    R, HWB, TRACK = 0.15, 0.6, 0.5
    V_CONST, OM_MAX = 0.28, 0.35

    layout = ForkiftWheelLayout(
        wheel_radius=R,
        half_wheelbase=HWB,
        track=TRACK,
        motor="forklift_drive",
        encoder_cpr=4096,
    )
    gt_state = np.zeros((3, 1))
    fk_state = np.zeros((3, 1))
    act_state = np.zeros((3, 1))
    gt_traj = [gt_state.flatten().tolist()]
    fk_traj = [fk_state.flatten().tolist()]
    times, cmds, fk_error, cmd_error = [], [], [], []
    wheel_omega: dict[str, list] = {"FL": [], "FR": []}
    wheel_enc: dict[str, list] = {"FL": [], "FR": []}
    wheel_delta: dict[str, list] = {"RC": []}
    prev_ticks: dict[str, int] = {"FL": 0, "FR": 0}

    for i in range(N):
        t = i * DT
        v_cmd = V_CONST
        om_cmd = OM_MAX * wave(t, T)
        vel = make_vel(v_cmd, om_cmd)

        layout.step(vel, DT)
        ws = layout.get_wheel_states()
        enc = layout.get_encoder_readings()

        wheel_omega["FL"].append(ws["FL"].omega_actual)
        wheel_omega["FR"].append(ws["FR"].omega_actual)
        wheel_enc["FL"].append(enc["FL"]["theta_enc"])
        wheel_enc["FR"].append(enc["FR"]["theta_enc"])
        wheel_delta["RC"].append(ws["RC"].delta_actual)
        times.append(t)
        cmds.append([v_cmd, om_cmd])

        gt_state = differential_kinematics(gt_state, vel, DT)
        gt_traj.append(gt_state.flatten().tolist())

        om_fl = omega_from_ticks(enc["FL"]["ticks"], prev_ticks["FL"], RAD_PER_TICK)
        om_fr = omega_from_ticks(enc["FR"]["ticks"], prev_ticks["FR"], RAD_PER_TICK)
        v_fk, omz_fk = diff_fwd_kin(om_fl, om_fr, R, TRACK)
        fk_state = differential_kinematics(fk_state, make_vel(v_fk, omz_fk), DT)
        fk_traj.append(fk_state.flatten().tolist())

        v_act, omz_act = diff_fwd_kin(
            ws["FL"].omega_actual, ws["FR"].omega_actual, R, TRACK
        )
        act_state = differential_kinematics(act_state, make_vel(v_act, omz_act), DT)

        fk_error.append(
            math.hypot(fk_state[0, 0] - gt_state[0, 0], fk_state[1, 0] - gt_state[1, 0])
        )
        cmd_error.append(
            math.hypot(
                act_state[0, 0] - gt_state[0, 0], act_state[1, 0] - gt_state[1, 0]
            )
        )

        prev_ticks["FL"] = enc["FL"]["ticks"]
        prev_ticks["FR"] = enc["FR"]["ticks"]

    return {
        "name": "Forklift",
        "times": times,
        "cmds": cmds,
        "cmd_labels": ["v (m/s)", "omega (rad/s)"],
        "gt_traj": gt_traj,
        "enc_traj": fk_traj,
        "fk_error": fk_error,
        "cmd_error": cmd_error,
        "wheel_omega": wheel_omega,
        "wheel_enc": wheel_enc,
        "wheel_delta": wheel_delta,
    }


# ---------------------------------------------------------------------------
# Dual Steer (2-wheel swerve: FC at +HWB, RC at -HWB)
# ---------------------------------------------------------------------------


def run_dual_steer() -> dict:
    """Simplified: both FC and RC wheels share the same delta=psi and omega=v/R.

    Sending [vx=v*cos(psi), vy=v*sin(psi), omz=0] to the layout makes both
    wheels receive identical commands via swerve IK -- equivalent to a single
    steering wheel pointing in direction psi.  Servo lag is clearly visible.
    """
    T, N = T_STEER, N_STEPS_STEER
    R, HWB = 0.10, 0.4
    V_MAX, PSI_MAX = 0.50, 0.60

    layout = DualSteerWheelLayout(
        wheel_radius=R, half_wheelbase=HWB, motor="agv_hub_motor", encoder_cpr=4096
    )
    gt_state = np.zeros((3, 1))
    fk_state = np.zeros((3, 1))
    act_state = np.zeros((3, 1))
    gt_traj = [gt_state.flatten().tolist()]
    fk_traj = [fk_state.flatten().tolist()]
    times, cmds, fk_error, cmd_error = [], [], [], []
    wheel_omega: dict[str, list] = {"FC": [], "RC": []}
    wheel_enc: dict[str, list] = {"FC": [], "RC": []}
    wheel_delta: dict[str, list] = {"FC": [], "RC": []}
    wheel_delta_cmd: dict[str, list] = {"FC": [], "RC": []}
    prev_ticks: dict[str, int] = {"FC": 0, "RC": 0}

    for i in range(N):
        t = i * DT
        v_cmd = V_MAX * bell(t, T)
        psi_cmd = PSI_MAX * wave(t, T)

        # Both wheels: same delta=psi_cmd, same omega=v_cmd/R
        vx_eff = v_cmd * math.cos(psi_cmd)
        vy_eff = v_cmd * math.sin(psi_cmd)
        vel_eff = make_vel(vx_eff, vy_eff, 0.0)

        layout.step(vel_eff, DT)
        ws = layout.get_wheel_states()
        enc = layout.get_encoder_readings()

        for nm in ["FC", "RC"]:
            wheel_omega[nm].append(ws[nm].omega_actual)
            wheel_enc[nm].append(enc[nm]["theta_enc"])
            wheel_delta[nm].append(ws[nm].delta_actual)
            wheel_delta_cmd[nm].append(psi_cmd)
        times.append(t)
        cmds.append([v_cmd, psi_cmd])

        # GT: commanded pointing motion (v in direction psi_cmd)
        gt_vel = make_vel(vx_eff, vy_eff, 0.0)
        gt_state = omni_angular_kinematics(gt_state, gt_vel, DT)
        gt_traj.append(gt_state.flatten().tolist())

        # FK: avg omega from ticks, steer angle from delta_actual of FC
        om_fc = omega_from_ticks(enc["FC"]["ticks"], prev_ticks["FC"], RAD_PER_TICK)
        om_rc = omega_from_ticks(enc["RC"]["ticks"], prev_ticks["RC"], RAD_PER_TICK)
        v_fk = (om_fc + om_rc) / 2.0 * R
        psi_fk = ws["FC"].delta_actual
        fk_state = omni_angular_kinematics(
            fk_state,
            make_vel(v_fk * math.cos(psi_fk), v_fk * math.sin(psi_fk), 0.0),
            DT,
        )
        fk_traj.append(fk_state.flatten().tolist())

        # Actual: same formula but from omega_actual (no quantisation)
        v_act = (ws["FC"].omega_actual + ws["RC"].omega_actual) / 2.0 * R
        psi_act = ws["FC"].delta_actual
        act_state = omni_angular_kinematics(
            act_state,
            make_vel(v_act * math.cos(psi_act), v_act * math.sin(psi_act), 0.0),
            DT,
        )

        fk_error.append(
            math.hypot(fk_state[0, 0] - gt_state[0, 0], fk_state[1, 0] - gt_state[1, 0])
        )
        cmd_error.append(
            math.hypot(
                act_state[0, 0] - gt_state[0, 0], act_state[1, 0] - gt_state[1, 0]
            )
        )

        prev_ticks["FC"] = enc["FC"]["ticks"]
        prev_ticks["RC"] = enc["RC"]["ticks"]

    return {
        "name": "Dual Steer",
        "times": times,
        "cmds": cmds,
        "cmd_labels": ["v (m/s)", "psi (rad)"],
        "gt_traj": gt_traj,
        "enc_traj": fk_traj,
        "fk_error": fk_error,
        "cmd_error": cmd_error,
        "wheel_omega": wheel_omega,
        "wheel_enc": wheel_enc,
        "wheel_delta": wheel_delta,
        "wheel_delta_cmd": wheel_delta_cmd,
    }


# ---------------------------------------------------------------------------
# Quad Steer / Swerve (4-wheel independent steer+drive)
# ---------------------------------------------------------------------------


def run_quad_steer() -> dict:
    """Simplified: all 4 wheels share the same delta=psi and omega=v/R."""
    T, N = T_STEER, N_STEPS_STEER
    R, HL, HW = 0.05, 0.15, 0.15
    V_MAX, PSI_MAX = 0.50, 0.60

    names = ["FL", "FR", "RL", "RR"]

    layout = QuadSteerWheelLayout(
        wheel_radius=R, half_length=HL, half_width=HW, motor="small_dc", encoder_cpr=4096
    )
    gt_state = np.zeros((3, 1))
    fk_state = np.zeros((3, 1))
    act_state = np.zeros((3, 1))
    gt_traj = [gt_state.flatten().tolist()]
    fk_traj = [fk_state.flatten().tolist()]
    times, cmds, fk_error, cmd_error = [], [], [], []
    wheel_omega: dict[str, list] = {n: [] for n in names}
    wheel_enc: dict[str, list] = {n: [] for n in names}
    wheel_delta: dict[str, list] = {n: [] for n in names}
    wheel_delta_cmd: dict[str, list] = {n: [] for n in names}
    prev_ticks: dict[str, int] = dict.fromkeys(names, 0)

    for i in range(N):
        t = i * DT
        v_cmd = V_MAX * bell(t, T)
        psi_cmd = PSI_MAX * wave(t, T)
        # All wheels: same delta=psi_cmd, same omega=v_cmd/R
        vx_eff = v_cmd * math.cos(psi_cmd)
        vy_eff = v_cmd * math.sin(psi_cmd)
        vel_eff = make_vel(vx_eff, vy_eff, 0.0)
        layout.step(vel_eff, DT)
        ws = layout.get_wheel_states()
        enc = layout.get_encoder_readings()

        for nm in names:
            wheel_omega[nm].append(ws[nm].omega_actual)
            wheel_enc[nm].append(enc[nm]["theta_enc"])
            wheel_delta[nm].append(ws[nm].delta_actual)
            wheel_delta_cmd[nm].append(psi_cmd)
        times.append(t)
        cmds.append([v_cmd, psi_cmd])

        # GT: commanded pointing motion
        gt_state = omni_angular_kinematics(gt_state, vel_eff, DT)
        gt_traj.append(gt_state.flatten().tolist())

        # FK: average omega from ticks across all 4 wheels, steer angle from delta_actual of FL
        om_sum = sum(
            omega_from_ticks(enc[nm]["ticks"], prev_ticks[nm], RAD_PER_TICK)
            for nm in names
        )
        v_fk = om_sum / len(names) * R
        psi_fk = ws["FL"].delta_actual
        fk_state = omni_angular_kinematics(
            fk_state,
            make_vel(v_fk * math.cos(psi_fk), v_fk * math.sin(psi_fk), 0.0),
            DT,
        )
        fk_traj.append(fk_state.flatten().tolist())

        # Actual: same formula but from omega_actual (no quantisation)
        v_act = sum(ws[nm].omega_actual for nm in names) / len(names) * R
        psi_act = ws["FL"].delta_actual
        act_state = omni_angular_kinematics(
            act_state,
            make_vel(v_act * math.cos(psi_act), v_act * math.sin(psi_act), 0.0),
            DT,
        )

        fk_error.append(
            math.hypot(fk_state[0, 0] - gt_state[0, 0], fk_state[1, 0] - gt_state[1, 0])
        )
        cmd_error.append(
            math.hypot(
                act_state[0, 0] - gt_state[0, 0], act_state[1, 0] - gt_state[1, 0]
            )
        )

        for nm in names:
            prev_ticks[nm] = enc[nm]["ticks"]

    return {
        "name": "Quad Steer",
        "times": times,
        "cmds": cmds,
        "cmd_labels": ["v (m/s)", "psi (rad)"],
        "gt_traj": gt_traj,
        "enc_traj": fk_traj,
        "fk_error": fk_error,
        "cmd_error": cmd_error,
        "wheel_omega": wheel_omega,
        "wheel_enc": wheel_enc,
        "wheel_delta": wheel_delta,
        "wheel_delta_cmd": wheel_delta_cmd,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save", action="store_true", help="Write sim_data.json to current directory"
    )
    args = parser.parse_args()

    results = {
        "dt": DT,
        "T_drive": T_DRIVE,
        "T_steer": T_STEER,
        "models": [
            run_diff(),
            run_mecanum(),
            run_acker(),
            run_forklift(),
            run_dual_steer(),
            run_quad_steer(),
        ],
    }

    print("Wheel FK Correctness Results")
    print("=" * 76)
    print(f"{'Model':<30}  {'FK error (max)':<20}  {'Cmd tracking (max)'}")
    print("-" * 76)
    for m in results["models"]:
        fe = m["fk_error"]
        ce = m["cmd_error"]
        max_fk = max(fe) * 1000
        rms_fk = (sum(e**2 for e in fe) / len(fe)) ** 0.5 * 1000
        max_cmd = max(ce) * 1000
        print(
            f"  {m['name']:<28}  max={max_fk:.3f} mm  rms={rms_fk:.3f} mm  |  {max_cmd:.3f} mm"
        )

    if args.save:
        out = "sim_data.json"
        with open(out, "w") as f:
            json.dump(results, f)
        print(f"\nData written to {out}")


if __name__ == "__main__":
    main()
