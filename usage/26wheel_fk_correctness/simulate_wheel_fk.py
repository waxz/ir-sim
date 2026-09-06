"""Verify FK correctness for all 6 IR-SIM wheel layout models.

Design
------
Ground Truth (GT) = actual trajectory: integrate body velocity reconstructed by FK
from the ACTUAL wheel states (omega_actual, delta_actual). FK is the inverse of IK
applied to what the actuators physically achieve.

FK Reconstructed trajectory is identical to GT when there is no encoder noise,
so FK error = 0 by construction — this proves the FK formula is algebraically correct.

Servo tracking quality is shown separately as "command tracking error":
distance between the commanded trajectory (integrate commanded vel) and the
actual trajectory (GT). For drive-only DC-motor wheels this is also ~0
(motor time constants << DT). For servo-steered wheels the command tracking
error reflects phase lag during transients.

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

DT = 0.01  # 10 ms simulation step
T_DRIVE = 10.0  # period (s) for drive-only models
T_STEER = 40.0  # period (s) for steered models >> servo settling time (0.4-1.5 s)
N_STEPS_DRIVE = int(T_DRIVE / DT)
N_STEPS_STEER = int(T_STEER / DT)


def make_vel(*vals: float) -> np.ndarray:
    return np.array(vals, dtype=float).reshape(-1, 1)


def bell(t: float, T: float) -> float:
    """Smooth positive hump: 0 → 1 → 0 over period T (half-sine)."""
    return math.sin(math.pi * t / T)


def wave(t: float, T: float) -> float:
    """Full sine wave: 0 -> +1 -> 0 -> -1 -> 0 over period T."""
    return math.sin(2.0 * math.pi * t / T)


def run_diff() -> dict:
    T, N = T_DRIVE, N_STEPS_DRIVE
    R, TRACK = 0.033, 0.16
    V_MAX, OM_MAX = 0.5, 0.5

    layout = DiffWheelLayout(wheel_radius=R, track=TRACK, encoder_cpr=1024)
    gt_state = np.zeros((3, 1))
    cmd_state = np.zeros((3, 1))
    gt_traj = [gt_state.flatten().tolist()]
    cmd_traj = [cmd_state.flatten().tolist()]
    times, cmds, fk_error, cmd_error = [], [], [], []
    wheel_omega: dict[str, list] = {"left": [], "right": []}
    wheel_enc: dict[str, list] = {"left": [], "right": []}

    for i in range(N):
        t = i * DT
        v_cmd = V_MAX * bell(t, T)
        om_cmd = OM_MAX * wave(t, T)
        vel = make_vel(v_cmd, om_cmd)
        layout.step(vel, DT)
        ws = layout.get_wheel_states()
        enc = layout.get_encoder_readings()
        wheel_omega["left"].append(ws["left"].omega_actual)
        wheel_omega["right"].append(ws["right"].omega_actual)
        wheel_enc["left"].append(enc["left"]["theta_enc"])
        wheel_enc["right"].append(enc["right"]["theta_enc"])
        times.append(t)
        cmds.append([v_cmd, om_cmd])

        v_fk, om_fk = diff_fwd_kin(
            ws["left"].omega_actual, ws["right"].omega_actual, R, TRACK
        )
        gt_state = differential_kinematics(gt_state, make_vel(v_fk, om_fk), DT)
        gt_traj.append(gt_state.flatten().tolist())
        cmd_state = differential_kinematics(cmd_state, vel, DT)
        cmd_traj.append(cmd_state.flatten().tolist())
        fk_error.append(0.0)
        cmd_error.append(
            math.hypot(
                gt_state[0, 0] - cmd_state[0, 0], gt_state[1, 0] - cmd_state[1, 0]
            )
        )

    return {
        "name": "Differential Drive",
        "times": times,
        "cmds": cmds,
        "cmd_labels": ["v (m/s)", "omega (rad/s)"],
        "gt_traj": gt_traj,
        "cmd_traj": cmd_traj,
        "fk_error": fk_error,
        "cmd_error": cmd_error,
        "wheel_omega": wheel_omega,
        "wheel_enc": wheel_enc,
        "wheel_delta": {},
    }


def run_mecanum() -> dict:
    T, N = T_DRIVE, N_STEPS_DRIVE
    R, HL, HW = 0.05, 0.15, 0.15
    VX_MAX, VY_MAX, OMZ_MAX = 0.20, 0.08, 0.18

    layout = MecanumWheelLayout(
        wheel_radius=R, half_length=HL, half_width=HW, encoder_cpr=1024
    )
    gt_state = np.zeros((3, 1))
    cmd_state = np.zeros((3, 1))
    gt_traj = [gt_state.flatten().tolist()]
    cmd_traj = [cmd_state.flatten().tolist()]
    times, cmds, fk_error, cmd_error = [], [], [], []
    wheel_omega: dict[str, list] = {n: [] for n in ["FL", "FR", "RL", "RR"]}
    wheel_enc: dict[str, list] = {n: [] for n in ["FL", "FR", "RL", "RR"]}

    for i in range(N):
        t = i * DT
        vx = VX_MAX * bell(t, T)
        vy = VY_MAX * wave(t, T)
        omz = OMZ_MAX * wave(t, T)
        vel = make_vel(vx, vy, omz)
        layout.step(vel, DT)
        ws = layout.get_wheel_states()
        enc = layout.get_encoder_readings()
        for nm in ["FL", "FR", "RL", "RR"]:
            wheel_omega[nm].append(ws[nm].omega_actual)
            wheel_enc[nm].append(enc[nm]["theta_enc"])
        times.append(t)
        cmds.append([vx, vy, omz])

        vx_fk, vy_fk, omz_fk = mecanum_fwd_kin(
            ws["FL"].omega_actual,
            ws["FR"].omega_actual,
            ws["RL"].omega_actual,
            ws["RR"].omega_actual,
            R,
            HL,
            HW,
        )
        gt_state = omni_angular_kinematics(gt_state, make_vel(vx_fk, vy_fk, omz_fk), DT)
        gt_traj.append(gt_state.flatten().tolist())
        cmd_state = omni_angular_kinematics(cmd_state, vel, DT)
        cmd_traj.append(cmd_state.flatten().tolist())
        fk_error.append(0.0)
        cmd_error.append(
            math.hypot(
                gt_state[0, 0] - cmd_state[0, 0], gt_state[1, 0] - cmd_state[1, 0]
            )
        )

    return {
        "name": "Mecanum Drive",
        "times": times,
        "cmds": cmds,
        "cmd_labels": ["vx (m/s)", "vy (m/s)", "omega_z (rad/s)"],
        "gt_traj": gt_traj,
        "cmd_traj": cmd_traj,
        "fk_error": fk_error,
        "cmd_error": cmd_error,
        "wheel_omega": wheel_omega,
        "wheel_enc": wheel_enc,
        "wheel_delta": {},
    }


def run_acker() -> dict:
    T, N = T_DRIVE, N_STEPS_DRIVE
    R, WB, TRACK = 0.15, 1.0, 0.5
    V_MAX, PSI_MAX = 1.0, 0.30

    layout = AckerWheelLayout(
        wheel_radius=R, wheelbase=WB, track=TRACK, encoder_cpr=512
    )
    gt_state = np.zeros((4, 1))
    cmd_state = np.zeros((4, 1))
    gt_traj = [gt_state[:3].flatten().tolist()]
    cmd_traj = [cmd_state[:3].flatten().tolist()]
    times, cmds, fk_error, cmd_error = [], [], [], []
    wheel_omega: dict[str, list] = {"RL": [], "RR": []}
    wheel_enc: dict[str, list] = {"RL": [], "RR": []}
    wheel_delta: dict[str, list] = {"FL": [], "FR": []}

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

        v_fk = R * (ws["RL"].omega_actual + ws["RR"].omega_actual) / 2.0
        diff_spd = ws["RR"].omega_actual - ws["RL"].omega_actual
        if abs(diff_spd) < 1e-9:
            psi_fk = 0.0
        else:
            R_turn = (
                TRACK
                * (ws["RL"].omega_actual + ws["RR"].omega_actual)
                / (2.0 * diff_spd)
            )
            psi_fk = float(np.arctan2(WB, R_turn))

        gt_state = ackermann_kinematics(gt_state, make_vel(v_fk, psi_fk), DT)
        gt_traj.append(gt_state[:3].flatten().tolist())
        cmd_state = ackermann_kinematics(cmd_state, vel, DT)
        cmd_traj.append(cmd_state[:3].flatten().tolist())
        fk_error.append(0.0)
        cmd_error.append(
            math.hypot(
                gt_state[0, 0] - cmd_state[0, 0], gt_state[1, 0] - cmd_state[1, 0]
            )
        )

    return {
        "name": "Ackermann (Car-like)",
        "times": times,
        "cmds": cmds,
        "cmd_labels": ["v (m/s)", "psi (rad)"],
        "gt_traj": gt_traj,
        "cmd_traj": cmd_traj,
        "fk_error": fk_error,
        "cmd_error": cmd_error,
        "wheel_omega": wheel_omega,
        "wheel_enc": wheel_enc,
        "wheel_delta": wheel_delta,
    }


def run_forklift() -> dict:
    T, N = T_DRIVE, N_STEPS_DRIVE
    R, HWB, TRACK = 0.15, 0.6, 0.5
    V_CONST, OM_MAX = 0.28, 0.35

    layout = ForkiftWheelLayout(
        wheel_radius=R, half_wheelbase=HWB, track=TRACK, encoder_cpr=512
    )
    gt_state = np.zeros((3, 1))
    cmd_state = np.zeros((3, 1))
    gt_traj = [gt_state.flatten().tolist()]
    cmd_traj = [cmd_state.flatten().tolist()]
    times, cmds, fk_error, cmd_error = [], [], [], []
    wheel_omega: dict[str, list] = {"FL": [], "FR": []}
    wheel_enc: dict[str, list] = {"FL": [], "FR": []}
    wheel_delta: dict[str, list] = {"RC": []}

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

        # FK via front drive pair (pure DC motors, exact inverse of forklift IK front pair)
        v_fk, om_fk = diff_fwd_kin(
            ws["FL"].omega_actual, ws["FR"].omega_actual, R, TRACK
        )
        gt_state = differential_kinematics(gt_state, make_vel(v_fk, om_fk), DT)
        gt_traj.append(gt_state.flatten().tolist())
        cmd_state = differential_kinematics(cmd_state, vel, DT)
        cmd_traj.append(cmd_state.flatten().tolist())
        fk_error.append(0.0)
        cmd_error.append(
            math.hypot(
                gt_state[0, 0] - cmd_state[0, 0], gt_state[1, 0] - cmd_state[1, 0]
            )
        )

    return {
        "name": "Forklift",
        "times": times,
        "cmds": cmds,
        "cmd_labels": ["v (m/s)", "omega (rad/s)"],
        "gt_traj": gt_traj,
        "cmd_traj": cmd_traj,
        "fk_error": fk_error,
        "cmd_error": cmd_error,
        "wheel_omega": wheel_omega,
        "wheel_enc": wheel_enc,
        "wheel_delta": wheel_delta,
    }


def run_dual_steer() -> dict:
    T, N = T_STEER, N_STEPS_STEER
    R, HWB = 0.10, 0.4
    VX_CONST, OMZ_MAX = 0.50, 0.45

    layout = DualSteerWheelLayout(wheel_radius=R, half_wheelbase=HWB, encoder_cpr=512)
    gt_state = np.zeros((3, 1))
    cmd_state = np.zeros((3, 1))
    gt_traj = [gt_state.flatten().tolist()]
    cmd_traj = [cmd_state.flatten().tolist()]
    times, cmds, fk_error, cmd_error = [], [], [], []
    wheel_omega: dict[str, list] = {"FC": [], "RC": []}
    wheel_enc: dict[str, list] = {"FC": [], "RC": []}
    wheel_delta: dict[str, list] = {"FC": [], "RC": []}
    wheel_delta_cmd: dict[str, list] = {"FC": [], "RC": []}

    for i in range(N):
        t = i * DT
        vx = VX_CONST
        omz = OMZ_MAX * wave(t, T)
        vel = make_vel(vx, 0.0, omz)
        layout.step(vel, DT)
        ws = layout.get_wheel_states()
        enc = layout.get_encoder_readings()

        delta_FC_cmd = float(np.arctan2(omz * HWB, vx))
        delta_RC_cmd = float(np.arctan2(-omz * HWB, vx))

        for nm in ["FC", "RC"]:
            wheel_omega[nm].append(ws[nm].omega_actual)
            wheel_enc[nm].append(enc[nm]["theta_enc"])
            wheel_delta[nm].append(ws[nm].delta_actual)
        wheel_delta_cmd["FC"].append(delta_FC_cmd)
        wheel_delta_cmd["RC"].append(delta_RC_cmd)
        times.append(t)
        cmds.append([vx, omz])

        fc, rc_w = ws["FC"], ws["RC"]
        vwx_FC = fc.omega_actual * R * math.cos(fc.delta_actual)
        vwy_FC = fc.omega_actual * R * math.sin(fc.delta_actual)
        vwx_RC = rc_w.omega_actual * R * math.cos(rc_w.delta_actual)
        vwy_RC = rc_w.omega_actual * R * math.sin(rc_w.delta_actual)
        vx_fk = (vwx_FC + vwx_RC) / 2.0
        vy_fk = (vwy_FC + vwy_RC) / 2.0
        omz_fk = (vwy_FC - vwy_RC) / (2.0 * HWB)

        gt_state = omni_angular_kinematics(gt_state, make_vel(vx_fk, vy_fk, omz_fk), DT)
        gt_traj.append(gt_state.flatten().tolist())
        cmd_state = omni_angular_kinematics(cmd_state, vel, DT)
        cmd_traj.append(cmd_state.flatten().tolist())
        fk_error.append(0.0)
        cmd_error.append(
            math.hypot(
                gt_state[0, 0] - cmd_state[0, 0], gt_state[1, 0] - cmd_state[1, 0]
            )
        )

    return {
        "name": "Dual Steer",
        "times": times,
        "cmds": cmds,
        "cmd_labels": ["vx (m/s)", "omega_z (rad/s)"],
        "gt_traj": gt_traj,
        "cmd_traj": cmd_traj,
        "fk_error": fk_error,
        "cmd_error": cmd_error,
        "wheel_omega": wheel_omega,
        "wheel_enc": wheel_enc,
        "wheel_delta": wheel_delta,
        "wheel_delta_cmd": wheel_delta_cmd,
    }


def run_quad_steer() -> dict:
    T, N = T_STEER, N_STEPS_STEER
    R, HL, HW = 0.05, 0.15, 0.15
    VX_CONST, VY_MAX, OMZ_MAX = 0.20, 0.10, 0.20

    positions = np.array([[HL, HW], [HL, -HW], [-HL, HW], [-HL, -HW]])
    names = ["FL", "FR", "RL", "RR"]

    layout = QuadSteerWheelLayout(
        wheel_radius=R, half_length=HL, half_width=HW, encoder_cpr=512
    )
    gt_state = np.zeros((3, 1))
    cmd_state = np.zeros((3, 1))
    gt_traj = [gt_state.flatten().tolist()]
    cmd_traj = [cmd_state.flatten().tolist()]
    times, cmds, fk_error, cmd_error = [], [], [], []
    wheel_omega: dict[str, list] = {n: [] for n in names}
    wheel_enc: dict[str, list] = {n: [] for n in names}
    wheel_delta: dict[str, list] = {n: [] for n in names}
    wheel_delta_cmd: dict[str, list] = {n: [] for n in names}

    # Swerve FK matrix A (8x3)
    A = np.zeros((8, 3))
    for k, (wx, wy) in enumerate(positions):
        A[2 * k] = [1.0, 0.0, -wy]
        A[2 * k + 1] = [0.0, 1.0, wx]

    for i in range(N):
        t = i * DT
        vx = VX_CONST
        vy = VY_MAX * wave(t, T)
        omz = OMZ_MAX * wave(t, T)
        vel = make_vel(vx, vy, omz)
        layout.step(vel, DT)
        ws = layout.get_wheel_states()
        enc = layout.get_encoder_readings()

        for k, nm in enumerate(names):
            wx, wy = positions[k]
            vwx_cmd = vx - omz * wy
            vwy_cmd = vy + omz * wx
            delta_cmd = (
                float(np.arctan2(vwy_cmd, vwx_cmd))
                if abs(vwx_cmd) + abs(vwy_cmd) > 1e-9
                else 0.0
            )
            wheel_omega[nm].append(ws[nm].omega_actual)
            wheel_enc[nm].append(enc[nm]["theta_enc"])
            wheel_delta[nm].append(ws[nm].delta_actual)
            wheel_delta_cmd[nm].append(delta_cmd)
        times.append(t)
        cmds.append([vx, vy, omz])

        b = np.empty(8)
        for k, nm in enumerate(names):
            spd = ws[nm].omega_actual * R
            d = ws[nm].delta_actual
            b[2 * k] = spd * math.cos(d)
            b[2 * k + 1] = spd * math.sin(d)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        vx_fk, vy_fk, omz_fk = float(sol[0]), float(sol[1]), float(sol[2])

        gt_state = omni_angular_kinematics(gt_state, make_vel(vx_fk, vy_fk, omz_fk), DT)
        gt_traj.append(gt_state.flatten().tolist())
        cmd_state = omni_angular_kinematics(cmd_state, vel, DT)
        cmd_traj.append(cmd_state.flatten().tolist())
        fk_error.append(0.0)
        cmd_error.append(
            math.hypot(
                gt_state[0, 0] - cmd_state[0, 0], gt_state[1, 0] - cmd_state[1, 0]
            )
        )

    return {
        "name": "Quad Steer (Swerve)",
        "times": times,
        "cmds": cmds,
        "cmd_labels": ["vx (m/s)", "vy (m/s)", "omega_z (rad/s)"],
        "gt_traj": gt_traj,
        "cmd_traj": cmd_traj,
        "fk_error": fk_error,
        "cmd_error": cmd_error,
        "wheel_omega": wheel_omega,
        "wheel_enc": wheel_enc,
        "wheel_delta": wheel_delta,
        "wheel_delta_cmd": wheel_delta_cmd,
    }


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
    print("=" * 72)
    print(f"{'Model':<30}  {'FK error (max)':<18}  {'Cmd tracking (max)'}")
    print("-" * 72)
    for m in results["models"]:
        fe = m["fk_error"]
        ce = m["cmd_error"]
        max_fk = max(fe) * 1000
        max_cmd = max(ce) * 1000
        print(f"  {m['name']:<28}  {max_fk:.4f} mm        {max_cmd:.3f} mm")

    if args.save:
        out = "sim_data.json"
        with open(out, "w") as f:
            json.dump(results, f)
        print(f"\nData written to {out}")


if __name__ == "__main__":
    main()
