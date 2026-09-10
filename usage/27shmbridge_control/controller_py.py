"""
controller_py.py — Python proportional-heading controller (shmbridge subscriber).

Reads robot state from the shared-memory segment published by sim.py and
writes back a [linear, angular] velocity command at each cycle.

Algorithm: proportional-heading (P-controller)
  - Compute bearing error to goal.
  - Set angular = K_ang * error (clamped).
  - Set linear  = V_max * max(0, cos(error))  (slow down when turning).

Usage (in a separate terminal while sim.py is running):
    python controller_py.py
"""

from __future__ import annotations

import math
import os
import sys
import time

_sb_src = os.path.join(os.path.dirname(__file__), "..", "..", "shmbridge", "src")
if os.path.isdir(_sb_src):
    sys.path.insert(0, os.path.abspath(_sb_src))

import shmbridge  # noqa: E402

SHM_NAME = "/irsim_shmbridge_demo"
RATE_HZ = 40.0       # controller runs at 40 Hz (faster than sim 20 Hz)
V_MAX = 1.0          # m/s
OMEGA_MAX = 1.5      # rad/s
K_ANG = 2.0          # proportional gain for heading error
GOAL_TOL = 0.3       # stop within 0.3 m of goal


def bearing_error(state: shmbridge.RobotState) -> float:
    """Signed angle from current heading to goal direction."""
    dx = state.goal_x - state.x
    dy = state.goal_y - state.y
    target_heading = math.atan2(dy, dx)
    err = target_heading - state.heading
    # Wrap to [-π, π]
    while err > math.pi:
        err -= 2 * math.pi
    while err < -math.pi:
        err += 2 * math.pi
    return err


def compute_cmd(state: shmbridge.RobotState, seq: int) -> shmbridge.RobotCmd:
    if state.reached or state.goal_dist < GOAL_TOL:
        return shmbridge.RobotCmd(linear=0.0, angular=0.0, seq=seq)

    err = bearing_error(state)
    angular = max(-OMEGA_MAX, min(OMEGA_MAX, K_ANG * err))
    # Reduce forward speed when heading is far off
    linear = V_MAX * max(0.0, math.cos(err))
    return shmbridge.RobotCmd(linear=linear, angular=angular, seq=seq)


def main() -> None:
    sub = shmbridge.ShmSubscriber(SHM_NAME)
    print(f"[ctrl-py] attaching to {SHM_NAME} …")
    sub.attach(timeout_ms=10_000)
    print(f"[ctrl-py] attached  backend={shmbridge._BACKEND}")

    dt = 1.0 / RATE_HZ
    cmd_seq = 0
    last_step = -1

    # Both C++ and Python backends use max_retries (not timeout_ms).
    while True:
        state = sub.read_state_spin(0, 4000)  # up to 4000 retries (~200 ms at 20 Hz)
        if state is None:
            print("[ctrl-py] timeout waiting for state — sim may have exited")
            break

        if state.reached or state.collision:
            status = "reached" if state.reached else "collision"
            print(f"[ctrl-py] {status}  x={state.x:.2f}  y={state.y:.2f}"
                  f"  step={state.step}")
            break

        if state.step != last_step:
            cmd = compute_cmd(state, seq=cmd_seq)
            # write_cmd(robot_idx, consumer_idx, linear, angular) — no seq kwarg
            sub.write_cmd(0, 0, cmd.linear, cmd.angular)
            cmd_seq += 1
            last_step = state.step

            if state.step % 20 == 0:
                print(f"[ctrl-py] step={state.step:4d}"
                      f"  x={state.x:.2f}  y={state.y:.2f}"
                      f"  h={math.degrees(state.heading):.1f}°"
                      f"  dist={state.goal_dist:.2f}"
                      f"  cmd=({cmd.linear:.2f},{cmd.angular:.2f})")

        time.sleep(dt)

    print("[ctrl-py] done")


if __name__ == "__main__":
    main()
