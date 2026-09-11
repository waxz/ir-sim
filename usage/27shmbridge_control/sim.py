"""
sim.py — IR-SIM simulator acting as shmbridge publisher.

Runs at 20 Hz, publishes robot state via shared memory, and applies the
velocity command received from an external controller (C++ or Python).
Falls back to a built-in dash behaviour when no controller is connected.

Usage:
    python sim.py              # headless, runs for 500 steps
    python sim.py --render     # with matplotlib window
    python sim.py --steps 200
"""

from __future__ import annotations

import argparse
import os
import sys

# Resolve the shmbridge package from the source tree when not installed.
_sb_src = os.path.join(os.path.dirname(__file__), "..", "..", "shmbridge", "src")
if os.path.isdir(_sb_src):
    sys.path.insert(0, os.path.abspath(_sb_src))

import irsim  # noqa: E402
import shmbridge  # noqa: E402

SHM_NAME = "/irsim_shmbridge_demo"
STEP_TIME = 0.05  # must match world.yaml


def _publish_robot_state(pub, robot, step: int, sim_time: float) -> None:
    """Write robot state — compatible with C++ and Python backends."""
    if shmbridge._BACKEND == "cpp":
        # C++ ShmPublisher.write_state(robot_idx, RobotState)
        import math as _m

        s = shmbridge.RobotState()
        st, vel = robot.state, robot.velocity
        s.x = float(st.item(0))
        s.y = float(st.item(1))
        s.heading = float(st.item(2))
        s.vx = float(vel.item(0)) if vel.size > 0 else 0.0
        s.vy = float(vel.item(1)) if vel.size > 1 else 0.0
        s.omega = float(vel.item(2)) if vel.size > 2 else 0.0
        goal = robot.goal
        if goal is not None:
            s.goal_x = float(goal.item(0))
            s.goal_y = float(goal.item(1))
            dx, dy = s.goal_x - s.x, s.goal_y - s.y
            s.goal_dist = _m.sqrt(dx * dx + dy * dy)
        s.step = step
        s.sim_time = sim_time
        s.reached = bool(robot.arrive)
        s.collision = bool(robot.collision)
        pub.write_state(0, s)
    else:
        # Python ShmBridge.write_state_from_robot
        pub.write_state_from_robot(robot, step=step, sim_time=sim_time)


def main(render: bool = False, steps: int = 500) -> None:
    yaml_path = os.path.join(os.path.dirname(__file__), "world.yaml")
    env = irsim.make(yaml_path, headless=not render)

    pub = shmbridge.ShmPublisher(SHM_NAME, n_robots=1)
    pub.open()
    print(f"[sim] shmbridge opened  name={SHM_NAME}  backend={shmbridge._BACKEND}")

    step = 0
    sim_time = 0.0

    for _ in range(steps):
        robot = env.robot_list[0]

        # Publish current robot state so the controller can read it.
        _publish_robot_state(pub, robot, step=step, sim_time=sim_time)

        # Read the velocity command from the controller (non-blocking).
        cmd = pub.read_best_cmd(robot_idx=0)

        if cmd is not None:
            action = [cmd.linear, cmd.angular]
            env.step(action)
        else:
            # No controller connected yet — let the built-in dash behaviour drive.
            env.step()

        if render:
            env.render(show_goal=True)

        if env.done():
            print(f"[sim] goal reached at step={step}  time={sim_time:.2f}s")
            break

        step += 1
        sim_time += STEP_TIME

    print(f"[sim] finished  steps={step}  time={sim_time:.2f}s")
    pub.close()
    env.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IR-SIM shmbridge demo simulator")
    parser.add_argument("--render", action="store_true", help="Show matplotlib window")
    parser.add_argument("--steps", type=int, default=500, help="Max simulation steps")
    args = parser.parse_args()
    main(render=args.render, steps=args.steps)
