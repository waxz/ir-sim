"""
demo.py -- IR-SIM + shared-memory bridge demo.

Two-process architecture:
  Terminal 1:  python usage/cpp_bridge_demo/demo.py
  Terminal 2:  ./cpp_bridge/build/controller

The simulator runs at 20 Hz.  The C++ controller reads robot state
from shared memory each step and writes back (linear, angular) velocity
commands.  Commands are applied on the next env.step() call.

No threads inside this process -- the shm reads/writes are wait-free
and happen inline in the step loop.  The C++ process runs concurrently
on a separate core.
"""

import pathlib
import sys
import time

# locate irsim regardless of install path
_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

import irsim  # noqa: E402
from irsim.util.shm_bridge import ShmBridge  # noqa: E402

SCENARIO = pathlib.Path(__file__).parent / "scenario.yaml"
RENDER = True  # set False for pure headless


def main() -> None:
    env = irsim.make(str(SCENARIO), display=RENDER)
    bridge = ShmBridge()
    bridge.open()

    print(f"\n[bridge] shm segment '{bridge._name.decode()}' ready.")
    print("[bridge] start the C++ controller in another terminal:")
    print(
        f"         cd {_root}/cpp_bridge && cmake -B build && "
        "cmake --build build && ./build/controller\n"
    )

    step = 0
    sim_time = 0.0
    cmd_linear = 0.0
    cmd_angular = 0.0
    step_dt = float(env.step_time)
    last_report = time.monotonic()

    try:
        while not env.done():
            t0 = time.monotonic()

            # apply the last command from the C++ controller
            env.step(action=[cmd_linear, cmd_angular])

            robot = env.robot_list[0]
            step += 1
            sim_time += step_dt

            # publish new state to shared memory
            bridge.write_state_from_robot(robot, step, sim_time)

            # read the latest command (non-blocking)
            cmd = bridge.read_cmd()
            if cmd is not None:
                cmd_linear, cmd_angular = cmd

            if RENDER:
                env.render()

            # terminal status every 2 s
            now = time.monotonic()
            if now - last_report >= 2.0:
                st = robot.state
                g = robot.goal
                dist = 0.0
                if g is not None:
                    import math

                    dist = math.sqrt(
                        (float(g[0]) - float(st[0])) ** 2
                        + (float(g[1]) - float(st[1])) ** 2
                    )
                print(
                    f"step={step:5d}  t={sim_time:.1f}s  "
                    f"pos=({float(st[0]):.2f},{float(st[1]):.2f})  "
                    f"dist={dist:.2f}m  "
                    f"cmd=({cmd_linear:.2f},{cmd_angular:.2f})"
                )
                last_report = now

            # wall-clock rate limiter (best-effort, doesn't block)
            elapsed = time.monotonic() - t0
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[bridge] interrupted by user")
    finally:
        bridge.close()
        env.end()
        print("[bridge] done")


if __name__ == "__main__":
    main()
