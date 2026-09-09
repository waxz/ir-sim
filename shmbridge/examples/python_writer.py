"""
python_writer.py - example: drive a fake robot state into shmbridge and read
back a cmd, without needing ir-sim installed.

Run:
    pip install -e ../../   # install shmbridge package
    python python_writer.py
"""

import time

from shmbridge import RobotState, ShmBridge

try:
    from shmbridge._core import LoopSleeper  # precise 100 Hz timing
    _HAS_LOOP_SLEEPER = True
except ImportError:
    _HAS_LOOP_SLEEPER = False


def main():
    bridge = ShmBridge()
    bridge.open()
    print("Segment open. Waiting for C++ controller to attach …")

    if _HAS_LOOP_SLEEPER:
        sleeper = LoopSleeper(100.0)
        print("Using LoopSleeper for precise 100 Hz timing.")
    else:
        print("C++ extension not available; falling back to time.sleep.")

    step = 0
    try:
        while True:
            if _HAS_LOOP_SLEEPER:
                sleeper.start()

            t = step * 0.01  # 100 Hz sim
            state = RobotState(
                x=0.1 * step,
                y=0.0,
                heading=0.0,
                goal_x=5.0,
                goal_y=0.0,
                goal_dist=max(0.0, 5.0 - 0.1 * step),
                step=step,
                sim_time=t,
            )
            bridge.write_state_obj(state, step=step, sim_time=t)

            cmd = bridge.read_cmd()
            if cmd is not None:
                print(
                    f"step {step:5d}  cmd linear={cmd.linear:.3f}  "
                    f"angular={cmd.angular:.3f}  seq={cmd.seq}"
                )
            else:
                print(f"step {step:5d}  no cmd")

            if not bridge.is_controller_alive(max_age_ms=200):
                print("WARNING: controller appears stale (> 200 ms)")

            step += 1
            if _HAS_LOOP_SLEEPER:
                sleeper.sleep()
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        print("Bridge closed.")


if __name__ == "__main__":
    main()
