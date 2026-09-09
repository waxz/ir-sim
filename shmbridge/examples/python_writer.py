"""
python_writer.py – example: drive a fake robot state into shmbridge and read
back a cmd, without needing ir-sim installed.

Run:
    pip install -e ../../   # install shmbridge package
    python python_writer.py
"""

import time
from shmbridge import ShmBridge, RobotState

def main():
    bridge = ShmBridge()
    bridge.open()
    print("Segment open. Waiting for C++ controller to attach …")

    step = 0
    try:
        while True:
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
                print(f"step {step:5d}  cmd linear={cmd.linear:.3f}  "
                      f"angular={cmd.angular:.3f}  seq={cmd.seq}")
            else:
                print(f"step {step:5d}  no cmd")

            if not bridge.is_controller_alive(max_age_ms=200):
                print("WARNING: controller appears stale (> 200 ms)")

            step += 1
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        print("Bridge closed.")

if __name__ == "__main__":
    main()
