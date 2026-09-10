"""
python_node_demo.py — in-process publisher + subscriber demo.

Runs a publisher thread and a subscriber thread in the same process.  The
subscriber calls attach(timeout_ms=5000) which retries until the publisher
has created the segment — no sleep hack, no fixed ordering required.

For a real multi-process setup see:
    publisher_demo.py   — run in one terminal
    subscriber_demo.py  — run in a second terminal (either order)

Usage:
    pip install -e ../..
    python python_node_demo.py
"""

import math
import threading
import time

from shmbridge import RobotState, ShmPublisher, ShmSubscriber

SHM_NAME = "/sb_demo"
N_STEPS = 500
HZ = 100
DT = 1.0 / HZ


def run_publisher(stop: threading.Event) -> None:
    with ShmPublisher(SHM_NAME, n_robots=1, n_consumers=1, heartbeat_every=1) as pub:
        print("[pub]  segment open — publishing at 100 Hz")
        step = 0
        while not stop.is_set() and step < N_STEPS:
            t = step * DT
            state = RobotState()
            state.x = math.cos(2 * math.pi * t / 5.0)
            state.y = math.sin(2 * math.pi * t / 5.0)
            state.heading = 2 * math.pi * t / 5.0
            state.step = step
            state.sim_time = t
            pub.write_state(0, state)

            cmd = pub.read_best_cmd(0)
            if cmd is not None and step % 50 == 0:
                print(
                    f"[pub]  step={step:4d}  cmd linear={cmd.linear:.2f}"
                    f"  angular={cmd.angular:.2f}  seq={cmd.seq}"
                )
            step += 1
            time.sleep(DT)
    print("[pub]  done")


def run_subscriber(stop: threading.Event) -> None:
    sub = ShmSubscriber(SHM_NAME, n_robots=1)
    # attach() retries for up to 5 s — no sleep hack needed
    print("[sub]  attaching…")
    sub.attach(timeout_ms=5000)
    print("[sub]  attached — reading state and writing commands")

    consumer_idx = 0
    seq = 0
    while not stop.is_set():
        state = sub.read_state_spin(0)
        if state is not None:
            err_x = -state.x
            err_y = -state.y
            dist = math.hypot(err_x, err_y)
            desired = math.atan2(err_y, err_x)
            heading_err = desired - state.heading
            while heading_err > math.pi:
                heading_err -= 2 * math.pi
            while heading_err < -math.pi:
                heading_err += 2 * math.pi

            linear = min(0.5 * dist, 1.0)
            angular = 1.5 * heading_err
            sub.write_cmd(0, consumer_idx, linear, angular)
            seq += 1

            if seq % 50 == 0:
                print(
                    f"[sub]  x={state.x:+.3f}  y={state.y:+.3f}"
                    f"  → lin={linear:.2f}  ang={angular:.2f}"
                )
        time.sleep(DT)

    sub.detach()
    print("[sub]  stopped")


def main() -> None:
    stop = threading.Event()

    # Start subscriber first to demonstrate that ordering doesn't matter
    sub_thread = threading.Thread(target=run_subscriber, args=(stop,), daemon=True)
    pub_thread = threading.Thread(target=run_publisher, args=(stop,), daemon=True)

    sub_thread.start()
    pub_thread.start()

    try:
        pub_thread.join()
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down")
    finally:
        stop.set()
        sub_thread.join(timeout=2.0)
        print("Done.")


if __name__ == "__main__":
    main()
