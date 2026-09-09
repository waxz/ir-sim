"""
python_node_demo.py — in-process publisher + subscriber demo using the
shmbridge v2 Python API (ShmPublisher / ShmSubscriber).

Runs a background thread as the "simulator" (publisher) and a foreground
loop as the "controller" (subscriber), exchanging robot state and velocity
commands through shared memory.

Usage:
    pip install -e ../../     # install shmbridge
    python python_node_demo.py
"""

import threading
import time
import math

from shmbridge import ShmPublisher, ShmSubscriber, RobotState

SHM_NAME   = "/sb_demo"
N_STEPS    = 500
HZ         = 100                   # simulation rate
DT         = 1.0 / HZ


# ── Simulator thread (publisher) ─────────────────────────────────────────────

def run_sim(stop_event: threading.Event) -> None:
    with ShmPublisher(SHM_NAME, n_robots=1, n_consumers=1,
                      heartbeat_every=1) as pub:
        print("[sim]  segment open — publishing at 100 Hz")
        step = 0
        while not stop_event.is_set() and step < N_STEPS:
            t = step * DT
            state = RobotState()
            state.x        = math.cos(2 * math.pi * t / 5.0)  # 0.2 Hz circle
            state.y        = math.sin(2 * math.pi * t / 5.0)
            state.heading  = 2 * math.pi * t / 5.0
            state.step     = step
            state.sim_time = t
            pub.write_state(0, state)

            cmd = pub.read_best_cmd(0)
            if cmd is not None and step % 50 == 0:
                print(f"[sim]  step={step:4d}  cmd linear={cmd.linear:.2f}"
                      f"  angular={cmd.angular:.2f}  seq={cmd.seq}")

            step += 1
            time.sleep(DT)

    print("[sim]  done")


# ── Controller loop (subscriber) ─────────────────────────────────────────────

def run_controller(stop_event: threading.Event) -> None:
    sub = ShmSubscriber(SHM_NAME, n_robots=1)

    # Wait for the publisher to create the segment (up to 5 s)
    print("[ctrl] attaching …")
    sub.attach(timeout_ms=5000)
    print("[ctrl] attached — reading state and sending commands")

    try:
        from shmbridge._core import LoopSleeper
        sleeper = LoopSleeper(HZ)
        use_sleeper = True
    except ImportError:
        use_sleeper = False

    consumer_idx = 0
    seq = 0

    while not stop_event.is_set():
        if use_sleeper:
            sleeper.start()

        state = sub.read_state_spin(0)
        if state is not None:
            age_us = (time.monotonic_ns() - state.write_ns) / 1e3 \
                     if state.write_ns else 0.0

            # Proportional heading controller: drive toward origin
            err_x    = -state.x
            err_y    = -state.y
            dist     = math.hypot(err_x, err_y)
            desired  = math.atan2(err_y, err_x)
            heading_err = desired - state.heading

            # Normalise to (−π, π]
            while heading_err >  math.pi: heading_err -= 2 * math.pi
            while heading_err < -math.pi: heading_err += 2 * math.pi

            linear  = min(0.5 * dist, 1.0)
            angular = 1.5 * heading_err

            sub.write_cmd(0, consumer_idx, linear, angular)
            seq += 1

            if seq % 50 == 0:
                print(f"[ctrl] x={state.x:+.3f}  y={state.y:+.3f}"
                      f"  age={age_us:5.1f} µs"
                      f"  → lin={linear:.2f}  ang={angular:.2f}")

        if use_sleeper:
            sleeper.sleep()
        else:
            time.sleep(DT)

    sub.detach()
    print("[ctrl] stopped")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    stop = threading.Event()

    sim_thread  = threading.Thread(target=run_sim,        args=(stop,), daemon=True)
    ctrl_thread = threading.Thread(target=run_controller, args=(stop,), daemon=True)

    sim_thread.start()
    time.sleep(0.05)          # let publisher create the segment first
    ctrl_thread.start()

    try:
        sim_thread.join()
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down")
    finally:
        stop.set()
        ctrl_thread.join(timeout=2.0)
        print("Done.")


if __name__ == "__main__":
    main()
