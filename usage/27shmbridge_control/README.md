# 27 — shmbridge control

Demonstrates running IR-SIM as the **publisher** (simulator) while an
external **controller** — written in Python or C++ — reads robot state and
writes velocity commands through shared memory via `shmbridge`.

```
┌─────────────────┐  write_state  ┌──────────────────────┐
│  sim.py         │ ─────────────►│  controller_py.py    │
│  (IR-SIM pub)   │               │  or ctrl_cpp         │
│                 │◄──────────────│  (shmbridge sub)     │
└─────────────────┘  write_cmd    └──────────────────────┘
        shared memory: /irsim_shmbridge_demo
```

## Files

| File | Description |
|---|---|
| `world.yaml` | IR-SIM environment (diff-drive robot, two static obstacles) |
| `sim.py` | Python simulator — publishes state, applies controller commands |
| `controller_py.py` | Python P-heading controller — subscriber |
| `controller_cpp.cpp` | C++ P-heading controller — subscriber |
| `CMakeLists.txt` | Builds `ctrl_cpp` from the C++ controller |
| `test_shmbridge_irsim.py` | pytest integration tests (no subprocesses) |

## Quick start

### 1. Install dependencies

```bash
# From the repo root
pip install -e .
pip install -e shmbridge/
```

### 2. Run with the Python controller

Open two terminals:

```bash
# Terminal 1 — simulator (headless)
python usage/27shmbridge_control/sim.py

# Terminal 2 — Python controller
python usage/27shmbridge_control/controller_py.py
```

Add `--render` to `sim.py` to open a matplotlib window.

### 3. Run with the C++ controller

Build the C++ controller first:

```bash
cmake -S usage/27shmbridge_control -B /tmp/ctrl_build
cmake --build /tmp/ctrl_build
```

Then run:

```bash
# Terminal 1 — simulator
python usage/27shmbridge_control/sim.py

# Terminal 2 — C++ controller
/tmp/ctrl_build/ctrl_cpp
```

### 4. Run the pytest integration tests

```bash
pytest usage/27shmbridge_control/test_shmbridge_irsim.py -v
```

## How it works

**Simulator (`sim.py`)**  
After each `env.step()` the simulator calls `pub.write_state_from_robot(robot, …)` to
publish pose, velocity, goal distance and flags via the seqlock segment.  It then calls
`pub.read_best_cmd()` to retrieve the latest controller command and passes
`[linear, angular]` as the action to the next `env.step()`.  Without a controller the
built-in `dash` behaviour drives the robot.

**Controller (Python or C++)**  
The controller attaches to the segment with `ShmSubscriber.attach()`, then loops:
1. `read_state_spin()` — spin-reads the latest seqlock state.
2. Runs a proportional-heading algorithm:  
   - `angular = K_ang * bearing_error` (clamped to ±1.5 rad/s)  
   - `linear = V_max * cos(error)` (slows when heading is off)
3. `write_cmd()` — posts the command into the consumer slot.

The C++ controller runs the same algorithm using `shmbridge/core.hpp` with no
dependency on Python or pybind11 at runtime.
