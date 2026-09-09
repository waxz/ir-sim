# shmbridge

POSIX shared-memory seqlock bridge for Python ↔ C++ robot control.

**shmbridge** lets a simulator (publisher) and one or more controllers
(subscribers) exchange robot state and velocity commands across process
boundaries with sub-microsecond write/read overhead and zero copies in
the critical path.

---

## Contents

- [Design](#design)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Notification modes](#notification-modes)
- [Performance](#performance)
- [Build](#build)
- [Examples](#examples)

---

## Design

### Shared-memory layout

```
offset 0          IrsimHeader       (128 B)
offset 128        IrsimStateSlot[N] (N × 128 B, one per robot)
offset 128+128·N  IrsimCmdSlot[N·C] (N·C × 128 B, C consumers per robot)
```

Each slot is exactly 128 bytes — one cache line on 64-byte systems plus one
padding line — so reads and writes never straddle a cache-line boundary.

### Seqlock protocol

Both state and command slots use a dual-sequence seqlock:

```
Writer                              Reader
------                              ------
seq  = ++counter   (now odd)        s1 = seq
_SB_FENCE_W()                       _SB_FENCE_R()
  ... write payload ...               ... read payload ...
writer_ts_ns = now_ns()             out.write_ns = writer_ts_ns
_SB_FENCE_W()                       _SB_FENCE_R()
seq  = ++counter   (now even)       s2 = seq2
seq2 = counter                      ok = (s1 == s2) && !(s1 & 1)
```

A torn read is detected when `seq ≠ seq2` or when `seq` is odd (writer in
progress). `read_state_spin` retries until consistent; `read_state` returns
`None` / `std::nullopt` immediately on failure.

### System timestamp (`write_ns`)

`writer_ts_ns` is captured **inside** the seqlock window, between the two
`_SB_FENCE_W()` calls, so every reader sees a (state, timestamp) pair that
is always consistent. Setting `heartbeat_every = 0` skips the clock call
entirely for minimum-latency write paths where staleness checking is not
needed.

**Usable latency** = `now_ns() - state.write_ns` — the actual age of the
sensor data when the controller uses it. This is the only latency figure
that matters for control quality.

### Memory fences

On x86 (TSO) the fences compile to compiler barriers only — no `mfence`.
On AArch64 they emit `dmb ishst` / `dmb ish`. Both are lighter than
`std::memory_order_seq_cst`.

---

## Quick start

### Python simulator → Python controller

```python
# Simulator process (publisher)
from shmbridge import ShmPublisher, RobotState

pub = ShmPublisher("/irsim_bridge_v2", n_robots=1, n_consumers=1,
                   heartbeat_every=1)    # write_ns enabled
with pub:                                # open() / close() via context manager
    state = RobotState()
    state.x, state.y, state.step = 1.0, 0.0, 42
    pub.write_state(0, state)            # robot index 0

    cmd = pub.read_best_cmd(0)           # highest-seq cmd across all consumers
    if cmd:
        print(cmd.linear, cmd.angular)
```

```python
# Controller process (subscriber)
from shmbridge import ShmSubscriber
import time

sub = ShmSubscriber("/irsim_bridge_v2")
with sub:                                # attach() / detach()
    state = sub.read_state_spin(0)       # blocks until clean read
    if state:
        age_us = (time.monotonic_ns() - state.write_ns) / 1e3
        print(f"state age: {age_us:.1f} µs")
        sub.write_cmd(0, 0, linear=0.5, angular=0.0)
```

### C++ controller with Python simulator

```cpp
// C++ controller — header-only, no library to link
#include <shmbridge/core.hpp>
#include <iostream>

int main() {
    shmbridge::ShmSubscriber sub("/irsim_bridge_v2");
    sub.attach();

    while (true) {
        auto opt = sub.read_state_spin(0);
        if (!opt) continue;
        const auto& s = *opt;

        uint64_t age_ns = shmbridge::detail::now_ns() - s.write_ns;
        std::cout << "x=" << s.x << " age=" << age_ns / 1000 << " µs\n";

        sub.write_cmd(0, 0, 0.5f, 0.1f);
    }
}
```

### Using `LoopSleeper` for fixed-rate control

```python
from shmbridge import _core

sleeper = _core.LoopSleeper(200.0)   # 200 Hz
while True:
    sleeper.start()
    # ... read state, compute command, write cmd ...
    sleeper.sleep()                   # releases GIL; hybrid nanosleep + spin
```

### Using `TaskScheduler` for multi-task control

```python
from shmbridge import _core

sched = _core.TaskScheduler(loop_hz=200.0)
sched.add_task("state_read", lambda: True, period_ms=5.0)   # 200 Hz
sched.add_task("cmd_write",  lambda: True, period_ms=10.0)  # 100 Hz
sched.run_threaded()
# ... sched.stop() when done ...
print(sched.report())   # markdown table of timing stats
```

---

## API reference

### `ShmPublisher`

```
ShmPublisher(name="/irsim_bridge_v2", n_robots=1, n_consumers=1,
             heartbeat_every=1)
```

| Method | Description |
|---|---|
| `open(mlock=False)` | Create and zero-init the shm segment. |
| `close()` | Unmap and unlink the segment. |
| `write_state(robot_idx, state)` | Seqlock-write robot state (releases GIL). |
| `read_cmd(robot_idx, consumer_idx)` | Non-blocking read of one consumer slot. |
| `read_cmd_blocking(timeout_ms, poll_sleep_ns, robot_idx, consumer_idx)` | Block until a cmd arrives (releases GIL). |
| `read_best_cmd(robot_idx)` | Return the valid cmd with the highest `seq` across all consumers. |
| `is_controller_alive(max_age_ms, robot_idx, consumer_idx)` | Check cmd heartbeat. |

### `ShmSubscriber`

```
ShmSubscriber(name="/irsim_bridge_v2", n_robots=1)
```

| Method | Description |
|---|---|
| `attach(timeout_ms=30000)` | Attach to an existing segment. |
| `detach()` | Unmap the segment. |
| `read_state(robot_idx)` | Single seqlock read; returns `None` on torn read. |
| `read_state_spin(robot_idx, max_retries=64)` | Retry until clean; returns `None` only on persistent failure. |
| `write_cmd(robot_idx, consumer_idx, linear, angular)` | Seqlock-write velocity command. |
| `is_publisher_alive(max_age_ms, robot_idx)` | Check state heartbeat. |

### `RobotState`

| Field | Type | Description |
|---|---|---|
| `x`, `y`, `heading` | `float64` | World-frame pose (m, m, rad) |
| `vx`, `vy`, `omega` | `float32` | World-frame velocity (m/s, rad/s) |
| `goal_x`, `goal_y`, `goal_dist` | `float32` | Current goal position and distance |
| `step` | `uint64` | Simulation step counter |
| `sim_time` | `float64` | Simulated time (s) |
| `reached` | `bool` | Goal reached flag |
| `collision` | `bool` | In-collision flag |
| `write_ns` | `uint64` | `CLOCK_MONOTONIC` nanoseconds captured at write time (0 if `heartbeat_every=0`) |

### `RobotCmd`

| Field | Type | Description |
|---|---|---|
| `linear` | `float32` | Forward velocity (m/s) |
| `angular` | `float32` | Angular velocity (rad/s, CCW+) |
| `seq` | `uint32` | Monotone command counter |

### `LoopSleeper`

```python
sleeper = _core.LoopSleeper(hz=200.0)
sleeper.start()   # mark the top of a loop iteration
sleeper.sleep()   # sleep for the remainder of the period (hybrid nanosleep+spin)
```

Achieves < 1 µs jitter via a Welford-adaptive nanosleep + `_SB_PAUSE()` spin
tail. Releases the GIL during the coarse phase.

### `TaskScheduler`

```python
sched = _core.TaskScheduler(loop_hz=200.0, max_prio=10)
sched.add_task(name, func, period_ms)   # func() -> bool (False = one-shot remove)
sched.run_threaded()
sched.stop()
sched.report()   # -> markdown timing table
```

---

## Notification modes

A subscriber can learn that new data is available in two ways:

### Mode A — shm + external notify (lowest usable latency)

Publisher writes shm, then sends a 1-byte signal over a side channel.
Subscriber blocks on the side channel, then reads shm.

Recommended channel: **POSIX semaphore** (`multiprocessing.Semaphore(0)`).

```python
# publisher side (after pub.write_state)
sem.release()

# subscriber side
sem.acquire()
state = sub.read_state_spin(0)
age_us = (time.monotonic_ns() - state.write_ns) / 1e3
```

Measured usable latency (100 Hz, `n=500`):

| Channel | p50 | p99 | p999 | Sub CPU% |
|---|---|---|---|---|
| POSIX sem | **58 µs** | 128 µs | 166 µs | **0.6%** |
| eventfd SCM_RIGHTS | 67 µs | 175 µs | 276 µs | 0.8% |
| UDS DGRAM | 84 µs | 182 µs | 531 µs | 0.9% |
| mp.Pipe | 84 µs | 191 µs | 813 µs | 1.2% |

Use `eventfd` via SCM_RIGHTS when the `forkserver` or `spawn` start method
is required (semaphore inheritance is unavailable).

### Mode B — shm poll + timestamp check (zero infrastructure)

Controller reads shm at its own fixed cycle rate and checks `write_ns` for
freshness. No side channel required.

```python
state = sub.read_state_spin(0)
if state and state.write_ns > 0:
    age_us = (time.monotonic_ns() - state.write_ns) / 1e3
    if age_us > max_age_us:
        # data is stale — handle appropriately
        pass
```

This introduces average staleness of half the sensor period (5 ms at 100 Hz).
Use Mode A when end-to-end latency matters; use Mode B when the controller
already runs at or above the sensor rate and simplicity is preferred.

---

## Performance

All measurements on Linux x86-64, Python 3.11, `n=10 000` samples.

### Write / read operation cost

| Operation | p50 | p99 | Note |
|---|---|---|---|
| `write_state` ts=off | **333 ns** | 385 ns | bare seqlock |
| `write_state` ts=on | **358 ns** | 848 ns | +25 ns for `clock_gettime` |
| `read_state_spin` | **502 ns** | 1063 ns | sequential, no contention |
| `write_state` (contention) | 547 ns | 1585 ns | concurrent reader in parallel |
| `read_state_spin` (contention) | 649 ns | 2205 ns | concurrent writer in parallel |

The timestamp overhead is ~25 ns (VDSO `clock_gettime`). The p99 jump for
`ts=on` (848 ns) reflects occasional kernel fallback — a rare, benign event
that has no impact at 100–1000 Hz control rates.

Throughput under full contention (two processes): ~610 K writes/s and
~548 K reads/s simultaneously.

### Usable latency by communication mode

Measured at 1000 Hz sensor rate, `n=500` controller cycles:

| Mode | p50 | p99 | p999 |
|---|---|---|---|
| shm + POSIX sem notify | **58 µs** | 128 µs | 166 µs |
| shm + eventfd notify | 67 µs | 175 µs | 276 µs |
| shm + UDS DGRAM notify | 84 µs | 182 µs | 531 µs |
| shm poll (no notify, 1:1 rate) | ~500 µs | — | — |
| shm poll (1:3 rate, sensor 30 Hz) | ~16 600 µs | — | — |

The shm-poll mode's high usable latency is a synchronization issue, not a
transport issue — the controller reads at a random phase within the last
sensor period, so average staleness ≈ half the period.

### Budget in a 100 Hz (10 ms) control cycle

| Item | Cost |
|---|---|
| `write_state` (ts=on) | ~360 ns |
| POSIX sem notify | ~20 ns (kernel futex wake) |
| Subscriber wake + `read_state_spin` | ~580 ns |
| **Total IPC overhead** | **< 1 µs** |
| Remaining for control algorithm | **> 9 999 µs** |

---

## Build

### Python package (with C++ extension)

```bash
pip install -e .
# or, with uv:
uv pip install -e .
```

Requires: `scikit-build-core >= 0.8`, `pybind11 >= 2.12`, a C++17 compiler.
Falls back to the pure-Python ctypes implementation if the C++ build is
skipped.

### C++ header-only usage

Copy `include/shmbridge/` into your project and add the directory to your
include path. No library to link.

```cmake
target_include_directories(my_controller PRIVATE path/to/shmbridge/include)
```

---

## Examples

| File | Description |
|---|---|
| `examples/python_writer.py` | Python publisher writing fake robot state at 100 Hz |
| `examples/cpp_controller/controller.cpp` | C++ subscriber reading state and writing velocity commands |
| `tests/bench_ops.py` | Write / read micro-benchmark (timestamp on/off, contention) |
| `tests/bench_notify.py` | Notification mechanism comparison (sem, pipe, eventfd, UDS) |
| `tests/bench_ipc_ctrl.py` | Controller-cycle usable latency benchmark |
| `tests/bench_ipc_rates.py` | Rate-mismatch scenarios (1:1, 10:1, 1:3) |

---

## Wire format

Version 2 is backwards-compatible with version 1:

| Field | v1 | v2 |
|---|---|---|
| `header.ready` (offset 0) | ✓ | ✓ |
| `header.magic` (offset 8) | — | `0x53484D42` |
| `header.schema_version` (offset 12) | — | `2` |
| `header.n_robots` (offset 16) | — | ✓ |
| `header.n_consumers` (offset 17) | — | ✓ |
| `state.writer_ts_ns` (offset 88) | padding | ✓ |
| `cmd.writer_ts_ns` (offset 32) | padding | ✓ |

A v1 C++ controller can read a v2 segment without recompilation: the new
fields occupy bytes that v1 treated as padding.

---

## License

MIT
