"""
bench_bidir.py -- Bidirectional ShmBridge overhead analysis in a realistic
simulator step loop.

Measures:
  1. Bridge call cost inside a step loop (absolute + % of step budget)
  2. Worst-case latency / jitter (99th and 99.9th percentile)
  3. Memory footprint: POSIX shm size + process RSS delta
  4. End-to-end C++ round-trip latency via a proxy reader process
  5. Headroom table: bridge overhead vs. target sim frequencies

Usage:
    python tests/bench_bidir.py

Saves results to tests/bench_bidir_results.json.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from irsim.util.shm_bridge import ShmBridge  # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────────

N_STEP = 50_000  # step-loop iterations
N_WARMUP = 1_000
STEP_FREQS_HZ = [50, 100, 200, 500, 1000]  # typical sim rates


def _rss_kb() -> int:
    """Current process RSS in KB (Linux /proc/self/status)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# ── 1. step-loop overhead ─────────────────────────────────────────────────────


def measure_step_overhead(
    shm_name: str, n: int = N_STEP, warmup: int = N_WARMUP
) -> dict:
    """
    Simulate a step loop: each iteration does write_state + read_cmd.
    Records per-call latency in ns; computes mean, p99, p99.9, max.
    """
    bridge = ShmBridge(shm_name=shm_name)
    bridge.open()

    # inject a valid command so read_cmd returns non-None
    bridge._cmd_slot.seq = 2
    bridge._cmd_slot.cmd.linear = 0.5
    bridge._cmd_slot.cmd.angular = 0.1
    bridge._cmd_slot.cmd.valid = 1
    bridge._cmd_slot.seq2 = 2

    # warmup
    for _ in range(warmup):
        bridge.write_state(1.0, 2.0, 0.5, 0.3, 0.0, 0.1, 8.0, 8.0, 9.0, 0, 0.0)
        bridge.read_cmd()

    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        t0 = time.perf_counter_ns()
        bridge.write_state(1.0, 2.0, 0.5, 0.3, 0.0, 0.1, 8.0, 8.0, 9.0, i, i * 0.01)
        bridge.read_cmd()
        samples[i] = time.perf_counter_ns() - t0

    bridge.close()

    return {
        "n": n,
        "mean_ns": float(np.mean(samples)),
        "median_ns": float(np.median(samples)),
        "p99_ns": float(np.percentile(samples, 99)),
        "p999_ns": float(np.percentile(samples, 99.9)),
        "max_ns": float(np.max(samples)),
        "std_ns": float(np.std(samples)),
    }


# ── 2. write_state_from_robot overhead ───────────────────────────────────────


class _MockRobot:
    def __init__(self) -> None:
        self.state = np.array([[2.0], [3.0], [0.5]])
        self.velocity = np.array([[0.3], [0.0]])
        self.goal = np.array([[8.5], [8.5]])
        self.arrive = False
        self.collision = False


def measure_from_robot_overhead(
    shm_name: str, n: int = N_STEP, warmup: int = N_WARMUP
) -> dict:
    """Overhead of write_state_from_robot + read_cmd (realistic sim step)."""
    bridge = ShmBridge(shm_name=shm_name)
    bridge.open()
    bridge._cmd_slot.seq = 2
    bridge._cmd_slot.cmd.linear = 0.5
    bridge._cmd_slot.cmd.angular = 0.1
    bridge._cmd_slot.cmd.valid = 1
    bridge._cmd_slot.seq2 = 2

    robot = _MockRobot()

    for _ in range(warmup):
        bridge.write_state_from_robot(robot, step=0, sim_time=0.0)
        bridge.read_cmd()

    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        t0 = time.perf_counter_ns()
        bridge.write_state_from_robot(robot, step=i, sim_time=i * 0.01)
        bridge.read_cmd()
        samples[i] = time.perf_counter_ns() - t0

    bridge.close()

    return {
        "n": n,
        "mean_ns": float(np.mean(samples)),
        "median_ns": float(np.median(samples)),
        "p99_ns": float(np.percentile(samples, 99)),
        "p999_ns": float(np.percentile(samples, 99.9)),
        "max_ns": float(np.max(samples)),
        "std_ns": float(np.std(samples)),
    }


# ── 3. memory footprint ───────────────────────────────────────────────────────


def measure_memory_footprint(shm_name: str) -> dict:
    """
    Measure RSS before and after opening the bridge to isolate shm overhead.
    Also report the shm segment sizes.
    """
    # RSS before any bridge is opened
    rss_before = _rss_kb()

    bridge = ShmBridge(shm_name=shm_name)
    bridge.open()
    rss_after_open = _rss_kb()

    # Write a few times to force OS to page in any lazy pages
    for i in range(100):
        bridge.write_state(1.0, 2.0, 0.5, 0.3, 0.0, 0.1, 8.0, 8.0, 9.0, i, 0.0)
    rss_after_write = _rss_kb()

    bridge.close()
    rss_after_close = _rss_kb()

    return {
        "rss_before_kb": rss_before,
        "rss_after_open_kb": rss_after_open,
        "rss_after_write_kb": rss_after_write,
        "rss_after_close_kb": rss_after_close,
        "rss_delta_open_kb": rss_after_open - rss_before,
        "rss_delta_write_kb": rss_after_write - rss_before,
        "shm_segment_bytes": 512,  # SHM_SIZE constant
        "shm_used_bytes": 384,  # IrsimBlock size
        "ext_shm_segment_bytes": 1_049_344,  # EXT_SHM_SIZE
    }


# ── 4. step-budget headroom ───────────────────────────────────────────────────


def compute_headroom(bridge_mean_ns: float, bridge_p99_ns: float) -> list[dict]:
    """
    For each target sim frequency, compute bridge overhead as % of step budget
    and remaining headroom for simulation compute.
    """
    rows = []
    for hz in STEP_FREQS_HZ:
        budget_ns = 1e9 / hz
        overhead_mean_pct = bridge_mean_ns / budget_ns * 100.0
        overhead_p99_pct = bridge_p99_ns / budget_ns * 100.0
        rows.append(
            {
                "hz": hz,
                "budget_us": budget_ns / 1e3,
                "bridge_mean_us": bridge_mean_ns / 1e3,
                "bridge_p99_us": bridge_p99_ns / 1e3,
                "overhead_mean_pct": overhead_mean_pct,
                "overhead_p99_pct": overhead_p99_pct,
                "headroom_mean_us": (budget_ns - bridge_mean_ns) / 1e3,
                "headroom_p99_us": (budget_ns - bridge_p99_ns) / 1e3,
                "rt_ok": overhead_p99_pct < 5.0,  # conservative: p99 < 5% of budget
            }
        )
    return rows


# ── 5. C++ round-trip via proxy reader ────────────────────────────────────────


def _proxy_reader(shm_name: str, n: int, ready_q: mp.Queue, done_q: mp.Queue) -> None:
    """
    Proxy for the C++ controller process: attaches to an already-open shm segment,
    spins watching for new steps, immediately writes a command back.
    """
    import mmap as _mmap

    from irsim.util.shm_bridge import _O_RDWR, SHM_SIZE, _IrsimBlock, _libc

    fd = _libc.shm_open(shm_name.encode(), _O_RDWR, 0o666)
    if fd < 0:
        ready_q.put("error")
        return
    mm = _mmap.mmap(fd, SHM_SIZE, _mmap.MAP_SHARED, _mmap.PROT_READ | _mmap.PROT_WRITE)
    os.close(fd)
    blk = _IrsimBlock.from_buffer(mm)
    cmd_slot = blk.cmd
    state_slot = blk.state

    ready_q.put("ready")

    last_step = -1
    count = 0
    t_end = time.monotonic() + 30.0
    cmd_seq = 2  # start with an even value

    while count < n and time.monotonic() < t_end:
        s1 = state_slot.seq
        cur_step = state_slot.state.step
        s2 = state_slot.seq2
        if s1 != s2 or (s1 & 1) or s1 == 0:
            continue
        if cur_step == last_step:
            continue
        last_step = cur_step

        # write command back (seqlock write)
        cmd_seq += 1
        cmd_slot.seq = cmd_seq  # odd — begin write
        cmd_slot.cmd.linear = 0.5
        cmd_slot.cmd.angular = 0.1
        cmd_slot.cmd.valid = 1
        cmd_seq += 1
        cmd_slot.seq = cmd_seq  # even — end write
        cmd_slot.seq2 = cmd_seq
        count += 1

    # release ctypes refs before closing mmap
    blk = None
    cmd_slot = None
    state_slot = None
    mm.close()
    done_q.put(count)


def measure_roundtrip(shm_name: str, n: int = 500) -> dict:
    """
    Open the shm bridge in the main process first, then spawn a proxy reader
    (C++ stand-in). Measures wall time from write_state() to read_cmd()
    returning a fresh command (one-step pipeline latency).
    """
    # Create segment before spawning reader
    bridge = ShmBridge(shm_name=shm_name)
    bridge.open()
    bridge._cmd_slot.cmd.valid = 0
    bridge._cmd_slot.seq = 0
    bridge._cmd_slot.seq2 = 0

    ready_q: mp.Queue = mp.Queue()
    done_q: mp.Queue = mp.Queue()
    rp = mp.Process(target=_proxy_reader, args=(shm_name, n, ready_q, done_q))
    rp.start()

    sig = ready_q.get(timeout=5.0)
    if sig != "ready":
        rp.terminate()
        bridge.close()
        return {"error": "proxy reader failed to attach"}

    samples = []
    prev_cmd_seq2 = bridge._cmd_slot.seq2

    for i in range(n):
        t_write = time.perf_counter_ns()
        bridge.write_state(1.0, 2.0, 0.5, 0.3, 0.0, 0.1, 8.0, 8.0, 9.0, i + 1, i * 0.01)

        # busy-wait until reader posts a new command (seq2 advances)
        deadline = t_write + 2_000_000  # 2 ms timeout per step
        replied = False
        while time.perf_counter_ns() < deadline:
            cur = bridge._cmd_slot.seq2
            if cur != prev_cmd_seq2 and bridge._cmd_slot.cmd.valid:
                replied = True
                prev_cmd_seq2 = cur
                break

        t_reply = time.perf_counter_ns()
        if replied:
            samples.append(float(t_reply - t_write))

        time.sleep(0.001)  # 1 ms inter-step so reader isn't swamped

    bridge.close()
    rp.join(timeout=5)
    total_seen = done_q.get(timeout=3) if not done_q.empty() else -1

    if not samples:
        return {"error": "no valid round-trips measured"}

    arr = np.array(samples)
    return {
        "n_measured": len(samples),
        "n_reader_saw": total_seen,
        "mean_ns": float(np.mean(arr)),
        "median_ns": float(np.median(arr)),
        "p99_ns": float(np.percentile(arr, 99)),
        "max_ns": float(np.max(arr)),
        "min_ns": float(np.min(arr)),
    }


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    mp.set_start_method("fork", force=True)

    pid = os.getpid()
    shm_name = f"/irsim_bidir_bench_{pid}"

    print(f"Python {sys.version.split()[0]}  PID={pid}")
    print(f"SHM name: {shm_name}")
    print()

    results: dict = {}

    # ── 1. step overhead (raw write_state + read_cmd) ─────────────────────────
    print("=== Step-loop overhead (write_state + read_cmd) ===")
    step = measure_step_overhead(shm_name)
    results["step_overhead"] = step
    print(f"  mean:    {step['mean_ns']:.1f} ns")
    print(f"  median:  {step['median_ns']:.1f} ns")
    print(f"  p99:     {step['p99_ns']:.1f} ns")
    print(f"  p99.9:   {step['p999_ns']:.1f} ns")
    print(f"  max:     {step['max_ns']:.1f} ns")
    print(f"  std:     {step['std_ns']:.1f} ns")
    print()

    # ── 2. write_state_from_robot + read_cmd ──────────────────────────────────
    print("=== From-robot overhead (write_state_from_robot + read_cmd) ===")
    robot_step = measure_from_robot_overhead(shm_name)
    results["robot_step_overhead"] = robot_step
    print(f"  mean:    {robot_step['mean_ns']:.1f} ns")
    print(f"  p99:     {robot_step['p99_ns']:.1f} ns")
    print(f"  p99.9:   {robot_step['p999_ns']:.1f} ns")
    print(f"  max:     {robot_step['max_ns']:.1f} ns")
    print()

    # ── 3. memory footprint ───────────────────────────────────────────────────
    print("=== Memory footprint ===")
    mem = measure_memory_footprint(shm_name)
    results["memory"] = mem
    print(f"  RSS before open:     {mem['rss_before_kb']:,} KB")
    print(
        f"  RSS after open:      {mem['rss_after_open_kb']:,} KB  (delta {mem['rss_delta_open_kb']:+d} KB)"
    )
    print(
        f"  RSS after writes:    {mem['rss_after_write_kb']:,} KB  (delta {mem['rss_delta_write_kb']:+d} KB)"
    )
    print(
        f"  SHM segment (base):  {mem['shm_segment_bytes']} B allocated, {mem['shm_used_bytes']} B used"
    )
    print(
        f"  SHM segment (ext):   {mem['ext_shm_segment_bytes']:,} B (~{mem['ext_shm_segment_bytes'] // 1024} KiB)"
    )
    print()

    # ── 4. step-budget headroom ───────────────────────────────────────────────
    print("=== Step-budget headroom analysis ===")
    print(
        f"  {'Hz':>6}  {'budget':>8}  {'bridge µs':>10}  {'bridge%':>8}  {'p99%':>8}  {'RT-OK':>7}"
    )
    print(f"  {'':->6}  {'':->8}  {'':->10}  {'':->8}  {'':->8}  {'':->7}")
    headroom = compute_headroom(step["mean_ns"], step["p99_ns"])
    results["headroom"] = headroom
    for row in headroom:
        ok_str = "YES" if row["rt_ok"] else "NO"
        print(
            f"  {row['hz']:>6}  "
            f"{row['budget_us']:>6.0f} µs  "
            f"{row['bridge_mean_us']:>8.1f} µs  "
            f"{row['overhead_mean_pct']:>7.3f}%  "
            f"{row['overhead_p99_pct']:>7.3f}%  "
            f"{ok_str:>7}"
        )
    print()

    # ── 5. C++ round-trip via proxy reader ────────────────────────────────────
    print("=== C++ round-trip latency (proxy reader, 2000 steps at 1 Hz) ===")
    rtt = measure_roundtrip(shm_name, n=500)
    results["roundtrip"] = rtt
    if "error" in rtt:
        print(f"  ERROR: {rtt['error']}")
    else:
        print(f"  n measured:  {rtt['n_measured']}")
        print(f"  mean:        {rtt['mean_ns'] / 1e3:.1f} µs")
        print(f"  median:      {rtt['median_ns'] / 1e3:.1f} µs")
        print(f"  p99:         {rtt['p99_ns'] / 1e3:.1f} µs")
        print(f"  max:         {rtt['max_ns'] / 1e3:.1f} µs")
        print(f"  min:         {rtt['min_ns'] / 1e3:.1f} µs")
    print()

    # ── save ──────────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "bench_bidir_results.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
