"""
bench_shm_bridge.py -- micro-benchmark for ShmBridge hot paths.

Usage:
    python tests/bench_shm_bridge.py

Measures per-call latency (ns) and throughput (ops/s) for:
  - write_state()
  - read_cmd()
  - write_state_from_robot() (includes numpy extraction)
  - combined loop: write_state + read_cmd (simulated step cycle)

Results are printed in a tabular summary and also saved to
bench_results.json for the report generator.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from irsim.util.shm_bridge import ShmBridge  # noqa: E402

N = 200_000  # iterations per measurement
WARMUP = 2_000  # warmup iterations (not timed)


# ── mock robot ────────────────────────────────────────────────────────────────


class _Robot:
    def __init__(self) -> None:
        self.state = np.array([[2.0], [3.0], [0.5]])
        self.velocity = np.array([[0.3], [0.0], [0.1]])
        self.goal = np.array([[8.5], [8.5]])
        self.arrive = False
        self.collision = False


# ── timing helper ─────────────────────────────────────────────────────────────


def _measure(fn, n: int = N, warmup: int = WARMUP) -> dict:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn()
    t1 = time.perf_counter_ns()
    elapsed_ns = t1 - t0
    per_call_ns = elapsed_ns / n
    ops_per_s = n / (elapsed_ns * 1e-9)
    return {
        "total_ms": elapsed_ns / 1e6,
        "per_call_ns": per_call_ns,
        "ops_per_s": ops_per_s,
        "n": n,
    }


# ── bench functions ───────────────────────────────────────────────────────────


def run_benchmarks(bridge_cls=ShmBridge, name: str = "baseline") -> dict:
    shm_name = f"/irsim_bench_{os.getpid()}"
    bridge = bridge_cls(shm_name=shm_name)
    bridge.open()

    robot = _Robot()

    # inject a valid command so read_cmd returns non-None
    bridge._cmd_slot.seq = 2
    bridge._cmd_slot.cmd.linear = 0.5
    bridge._cmd_slot.cmd.angular = 0.1
    bridge._cmd_slot.cmd.valid = 1
    bridge._cmd_slot.seq2 = 2

    def _write():
        bridge.write_state(
            2.0,
            3.0,
            0.5,
            0.3,
            0.0,
            0.1,
            8.5,
            8.5,
            8.48,
            1,
            0.05,
        )

    def _read():
        bridge.read_cmd()

    def _write_from_robot():
        bridge.write_state_from_robot(robot, step=1, sim_time=0.05)

    def _combined():
        bridge.write_state(2.0, 3.0, 0.5, 0.3, 0.0, 0.1, 8.5, 8.5, 8.48, 1, 0.05)
        bridge.read_cmd()

    results = {
        "name": name,
        "write_state": _measure(_write),
        "read_cmd": _measure(_read),
        "write_state_from_robot": _measure(_write_from_robot),
        "combined_step": _measure(_combined),
    }

    bridge.close()
    return results


# ── main ──────────────────────────────────────────────────────────────────────


def print_results(r: dict) -> None:
    print(f"\n{'=' * 64}")
    print(f"  ShmBridge benchmark  [{r['name']}]  N={N:,}")
    print(f"{'=' * 64}")
    print(f"  {'Operation':<30} {'ns/call':>10}  {'Mops/s':>8}")
    print(f"  {'-' * 50}")
    for key in ("write_state", "read_cmd", "write_state_from_robot", "combined_step"):
        v = r[key]
        print(f"  {key:<30} {v['per_call_ns']:>10.1f}  {v['ops_per_s'] / 1e6:>8.2f}")
    print(f"{'=' * 64}\n")


def main() -> None:
    print(f"Python {sys.version}")
    print(f"Iterations: {N:,}  Warmup: {WARMUP:,}")

    baseline = run_benchmarks(name="baseline")
    print_results(baseline)

    # Save results
    out = Path(__file__).parent / "bench_results.json"
    with out.open("w") as f:
        json.dump({"baseline": baseline}, f, indent=2)
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
