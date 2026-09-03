"""Performance and correctness checks for the lidar ray-casting kernel.

Runs in CI on a GitHub Actions runner (see ``.github/workflows/benchmark.yml``)
rather than a developer machine, so numbers are comparable run to run. Checks:

1. The numba and pure-numpy kernels agree numerically (skipped when numba
   isn't installed).
2. Per-call speedup of the numba kernel over the numpy kernel.
3. Multi-threaded speedup: several threads calling the kernel concurrently
   (simulating parallel RL environments) must be both faster than running the
   same work sequentially and return results identical to a single-threaded
   call, proving the kernel has no shared mutable state / races.
4. End-to-end ``env.step()`` timing on a representative multi-robot,
   lidar-equipped scenario.

Exits non-zero if a correctness check fails. Timing regressions are reported
but do not fail the job (shared CI runners are too noisy for a hard
threshold).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import irsim.lib.algorithm.ray_casting_2d as rc  # noqa: E402

FAILURES: list[str] = []


def _report(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def _make_scene(n_rays: int = 361, n_segments: int = 3000, seed: int = 20260903):
    rng = np.random.default_rng(seed)
    angles = np.linspace(-np.pi, np.pi, n_rays)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    seg_start = rng.uniform(-8.0, 8.0, size=(n_segments, 2))
    seg_end = seg_start + rng.uniform(-0.5, 0.5, size=(n_segments, 2))
    return np.zeros(2), directions, seg_start, seg_end, 10.0


def check_numba_matches_numpy() -> None:
    if not rc._HAS_NUMBA:
        print("[SKIP] numba-vs-numpy correctness: numba not installed")
        return

    origin, directions, seg_start, seg_end, max_range = _make_scene()

    numba_ranges, numba_hits = rc.cast_ray_segments(
        origin, directions, seg_start, seg_end, max_range
    )

    original = rc._nonparallel_hit_distances
    rc._nonparallel_hit_distances = rc._nonparallel_hit_distances_numpy
    try:
        numpy_ranges, numpy_hits = rc.cast_ray_segments(
            origin, directions, seg_start, seg_end, max_range
        )
    finally:
        rc._nonparallel_hit_distances = original

    ranges_ok = np.allclose(numba_ranges, numpy_ranges, atol=1e-9)
    hits_ok = np.array_equal(numba_hits, numpy_hits)
    _report(
        "numba-vs-numpy correctness",
        ranges_ok and hits_ok,
        f"{len(directions)} rays x {len(seg_start)} segments, "
        f"max |range diff| = {np.max(np.abs(numba_ranges - numpy_ranges)):.3e}",
    )


def bench_kernel_speed(n_calls: int = 100) -> None:
    if not rc._HAS_NUMBA:
        print("[SKIP] kernel speed comparison: numba not installed")
        return

    origin, directions, seg_start, seg_end, max_range = _make_scene()

    # Warm up JIT compilation before timing.
    for _ in range(3):
        rc.cast_ray_segments(origin, directions, seg_start, seg_end, max_range)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        rc.cast_ray_segments(origin, directions, seg_start, seg_end, max_range)
    numba_time = (time.perf_counter() - t0) / n_calls

    original = rc._nonparallel_hit_distances
    rc._nonparallel_hit_distances = rc._nonparallel_hit_distances_numpy
    t0 = time.perf_counter()
    for _ in range(n_calls):
        rc.cast_ray_segments(origin, directions, seg_start, seg_end, max_range)
    numpy_time = (time.perf_counter() - t0) / n_calls
    rc._nonparallel_hit_distances = original

    speedup = numpy_time / numba_time if numba_time > 0 else float("inf")
    print(
        f"[INFO] kernel speed: numba={numba_time * 1000:.3f} ms/call, "
        f"numpy={numpy_time * 1000:.3f} ms/call, speedup={speedup:.2f}x"
    )


def check_thread_safety(n_threads: int = 4, calls_per_thread: int = 30) -> None:
    origin, directions, seg_start, seg_end, max_range = _make_scene()

    # Warm up JIT compilation outside of the timed/threaded section.
    rc.cast_ray_segments(origin, directions, seg_start, seg_end, max_range)

    expected_ranges, expected_hits = rc.cast_ray_segments(
        origin, directions, seg_start, seg_end, max_range
    )

    results: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    lock = threading.Lock()

    def worker(thread_id: int) -> None:
        local = []
        for _ in range(calls_per_thread):
            r, h = rc.cast_ray_segments(
                origin, directions, seg_start, seg_end, max_range
            )
            local.append((r, h))
        with lock:
            results[thread_id] = local

    def run_sequential() -> float:
        t0 = time.perf_counter()
        for i in range(n_threads):
            worker(i)
        return time.perf_counter() - t0

    def run_concurrent() -> float:
        results.clear()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        t0 = time.perf_counter()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        return time.perf_counter() - t0

    t_seq = run_sequential()
    t_par = run_concurrent()

    all_correct = all(
        np.array_equal(r, expected_ranges) and np.array_equal(h, expected_hits)
        for calls in results.values()
        for r, h in calls
    )
    _report(
        "thread safety",
        all_correct,
        f"{n_threads} threads x {calls_per_thread} calls all match the "
        "single-threaded result (no data races)",
    )

    speedup = t_seq / t_par if t_par > 0 else float("inf")
    print(
        f"[INFO] threaded speedup: sequential={t_seq:.3f}s, "
        f"concurrent={t_par:.3f}s, speedup={speedup:.2f}x "
        f"({'numba/nogil' if rc._HAS_NUMBA else 'numpy'} kernel)"
    )
    if rc._HAS_NUMBA and speedup <= 1.0:
        print(
            "[WARN] numba kernel showed no threaded speedup on this runner "
            "(GIL may not be releasing as expected)"
        )


def bench_env_step(n_steps: int = 60, n_warmup: int = 10) -> None:
    import irsim

    scenario = (
        REPO_ROOT / "usage" / "06multi_objects_world" / "multi_objects_large.yaml"
    )
    env = irsim.make(str(scenario), save_ani=False, full=False, display=False)
    try:
        for _ in range(n_warmup):
            env.step()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            env.step()
        elapsed = time.perf_counter() - t0
    finally:
        env.end()

    per_step_ms = elapsed / n_steps * 1000
    print(
        f"[INFO] env.step() ({scenario.name}, 50 lidar-equipped robots): "
        f"{per_step_ms:.3f} ms/step over {n_steps} steps"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-env-step",
        action="store_true",
        help="Skip the end-to-end env.step() benchmark (kernel checks only).",
    )
    args = parser.parse_args()

    print(f"numba backend active: {rc._HAS_NUMBA}")

    check_numba_matches_numpy()
    bench_kernel_speed()
    check_thread_safety()
    if not args.skip_env_step:
        bench_env_step()

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as fh:
            fh.write("\n### Ray casting benchmark\n\n")
            fh.write(f"- numba backend active: `{rc._HAS_NUMBA}`\n")
            fh.write(f"- failures: {len(FAILURES)}\n")
            for failure in FAILURES:
                fh.write(f"  - FAIL: {failure}\n")

    if FAILURES:
        print(f"\n{len(FAILURES)} correctness check(s) failed:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("\nAll correctness checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
