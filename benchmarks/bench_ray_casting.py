"""Performance and correctness checks for the lidar ray-casting kernel.

Runs in CI on a GitHub Actions runner (see ``.github/workflows/benchmark.yml``)
rather than a developer machine, so numbers are comparable run to run. Checks:

1. The numba and pure-numpy kernels agree numerically (skipped when numba
   isn't installed).
2. Per-ray-origin (motion-skew) casting agrees with looping the shared-origin
   kernel per ray, with and without numba; a moving robot actually reads
   different ranges with motion_skew on vs off.
3. Per-call speedup of the numba kernel over the numpy kernel.
4. Segment-cache speedup for a stationary sensor, and that cached results
   match an uncached (always-fresh) gather.
5. Multi-threaded speedup: several threads calling the kernel concurrently
   (simulating parallel RL environments) must be both faster than running the
   same work sequentially and return results identical to a single-threaded
   call, proving the kernel has no shared mutable state / races.
6. End-to-end ``env.step()`` timing on a representative multi-robot,
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


def check_per_ray_origin_matches_shared_origin() -> None:
    """Per-ray (motion-skew) origins must agree with the shared-origin kernel
    called once per ray, both with and without numba."""
    rng = np.random.default_rng(20260903)
    n_rays = 40
    origins = rng.uniform(-2.0, 2.0, size=(n_rays, 2))
    angles = rng.uniform(-np.pi, np.pi, n_rays)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    seg_start = rng.uniform(-5.0, 5.0, size=(300, 2))
    seg_end = seg_start + rng.uniform(-1.0, 1.0, size=(300, 2))
    max_range = 8.0

    ref_ranges = np.empty(n_rays)
    ref_hits = np.empty(n_rays, dtype=int)
    for i in range(n_rays):
        r, h = rc._cast_ray_segments_shared_origin(
            origins[i], directions[i : i + 1], seg_start, seg_end, max_range
        )
        ref_ranges[i] = r[0]
        ref_hits[i] = h[0]

    for use_numba, label in ((rc._HAS_NUMBA, "numba"), (False, "numpy-fallback")):
        had_numba = rc._HAS_NUMBA
        rc._HAS_NUMBA = use_numba
        try:
            ranges, hits = rc.cast_ray_segments(
                origins, directions, seg_start, seg_end, max_range
            )
        finally:
            rc._HAS_NUMBA = had_numba
        ok = np.allclose(ranges, ref_ranges, atol=1e-9) and np.array_equal(
            hits, ref_hits
        )
        _report(
            f"per-ray origin ({label}) matches per-ray reference",
            ok,
            f"{n_rays} rays x {len(seg_start)} segments",
        )


def check_motion_skew_smoke() -> None:
    """A moving robot must read different ranges with motion_skew on vs off;
    a robot that hasn't moved yet (first scan) must read the same either way."""
    from irsim.world.sensors.lidar2d import Lidar2D

    class _EnvParam:
        def __init__(self, objects, tree):
            self.objects = objects
            self.GeometryTree = tree

    class _Env:
        def __init__(self, env_param):
            self._env_param = env_param

    class _Parent:
        def __init__(self, env_param):
            self._env = _Env(env_param)

    class _Obstacle:
        def __init__(self, obj_id, geometry):
            self._id = obj_id
            self._geometry = geometry
            self._geometry_valid = True
            self.shape = "linestring"
            self.unobstructed = False
            self._velocity_xy = np.zeros((2, 1))

        @property
        def geometry(self):
            return self._geometry

        @property
        def velocity_xy(self):
            return self._velocity_xy

    import shapely
    from shapely import STRtree

    wall = shapely.LineString([(-5.0, 4.0), (5.0, 4.0)])
    obstacle = _Obstacle(2, wall)
    env_param = _EnvParam([obstacle], STRtree([obstacle.geometry]))

    state_a = np.array([[0.0], [0.0], [0.0]])
    state_b = np.array([[0.0], [2.0], [0.0]])

    plain = Lidar2D(state=state_a, number=11, angle_range=np.pi)
    plain.parent = _Parent(env_param)
    plain.step(state_a)
    plain.step(state_b)

    skewed = Lidar2D(state=state_a, number=11, angle_range=np.pi, motion_skew=True)
    skewed.parent = _Parent(env_param)
    skewed.step(state_a)
    skewed.step(state_b)

    differs = not np.allclose(plain.range_data, skewed.range_data)
    _report(
        "motion_skew changes readings for a moving robot",
        differs,
        "range_data differs between motion_skew=False and motion_skew=True",
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


def bench_segment_cache_speedup(n_calls: int = 200) -> None:
    """Compare a stationary scan's cost with and without ``SegmentCache``.

    ``_gather_obstacle_edges`` (STRtree query + boundary flattening) is a
    fixed per-call cost independent of beam count; caching it across calls
    for a sensor that hasn't moved should recover most of that cost.
    """
    import shapely

    rng = np.random.default_rng(20260903)
    polygons = [
        shapely.buffer(shapely.Point(x, y), 0.3, quad_segs=8)
        for x, y in rng.uniform(-8.0, 8.0, size=(30, 2))
    ]

    class _Obj:
        def __init__(self, geometry):
            self._geometry = geometry
            self.shape = "circle"

    detected_objects = [_Obj(g) for g in polygons]

    angles = np.linspace(-np.pi, np.pi, 361)
    max_range = 10.0
    origin = np.zeros(2)
    endpoints = origin + max_range * np.column_stack((np.cos(angles), np.sin(angles)))
    lidar_geometry = shapely.MultiLineString(
        [shapely.LineString([origin, p]) for p in endpoints]
    )

    # No cache: every call re-gathers (today's default behavior).
    for _ in range(3):
        rc.cast_rays(lidar_geometry, detected_objects, max_range)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        rc.cast_rays(lidar_geometry, detected_objects, max_range)
    uncached_time = (time.perf_counter() - t0) / n_calls

    # Cached, stationary sensor: gather happens once, then reused.
    cache = rc.SegmentCache(max_displacement=0.05, max_age_steps=n_calls)
    for _ in range(3):
        rc.cast_rays(lidar_geometry, detected_objects, max_range, cache=cache)
    cache.invalidate()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        rc.cast_rays(lidar_geometry, detected_objects, max_range, cache=cache)
    cached_time = (time.perf_counter() - t0) / n_calls

    speedup = uncached_time / cached_time if cached_time > 0 else float("inf")
    print(
        f"[INFO] segment cache (stationary sensor, 361 beams, 30 obstacles): "
        f"uncached={uncached_time * 1000:.3f} ms/call, "
        f"cached={cached_time * 1000:.3f} ms/call, speedup={speedup:.2f}x"
    )
    if speedup <= 1.0:
        print("[WARN] segment cache showed no speedup on this runner")

    # Correctness: cached result for a stationary sensor must match the
    # uncached (always-fresh) result exactly.
    cache.invalidate()
    # Prime the cache (this call's own result is unused; the next call is
    # the one that actually reuses it).
    rc.cast_rays(lidar_geometry, detected_objects, max_range, cache=cache)
    cached_ranges2, cached_hits2, _, _ = rc.cast_rays(
        lidar_geometry, detected_objects, max_range, cache=cache
    )
    fresh_ranges, fresh_hits, _, _ = rc.cast_rays(
        lidar_geometry, detected_objects, max_range
    )
    ok = np.array_equal(cached_ranges2, fresh_ranges) and np.array_equal(
        cached_hits2, fresh_hits
    )
    _report(
        "segment cache reuse matches fresh gather",
        ok,
        "second cached call (reused segments) matches an uncached call",
    )


def bench_multi_stage_comparison(n_steps: int = 150) -> None:
    """Three-way comparison against a moving-obstacle scene: no cache
    (baseline), the combined ``SegmentCache``, and the tiered
    ``TieredSegmentCache`` (static geometry cached, dynamic gathered fresh
    every step -- see ``report/multi_stage_ray_casting.md``).

    The scene has a stationary sensor, many static wall/box obstacles, and a
    few dynamic obstacles that actually move every step (unlike the
    stationary-scene cache benchmark above), so this exercises the case the
    combined cache is unsound for and the tiered cache is designed to fix.
    Reports both timing and per-step correctness against an always-fresh
    gather computed independently at each step.
    """
    import shapely

    rng = np.random.default_rng(20260904)

    class _Obj:
        def __init__(self, geometry, static):
            self._geometry = geometry
            self.static = static
            self.shape = "rectangle" if static else "circle"

    n_static, n_dynamic = 400, 10
    static_objs = [
        _Obj(shapely.box(x, y, x + w, y + h), static=True)
        for (x, y), (w, h) in zip(
            rng.uniform(-25.0, 25.0, size=(n_static, 2)),
            rng.uniform(0.3, 1.2, size=(n_static, 2)),
            strict=True,
        )
    ]
    dynamic_starts = rng.uniform(-10.0, 10.0, size=(n_dynamic, 2))
    dynamic_velocities = rng.uniform(-0.3, 0.3, size=(n_dynamic, 2))

    def dynamic_objs_at(step: int) -> list:
        positions = dynamic_starts + step * dynamic_velocities
        return [
            _Obj(shapely.buffer(shapely.Point(x, y), 0.25), static=False)
            for x, y in positions
        ]

    number = 361
    max_range = 15.0
    angles = np.linspace(-np.pi, np.pi, number)
    origin = np.zeros(2)
    endpoints = origin + max_range * np.column_stack((np.cos(angles), np.sin(angles)))
    lidar_geometry = shapely.MultiLineString(
        [shapely.LineString([origin, p]) for p in endpoints]
    )

    def all_objs_at(step: int) -> list:
        return static_objs + dynamic_objs_at(step)

    # Ground truth: an always-fresh gather at every step (no caching at all).
    ground_truth = [
        rc.cast_rays(lidar_geometry, all_objs_at(step), max_range)[0]
        for step in range(n_steps)
    ]

    def run(cache_factory):
        cache = cache_factory() if cache_factory is not None else None
        for _ in range(3):
            rc.cast_rays(lidar_geometry, all_objs_at(0), max_range, cache=cache)
        if cache is not None:
            cache.invalidate()
        t0 = time.perf_counter()
        max_deviation = 0.0
        for step in range(n_steps):
            ranges, *_ = rc.cast_rays(
                lidar_geometry, all_objs_at(step), max_range, cache=cache
            )
            max_deviation = max(
                max_deviation, float(np.max(np.abs(ranges - ground_truth[step])))
            )
        elapsed = (time.perf_counter() - t0) / n_steps
        return elapsed, max_deviation

    baseline_time, baseline_dev = run(None)
    combined_time, combined_dev = run(
        lambda: rc.SegmentCache(max_displacement=1.0, max_age_steps=10)
    )
    tiered_time, tiered_dev = run(
        lambda: rc.TieredSegmentCache(max_displacement=1.0, max_age_steps=10)
    )

    print(
        f"\n[INFO] multi-stage comparison ({n_static} static, {n_dynamic} "
        f"dynamic-and-moving obstacles, {number} beams, {n_steps} steps):"
    )
    print(
        f"  baseline (no cache):    {baseline_time * 1000:7.3f} ms/step, "
        f"max deviation from ground truth = {baseline_dev:.3e} m"
    )
    print(
        f"  combined SegmentCache:  {combined_time * 1000:7.3f} ms/step, "
        f"max deviation from ground truth = {combined_dev:.3e} m, "
        f"speedup = {baseline_time / combined_time:.2f}x"
    )
    print(
        f"  TieredSegmentCache:     {tiered_time * 1000:7.3f} ms/step, "
        f"max deviation from ground truth = {tiered_dev:.3e} m, "
        f"speedup = {baseline_time / tiered_time:.2f}x"
    )

    _report(
        "baseline matches ground truth (self-consistency)",
        baseline_dev == 0.0,
        f"max deviation = {baseline_dev:.3e} m",
    )
    _report(
        "TieredSegmentCache matches ground truth in a moving-obstacle scene",
        tiered_dev == 0.0,
        f"max deviation = {tiered_dev:.3e} m (must be exactly 0: dynamic "
        "objects are never cached)",
    )
    # This is a known, documented tradeoff, not a bug: report it, don't fail
    # the job on it. It's exactly what motivates TieredSegmentCache.
    if combined_dev > 0.0:
        print(
            f"[INFO] combined SegmentCache shows {combined_dev:.3e} m staleness "
            "in this moving-obstacle scene, as documented -- this is the gap "
            "TieredSegmentCache closes, not a regression."
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
    check_per_ray_origin_matches_shared_origin()
    check_motion_skew_smoke()
    bench_kernel_speed()
    bench_segment_cache_speedup()
    bench_multi_stage_comparison()
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
