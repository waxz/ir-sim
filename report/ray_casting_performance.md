# Lidar Ray-Casting Performance Report

**Date:** 2026-09-03
**Scope:** `irsim/lib/algorithm/ray_casting_2d.py`, `irsim/world/sensors/lidar2d.py`
**Measured on:** this session's sandbox (single machine, shared CPU) -- treat absolute
numbers as illustrative, not a guaranteed multiplier on every machine. All
numbers below are reproducible via `benchmarks/bench_ray_casting.py`.

## Summary

Lidar ray casting is the dominant per-step cost in lidar-equipped scenes (profiling
showed it at roughly half of `env.step()` time in a 50-robot scenario, and lidar
scan cost splits roughly 50/50 between the obstacle-boundary *gather* -- a
Shapely/STRtree spatial query, independent of beam count -- and the *numeric*
ray/segment kernel, which scales with beams x segments). Three independent
optimizations were made this session:

1. **A JIT-compiled, GIL-releasing kernel** for the numeric ray/segment math (optional,
   via `numba`).
2. **Per-beam motion skew**, modeling the fact that a real spinning lidar's beams are
   each measured from a slightly different robot pose -- and, as a side effect of how
   it's implemented, a per-ray-origin kernel that keeps O(rays) memory.
3. **A segment gather cache** (`SegmentCache`), reusing the STRtree query and boundary
   flattening across steps for a stationary or slow-moving sensor.

All three are **opt-in** (`numba` optional dependency; `motion_skew` and
`cache_max_displacement` default off/0). A scan built with none of them enabled is
byte-for-byte identical to before this work -- verified by the full existing test
suite (965 passed, 45 skipped, 0 failures) both with and without `numba` installed.

## 1. Numba-accelerated numeric kernel

The chunked numpy ray/segment intersection math (`_nonparallel_hit_distances`) was
re-expressed as a fused, `@njit(nogil=True, cache=True)` loop over
`(segment, ray)` pairs, avoiding the several full-size temporary matrices the numpy
version allocates.

| Metric | Result |
| --- | --- |
| Per-call kernel speed (361 rays x 3000 segments) | numba ~9-11 ms/call vs numpy ~16-18 ms/call, **~1.6-2.4x** |
| Threaded speedup (4 threads, simulating 4 parallel RL envs) | numba/nogil: **2.6-4.2x**; numpy-only: **1.3-1.5x** |
| Correctness | numba and numpy kernels agree to float64 precision (max diff ~2e-16) on every scenario tested |

The threaded-speedup gap (numba ~3-4x vs numpy ~1.3-1.5x on 4 threads) is the
practical payoff for parallel RL training: numpy's C-level ops release the GIL
inconsistently across the pipeline, while the numba kernel releases it for the
entire hot loop, so several environments stepped from a `ThreadPoolExecutor`
actually run concurrently instead of serializing.

`numba` is an optional dependency (`pip install ir-sim[fast]` / `ir-sim[all]`);
without it, the original pure-numpy kernel runs unchanged.

## 2. Per-beam motion skew

`Lidar2D(motion_skew=True)` interpolates each beam's sensor pose linearly between
the previous and current simulation state, at that beam's fraction of the sweep,
instead of every beam sharing one end-of-step pose. This required generalizing
`cast_ray_segments`/`cast_rays` to accept either a single shared origin (existing
fast path, unchanged) or one origin per ray.

- The per-ray case is a fused numba kernel that keeps O(rays) memory regardless of
  segment count (no `(segments, rays)` matrix at all), or a Python loop over the
  existing shared-origin kernel when numba isn't installed (correct, slower).
- Default `False`; a scan with it off is identical to the pre-existing single-pose
  behavior (regression test: `test_motion_skew_disabled_uses_single_shared_pose`).
- Verified against a hand-computed reference (linear position/heading
  interpolation) and shown to produce different, expected readings for a moving
  robot scanning a wall (`test_motion_skew_moving_robot_differs_from_instantaneous_scan`).

## 3. Segment gather cache

`SegmentCache` reuses the last `_gather_obstacle_edges` result (STRtree query +
`shapely.boundary` flattening) across `cast_rays` calls, as long as the query
origin has moved less than `cache_max_displacement` and the cache hasn't been
reused more than `cache_max_age_steps` times.

| Scenario (30 circular obstacles, 361 beams, stationary sensor) | Result |
| --- | --- |
| Uncached | ~8-11 ms/call |
| Cached (`max_displacement=0.05`) | ~4-8 ms/call |
| Speedup | **1.4-1.9x** |

**Known limitation (by design, documented in code and docs):** the cache is keyed
on the *sensor's* displacement, not on whether nearby objects moved. A dynamic
obstacle that drifts close to a stationary sensor is invisible until the cache
next refreshes. This makes the cache, as shipped, safe mainly for static or
slowly changing scenes -- see `report/multi_stage_ray_casting.md` for a design
that removes this limitation for the common case (a mostly-static map with a
handful of moving robots/obstacles), which is not yet implemented.

## Reproducing these numbers

```bash
uv sync --locked --all-extras --dev   # installs numba
uv run python benchmarks/bench_ray_casting.py
uv pip uninstall numba llvmlite
uv run python benchmarks/bench_ray_casting.py --skip-env-step   # numpy-fallback path
```

The same script runs in CI on a GitHub-hosted runner
(`.github/workflows/benchmark.yml`), both with and without `numba`, so these
checks aren't only self-reported from a developer machine.

## Test coverage added this session

- `tests/test_sensors.py`: 7 new tests covering motion-skew interpolation math,
  the no-skew regression path, moving-robot skew divergence, cache call-count
  reduction, cache correctness, cache displacement-invalidation, and
  `reset()` clearing skew/cache state.
- `benchmarks/bench_ray_casting.py`: correctness + speed checks for the numba
  kernel, the per-ray-origin kernel, and the segment cache, runnable standalone
  or in CI.
