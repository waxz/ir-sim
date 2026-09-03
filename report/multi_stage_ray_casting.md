# Multi-Stage Ray Casting (Static Walls Cached, Dynamic Objects Gathered Fresh)

**Date:** 2026-09-03 (evaluated) / 2026-09-04 (implemented)
**Status:** Implemented as `TieredSegmentCache` in
`irsim/lib/algorithm/ray_casting_2d.py`, wired into `Lidar2D` via
`cache_split_static=True`. Opt-in; default behavior is unchanged.

## The idea

`cast_rays` gathers boundary segments from *every* detected object in one
pass (`_gather_obstacle_edges`), then casts all beams against that combined
segment set. In a typical scene, most objects are static (walls, fixed
obstacles, a loaded map) and a few are dynamic (other robots, moving
obstacles). Static geometry never needs re-gathering once built; only the
dynamic objects' segments need to be fresh every step.

This is a refinement of the `SegmentCache` mechanism from
`report/ray_casting_performance.md`: that cache treats all detected objects
as one blob and is keyed only on sensor displacement, so it is only
correctness-safe for scenes with no obstacles moving near the sensor.
Splitting by `obj.static` (an existing `ObjectBase` attribute) removes that
restriction for the common case.

## Implementation

`TieredSegmentCache` (`irsim/lib/algorithm/ray_casting_2d.py`) partitions
`detected_objects` into a static tier (truthy `.static`) and a dynamic tier
(everything else, including objects with no `.static` attribute at all --
the conservative default) on every call:

- **Static tier**: delegated to an internal `SegmentCache`, keyed on the
  query origin's displacement exactly like the combined cache. This is
  *exact*, not an approximation: static geometry never changes, so the only
  valid reason to re-gather it is the sensor moving far enough for different
  static objects to enter or leave detection range -- precisely what the
  displacement check re-triggers on.
- **Dynamic tier**: gathered unconditionally on every call via
  `_gather_obstacle_edges`, so moving objects are never stale.
- Owner indices from each tier are re-mapped back to positions in the
  original `detected_objects` list before the segments are concatenated, so
  `cast_rays`'s `hit_object_indices` output is unaffected by the split.

Enabled per-sensor via two new `Lidar2D` constructor arguments:

```yaml
sensors:
  - name: 'lidar2d'
    cache_max_displacement: 0.05   # as before: enables caching at all
    cache_max_age_steps: 8
    cache_split_static: True       # new: use the tiered cache
```

`cache_split_static=False` (default) keeps today's combined `SegmentCache`
behavior unchanged.

## Performance and correctness: three-way comparison

Benchmarked in `benchmarks/bench_ray_casting.py::bench_multi_stage_comparison`
(also runs in CI, `.github/workflows/benchmark.yml`) against a scene with 400
static box obstacles and 10 dynamic circular obstacles that **actually move
every step** (unlike a purely stationary-scene benchmark, this exercises the
exact case the combined cache is unsound for), a stationary 361-beam sensor,
150 steps. Correctness is measured as the maximum per-beam range deviation
from an always-fresh, uncached gather computed independently at every step
("ground truth"):

| Option | Time/step | Speedup vs. baseline | Max deviation from ground truth |
| --- | ---: | ---: | ---: |
| **Baseline** (no cache, fresh gather every step) | 14.66 ms | 1.00x | **0.0 m** (exact, by construction) |
| **Combined `SegmentCache`** (`max_age_steps=10`) | 3.44 ms | **4.26x** | **14.75 m** (wrong -- serves a stale, out-of-date obstacle position) |
| **`TieredSegmentCache`** (`max_age_steps=10`) | 3.96 ms | **3.70x** | **0.0 m** (exact) |

The combined cache's 14.75 m deviation is not a fluke or a tuning mistake --
it is the documented, structural tradeoff of caching a mixed static+dynamic
object list keyed only on sensor displacement: a stationary sensor never
invalidates the cache no matter how far the dynamic obstacles move, so a
scene with any moving obstacle near a slow or stationary sensor gets stale
readings under that cache. `TieredSegmentCache` closes this gap entirely
(exactly 0.0 m deviation, every step, by construction -- the dynamic tier is
never cached) while still recovering most of the combined cache's speedup
(3.70x vs. 4.26x here; the ~13% gap is the cost of partitioning objects into
two tiers and gathering the dynamic tier as a separate call each step).

A second sweep (single, stationary-obstacle scenes, no per-step obstacle
motion -- see "Prototype and earlier benchmark" below) additionally shows how
the tiered cache's advantage over a full re-gather scales with the
static:dynamic object ratio, from 1.49x (dynamic-heavy scenes) up to 5.69x
(static-heavy scenes, the common navigation/RL case).

### Test coverage

`tests/test_sensors.py` adds:

- `test_tiered_cache_never_misses_a_moving_obstacle`: a dynamic obstacle
  sweeps toward and away from a stationary sensor across 4 steps; the tiered
  cache's `range_data` matches an uncached lidar exactly at every step, and
  the static tier's internal age confirms it was gathered exactly once.
- `test_tiered_cache_beats_combined_cache_correctness_in_dynamic_scene`:
  reproduces the staleness gap above directly through the public `Lidar2D`
  API -- after an obstacle moves out of range, the combined cache still
  reports a hit (stale) while the tiered cache correctly reports none.

Both, plus the full existing suite, pass: **967 passed, 45 skipped, 0
failures** (`pytest tests/`), unchanged from before this feature (it is
purely additive; `cache_split_static` defaults to `False`).

## Interpretation

- **Speedup scales with the static:dynamic ratio and with the dynamic tier's
  gather cost.** A mostly-static map with a handful of moving robots -- the
  common navigation/RL scenario -- sees the largest win. A scene where most
  objects are dynamic sees a smaller benefit, since per-step cost is then
  dominated by the (never-cached-by-design) dynamic gather.
- **No correctness compromise for the static tier**, unlike the combined
  cache: static geometry is genuinely immutable, so caching it forever (for
  a fixed sensor position) is exact. The measured 0.0 m deviation across all
  150 steps of a moving-obstacle scene confirms this empirically, not just
  by argument.
- **Composable with the existing motion-skew and numba work**: the combined
  segment array (static + fresh dynamic) is handed to the same
  `cast_ray_segments`/numba kernel unchanged, so it works with per-ray
  origins from `motion_skew` and gets the same JIT speedup.

## Prototype and earlier benchmark (historical)

Before implementation, a standalone prototype (not merged) was built
directly against `_gather_obstacle_edges` / `_ray_parameters` /
`cast_ray_segments` to validate the idea and measure it across a range of
static:dynamic ratios on stationary-obstacle scenes:

| Static objects | Dynamic objects | Baseline | Multi-stage prototype | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 400 | 10 | 17.5 ms/step | 4.5 ms/step | 3.84x |
| 100 | 10 | 4.9 ms/step | 1.8 ms/step | 2.74x |
| 1000 | 10 | 37.1 ms/step | 6.5 ms/step | 5.69x |
| 400 | 50 | 29.6 ms/step | 15.2 ms/step | 1.95x |
| 400 | 100 | 40.8 ms/step | 27.3 ms/step | 1.49x |

These numbers guided the implementation but were produced by hand-rolled
script code, not the shipped `TieredSegmentCache` class; the three-way
comparison above supersedes them as the authoritative, moving-obstacle,
production-code benchmark.

## Reproducing these numbers

```bash
uv sync --locked --all-extras --dev
uv run python benchmarks/bench_ray_casting.py --skip-env-step
uv run pytest tests/test_sensors.py -k tiered
```
