# Evaluation: Multi-Stage Ray Casting (Static Walls First, Then Dynamic Objects)

**Date:** 2026-09-03
**Status:** Evaluated and benchmarked; **not yet implemented** in `irsim/`. This
report documents the idea, a working prototype used to measure it, and a
recommendation.

## The idea

`irsim.lib.algorithm.ray_casting_2d.cast_rays` currently gathers boundary
segments from *every* detected object in one pass
(`_gather_obstacle_edges`), then casts all beams against that combined segment
set. In a typical scene, most objects are static (walls, fixed obstacles, a
loaded map) and a few are dynamic (other robots, moving obstacles). Static
geometry never needs re-gathering once built; only the dynamic objects'
segments need to be fresh every step.

Splitting the gather into two stages -- static segments gathered once and
reused indefinitely (for a given sensor position), dynamic segments gathered
fresh every step -- means the per-step gather cost scales with the number of
*dynamic* objects only, not the whole scene.

This is a refinement of the `SegmentCache` mechanism added earlier this
session (see `report/ray_casting_performance.md`): that cache treats all
detected objects as one blob and is keyed only on sensor displacement, so it
is only safe for scenes with no nearby moving obstacles. Splitting by
`obj.static` (an existing `ObjectBase` attribute) removes that restriction for
the common case.

## Prototype and benchmark

A prototype (not merged) was built directly against the existing internals
(`_gather_obstacle_edges`, `_ray_parameters`, `cast_ray_segments`), exercising
exactly the code path `cast_rays` already uses, just split in two:

```python
static_start, static_end, static_owner = _gather_obstacle_edges(lidar_geometry, static_objs)  # once

def step():
    dyn_start, dyn_end, dyn_owner = _gather_obstacle_edges(lidar_geometry, dynamic_objs)  # every step
    seg_start = np.concatenate([static_start, dyn_start])
    seg_end = np.concatenate([static_end, dyn_end])
    origin, directions = _ray_parameters(lidar_geometry, max_range)
    return cast_ray_segments(origin, directions, seg_start, seg_end, max_range)
```

**Correctness:** the prototype's output was checked to match a full, fresh,
uncached `cast_rays` call over the combined object list exactly
(`np.testing.assert_allclose` / `assert_array_equal`, tolerance 1e-9) -- the
split changes nothing about *what* is computed, only *when* the static half is
re-gathered.

**Performance** (361-beam scan, stationary sensor, rectangular static
obstacles + circular dynamic obstacles, mean of 150-300 steps after JIT/cache
warmup):

| Static objects | Dynamic objects | Baseline (fresh gather every step) | Multi-stage (static cached) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 400 | 10 | 17.5 ms/step | 4.5 ms/step | **3.84x** |
| 100 | 10 | 4.9 ms/step | 1.8 ms/step | **2.74x** |
| 1000 | 10 | 37.1 ms/step | 6.5 ms/step | **5.69x** |
| 400 | 50 | 29.6 ms/step | 15.2 ms/step | **1.95x** |
| 400 | 100 | 40.8 ms/step | 27.3 ms/step | **1.49x** |

For reference, today's combined `SegmentCache` (`max_age_steps=1`) on the
400-static/10-dynamic scene measured 10.7 ms/step -- faster than the
uncached baseline, but **2.38x slower than the multi-stage split, and not
actually correctness-safe here**: at `max_age_steps=1` it still reuses the
cache every other call, so it would silently serve one-step-stale dynamic
obstacle positions half the time. The multi-stage split has no such caveat --
dynamic segments are gathered fresh on every single step, by construction.

## Interpretation

- **Speedup scales with the static:dynamic ratio.** A mostly-static map (a
  building, a warehouse) with a handful of moving robots -- the common
  navigation/RL scenario -- sees the largest win (up to ~5.7x at 1000
  static : 10 dynamic in this sweep). A scene where most objects are dynamic
  (e.g. dense multi-robot swarms with few walls) sees a much smaller benefit,
  since the per-step cost is then dominated by the dynamic gather, which this
  design doesn't speed up.
- **No correctness compromise**, unlike the existing combined `SegmentCache`:
  static geometry is genuinely immutable, so caching it forever (for a fixed
  sensor position) is exact, not an approximation. Dynamic objects are never
  stale.
- **Composable with the existing motion-skew and numba work**: the combined
  segment array (static + fresh dynamic) is just handed to the same
  `cast_ray_segments`, so it works unchanged with per-ray origins and the
  numba kernel from `report/ray_casting_performance.md`.

## Recommendation

Worth implementing as a follow-up: extend `SegmentCache` (or add a sibling
class) to gather `static` and non-`static` detected objects separately, cache
the static half keyed on sensor displacement (as today, since the sensor
still needs to re-query when it moves far enough for new static geometry to
enter range) with no `max_age_steps` staleness concern, and always gather the
dynamic half fresh. Suggested scope for that change:

1. `Lidar2D._get_detected_objects` (or a new helper) partitions
   `detected_objects` by `obj.static`.
2. `SegmentCache` (or a new `TieredSegmentCache`) caches the static partition
   exactly like today, and gathers the dynamic partition unconditionally on
   every `get()` call, then concatenates.
3. Default behavior stays unchanged (no cache configured); this is additive,
   like `motion_skew` and `cache_max_displacement` before it.
4. Needs the same reset-safety treatment as the current cache
   (`SegmentCache.invalidate()` on `Lidar2D.reset()`), plus invalidation when
   an object's `static` flag or the scene's object set changes (object
   add/remove), which the current sensor-displacement-only cache does not
   need to handle since it treats the whole scene as one unit.

This report intentionally stops at evaluation + prototype-level benchmarking,
per this session's task scope; implementation is left for a follow-up change.
