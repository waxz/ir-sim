"""
Benchmark ray-casting kernels and print a structured performance report.

Measures throughput (rays/sec) and latency (ms/scan) for:
  - NumPy reference kernel
  - OMP scalar kernel (if available)
  - AVX2 SIMD kernel (if available)

Workload: 360 rays x 200 segments, repeated REPS times.
"""

import platform
import sys
import timeit

import numpy as np

from irsim.lib.algorithm.ray_casting_2d import cast_ray_segments
from irsim.lib.algorithm.ray_casting_2d_omp import (
    cast_ray_segments_avx2,
    cast_ray_segments_omp,
    is_avx2_available,
    is_omp_available,
)

# ── Workload parameters ────────────────────────────────────────────────────────
N_RAYS = 360
N_SEGS = 200
MAX_RANGE = 10.0
WARMUP = 5
REPS = 200

rng = np.random.default_rng(42)
origin = np.array([0.0, 0.0])
angles = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
seg_start = rng.uniform(-8, 8, (N_SEGS, 2))
seg_end = rng.uniform(-8, 8, (N_SEGS, 2))


def bench(fn, *args):
    """Return (mean_ms, std_ms, rays_per_sec) over REPS timed calls."""
    # warmup
    for _ in range(WARMUP):
        fn(*args)

    times = timeit.repeat(lambda: fn(*args), number=1, repeat=REPS)
    arr = np.array(times) * 1e3  # seconds → milliseconds
    mean_ms = arr.mean()
    std_ms = arr.std()
    rays_per_sec = N_RAYS / (mean_ms * 1e-3)
    return mean_ms, std_ms, rays_per_sec


# ── Platform info ──────────────────────────────────────────────────────────────
print("=" * 62)
print("Ray-Casting Kernel Benchmark")
print("=" * 62)
print(f"Platform : {platform.system()} {platform.machine()}")
print(f"Python   : {sys.version.split()[0]}")
print(f"NumPy    : {np.__version__}")
print(f"OMP      : {is_omp_available()}")
print(f"AVX2     : {is_avx2_available()}")
print(f"Workload : {N_RAYS} rays x {N_SEGS} segments x {REPS} reps")
print("-" * 62)
print(f"{'Kernel':<18} {'mean ms':>10} {'± std':>8} {'Mrays/s':>10} {'speedup':>9}")
print("-" * 62)

results = {}

# NumPy reference
m, s, rps = bench(cast_ray_segments, origin, dirs, seg_start, seg_end, MAX_RANGE)
results["NumPy"] = (m, s, rps)
print(f"{'NumPy':<18} {m:>10.3f} {s:>8.3f} {rps/1e6:>10.3f} {'1.00x':>9}")

# OMP scalar
m, s, rps = bench(cast_ray_segments_omp, origin, dirs, seg_start, seg_end, MAX_RANGE)
results["OMP"] = (m, s, rps)
speedup = results["NumPy"][0] / m
tag = "" if is_omp_available() else " (fallback)"
print(f"{'OMP' + tag:<18} {m:>10.3f} {s:>8.3f} {rps/1e6:>10.3f} {speedup:>8.2f}x")

# AVX2 SIMD
m, s, rps = bench(cast_ray_segments_avx2, origin, dirs, seg_start, seg_end, MAX_RANGE)
results["AVX2"] = (m, s, rps)
speedup = results["NumPy"][0] / m
tag = "" if is_avx2_available() else " (fallback)"
print(f"{'AVX2' + tag:<18} {m:>10.3f} {s:>8.3f} {rps/1e6:>10.3f} {speedup:>8.2f}x")

print("=" * 62)

# Correctness check
r_ref, h_ref = cast_ray_segments(origin, dirs, seg_start, seg_end, MAX_RANGE)
r_omp, _ = cast_ray_segments_omp(origin, dirs, seg_start, seg_end, MAX_RANGE)
r_avx, _ = cast_ray_segments_avx2(origin, dirs, seg_start, seg_end, MAX_RANGE)
assert np.allclose(r_ref, r_omp, atol=1e-12), "OMP range mismatch"
assert np.allclose(r_ref, r_avx, atol=1e-12), "AVX2 range mismatch"
print("Correctness: PASS (all kernels agree to 1e-12)")
