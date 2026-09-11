"""CI helper: verify that the NumPy fallback works when no C extension is built."""

import numpy as np

from irsim.lib.algorithm.ray_casting_2d import cast_ray_segments
from irsim.lib.algorithm.ray_casting_2d_omp import (
    cast_ray_segments_avx2,
    cast_ray_segments_omp,
    is_avx2_available,
    is_omp_available,
)

print(f"OMP  available: {is_omp_available()}")
print(f"AVX2 available: {is_avx2_available()}")

rng = np.random.default_rng(7)
origin = np.array([0.0, 0.0])
angles = np.linspace(0, 2 * np.pi, 90, endpoint=False)
dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
ss = rng.uniform(-5, 5, (20, 2))
se = rng.uniform(-5, 5, (20, 2))
mr = 8.0

r_ref, h_ref = cast_ray_segments(origin, dirs, ss, se, mr)
r_omp, h_omp = cast_ray_segments_omp(origin, dirs, ss, se, mr)
r_avx, h_avx = cast_ray_segments_avx2(origin, dirs, ss, se, mr)

assert np.allclose(r_ref, r_omp, atol=1e-12), "OMP fallback mismatch"
assert np.allclose(r_ref, r_avx, atol=1e-12), "AVX2 fallback mismatch"
print("PASS: NumPy fallback is transparent")
