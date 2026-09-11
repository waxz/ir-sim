"""CI helper: smoke-test cast_ray_segments_avx2 against the NumPy reference."""

import numpy as np

from irsim.lib.algorithm.ray_casting_2d import cast_ray_segments
from irsim.lib.algorithm.ray_casting_2d_omp import cast_ray_segments_avx2

rng = np.random.default_rng(42)
origin = np.array([0.0, 0.0])
angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
ss = rng.uniform(-5, 5, (30, 2))
se = rng.uniform(-5, 5, (30, 2))
max_range = 10.0

r_ref, h_ref = cast_ray_segments(origin, dirs, ss, se, max_range)
r_avx, h_avx = cast_ray_segments_avx2(origin, dirs, ss, se, max_range)

assert np.allclose(r_ref, r_avx, atol=1e-12), (
    f"Range mismatch: max_err={np.abs(r_ref - r_avx).max()}"
)
assert np.array_equal(h_ref, h_avx), "Hit-index mismatch"
print("PASS: results match NumPy reference")
