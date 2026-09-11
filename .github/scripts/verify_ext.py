"""CI helper: verify that OMP and AVX2 availability match expectations.

Usage:
    python verify_ext.py <expect_omp> <expect_avx2>

Arguments must be the strings "true" or "false".
"""

import sys

from irsim.lib.algorithm.ray_casting_2d_omp import (
    is_avx2_available,
    is_omp_available,
)

omp = is_omp_available()
avx2 = is_avx2_available()
expect_omp = sys.argv[1] == "true"
expect_avx2 = sys.argv[2] == "true"

print(f"OMP  available: {omp}  (expected: {expect_omp})")
print(f"AVX2 available: {avx2} (expected: {expect_avx2})")

ok = True
if omp != expect_omp:
    print(f"FAIL: OMP mismatch — got {omp}, expected {expect_omp}")
    ok = False
if avx2 != expect_avx2:
    print(f"FAIL: AVX2 mismatch — got {avx2}, expected {expect_avx2}")
    ok = False

if not ok:
    sys.exit(1)
print("PASS")
