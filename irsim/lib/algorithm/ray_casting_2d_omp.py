"""Optional C+OpenMP-accelerated ray-casting kernels.

Drop-in replacement for :func:`~irsim.lib.algorithm.ray_casting_2d.cast_ray_segments`
when a C compiler with OpenMP is available.  If compilation fails the module still
imports cleanly and :func:`cast_ray_segments_omp` falls back to the NumPy kernel.

Two kernels are available:

* :func:`cast_ray_segments_omp` — scalar OpenMP (AoS layout); always available
  when the C library is loaded.
* :func:`cast_ray_segments_avx2` — AVX2 SIMD 4-wide OpenMP (SoA layout);
  available only when the compiled library exposes
  ``cast_ray_segments_avx2_soa`` (requires ``-mavx2`` or ``-march=native`` on
  an AVX2-capable CPU).  Falls back to :func:`cast_ray_segments_omp` when
  unavailable.

Build
-----
The preferred path is the pre-compiled ``_ray_casting_omp`` extension built by
setuptools / cibuildwheel and installed alongside the package.  When the extension
is absent (e.g. an sdist install on a machine without a compiler), the first call
to :func:`ensure_built` attempts a legacy ``gcc`` compile of ``ray_casting_omp.c``
into ``ray_casting_omp.so`` in the same directory.

Usage::

    from irsim.lib.algorithm.ray_casting_2d_omp import (
        cast_ray_segments_avx2,
        cast_ray_segments_omp,
        is_avx2_available,
        is_omp_available,
        ensure_built,
    )

    if is_avx2_available():
        ranges, hits = cast_ray_segments_avx2(
            origin, directions, seg_start, seg_end, max_range
        )
    elif is_omp_available():
        ranges, hits = cast_ray_segments_omp(
            origin, directions, seg_start, seg_end, max_range
        )
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_C_SRC = _HERE / "ray_casting_omp.c"
# Legacy output path (runtime gcc compile); kept for backward compatibility.
_SO_OUT = _HERE / "ray_casting_omp.so"

_lib: ctypes.CDLL | None = None
_OMP_AVAILABLE: bool | None = None  # None = not yet probed
_AVX2_AVAILABLE: bool = False


def _find_compiled_ext() -> Path | None:
    """Locate the setuptools-compiled ``_ray_casting_omp`` extension.

    Checks the installed package first (via importlib), then looks for an
    in-tree build artifact next to this file.
    """
    import importlib.util

    spec = importlib.util.find_spec("irsim.lib.algorithm._ray_casting_omp")
    if spec is not None and spec.origin:
        return Path(spec.origin)
    # In-tree / editable install: search for the compiled file beside this module.
    for pat in ("_ray_casting_omp*.so", "_ray_casting_omp*.pyd"):
        found = list(_HERE.glob(pat))
        if found:
            return found[0]
    return None


def _try_build(force: bool = False) -> bool:
    """Compile the C source via gcc (legacy fallback).  Returns True on success."""
    if not force and _SO_OUT.exists():
        return True
    if not _C_SRC.exists():
        return False
    cmd = [
        "gcc",
        "-O3",
        "-march=native",
        "-fopenmp",
        "-shared",
        "-fPIC",
        str(_C_SRC),
        "-o",
        str(_SO_OUT),
        "-lm",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _setup_lib(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Attach argtypes/restype to *lib* and return it."""
    global _AVX2_AVAILABLE
    lib.cast_ray_segments_omp.restype = None
    lib.cast_ray_segments_omp.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # origin
        ctypes.POINTER(ctypes.c_double),  # directions
        ctypes.POINTER(ctypes.c_double),  # seg_start
        ctypes.POINTER(ctypes.c_double),  # seg_end
        ctypes.c_int,  # N
        ctypes.c_int,  # M
        ctypes.c_double,  # max_range
        ctypes.POINTER(ctypes.c_double),  # out_ranges
        ctypes.POINTER(ctypes.c_int64),  # out_hit
    ]
    # Wire up AVX2 SoA function when the compiled library exposes it
    try:
        fn = lib.cast_ray_segments_avx2_soa
        fn.restype = None
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # origin
            ctypes.POINTER(ctypes.c_double),  # dir_dx
            ctypes.POINTER(ctypes.c_double),  # dir_dy
            ctypes.POINTER(ctypes.c_double),  # seg_sx
            ctypes.POINTER(ctypes.c_double),  # seg_sy
            ctypes.POINTER(ctypes.c_double),  # seg_ex
            ctypes.POINTER(ctypes.c_double),  # seg_ey
            ctypes.c_int,  # N
            ctypes.c_int,  # M
            ctypes.c_double,  # max_range
            ctypes.POINTER(ctypes.c_double),  # out_ranges
            ctypes.POINTER(ctypes.c_int64),  # out_hit
        ]
        _AVX2_AVAILABLE = True
    except AttributeError:
        _AVX2_AVAILABLE = False
    return lib


def _load_lib() -> ctypes.CDLL | None:
    """Load the compiled shared library and set up argtypes.

    Tries the setuptools extension first, then the legacy runtime-compiled .so.
    """
    # 1) setuptools-compiled extension (preferred, works on all platforms)
    ext_path = _find_compiled_ext()
    if ext_path is not None:
        try:
            return _setup_lib(ctypes.CDLL(str(ext_path)))
        except OSError:
            pass
    # 2) Legacy .so compiled by _try_build()
    if _SO_OUT.exists():
        try:
            return _setup_lib(ctypes.CDLL(str(_SO_OUT)))
        except OSError:
            pass
    return None


def build_omp_lib(force: bool = False) -> bool:
    """Build (or rebuild) the OpenMP shared library.

    Attempts to use the setuptools-compiled extension first.  If unavailable,
    falls back to a runtime ``gcc`` compile.

    Args:
        force: Recompile the legacy gcc artefact even if it already exists.
               Has no effect on the setuptools extension.

    Returns:
        ``True`` if the library is available (pre-compiled or just built).
    """
    global _lib, _OMP_AVAILABLE
    # Try to load an already-available library (setuptools ext or legacy .so)
    candidate = _load_lib()
    if candidate is not None:
        _lib = candidate
        _OMP_AVAILABLE = True
        return True
    # Fall back to runtime gcc compile (Linux only; ignored on Windows/macOS)
    ok = _try_build(force=force)
    if ok:
        _lib = _load_lib()
        _OMP_AVAILABLE = _lib is not None
    else:
        _OMP_AVAILABLE = False
    if not _OMP_AVAILABLE:
        global _AVX2_AVAILABLE
        _AVX2_AVAILABLE = False
    return bool(_OMP_AVAILABLE)


def ensure_built() -> bool:
    """Build if not already done.  Returns availability."""
    global _OMP_AVAILABLE
    if _OMP_AVAILABLE is None:
        build_omp_lib()
    return bool(_OMP_AVAILABLE)


def is_omp_available() -> bool:
    """Return ``True`` when the OpenMP kernel is compiled and loaded."""
    if _OMP_AVAILABLE is None:
        ensure_built()
    return bool(_OMP_AVAILABLE)


def is_avx2_available() -> bool:
    """Return ``True`` when the AVX2 SoA kernel is compiled and loaded."""
    if _OMP_AVAILABLE is None:
        ensure_built()
    return _AVX2_AVAILABLE


def _np_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _i64_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def cast_ray_segments_omp(
    origin: np.ndarray,
    directions: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    max_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """C+OpenMP ray-segment intersection -- same interface as :func:`cast_ray_segments`.

    Falls back to the NumPy kernel when the C library is unavailable.

    Args:
        origin: Shared ray origin ``(2,)``.
        directions: Unit ray directions ``(N, 2)``.
        seg_start: Segment start points ``(M, 2)``.
        seg_end: Segment end points ``(M, 2)``.
        max_range: Maximum ray length; misses return this.

    Returns:
        ``(ranges, hit_index)`` -- same semantics as
        :func:`~irsim.lib.algorithm.ray_casting_2d.cast_ray_segments`.
    """
    if not is_omp_available():
        from irsim.lib.algorithm.ray_casting_2d import cast_ray_segments

        return cast_ray_segments(origin, directions, seg_start, seg_end, max_range)

    if len(seg_start) == 0:
        return (
            np.full(len(directions), max_range, dtype=np.float64),
            np.full(len(directions), -1, dtype=np.int64),
        )

    N = len(directions)
    M = len(seg_start)

    origin_c = np.ascontiguousarray(origin, dtype=np.float64)
    directions_c = np.ascontiguousarray(directions, dtype=np.float64)
    seg_start_c = np.ascontiguousarray(seg_start, dtype=np.float64)
    seg_end_c = np.ascontiguousarray(seg_end, dtype=np.float64)

    out_ranges = np.empty(N, dtype=np.float64)
    out_hit = np.empty(N, dtype=np.int64)

    _lib.cast_ray_segments_omp(
        _np_ptr(origin_c),
        _np_ptr(directions_c),
        _np_ptr(seg_start_c),
        _np_ptr(seg_end_c),
        ctypes.c_int(N),
        ctypes.c_int(M),
        ctypes.c_double(float(max_range)),
        _np_ptr(out_ranges),
        _i64_ptr(out_hit),
    )
    return out_ranges, out_hit.astype(int)


def cast_ray_segments_avx2(
    origin: np.ndarray,
    directions: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    max_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """C+OpenMP+AVX2 SoA ray-segment intersection.

    Processes 4 beams simultaneously using 256-bit double-precision registers.
    Converts AoS ``directions`` / ``seg_start`` / ``seg_end`` to SoA layout
    internally before calling the C kernel.

    Falls back to :func:`cast_ray_segments_omp` when the AVX2 kernel is
    unavailable.

    Args:
        origin: Shared ray origin ``(2,)``.
        directions: Unit ray directions ``(N, 2)``.
        seg_start: Segment start points ``(M, 2)``.
        seg_end: Segment end points ``(M, 2)``.
        max_range: Maximum ray length; misses return this.

    Returns:
        ``(ranges, hit_index)`` with the same semantics as
        :func:`~irsim.lib.algorithm.ray_casting_2d.cast_ray_segments`.
    """
    if not _AVX2_AVAILABLE or _lib is None:
        return cast_ray_segments_omp(origin, directions, seg_start, seg_end, max_range)

    if len(seg_start) == 0:
        return (
            np.full(len(directions), max_range, dtype=np.float64),
            np.full(len(directions), -1, dtype=np.int64),
        )

    N = len(directions)
    M = len(seg_start)

    origin_c = np.ascontiguousarray(origin, dtype=np.float64)
    directions_c = np.ascontiguousarray(directions, dtype=np.float64)
    seg_start_c = np.ascontiguousarray(seg_start, dtype=np.float64)
    seg_end_c = np.ascontiguousarray(seg_end, dtype=np.float64)

    # AoS → SoA conversion
    dir_dx = np.ascontiguousarray(directions_c[:, 0])
    dir_dy = np.ascontiguousarray(directions_c[:, 1])
    seg_sx = np.ascontiguousarray(seg_start_c[:, 0])
    seg_sy = np.ascontiguousarray(seg_start_c[:, 1])
    seg_ex = np.ascontiguousarray(seg_end_c[:, 0])
    seg_ey = np.ascontiguousarray(seg_end_c[:, 1])

    out_ranges = np.empty(N, dtype=np.float64)
    out_hit = np.empty(N, dtype=np.int64)

    _lib.cast_ray_segments_avx2_soa(
        _np_ptr(origin_c),
        _np_ptr(dir_dx),
        _np_ptr(dir_dy),
        _np_ptr(seg_sx),
        _np_ptr(seg_sy),
        _np_ptr(seg_ex),
        _np_ptr(seg_ey),
        ctypes.c_int(N),
        ctypes.c_int(M),
        ctypes.c_double(float(max_range)),
        _np_ptr(out_ranges),
        _i64_ptr(out_hit),
    )
    return out_ranges, out_hit.astype(int)
