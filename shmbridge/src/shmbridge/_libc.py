"""Thin ctypes wrapper around the libc POSIX shm/fd calls."""

import ctypes
import sys

_libc = ctypes.CDLL(None, use_errno=True)

_libc.shm_open.restype = ctypes.c_int
_libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]

_libc.shm_unlink.restype = ctypes.c_int
_libc.shm_unlink.argtypes = [ctypes.c_char_p]

_libc.ftruncate.restype = ctypes.c_int
_libc.ftruncate.argtypes = [ctypes.c_int, ctypes.c_long]

# clock_gettime is needed for liveness timestamps (monotonic ns)
if sys.platform.startswith("linux"):
    CLOCK_MONOTONIC = 1

    class _Timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

    _libc.clock_gettime.restype = ctypes.c_int
    _libc.clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(_Timespec)]

    def _monotonic_ns() -> int:
        ts = _Timespec()
        _libc.clock_gettime(CLOCK_MONOTONIC, ctypes.byref(ts))
        return ts.tv_sec * 1_000_000_000 + ts.tv_nsec

elif sys.platform == "darwin":
    import time as _time

    def _monotonic_ns() -> int:
        return int(_time.monotonic_ns())
else:
    import time as _time

    def _monotonic_ns() -> int:
        return _time.monotonic_ns()
