"""Optional C+OpenMP-accelerated IMU integration algorithms.

Provides batch versions of the four 2-D dead-reckoning integrators
(Euler, Midpoint, RK4, Strapdown+Sculling) and an OpenMP-parallel
Monte Carlo runner as compiled C counterparts to the pure-Python
classes in :mod:`irsim.lib.algorithm.imu_pose_estimator`.

When the compiled extension is unavailable the module still imports
cleanly and :data:`is_available` returns ``False``.

Build
-----
The preferred path is the pre-compiled ``_imu_c_ext`` extension built
by setuptools / cibuildwheel and installed alongside the package.  When
that is absent (e.g. an sdist install without a C compiler), the first
call to :func:`ensure_built` tries a legacy ``gcc`` compile of
``imu_c_ext.c`` into ``imu_c_ext.so`` in the same directory.

Usage::

    from irsim.lib.algorithm.imu_c_integrators import (
        ensure_built,
        is_available,
        integrate_midpoint,
    )

    if ensure_built():
        px, py, theta = integrate_midpoint(dt, omega_arr, ax_arr, ay_arr)

All batch functions start the integrator at the zero state and return
NumPy arrays of shape ``(N,)`` for each output channel.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_C_SRC = _HERE / "imu_c_ext.c"
# Legacy output path (runtime gcc compile).
_SO_OUT = _HERE / "imu_c_ext.so"

_lib: ctypes.CDLL | None = None
_AVAILABLE: bool | None = None  # None = not yet probed


# ── Library loading ───────────────────────────────────────────────────────────


def _find_compiled_ext() -> Path | None:
    """Locate the setuptools-compiled ``_imu_c_ext`` extension."""
    import importlib.util

    spec = importlib.util.find_spec("irsim.lib.algorithm._imu_c_ext")
    if spec is not None and spec.origin:
        return Path(spec.origin)
    for pat in ("_imu_c_ext*.so", "_imu_c_ext*.pyd"):
        found = list(_HERE.glob(pat))
        if found:
            return found[0]
    return None


def _try_build(force: bool = False) -> bool:
    """Compile via gcc (legacy fallback).  Returns True on success."""
    if not force and _SO_OUT.exists():
        return True
    if not _C_SRC.exists():
        return False
    cmd = [
        "gcc",
        "-O3",
        "-march=native",
        "-ffast-math",
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
        # Try without OpenMP (serial fallback)
        cmd_serial = [
            "gcc",
            "-O3",
            "-march=native",
            "-ffast-math",
            "-shared",
            "-fPIC",
            str(_C_SRC),
            "-o",
            str(_SO_OUT),
            "-lm",
        ]
        try:
            subprocess.run(cmd_serial, check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


def _dbl_p(arr: np.ndarray) -> ctypes.POINTER:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _setup_lib(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Attach argtypes/restype signatures and return the library handle."""
    _batch_args = [
        ctypes.c_int,  # n
        ctypes.c_double,  # dt
        ctypes.POINTER(ctypes.c_double),  # omega[n]
        ctypes.POINTER(ctypes.c_double),  # ax[n]
        ctypes.POINTER(ctypes.c_double),  # ay[n]
        ctypes.POINTER(ctypes.c_double),  # out_px[n]
        ctypes.POINTER(ctypes.c_double),  # out_py[n]
        ctypes.POINTER(ctypes.c_double),  # out_th[n]
    ]
    for name in ("bench_euler", "bench_midpoint", "bench_rk4", "bench_strap"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = _batch_args

    lib.bench_mc_midpoint.restype = None
    lib.bench_mc_midpoint.argtypes = [
        ctypes.c_int,  # n_trials
        ctypes.c_int,  # n_steps
        ctypes.c_double,  # dt
        ctypes.POINTER(ctypes.c_double),  # omega[n_steps]
        ctypes.POINTER(ctypes.c_double),  # ax[n_steps]
        ctypes.POINTER(ctypes.c_double),  # ay[n_steps]
        ctypes.POINTER(ctypes.c_double),  # rmse_out[n_trials]
    ]
    return lib


def _load_lib() -> ctypes.CDLL | None:
    ext_path = _find_compiled_ext()
    if ext_path is not None:
        try:
            return _setup_lib(ctypes.CDLL(str(ext_path)))
        except OSError:
            pass
    if _SO_OUT.exists():
        try:
            return _setup_lib(ctypes.CDLL(str(_SO_OUT)))
        except OSError:
            pass
    return None


def ensure_built(force: bool = False) -> bool:
    """Load the compiled library, compiling via gcc if necessary.

    Args:
        force: Recompile the legacy gcc artefact even if it already exists.

    Returns:
        ``True`` if the C library is now available.
    """
    global _lib, _AVAILABLE
    if _AVAILABLE is not None and not force:
        return _AVAILABLE

    candidate = _load_lib()
    if candidate is not None:
        _lib = candidate
        _AVAILABLE = True
        return True

    ok = _try_build(force=force)
    if ok:
        _lib = _load_lib()
        _AVAILABLE = _lib is not None
    else:
        _AVAILABLE = False
    return bool(_AVAILABLE)


def is_available() -> bool:
    """Return ``True`` when the C extension is compiled and loaded."""
    if _AVAILABLE is None:
        ensure_built()
    return bool(_AVAILABLE)


# ── Public batch integration API ─────────────────────────────────────────────


def _check_available() -> None:
    if not is_available():
        raise RuntimeError(
            "imu_c_ext shared library not available. "
            "Call ensure_built() or install the package with a C compiler."
        )


def _prepare_inputs(
    omega: np.ndarray, ax: np.ndarray, ay: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    omega = np.ascontiguousarray(omega, dtype=np.float64)
    ax = np.ascontiguousarray(ax, dtype=np.float64)
    ay = np.ascontiguousarray(ay, dtype=np.float64)
    n = len(omega)
    if len(ax) != n or len(ay) != n:
        raise ValueError("omega, ax, ay must have equal length")
    return omega, ax, ay, n


def _run_batch(
    fn_name: str,
    dt: float,
    omega: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _check_available()
    omega, ax, ay, n = _prepare_inputs(omega, ax, ay)
    out_px = np.empty(n, dtype=np.float64)
    out_py = np.empty(n, dtype=np.float64)
    out_th = np.empty(n, dtype=np.float64)
    getattr(_lib, fn_name)(
        ctypes.c_int(n),
        ctypes.c_double(float(dt)),
        _dbl_p(omega),
        _dbl_p(ax),
        _dbl_p(ay),
        _dbl_p(out_px),
        _dbl_p(out_py),
        _dbl_p(out_th),
    )
    return out_px, out_py, out_th


def integrate_euler(
    dt: float,
    omega: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run N steps of 1st-order Euler integration in C.

    Args:
        dt: Fixed timestep (s).
        omega: Angular velocity measurements, shape ``(N,)``.
        ax: Body-frame x-acceleration measurements, shape ``(N,)``.
        ay: Body-frame y-acceleration measurements, shape ``(N,)``.

    Returns:
        ``(px, py, theta)`` — each a ``float64`` array of shape ``(N,)``.
    """
    return _run_batch("bench_euler", dt, omega, ax, ay)


def integrate_midpoint(
    dt: float,
    omega: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run N steps of 2nd-order midpoint integration in C.

    Args:
        dt: Fixed timestep (s).
        omega: Angular velocity measurements, shape ``(N,)``.
        ax: Body-frame x-acceleration measurements, shape ``(N,)``.
        ay: Body-frame y-acceleration measurements, shape ``(N,)``.

    Returns:
        ``(px, py, theta)`` — each a ``float64`` array of shape ``(N,)``.
    """
    return _run_batch("bench_midpoint", dt, omega, ax, ay)


def integrate_rk4(
    dt: float,
    omega: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run N steps of 4th-order Runge-Kutta integration in C.

    Args:
        dt: Fixed timestep (s).
        omega: Angular velocity measurements, shape ``(N,)``.
        ax: Body-frame x-acceleration measurements, shape ``(N,)``.
        ay: Body-frame y-acceleration measurements, shape ``(N,)``.

    Returns:
        ``(px, py, theta)`` — each a ``float64`` array of shape ``(N,)``.
    """
    return _run_batch("bench_rk4", dt, omega, ax, ay)


def integrate_strapdown(
    dt: float,
    omega: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run N steps of strapdown+sculling integration in C.

    Args:
        dt: Fixed timestep (s).
        omega: Angular velocity measurements, shape ``(N,)``.
        ax: Body-frame x-acceleration measurements, shape ``(N,)``.
        ay: Body-frame y-acceleration measurements, shape ``(N,)``.

    Returns:
        ``(px, py, theta)`` — each a ``float64`` array of shape ``(N,)``.
    """
    return _run_batch("bench_strap", dt, omega, ax, ay)


def monte_carlo_midpoint(
    n_trials: int,
    dt: float,
    omega: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
) -> np.ndarray:
    """Run N independent midpoint integrations in parallel (OpenMP).

    Each trial uses the same IMU input sequence starting from the zero
    state.  In a real application each trial would use independent noise
    samples drawn before calling this function.

    Args:
        n_trials: Number of independent trials.
        dt: Fixed timestep (s).
        omega: Angular velocity for one trial, shape ``(n_steps,)``.
        ax: Body-frame x-acceleration, shape ``(n_steps,)``.
        ay: Body-frame y-acceleration, shape ``(n_steps,)``.

    Returns:
        ``rmse`` — RMSE of position from origin per trial, shape ``(n_trials,)``.
    """
    _check_available()
    omega, ax, ay, n_steps = _prepare_inputs(omega, ax, ay)
    rmse = np.empty(n_trials, dtype=np.float64)
    _lib.bench_mc_midpoint(
        ctypes.c_int(n_trials),
        ctypes.c_int(n_steps),
        ctypes.c_double(float(dt)),
        _dbl_p(omega),
        _dbl_p(ax),
        _dbl_p(ay),
        _dbl_p(rmse),
    )
    return rmse
